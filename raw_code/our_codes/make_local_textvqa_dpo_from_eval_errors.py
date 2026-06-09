import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, load_from_disk


def normalize(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_answers(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        return [value]
    return [str(value)]


def answer_match(pred, answers):
    pred_norm = normalize(pred)
    if not pred_norm:
        return False
    for ans in answers:
        ans_norm = normalize(ans)
        if ans_norm and (pred_norm == ans_norm or ans_norm in pred_norm or pred_norm in ans_norm):
            return True
    return False


def answer_in_text(answer, text):
    ans = normalize(answer)
    ctx = normalize(text)
    if not ans or not ctx:
        return False
    if ans in ctx:
        return True
    toks = ans.split()
    if len(toks) <= 1:
        return False
    ctx_toks = set(ctx.split())
    return sum(tok in ctx_toks for tok in toks) / len(toks) >= 0.8


def extract_aux_text(prompt):
    marker = "please use it with caution:"
    sep = "\n----------\n"
    prompt = prompt or ""
    lower = prompt.lower()
    start = lower.find(marker)
    rest = prompt[start + len(marker) :] if start >= 0 else prompt
    end = rest.find(sep)
    return rest[:end].strip() if end >= 0 else rest.strip()


def make_key(question, answers, full_prompt):
    answer_key = "||".join(sorted(a.strip().lower() for a in parse_answers(answers)))
    return f"{str(question).strip()}###ANS={answer_key}###PROMPT={(full_prompt or '').strip()}"


def choose_chosen(answers):
    valid = [a.strip() for a in parse_answers(answers) if normalize(a)]
    if not valid:
        return ""
    normalized = [normalize(a) for a in valid]
    majority = Counter(normalized).most_common(1)[0][0]
    for raw in valid:
        if normalize(raw) == majority:
            return raw
    return valid[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    eval_rows = []
    skipped = {"correct": 0, "empty_pred": 0, "pred_not_in_aux": 0, "gt_like_pred": 0}
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("is_correct") is not False:
                skipped["correct"] += 1
                continue
            pred = str(row.get("pred_answer", "")).strip()
            if not pred or pred.lower().startswith("error"):
                skipped["empty_pred"] += 1
                continue
            answers = parse_answers(row.get("gt_answers"))
            if answer_match(pred, answers):
                skipped["gt_like_pred"] += 1
                continue
            aux_text = extract_aux_text(row.get("full_prompt", ""))
            if not answer_in_text(pred, aux_text):
                skipped["pred_not_in_aux"] += 1
                continue
            row["aux_text"] = aux_text
            row["misleading_answer"] = pred
            eval_rows.append(row)

    if args.max_examples:
        eval_rows = eval_rows[: args.max_examples]
    wanted = {
        make_key(row.get("question", ""), row.get("gt_answers", []), row.get("full_prompt", "")): row
        for row in eval_rows
    }
    if not wanted:
        raise RuntimeError(f"No text-following errors found. skipped={skipped}")

    ds = load_from_disk(args.dataset_dir)
    examples = []
    for sample in ds:
        key = make_key(sample.get("question", ""), sample.get("answers", []), sample.get("full_prompt", ""))
        row = wanted.get(key)
        if row is None:
            continue
        chosen = choose_chosen(row.get("gt_answers", []))
        rejected = str(row.get("misleading_answer", "")).strip()
        if not chosen or not rejected or answer_match(rejected, [chosen]):
            continue
        examples.append(
            {
                "question_id": str(row.get("question_id", "")),
                "text_type": str(sample.get("text_type", "corrupted")),
                "prompt": row.get("full_prompt") or sample.get("full_prompt") or sample.get("question", ""),
                "chosen": chosen,
                "rejected": rejected,
                "misleading_answer": rejected,
                "sample_weight": 1.0,
                "image": sample["image"],
            }
        )

    if not examples:
        raise RuntimeError("No matched DPO examples.")

    features = Features(
        {
            "question_id": Value("string"),
            "text_type": Value("string"),
            "prompt": Value("string"),
            "chosen": Value("string"),
            "rejected": Value("string"),
            "misleading_answer": Value("string"),
            "sample_weight": Value("float32"),
            "image": Image(),
        }
    )
    dataset = Dataset.from_list(examples, features=features).shuffle(seed=args.seed)
    if args.test_size > 0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
        out = DatasetDict({"train": split["train"], "test": split["test"]})
    else:
        out = DatasetDict({"train": dataset})

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_dir))
    print(out)
    print(f"text-following eval rows: {len(eval_rows)}")
    print(f"matched DPO examples: {len(examples)}")
    print(f"skipped: {skipped}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
