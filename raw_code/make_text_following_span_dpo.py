import argparse
import ast
import json
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset
from tqdm import tqdm


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


def make_key(question, answers, full_prompt=None):
    answers = parse_answers(answers)
    answer_key = "||".join(sorted(a.strip().lower() for a in answers))
    prompt_key = (full_prompt or "").strip()
    return f"{question.strip()}###ANS={answer_key}###PROMPT={prompt_key}"


def load_text_following_error_rows(path):
    rows = []
    skipped = {"correct": 0, "empty_pred": 0, "pred_not_in_aux": 0, "gt_like_pred": 0}
    with open(path, "r", encoding="utf-8") as f:
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
            row["misleading_answer"] = pred
            row["aux_text"] = aux_text
            rows.append(row)
    return rows, skipped


def pick_split(dataset, subset=""):
    if subset:
        if subset not in dataset:
            raise KeyError(f"Subset/split {subset!r} not found. Available: {list(dataset.keys())}")
        return dataset[subset]
    for name in ["validation", "test", "train"]:
        if name in dataset:
            return dataset[name]
    return dataset[list(dataset.keys())[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ds_name", default="dal-289/word_or_vision")
    parser.add_argument("--subset", default="")
    parser.add_argument("--text_type", default="corrupted")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=0)
    args = parser.parse_args()

    error_rows, skipped = load_text_following_error_rows(args.eval_file)
    if args.max_examples:
        error_rows = error_rows[: args.max_examples]

    wanted = {
        make_key(r.get("question", ""), r.get("gt_answers", []), r.get("full_prompt")): r
        for r in error_rows
    }
    if not wanted:
        raise RuntimeError("No text-following incorrect eval rows found.")

    ds = pick_split(load_dataset(args.ds_name), args.subset)
    if args.text_type:
        ds = ds.filter(lambda x: x.get("text_type") == args.text_type)

    examples = []
    for sample in tqdm(ds, desc="match text-following errors to dataset"):
        key = make_key(
            str(sample.get("question", "")),
            sample.get("answers") or sample.get("gt_answers") or sample.get("label"),
            sample.get("full_prompt"),
        )
        eval_row = wanted.get(key)
        if eval_row is None:
            continue

        answers = parse_answers(eval_row.get("gt_answers"))
        if not answers:
            continue
        chosen = answers[0].strip()
        rejected = str(eval_row.get("misleading_answer", "")).strip()
        if not chosen or not rejected or answer_match(rejected, [chosen]):
            continue

        examples.append(
            {
                "question_id": str(eval_row.get("question_id", "")),
                "text_type": args.text_type,
                "prompt": eval_row.get("full_prompt") or eval_row.get("question", ""),
                "chosen": chosen,
                "rejected": rejected,
                "misleading_answer": rejected,
                "sample_weight": 1.0,
                "image": sample["image"],
            }
        )

    if not examples:
        raise RuntimeError("No examples matched between text-following errors and dataset rows.")

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
    print(f"text-following rows: {len(error_rows)}")
    print(f"matched examples: {len(examples)}")
    print(f"skipped: {skipped}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
