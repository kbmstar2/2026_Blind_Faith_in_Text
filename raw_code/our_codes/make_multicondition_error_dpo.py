import argparse
import ast
import json
import re
from collections import Counter, defaultdict
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
    for ans in parse_answers(answers):
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


def strip_aux_prompt(prompt, question):
    sep = "\n----------\n"
    if prompt and sep in prompt:
        return prompt.split(sep, 1)[1].strip()
    return f"{str(question).strip()} Please only output the answer with a single word or phrase."


def make_key(question, answers, full_prompt=None):
    answer_key = "||".join(sorted(a.strip().lower() for a in parse_answers(answers)))
    return f"{str(question).strip()}###ANS={answer_key}###PROMPT={(full_prompt or '').strip()}"


def load_error_rows(path, condition, require_pred_in_aux=False):
    rows = []
    skipped = Counter()
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
            if not answers:
                skipped["no_gt"] += 1
                continue
            if answer_match(pred, answers):
                skipped["gt_like_pred"] += 1
                continue
            aux_text = extract_aux_text(row.get("full_prompt", ""))
            pred_in_aux = answer_in_text(pred, aux_text)
            if require_pred_in_aux and not pred_in_aux:
                skipped["pred_not_in_aux"] += 1
                continue
            row["condition"] = condition
            row["misleading_answer"] = pred
            row["aux_text"] = aux_text
            row["pred_in_aux"] = pred_in_aux
            rows.append(row)
    return rows, skipped


def pick_split(dataset, subset):
    if subset and subset in dataset:
        return dataset[subset]
    for name in ["validation", "test", "train"]:
        if name in dataset:
            return dataset[name]
    return dataset[list(dataset.keys())[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ds_name", default="dal-289/word_or_vision")
    parser.add_argument("--subset", default="DocVQA")
    parser.add_argument("--corrupted_eval", required=True)
    parser.add_argument("--match_eval", default="")
    parser.add_argument("--irrelevant_eval", default="")
    parser.add_argument("--no_text_eval", default="")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corrupted_weight", type=float, default=1.0)
    parser.add_argument("--match_weight", type=float, default=0.25)
    parser.add_argument("--irrelevant_weight", type=float, default=0.5)
    parser.add_argument("--no_text_weight", type=float, default=0.5)
    args = parser.parse_args()

    condition_specs = [
        ("corrupted", args.corrupted_eval, args.corrupted_weight, True),
        ("match", args.match_eval, args.match_weight, False),
        ("irrelevant", args.irrelevant_eval, args.irrelevant_weight, False),
        ("no_text", args.no_text_eval, args.no_text_weight, False),
    ]

    eval_rows = []
    skipped_by_condition = {}
    for condition, path, weight, require_pred_in_aux in condition_specs:
        if not path:
            continue
        rows, skipped = load_error_rows(path, condition, require_pred_in_aux=require_pred_in_aux)
        for row in rows:
            row["sample_weight"] = weight
        eval_rows.extend(rows)
        skipped_by_condition[condition] = dict(skipped)

    if not eval_rows:
        raise RuntimeError("No condition-specific error rows found.")

    ds = pick_split(load_dataset(args.ds_name), args.subset)
    by_prompt_key = {}
    by_question_answer = defaultdict(dict)
    for sample in tqdm(ds, desc="index dataset"):
        text_type = sample.get("text_type", "")
        answers = sample.get("answers") or sample.get("gt_answers") or sample.get("label")
        key = make_key(sample.get("question", ""), answers, sample.get("full_prompt"))
        by_prompt_key[key] = sample
        qa_key = make_key(sample.get("question", ""), answers, "")
        by_question_answer[qa_key][text_type] = sample

    examples = []
    missing = Counter()
    source_counts = Counter()
    pred_in_aux_counts = Counter()

    for row in eval_rows:
        condition = row["condition"]
        answers = parse_answers(row.get("gt_answers"))
        chosen = answers[0].strip() if answers else ""
        rejected = str(row.get("misleading_answer", "")).strip()
        if not chosen or not rejected or answer_match(rejected, [chosen]):
            continue

        prompt = row.get("full_prompt") or row.get("question", "")
        key = make_key(row.get("question", ""), row.get("gt_answers", []), row.get("full_prompt"))
        sample = by_prompt_key.get(key)

        if condition == "no_text":
            qa_key = make_key(row.get("question", ""), row.get("gt_answers", []), "")
            group = by_question_answer.get(qa_key, {})
            sample = group.get("corrupted") or group.get("match") or group.get("irrelevant") or sample
            prompt = strip_aux_prompt(row.get("full_prompt"), row.get("question", ""))

        if sample is None:
            missing[condition] += 1
            continue

        weight = float(row.get("sample_weight", 1.0))
        if condition == "irrelevant" and row.get("pred_in_aux"):
            weight = max(weight, 0.75)

        examples.append(
            {
                "question_id": str(row.get("question_id", "")),
                "text_type": condition,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "misleading_answer": rejected,
                "sample_weight": weight,
                "image": sample["image"],
            }
        )
        source_counts[condition] += 1
        if row.get("pred_in_aux"):
            pred_in_aux_counts[condition] += 1

    if not examples:
        raise RuntimeError("No examples matched to dataset images.")

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
    print(f"examples: {len(examples)}")
    print(f"source_counts: {dict(source_counts)}")
    print(f"pred_in_aux_counts: {dict(pred_in_aux_counts)}")
    print(f"missing: {dict(missing)}")
    print(f"skipped_by_condition: {skipped_by_condition}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
