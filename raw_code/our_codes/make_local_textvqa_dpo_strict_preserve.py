import argparse
import ast
import json
import re
from collections import Counter, defaultdict
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


def exactish_hint_match(answer, hint):
    ans = normalize(answer)
    h = normalize(hint)
    if not ans or not h:
        return False
    if ans == h:
        return True
    toks = ans.split()
    htoks = set(h.split())
    return len(toks) > 1 and sum(tok in htoks for tok in toks) / len(toks) >= 0.8


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


def make_key(question, answers, full_prompt):
    answer_key = "||".join(sorted(a.strip().lower() for a in parse_answers(answers)))
    return f"{str(question).strip()}###ANS={answer_key}###PROMPT={(full_prompt or '').strip()}"


def extract_hint(row):
    info = row.get("additional_info") or {}
    if isinstance(info, dict):
        wrong = info.get("wrong_answer") or ""
        qtype = info.get("question_type") or ""
    else:
        wrong = ""
        qtype = ""
    if not wrong:
        prompt = row.get("full_prompt", "")
        match = re.search(r'predicts the answer is: "(.*?)"', prompt)
        wrong = match.group(1) if match else ""
    return str(wrong).strip(), str(qtype).strip()


def row_to_example(row, sample, chosen, rejected, kind, weight):
    return {
        "question_id": str(row.get("question_id", "")),
        "text_type": f"corrupted:{kind}",
        "prompt": row.get("full_prompt") or sample.get("full_prompt") or sample.get("question", ""),
        "chosen": str(chosen).strip(),
        "rejected": str(rejected).strip(),
        "misleading_answer": str(rejected).strip(),
        "pair_kind": kind,
        "question_type": str(extract_hint(row)[1]),
        "sample_weight": float(weight),
        "image": sample["image"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preserve_weight", type=float, default=0.5)
    parser.add_argument("--max_preserve_per_correction", type=float, default=1.0)
    parser.add_argument("--exclude_qtypes", default="")
    args = parser.parse_args()

    excluded = {q.strip() for q in args.exclude_qtypes.split(",") if q.strip()}
    rows = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    wanted = {
        make_key(row.get("question", ""), row.get("gt_answers", []), row.get("full_prompt", "")): row
        for row in rows
    }
    ds = load_from_disk(args.dataset_dir)

    corrections = []
    preserve_candidates = []
    skipped = defaultdict(int)

    for sample in ds:
        key = make_key(sample.get("question", ""), sample.get("answers", []), sample.get("full_prompt", ""))
        row = wanted.get(key)
        if row is None:
            continue
        hint, qtype = extract_hint(row)
        if qtype in excluded:
            skipped["excluded_qtype"] += 1
            continue
        answers = parse_answers(row.get("gt_answers"))
        chosen = choose_chosen(answers)
        if not chosen or answer_match(hint, [chosen]):
            skipped["bad_chosen_or_hint_matches_gt"] += 1
            continue

        pred = str(row.get("pred_answer", "")).strip()
        if row.get("is_correct") is False:
            if not pred:
                skipped["empty_pred"] += 1
                continue
            if answer_match(pred, answers):
                skipped["gt_like_pred"] += 1
                continue
            if not exactish_hint_match(pred, hint):
                skipped["rejected_not_strict_hint"] += 1
                continue
            corrections.append(row_to_example(row, sample, chosen, pred, "strict_error", 1.0))
        elif row.get("is_correct") is True:
            # Teach the model to preserve a base-correct visual answer over a plausible corrupted hint.
            good = pred if answer_match(pred, answers) else chosen
            if not good or answer_match(hint, [good]):
                skipped["bad_preserve"] += 1
                continue
            if not answer_in_text(hint, row.get("full_prompt", "")):
                skipped["hint_missing"] += 1
                continue
            preserve_candidates.append(
                row_to_example(row, sample, good, hint, "base_correct_preserve", args.preserve_weight)
            )
        else:
            skipped["unknown_correctness"] += 1

    preserve_limit = int(len(corrections) * args.max_preserve_per_correction)
    preserve = preserve_candidates[:preserve_limit] if preserve_limit else []
    examples = corrections + preserve
    if not examples:
        raise RuntimeError(f"No examples created. skipped={dict(skipped)}")

    features = Features(
        {
            "question_id": Value("string"),
            "text_type": Value("string"),
            "prompt": Value("string"),
            "chosen": Value("string"),
            "rejected": Value("string"),
            "misleading_answer": Value("string"),
            "pair_kind": Value("string"),
            "question_type": Value("string"),
            "sample_weight": Value("float32"),
            "image": Image(),
        }
    )
    dataset = Dataset.from_list(examples, features=features).shuffle(seed=args.seed)
    split = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    out = DatasetDict({"train": split["train"], "test": split["test"]})

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_dir))
    print(out)
    print(f"strict correction pairs: {len(corrections)}")
    print(f"preservation candidates: {len(preserve_candidates)}")
    print(f"used preservation pairs: {len(preserve)}")
    print(f"total pairs: {len(examples)}")
    print(f"skipped: {dict(skipped)}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
