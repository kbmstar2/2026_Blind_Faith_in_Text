import argparse
import json
import re
import string
from pathlib import Path


def normalize(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_aux_text(full_prompt: str) -> str:
    if not full_prompt:
        return ""
    marker = "please use it with caution:"
    sep = "\n----------\n"
    lower = full_prompt.lower()
    start = lower.find(marker)
    if start >= 0:
        start += len(marker)
        rest = full_prompt[start:]
    else:
        rest = full_prompt
    end = rest.find(sep)
    return rest[:end] if end >= 0 else rest


def answer_in_text(answer: str, text: str) -> bool:
    ans = normalize(answer)
    ctx = normalize(text)
    if not ans or not ctx:
        return False
    if ans in ctx:
        return True
    ans_tokens = ans.split()
    if len(ans_tokens) <= 1:
        return False
    ctx_tokens = set(ctx.split())
    overlap = sum(tok in ctx_tokens for tok in ans_tokens)
    return overlap / len(ans_tokens) >= 0.8


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize(path: Path) -> dict:
    rows = list(load_jsonl(path))
    incorrect = [r for r in rows if not r.get("is_correct")]
    pred_in_context = []
    correct_gt_in_context = []
    for r in rows:
        aux_text = extract_aux_text(r.get("full_prompt", ""))
        pred_in_context.append(answer_in_text(r.get("pred_answer", ""), aux_text))
        gts = r.get("gt_answers") or []
        correct_gt_in_context.append(any(answer_in_text(gt, aux_text) for gt in gts))

    incorrect_pred_in_context = [
        p for p, r in zip(pred_in_context, rows) if not r.get("is_correct")
    ]
    return {
        "file": str(path),
        "num": len(rows),
        "accuracy": sum(bool(r.get("is_correct")) for r in rows) / len(rows) if rows else 0.0,
        "incorrect": len(incorrect),
        "pred_in_aux_text_rate_all": sum(pred_in_context) / len(rows) if rows else 0.0,
        "pred_in_aux_text_rate_incorrect": (
            sum(incorrect_pred_in_context) / len(incorrect_pred_in_context)
            if incorrect_pred_in_context
            else 0.0
        ),
        "gt_in_aux_text_rate_all": sum(correct_gt_in_context) / len(rows) if rows else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.files]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
