import argparse
import ast
import random
import re
import string
from collections import Counter
from pathlib import Path

from datasets import Dataset, Features, Image, Sequence, Value, load_dataset


def parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def norm(text):
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def valid_answer(answer):
    answer = norm(answer)
    if answer in {"", "yes", "no", "none", "unknown", "unanswerable"}:
        return False
    return any(ch.isalnum() for ch in answer) and len(answer) >= 2


def choose_answer(answers):
    normalized = [norm(a) for a in answers if valid_answer(a)]
    if not normalized:
        return None
    return Counter(normalized).most_common(1)[0][0]


def answer_overlap(candidate, answers):
    c = norm(candidate)
    if not c:
        return True
    for ans in answers:
        a = norm(ans)
        if not a:
            continue
        if c == a or c in a or a in c:
            return True
    return False


def choose_wrong_answer(ocr_tokens, answers, fallback_pool, rng, corruption_mode):
    if corruption_mode == "cross_sample_answer":
        fallback = [a for a in fallback_pool if valid_answer(a) and not answer_overlap(a, answers)]
        if fallback:
            return rng.choice(fallback), "cross_sample_answer"
        return None, None

    candidates = []
    for tok in ocr_tokens:
        t = norm(tok)
        if valid_answer(t) and not answer_overlap(t, answers):
            candidates.append(t)
    if candidates:
        return rng.choice(candidates), "same_image_ocr"

    fallback = [a for a in fallback_pool if valid_answer(a) and not answer_overlap(a, answers)]
    if fallback:
        return rng.choice(fallback), "other_textvqa_answer"
    return None, None


def build_aux_text(kind, answer, wrong, ocr_tokens, answer_variants):
    clean_ocr = []
    for tok in ocr_tokens:
        t = norm(tok)
        if valid_answer(t) and t not in clean_ocr and not answer_overlap(t, answer_variants):
            clean_ocr.append(t)
    clean_ocr = clean_ocr[:4]

    if kind == "match":
        tokens = [answer] + clean_ocr[:2]
    else:
        tokens = [wrong] + clean_ocr[:2]
    quoted = ", ".join(f'"{t}"' for t in tokens if t)
    return f"A possibly noisy scene-text parser says the visible text includes: {quoted}."


def make_prompt(question, aux_text):
    return (
        "You are given an image and auxiliary text extracted from the image. "
        "The auxiliary text may be incorrect, so please use it with caution:\n"
        f"{aux_text}\n"
        "----------\n"
        f"Question: {question}\n"
        "Please only output the answer with a single word or short phrase."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="data/textvqa_corruption_pilot_seed0_200")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--corruption_mode",
        choices=["same_image_ocr", "cross_sample_answer"],
        default="same_image_ocr",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    fallback_pool = []
    for i in indices[:5000]:
        ans = choose_answer(parse_list(ds[i].get("answers")))
        if ans:
            fallback_pool.append(ans)

    rows_by_kind = {"corrupted": [], "match": []}
    manifest = []
    for idx in indices:
        item = ds[idx]
        answers = parse_list(item.get("answers"))
        answer = choose_answer(answers)
        if not answer:
            continue
        ocr_tokens = parse_list(item.get("ocr_tokens"))
        wrong, wrong_source = choose_wrong_answer(
            ocr_tokens, [answer] + answers, fallback_pool, rng, args.corruption_mode
        )
        if not wrong:
            continue

        base = {
            "question_id": str(item.get("question_id")),
            "image_id": str(item.get("image_id")),
            "question": item.get("question"),
            "answers": [str(a) for a in answers],
            "canonical_answer": answer,
            "wrong_answer": wrong,
            "wrong_source": wrong_source,
            "ocr_tokens": [str(t) for t in ocr_tokens],
            "image": item.get("image"),
        }
        for kind in rows_by_kind:
            aux = build_aux_text(kind, answer, wrong, ocr_tokens, [answer] + answers)
            row = dict(base)
            row["text_type"] = kind
            row["additional_info"] = {
                "aux_text": aux,
                "wrong_answer": wrong,
                "wrong_source": wrong_source,
                "canonical_answer": answer,
            }
            row["full_prompt"] = make_prompt(item.get("question"), aux)
            rows_by_kind[kind].append(row)
        manifest.append({k: v for k, v in base.items() if k != "image"})
        if len(rows_by_kind["corrupted"]) >= args.max_samples:
            break

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    features = Features(
        {
            "question_id": Value("string"),
            "image_id": Value("string"),
            "question": Value("string"),
            "answers": Sequence(Value("string")),
            "canonical_answer": Value("string"),
            "wrong_answer": Value("string"),
            "wrong_source": Value("string"),
            "ocr_tokens": Sequence(Value("string")),
            "image": Image(),
            "text_type": Value("string"),
            "additional_info": {
                "aux_text": Value("string"),
                "wrong_answer": Value("string"),
                "wrong_source": Value("string"),
                "canonical_answer": Value("string"),
            },
            "full_prompt": Value("string"),
        }
    )
    for kind, rows in rows_by_kind.items():
        out_dir = out_root / kind
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)
        Dataset.from_list(rows, features=features).save_to_disk(str(out_dir))

    with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        import json

        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {len(rows_by_kind['corrupted'])} samples under {out_root}")


if __name__ == "__main__":
    main()
