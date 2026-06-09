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


def answer_shape(answer):
    answer = norm(answer)
    if re.fullmatch(r"[$€£]?\d+([.,:]\d+)*[%]?", answer):
        return "number"
    if re.fullmatch(r"\d{4}", answer):
        return "year"
    if answer in {
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "grey",
        "silver",
        "gold",
    }:
        return "color"
    if len(answer.split()) <= 3:
        return "short_text"
    return "phrase"


def question_type(question, answer=""):
    q = norm(question)
    shape = answer_shape(answer)
    if any(key in q for key in ["how many", "what number", "which number", "what digit", "what score"]):
        return "number"
    if any(key in q for key in ["what year", "which year", "what date", "expiration", "expire"]):
        return "year"
    if any(key in q for key in ["price", "cost", "amount", "dollar", "money"]):
        return "number"
    if "color" in q or "colour" in q:
        return "color"
    if any(key in q for key in ["brand", "logo", "company", "restaurant", "store", "business"]):
        return "brand"
    if any(key in q for key in ["what word", "what is written", "what does", "what text", "what name"]):
        return "short_text"
    return shape


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


def pick_from_pool(pool, answers, rng):
    candidates = [a for a in pool if valid_answer(a) and not answer_overlap(a, answers)]
    if candidates:
        return rng.choice(candidates)
    return None


def choose_wrong_answer(ocr_tokens, answers, fallback_pool, typed_pools, qtype, rng, corruption_mode):
    if corruption_mode == "question_type_matched_answer":
        wrong = pick_from_pool(typed_pools.get(qtype, []), answers, rng)
        if wrong:
            return wrong, f"question_type_matched_answer:{qtype}"
        wrong = pick_from_pool(typed_pools.get(answer_shape(answers[0] if answers else ""), []), answers, rng)
        if wrong:
            return wrong, "answer_shape_matched_answer"
        wrong = pick_from_pool(fallback_pool, answers, rng)
        if wrong:
            return wrong, "cross_sample_answer_fallback"
        return None, None

    if corruption_mode == "cross_sample_answer":
        wrong = pick_from_pool(fallback_pool, answers, rng)
        if wrong:
            return wrong, "cross_sample_answer"
        return None, None

    candidates = []
    for tok in ocr_tokens:
        t = norm(tok)
        if valid_answer(t) and not answer_overlap(t, answers):
            candidates.append(t)
    if candidates:
        return rng.choice(candidates), "same_image_ocr"

    wrong = pick_from_pool(fallback_pool, answers, rng)
    if wrong:
        return wrong, "other_textvqa_answer"
    return None, None


def build_aux_text(kind, answer, wrong, ocr_tokens, answer_variants, corruption_mode):
    if corruption_mode == "question_type_matched_answer":
        token = answer if kind == "match" else wrong
        return f'A possibly noisy scene-text QA parser predicts the answer is: "{token}".'

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
        choices=["same_image_ocr", "cross_sample_answer", "question_type_matched_answer"],
        default="same_image_ocr",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset("lmms-lab/textvqa", split="validation")
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    fallback_pool = []
    typed_pools = {}
    for i in indices[:5000]:
        item = ds[i]
        ans = choose_answer(parse_list(item.get("answers")))
        if ans:
            fallback_pool.append(ans)
            typed_pools.setdefault(question_type(item.get("question"), ans), []).append(ans)
            typed_pools.setdefault(answer_shape(ans), []).append(ans)

    rows_by_kind = {"corrupted": [], "match": []}
    manifest = []
    for idx in indices:
        item = ds[idx]
        answers = parse_list(item.get("answers"))
        answer = choose_answer(answers)
        if not answer:
            continue
        ocr_tokens = parse_list(item.get("ocr_tokens"))
        qtype = question_type(item.get("question"), answer)
        wrong, wrong_source = choose_wrong_answer(
            ocr_tokens,
            [answer] + answers,
            fallback_pool,
            typed_pools,
            qtype,
            rng,
            args.corruption_mode,
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
            "question_type": qtype,
            "ocr_tokens": [str(t) for t in ocr_tokens],
            "image": item.get("image"),
        }
        for kind in rows_by_kind:
            aux = build_aux_text(
                kind, answer, wrong, ocr_tokens, [answer] + answers, args.corruption_mode
            )
            row = dict(base)
            row["text_type"] = kind
            row["additional_info"] = {
                "aux_text": aux,
                "wrong_answer": wrong,
                "wrong_source": wrong_source,
                "canonical_answer": answer,
                "question_type": qtype,
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
            "question_type": Value("string"),
            "ocr_tokens": Sequence(Value("string")),
            "image": Image(),
            "text_type": Value("string"),
            "additional_info": {
                "aux_text": Value("string"),
                "wrong_answer": Value("string"),
                "wrong_source": Value("string"),
                "canonical_answer": Value("string"),
                "question_type": Value("string"),
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
