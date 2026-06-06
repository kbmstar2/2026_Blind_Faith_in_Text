import argparse
import ast
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, load_dataset


def parse_answers(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        return [value] if value else []
    return [str(value)]


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
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ds_name", default="dal-289/word_or_vision")
    parser.add_argument("--subset", default="")
    parser.add_argument("--text_types", nargs="+", default=["match", "irrelevant"])
    parser.add_argument("--max_per_type", type=int, default=200)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = pick_split(load_dataset(args.ds_name), args.subset)
    examples = []
    counts = {}
    for text_type in args.text_types:
        part = ds.filter(lambda x, tt=text_type: x.get("text_type") == tt).shuffle(seed=args.seed)
        if args.sample_offset:
            part = part.select(range(args.sample_offset, len(part)))
        if args.max_per_type > 0:
            part = part.select(range(min(args.max_per_type, len(part))))
        counts[text_type] = len(part)
        for idx, sample in enumerate(part):
            answers = parse_answers(sample.get("answers") or sample.get("gt_answers") or sample.get("label"))
            if not answers:
                continue
            chosen = answers[0].strip()
            prompt = sample.get("full_prompt") or sample.get("question") or ""
            if not chosen or not prompt:
                continue
            examples.append(
                {
                    "question_id": f"{text_type}-{idx}-{sample.get('question', '')}",
                    "text_type": str(text_type),
                    "prompt": str(prompt),
                    "chosen": chosen,
                    "sample_weight": 1.0,
                    "image": sample["image"],
                }
            )

    if not examples:
        raise RuntimeError("No preservation SFT examples created.")

    features = Features(
        {
            "question_id": Value("string"),
            "text_type": Value("string"),
            "prompt": Value("string"),
            "chosen": Value("string"),
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
    print(f"counts: {counts}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
