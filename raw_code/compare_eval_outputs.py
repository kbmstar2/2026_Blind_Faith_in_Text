import argparse
import json
from collections import Counter


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(row):
    return (row.get("question"), row.get("full_prompt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    base = {key(row): row for row in load_rows(args.base)}
    new = {key(row): row for row in load_rows(args.new)}
    keys = [k for k in base if k in new]

    corrected = []
    regressed = []
    changed = []
    same = 0
    for k in keys:
        b = base[k]
        n = new[k]
        if b.get("pred_answer") != n.get("pred_answer"):
            changed.append((b, n))
        else:
            same += 1
        if not b.get("is_correct") and n.get("is_correct"):
            corrected.append((b, n))
        if b.get("is_correct") and not n.get("is_correct"):
            regressed.append((b, n))

    print(
        json.dumps(
            {
                "matched": len(keys),
                "base_acc": sum(bool(base[k].get("is_correct")) for k in keys) / len(keys),
                "new_acc": sum(bool(new[k].get("is_correct")) for k in keys) / len(keys),
                "changed_predictions": len(changed),
                "same_predictions": same,
                "corrected": len(corrected),
                "regressed": len(regressed),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== corrected examples ===")
    for b, n in corrected[: args.show]:
        print(
            json.dumps(
                {
                    "question": b.get("question"),
                    "gt": b.get("gt_answers"),
                    "base": b.get("pred_answer"),
                    "new": n.get("pred_answer"),
                },
                ensure_ascii=False,
            )
        )

    print("\n=== regressed examples ===")
    for b, n in regressed[: args.show]:
        print(
            json.dumps(
                {
                    "question": b.get("question"),
                    "gt": b.get("gt_answers"),
                    "base": b.get("pred_answer"),
                    "new": n.get("pred_answer"),
                },
                ensure_ascii=False,
            )
        )

    print("\n=== changed prediction correctness transitions ===")
    print(Counter((bool(b.get("is_correct")), bool(n.get("is_correct"))) for b, n in changed))


if __name__ == "__main__":
    main()
