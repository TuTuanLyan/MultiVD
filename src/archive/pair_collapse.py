#!/usr/bin/env python3
"""How many vulnerable/fixed pairs become the same token sequence after truncation?

Both encoders stop at 512 positions and cannot be configured past it, while 70%
of PrimeVul functions are longer. The worry is that truncation deletes the lines
that separate a vulnerable function from its fix, leaving two identical inputs
under opposite labels -- training on pure noise.

That is a mechanism, and mechanisms can be measured rather than assumed. This
tokenises each pair under the truncation strategy actually in use and counts how
often the two halves collapse onto the same ids.
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CodeDataset  # noqa: E402


def collapse_rate(path, tokenizer, max_length, strategy, limit):
    with open(path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()][:limit]
    dataset = CodeDataset(records, tokenizer, max_length, strategy)
    ids = [tuple(dataset[i]["input_ids"].tolist()) for i in range(len(records))]

    pairs = collapsed = 0
    index = 0
    while index < len(records) - 1:
        left, right = records[index], records[index + 1]
        if left.get("label") != right.get("label") and left.get("cwe") == right.get("cwe"):
            pairs += 1
            collapsed += ids[index] == ids[index + 1]
            index += 2
        else:
            index += 1
    return pairs, collapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--model_name", default="microsoft/codebert-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--truncation_strategy", default="head_middle_tail")
    parser.add_argument("--limit", type=int, default=1200,
                        help="rows per file; tokenising every row buys no extra precision")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    for path in args.inputs:
        pairs, collapsed = collapse_rate(
            path, tokenizer, args.max_length, args.truncation_strategy, args.limit
        )
        rate = 100 * collapsed / max(pairs, 1)
        print(f"{Path(path).stem:<34}{pairs:>5} pairs{collapsed:>6} collapsed{rate:>8.1f}%")


if __name__ == "__main__":
    main()
