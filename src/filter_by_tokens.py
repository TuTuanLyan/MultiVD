#!/usr/bin/env python3
"""Keep only rows that fit the encoder's context window.

70% of PrimeVul functions exceed 510 tokens against a hard 512-token limit that
CodeBERT and CodeT5+ cannot be configured past, so the model sees under half of
a typical function. For a vulnerable/fixed pair differing by a few lines, losing
the changed lines makes the two rows tokenize almost identically under opposite
labels -- the model is then trained on noise.

That gives two competing explanations for why PrimeVul transfers worse than a
corpus seven times smaller: domain distance, or simply not fitting. Filtering
separates them, because a length-filtered PrimeVul keeps the domain and drops
the truncation.

Pairs are kept or dropped together: keeping only the half that fits would leave
an unpaired row whose partner the model never sees.
"""

import argparse
import json
from collections import Counter

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", default="microsoft/codebert-base")
    parser.add_argument("--max_tokens", type=int, default=510,
                        help="512 minus the two special tokens")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    with open(args.input, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    lengths = [
        len(tokenizer(record["code"], add_special_tokens=False)["input_ids"])
        for record in records
    ]

    # Rows arrive as adjacent (vulnerable, fixed) pairs. Drop a pair whenever
    # either half overflows, so no row is left without its counterpart.
    keep = [False] * len(records)
    index = 0
    while index < len(records):
        if (
            index + 1 < len(records)
            and records[index].get("label") != records[index + 1].get("label")
            and records[index].get("cwe") == records[index + 1].get("cwe")
        ):
            fits = lengths[index] <= args.max_tokens and lengths[index + 1] <= args.max_tokens
            keep[index] = keep[index + 1] = fits
            index += 2
        else:
            keep[index] = lengths[index] <= args.max_tokens
            index += 1

    kept = [record for record, flag in zip(records, keep) if flag]
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    labels = Counter(record["label"] for record in kept)
    cwes = Counter(record["cwe"] for record in kept)
    over = sum(1 for length in lengths if length > args.max_tokens)
    print(f"input   : {len(records)} rows, {100 * over / len(records):.0f}% over {args.max_tokens} tokens")
    print(f"kept    : {len(kept)} rows ({100 * len(kept) / len(records):.0f}%)")
    print(f"labels  : {dict(sorted(labels.items()))}")
    print(f"CWEs    : {len(cwes)} distinct, top {cwes.most_common(3)}")
    print(f"written : {args.output}")


if __name__ == "__main__":
    main()
