#!/usr/bin/env python3
"""Create a small LIAR-RAW style dataset for FaC-CofCED smoke runs."""

import argparse
import json
import shutil
from pathlib import Path


def stratified_prefix(data, limit):
    if limit <= 0 or limit >= len(data):
        return data
    by_label = {}
    for item in data:
        by_label.setdefault(item.get("label", "__missing__"), []).append(item)
    labels = sorted(by_label)
    selected = []
    cursor = 0
    while len(selected) < limit and labels:
        label = labels[cursor % len(labels)]
        bucket = by_label[label]
        if bucket:
            selected.append(bucket.pop(0))
        labels = [la for la in labels if by_label[la]]
        cursor += 1
    return selected


def copy_split(src_dir, dst_dir, split, limit, stratified):
    with (src_dir / f"{split}.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    if stratified:
        data = stratified_prefix(data, limit)
    else:
        data = data[:limit]
    with (dst_dir / f"{split}.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("Codes/dataset/LIAR-RAW"))
    parser.add_argument("--dst", type=Path, default=Path("Codes/dataset/LIAR-RAW_SMOKE"))
    parser.add_argument("--train", type=int, default=4)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    parser.add_argument("--no-stratify", action="store_true")
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    stratified = not args.no_stratify
    copy_split(args.src, args.dst, "train", args.train, stratified)
    copy_split(args.src, args.dst, "val", args.val, stratified)
    copy_split(args.src, args.dst, "test", args.test, stratified)

    vocab = args.src / "vocab_article_source.json"
    if vocab.exists():
        shutil.copy2(vocab, args.dst / vocab.name)

    print(f"created smoke dataset at {args.dst}")


if __name__ == "__main__":
    main()
