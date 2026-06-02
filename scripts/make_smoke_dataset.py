#!/usr/bin/env python3
"""Create a tiny LIAR-RAW style dataset for FaC-CofCED smoke runs."""

import argparse
import json
import shutil
from pathlib import Path


def copy_split(src_dir, dst_dir, split, limit):
    with (src_dir / f"{split}.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    with (dst_dir / f"{split}.json").open("w", encoding="utf-8") as f:
        json.dump(data[:limit], f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("Codes/dataset/LIAR-RAW"))
    parser.add_argument("--dst", type=Path, default=Path("Codes/dataset/LIAR-RAW_SMOKE"))
    parser.add_argument("--train", type=int, default=4)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=2)
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    copy_split(args.src, args.dst, "train", args.train)
    copy_split(args.src, args.dst, "val", args.val)
    copy_split(args.src, args.dst, "test", args.test)

    vocab = args.src / "vocab_article_source.json"
    if vocab.exists():
        shutil.copy2(vocab, args.dst / vocab.name)

    print(f"created smoke dataset at {args.dst}")


if __name__ == "__main__":
    main()
