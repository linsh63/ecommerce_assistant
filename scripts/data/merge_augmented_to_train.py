"""Merge DeepSeek-rewritten augmented samples into the training set."""

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REWRITTEN = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/04_deepseek_rewritten/augmentation_rewritten.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/sft_train.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rewritten", default=str(DEFAULT_REWRITTEN))
    p.add_argument("--train-file", default=str(DEFAULT_TRAIN))
    p.add_argument("--no-backup", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rewritten_path = Path(args.rewritten)
    train_path = Path(args.train_file)

    rewritten = read_jsonl(rewritten_path)
    if not rewritten:
        print(f"No rewritten samples found at {rewritten_path}")
        return

    train = read_jsonl(train_path)
    print(f"Before: {len(train)} train samples")

    # Dedup: skip if id already exists in training set
    existing_ids = {r["id"] for r in train}
    new_samples = [r for r in rewritten if r["id"] not in existing_ids]
    skipped = len(rewritten) - len(new_samples)
    if skipped:
        print(f"Skipped {skipped} duplicates")

    # Backup
    if not args.no_backup:
        backup_path = train_path.with_suffix(".jsonl.bak")
        write_jsonl(train, backup_path)
        print(f"Backup: {backup_path}")

    # Merge
    train.extend(new_samples)
    write_jsonl(train, train_path)

    from collections import Counter
    counts = Counter(r.get("meta", {}).get("scenario", "?") for r in train)

    print(f"After: {len(train)} train samples (+{len(new_samples)})")
    print()
    print("=== New distribution ===")
    for intent, count in counts.most_common():
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()
