"""Shrink eval set to ~150 and redistribute excess to train/dev, maintaining distribution."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_DIR = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2" / "05_final"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def get_intent(row: dict[str, Any]) -> str:
    return str(row.get("meta", {}).get("scenario") or row.get("meta", {}).get("intent") or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", default=str(DEFAULT_FINAL_DIR))
    parser.add_argument("--target-eval", type=int, default=150)
    parser.add_argument("--min-per-intent", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    final_dir = Path(args.final_dir)

    train_path = final_dir / "sft_train.jsonl"
    dev_path = final_dir / "sft_dev.jsonl"
    eval_path = final_dir / "eval_test.jsonl"

    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    eval_rows = read_jsonl(eval_path)

    # Group eval by intent
    eval_by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        eval_by_intent[get_intent(row)].append(row)

    # Calculate target per intent with proportional cut distribution.
    # The goal: shrink each category proportionally, then apply min-per-intent
    # by borrowing from larger categories, not gutting the biggest one.
    total_eval = len(eval_rows)
    target_per_intent: dict[str, int] = {}

    # Step 1: raw proportional targets
    raw: dict[str, float] = {}
    for intent, rows in eval_by_intent.items():
        raw[intent] = args.target_eval * len(rows) / total_eval

    # Step 2: clamp to [available, min_per_intent...available]
    for intent in raw:
        avail = len(eval_by_intent[intent])
        floored = max(raw[intent], min(args.min_per_intent, avail))
        target_per_intent[intent] = round(min(floored, avail))

    # Step 3: if sum overshoots, cut proportionally from categories that have
    # room above min_per_intent, weighted by their surplus share.
    current_sum = sum(target_per_intent.values())
    if current_sum > args.target_eval:
        excess = current_sum - args.target_eval
        # Build surplus map: how much each category can give
        surplus: dict[str, int] = {}
        for intent in target_per_intent:
            s = target_per_intent[intent] - min(args.min_per_intent, len(eval_by_intent[intent]))
            if s > 0:
                surplus[intent] = s
        total_surplus = sum(surplus.values())
        remaining = excess
        # Distribute cuts proportionally to surplus
        cuts_applied: dict[str, int] = {}
        for intent in sorted(surplus, key=lambda x: surplus[x], reverse=True):
            if remaining <= 0:
                break
            cut = max(1, round(excess * surplus[intent] / total_surplus))
            cut = min(cut, surplus[intent], remaining)
            cuts_applied[intent] = cut
            remaining -= cut
        # If rounding left a small residue, clean up from largest surplus
        for intent in sorted(surplus, key=lambda x: surplus[x] - cuts_applied.get(x, 0), reverse=True):
            if remaining <= 0:
                break
            already = cuts_applied.get(intent, 0)
            if already < surplus[intent]:
                take = min(1, remaining)
                cuts_applied[intent] = already + take
                remaining -= take
        for intent, cut in cuts_applied.items():
            target_per_intent[intent] -= cut

    elif current_sum < args.target_eval:
        deficit = args.target_eval - current_sum
        for intent in sorted(target_per_intent, key=lambda x: len(eval_by_intent[x]) - target_per_intent[x], reverse=True):
            if deficit <= 0:
                break
            room = len(eval_by_intent[intent]) - target_per_intent[intent]
            if room > 0:
                add = min(room, deficit)
                target_per_intent[intent] += add
                deficit -= add

    # Select eval rows and build transfer pool
    new_eval: list[dict[str, Any]] = []
    transfer_pool: list[dict[str, Any]] = []
    for intent, rows in eval_by_intent.items():
        rng.shuffle(rows)
        keep = target_per_intent.get(intent, 0)
        new_eval.extend(rows[:keep])
        transfer_pool.extend(rows[keep:])

    rng.shuffle(new_eval)

    # Split transfer pool between train and dev, proportional to current sizes
    new_train: list[dict[str, Any]] = list(train_rows)
    new_dev: list[dict[str, Any]] = list(dev_rows)
    rng.shuffle(transfer_pool)
    train_ratio = len(train_rows) / (len(train_rows) + len(dev_rows))
    split_point = round(len(transfer_pool) * train_ratio)
    new_train.extend(transfer_pool[:split_point])
    new_dev.extend(transfer_pool[split_point:])
    rng.shuffle(new_train)
    rng.shuffle(new_dev)

    # Build report
    def summarize(rows):
        c = Counter(get_intent(r) for r in rows)
        return {"total": len(rows), "by_intent": dict(c.most_common())}

    report = {
        "before": {"train": summarize(train_rows), "dev": summarize(dev_rows), "eval": summarize(eval_rows)},
        "after": {"train": summarize(new_train), "dev": summarize(new_dev), "eval": summarize(new_eval)},
        "transferred_from_eval": len(transfer_pool),
        "target_per_intent": target_per_intent,
    }

    if args.dry_run:
        print("=== DRY RUN ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    write_jsonl(new_train, train_path)
    write_jsonl(new_dev, dev_path)
    write_jsonl(new_eval, eval_path)
    print("Done. Summary:")
    print(json.dumps(report["after"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
