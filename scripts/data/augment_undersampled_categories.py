"""Pick extra samples from cleaned pool for undersampled categories to augment training set.

This only looks at the TRAINING data distribution — it never touches the eval set.
Targets are based purely on class balance, not on eval performance.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEANED = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/02_cleaned.jsonl"
DEFAULT_SELECTED = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/02_selected_for_rewrite.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/augmentation_source.jsonl"

# Target counts: what we want each undersampled category to reach in training
# These are based on the BUILD REPORT class distribution, not eval results.
AUGMENT_TARGETS = {
    "投诉-安抚升级": 120,
    "商品咨询-参数": 150,
    "购买决策-推荐对比": 200,
    "售后-质量异常": 200,
    "价格活动-优惠赠品": 500,
    "商品咨询-使用方法": 500,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def candidate_score(row: dict[str, Any]) -> float:
    """Same scoring heuristic as the original build script."""
    flags = set(row.get("meta", {}).get("quality_flags") or [])
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    prompt_len = sum(1 for c in prompt if c.strip() and ord(c) > 127)
    reply_len = sum(1 for c in response if c.strip() and ord(c) > 127)

    score = 0.0
    if 6 <= prompt_len <= 80:
        score += 3
    if 18 <= reply_len <= 180:
        score += 4
    if row.get("history"):
        score += 1
    penalties = {
        "placeholder_rewrite": 0.8,
        "boilerplate_rewrite": 1.3,
        "short_reply_rewrite": 1.1,
        "long_digit_rewrite": 0.8,
        "privacy_or_private_info_rewrite": 1.0,
        "overpromise_rewrite": 1.2,
        "stutter_rewrite": 1.5,
        "long_reply_rewrite": 1.0,
        "possible_conflict_rewrite": 1.5,
        "unknown_intent": 2.0,
    }
    for flag, penalty in penalties.items():
        if flag in flags:
            score -= penalty
    return score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cleaned", default=str(DEFAULT_CLEANED))
    p.add_argument("--selected", default=str(DEFAULT_SELECTED))
    p.add_argument("--train-file", default=str(PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/sft_train.jsonl"))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    cleaned = read_jsonl(Path(args.cleaned))
    selected_ids = {r["id"] for r in read_jsonl(Path(args.selected))}

    # Read current train set to check actual counts
    train_rows = read_jsonl(Path(args.train_file))
    current_counts = defaultdict(int)
    for row in train_rows:
        intent = row.get("meta", {}).get("scenario", "")
        current_counts[intent] += 1

    print("=== Current training set distribution ===")
    for intent, count in sorted(current_counts.items(), key=lambda x: -x[1]):
        target = AUGMENT_TARGETS.get(intent)
        marker = f" → target {target}" if target else ""
        print(f"  {intent}: {count}{marker}")
    print()

    # Pick extra samples from cleaned pool (not already selected)
    extra_pool = defaultdict(list)
    for row in cleaned:
        intent = row.get("meta", {}).get("scenario", "")
        if intent not in AUGMENT_TARGETS:
            continue
        if row["id"] in selected_ids:
            continue
        extra_pool[intent].append(row)

    # Score and pick best
    picked = []
    for intent, target in AUGMENT_TARGETS.items():
        current = current_counts.get(intent, 0)
        need = max(0, target - current)
        pool = extra_pool[intent]
        if need == 0:
            print(f"{intent}: already at target ({current}), skipping")
            continue
        if not pool:
            print(f"{intent}: need {need} but 0 available in pool")
            continue

        for row in pool:
            row.setdefault("meta", {})["augment_score"] = round(candidate_score(row), 3)
        pool.sort(key=lambda r: r["meta"]["augment_score"], reverse=True)
        take = min(need, len(pool))
        chosen = pool[:take]
        for row in chosen:
            row.setdefault("meta", {})["augmentation_source"] = True
        picked.extend(chosen)
        print(f"{intent}: {current} → {current+take} (took {take}/{len(pool)} from pool, "
              f"best score={chosen[0]['meta']['augment_score']:.1f}, "
              f"worst score={chosen[-1]['meta']['augment_score']:.1f})")

    rng.shuffle(picked)
    write_jsonl(picked, Path(args.output))
    print(f"\nTotal new samples for DeepSeek rewrite: {len(picked)}")
    print(f"Output: {args.output}")
    print()
    print("Next step: rewrite with DeepSeek, then merge into sft_train.jsonl")


if __name__ == "__main__":
    main()
