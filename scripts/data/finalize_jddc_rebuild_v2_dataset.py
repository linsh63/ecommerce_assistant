"""Finalize rewritten JDDC rebuild-v2 files into train/dev/eval SFT files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REWRITE_DIR = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2" / "04_deepseek_rewritten"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2" / "03_split"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2" / "05_final"


# 功能：读取 JSONL。
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


# 功能：写出 JSONL。
def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


# 功能：写出 JSON。
def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 功能：从 source 记录转成未改写的 fallback SFT 记录。
def source_to_sft(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "messages": row.get("messages") or [],
        "meta": {
            **(row.get("meta") or {}),
            "rewritten": False,
            "finalize_source": "source_fallback",
        },
    }


# 功能：读取改写文件；没有改写文件时可按参数回退到 source。
def load_split(split: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    rewrite_dir = Path(args.rewrite_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    candidates = [
        rewrite_dir / f"{split}_source_rewritten.jsonl",
        rewrite_dir / f"{split}_rewritten.jsonl",
        rewrite_dir / f"{split}.jsonl",
    ]
    for path in candidates:
        rows = read_jsonl(path)
        if rows:
            return rows, str(path)
    if args.allow_source_fallback:
        source_rows = read_jsonl(source_dir / f"{split}_source.jsonl")
        return [source_to_sft(row) for row in source_rows], str(source_dir / f"{split}_source.jsonl")
    return [], ""


# 功能：写出人工审阅 Markdown。
def write_eval_review(rows: list[dict[str, Any]], path: Path, limit: int, seed: int) -> None:
    rng = random.Random(seed)
    sampled = rows if len(rows) <= limit else rng.sample(rows, k=limit)
    lines = ["# JDDC Rebuild V2 Eval Review", "", f"- total: {len(rows)}", f"- shown: {len(sampled)}", ""]
    for index, row in enumerate(sampled, start=1):
        lines.extend([f"## eval_{index:03d} | {row.get('id')}", "", "```text"])
        for message in row.get("messages") or []:
            role = message.get("role")
            content = message.get("content")
            lines.append(f"[{role}] {content}")
        lines.extend(
            [
                "```",
                "",
                "预期必须覆盖：",
                "- ",
                "",
                "禁止出现：",
                "- ",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# 功能：解析参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite-dir", default=str(DEFAULT_REWRITE_DIR))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow-source-fallback", action="store_true", help="Use source files when rewrite files are missing.")
    parser.add_argument("--eval-review-limit", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# 功能：主流程。
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_source = load_split("train", args)
    dev_rows, dev_source = load_split("dev", args)
    test_rows, test_source = load_split("test", args)

    write_jsonl(train_rows, output_dir / "sft_train.jsonl")
    write_jsonl(dev_rows, output_dir / "sft_dev.jsonl")
    write_jsonl(test_rows, output_dir / "eval_test.jsonl")
    write_eval_review(test_rows, output_dir / "eval_test_review.md", args.eval_review_limit, args.seed)
    write_json([], output_dir / "eval_test_annotation.json")

    report = {
        "train_total": len(train_rows),
        "dev_total": len(dev_rows),
        "test_total": len(test_rows),
        "train_source": train_source,
        "dev_source": dev_source,
        "test_source": test_source,
        "allow_source_fallback": args.allow_source_fallback,
        "files": {
            "sft_train": str(output_dir / "sft_train.jsonl"),
            "sft_dev": str(output_dir / "sft_dev.jsonl"),
            "eval_test": str(output_dir / "eval_test.jsonl"),
            "eval_test_review": str(output_dir / "eval_test_review.md"),
            "eval_test_annotation": str(output_dir / "eval_test_annotation.json"),
        },
    }
    write_json(report, output_dir / "build_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
