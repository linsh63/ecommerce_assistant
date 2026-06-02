"""Sync edited annotation_sheet.md back to eval_test.jsonl and eval_test_annotation.json."""

import json, re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MD = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/annotation_sheet.md"
DEFAULT_EVAL = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/eval_test.jsonl"
DEFAULT_ANN = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/eval_test_annotation.json"

INTENT_ORDER = [
    "物流-查件催发", "物流-配送调整", "售后-退换维修", "售后-质量异常",
    "退款-取消价保", "商品咨询-参数", "商品咨询-使用方法",
    "购买决策-推荐对比", "价格活动-优惠赠品", "发票-资质服务",
    "投诉-安抚升级", "闲聊-礼貌收尾",
]


def get_intent(row):
    return row.get("meta", {}).get("scenario") or row.get("meta", {}).get("intent", "unknown")


def main():
    md_path = Path(DEFAULT_MD)
    eval_path = Path(DEFAULT_EVAL)
    ann_path = Path(DEFAULT_ANN)

    # Read eval rows
    with open(eval_path, encoding="utf-8") as f:
        eval_rows = [json.loads(l) for l in f if l.strip()]

    # Group by intent matching md order
    grouped = defaultdict(list)
    for row in eval_rows:
        grouped[get_intent(row)].append(row)
    flat_rows = []
    for intent in INTENT_ORDER:
        flat_rows.extend(grouped.get(intent, []))

    # Parse md
    content = md_path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=### eval_\d{3} \|)", content)
    md_anns = []
    for section in sections:
        m = re.match(r"### eval_(\d{3}) \| (.+?) \| .+", section)
        if not m:
            continue
        code_match = re.search(r"```\n(.*?)```", section, re.DOTALL)
        messages = []
        if code_match:
            for line in code_match.group(1).strip().split("\n"):
                line = line.strip()
                if line.startswith("[用户]"):
                    messages.append({"role": "user", "content": line[4:].strip()})
                elif line.startswith("[客服]"):
                    messages.append({"role": "assistant", "content": line[4:].strip()})

        exp_match = re.search(r"\*\*预期[：:]\*\*\s*(.+?)(?:\n|$)", section)
        expected = exp_match.group(1).strip() if exp_match else ""
        kw_match = re.search(r"\*\*关键词[：:]\*\*\s*(.+?)(?:\n|$)", section)
        keywords = kw_match.group(1).strip() if kw_match else ""

        md_anns.append({"messages": messages, "expected": expected, "keywords": keywords})

    if len(md_anns) != len(flat_rows):
        print(f"ERROR: md has {len(md_anns)} evals but jsonl has {len(flat_rows)} rows")
        return

    # Update jsonl rows
    msgs_updated = 0
    annotations = []
    for md_ann, row in zip(md_anns, flat_rows):
        if row.get("messages") != md_ann["messages"]:
            row["messages"] = md_ann["messages"]
            msgs_updated += 1
        annotations.append({
            "id": row.get("id"),
            "scenario": get_intent(row),
            "expectation": {"must_cover": [], "should_cover": [], "avoid": []},
            "expected_text": md_ann["expected"],
            "keywords": md_ann["keywords"],
        })

    # Map back to original eval order
    orig_rows = [None] * len(eval_rows)
    orig_anns = [None] * len(eval_rows)
    for i, row in enumerate(eval_rows):
        for j, flat_row in enumerate(flat_rows):
            if flat_row.get("id") == row.get("id"):
                orig_rows[i] = flat_row
                orig_anns[i] = annotations[j]
                break

    with open(eval_path, "w", encoding="utf-8") as f:
        for row in orig_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(orig_anns, f, ensure_ascii=False, indent=2)

    print(f"Synced: {msgs_updated} conversations updated, {len(orig_anns)} annotations written")
    print(f"  {eval_path}")
    print(f"  {ann_path}")


if __name__ == "__main__":
    main()
