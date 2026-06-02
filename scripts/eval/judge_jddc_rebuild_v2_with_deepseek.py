"""LLM judge for JDDC rebuild v2: score model generations against human-annotated expectations."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATIONS = PROJECT_ROOT / "outputs/sft_lora_qwen3_8b_jddc_rebuild_v2/post_train_test_generations.jsonl"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/eval_test_annotation.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/sft_lora_qwen3_8b_jddc_rebuild_v2/judge_results"

# ── Judge prompt ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = """你是电商客服质检专家。你需要评判一个AI客服的回复是否合格。
不要对输出格式做任何要求——纯文本回复、不包含分类行都是完全正常的。

评判维度：
1. 需求解决：回复是否直接解决了用户最后一个问题/诉求？是否给出了可执行的路径或方向？
2. 回答准确：回复是否准确回应了问题？有没有答非所问或错误引导？
3. 有帮助性：回复是否给出了操作指导或有用信息？用户看完知道可以怎么做？
4. 语气体验：是否礼貌、有同理心？
5. 风险控制：是否编造了只有后台系统才能查到的信息（具体订单状态、精确物流节点、精确退款金额、精确到小时的时效）？
6. 历史利用（仅多轮）：多轮对话是否使用了历史中的关键信息？

评分标准（核心原则：给了入口或路径就算具体）：
- 5: 回复给出了明确可执行路径，用户看完就能照着做。完美。
- 4: 回复方向正确且给出了至少一个具体操作入口或步骤（如"订单详情页点修改"、"联系快递转到xxx"）。有这些就算4分，不需要完美。
- 3: 回复方向正确，但只说了做什么没说怎么做（如只说"建议申请售后"不说入口在哪、"联系客服"不说怎么联系）。信息不够用。
- 2: 回复模糊、回避核心问题、或只是索要信息（如"请提供订单号"）而不给任何实质性帮助。
- 1: 答非所问、完全错误、或输出大量无关内容。

通过条件：需求解决>=4 且 回答准确>=4

重要说明：
- 给出通用电商流程不算编造（如"订单详情页查看物流"、"1-3个工作日退款"是行业常识）。
- 编造是指给出只有后台系统能查到的具体事实（如"您的订单昨天下午3点从上海仓发出"、"退款金额为283.5元"）。
- 闲聊-礼貌收尾类：简短礼貌回应即可得4-5分，不需要展开流程。
- 不要求任何特定输出格式，不因缺少分类/格式而扣分。"""


def build_judge_prompt(gen: dict, ann: dict) -> str:
    history = gen.get("history") or gen.get("source_record", {}).get("history") or []
    source = gen.get("source_record") or {}
    messages = source.get("messages") or []

    # Extract last user question and build history context
    history_text = ""
    last_user = gen.get("prompt", "")
    if messages and len(messages) >= 2:
        last_user = messages[-2].get("content", last_user)
        history_msgs = messages[:-2]
        if history_msgs:
            lines = []
            for m in history_msgs:
                role = "用户" if m.get("role") == "user" else "客服"
                lines.append(f"{role}：{m.get('content', '')}")
            history_text = "\n".join(lines)

    multi_label = "多轮" if history_text else "单轮"

    expected = ann.get("expected_text") or ""
    keywords = ann.get("keywords") or ""
    scenario = ann.get("scenario") or source.get("meta", {}).get("scenario", "")

    parts = [
        f"场景类别：{scenario}",
        f"对话类型：{multi_label}",
        "",
    ]
    if history_text:
        parts += ["历史对话：", history_text, ""]
    parts += [
        f"用户最后问题：{last_user}",
        "",
        f"AI客服回复：{gen.get('generated_reply') or gen.get('generated_response', '')}",
        "",
        f"人工预期标准：{expected}",
        f"关键参考词：{keywords}",
        "",
        "请按以下JSON格式输出评分：",
        "{",
        '  "需求解决": <1-5>,',
        '  "回答准确": <1-5>,',
        '  "有帮助性": <1-5>,',
        '  "语气体验": <1-5>,',
        '  "风险控制": <1-5>,',
        '  "历史利用": <1-5或null(单轮填null)>,',
        '  "通过": <true/false, 需求解决>=4且回答准确>=4为通过>,',
        '  "失败标签": "<不通过时从以下选一个：off_topic/too_generic/missing_action/wrong_intent/overpromise/placeholder_leak/history_ignored/tone_poor/unsafe_policy, 通过时填null>",',
        '  "解释": "<简短解释，50字以内>"',
        "}",
        "",
        "只输出JSON，不要Markdown代码块。",
    ]
    return "\n".join(parts)


# ── API call ──────────────────────────────────────────────────────────────

def parse_api_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def call_judge(args: argparse.Namespace, system: str, user_prompt: str) -> dict[str, Any]:
    endpoint = args.api_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    }
    if not args.no_response_format:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return parse_api_json(body["choices"][0]["message"]["content"])


# ── Scoring helpers ──────────────────────────────────────────────────────

DIMENSIONS = ["需求解决", "回答准确", "有帮助性", "语气体验", "风险控制", "历史利用"]
FAIL_TAGS = [
    "off_topic", "too_generic", "missing_action", "wrong_intent",
    "overpromise", "placeholder_leak", "history_ignored", "tone_poor", "unsafe_policy",
]


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    passed = sum(1 for r in results if r.get("通过"))
    dim_scores = {d: [] for d in DIMENSIONS}
    for r in results:
        for d in DIMENSIONS:
            v = r.get(d)
            if v is not None:
                dim_scores[d].append(v)

    dim_avg = {d: round(sum(v) / len(v), 2) if v else 0 for d, v in dim_scores.items()}

    # Auto-resolution rate: "需求解决" >= 3
    resolved = sum(1 for r in results if (r.get("需求解决") or 0) >= 3)

    # Satisfaction rate: 通过率
    # Accuracy rate: "回答准确" >= 3
    accurate = sum(1 for r in results if (r.get("回答准确") or 0) >= 3)

    fail_tags = Counter()
    for r in results:
        tag = r.get("失败标签")
        if tag and tag != "null":
            fail_tags[str(tag)] += 1

    return {
        "total": total,
        "通过数": passed,
        "通过率": round(passed / total * 100, 1),
        "自动解决率": round(resolved / total * 100, 1),
        "准确率": round(accurate / total * 100, 1),
        "各维度均分": dim_avg,
        "失败标签分布": dict(fail_tags.most_common()),
    }


def compute_metrics_by_scenario(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[r.get("scenario", "unknown")].append(r)
    return {s: compute_metrics(rows) for s, rows in sorted(groups.items())}


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generations", default=str(DEFAULT_GENERATIONS))
    p.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--no-response-format", action="store_true")
    p.add_argument("--api-base-url", default=os.environ.get("DEEPSEEK_API_BASE_URL") or "https://api.deepseek.com")
    p.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    p.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    gen_path = Path(args.generations).expanduser().resolve()
    ann_path = Path(args.annotations).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gen_path.exists():
        raise FileNotFoundError(f"Generations file not found: {gen_path}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_path}")

    # Read inputs
    generations = []
    with gen_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line))
    if args.limit:
        generations = generations[: args.limit]

    with ann_path.open(encoding="utf-8") as f:
        annotations_list = json.load(f)
    ann_by_id = {a["id"]: a for a in annotations_list}

    # Match generations to annotations
    matched = []
    skipped = 0
    for gen in generations:
        gid = gen.get("id", "")
        ann = ann_by_id.get(gid)
        if not ann:
            # Try matching by source_record id
            src = gen.get("source_record") or {}
            gid2 = src.get("id", "")
            ann = ann_by_id.get(gid2)
        if ann:
            matched.append((gen, ann))
        else:
            skipped += 1
    print(f"Matched: {len(matched)}, skipped (no annotation): {skipped}")

    # Diagnostic: show why matching failed
    if skipped > 0 and len(matched) == 0:
        print("DIAGNOSTIC: ID mismatch detected")
        gen_ids = [gen.get("id", "") for gen in generations[:3]]
        ann_ids = list(ann_by_id.keys())[:3]
        print(f"  First 3 generation ids: {gen_ids}")
        print(f"  First 3 annotation ids: {ann_ids}")
        if generations:
            g0 = generations[0]
            src_id = (g0.get("source_record") or {}).get("id", "")
            print(f"  First gen source_record.id: {src_id}")
            print(f"  First gen keys: {list(g0.keys())}")

    # Resume support
    results_path = output_dir / "judge_results.jsonl"
    done_ids = set()
    if not args.no_resume and results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line).get("id", ""))

    # Judge each sample
    results = []
    audit_lines = []
    failed = 0
    for gen, ann in matched:
        gid = gen.get("id", "")
        if gid in done_ids:
            # Re-read existing result
            continue

        user_prompt = build_judge_prompt(gen, ann)
        success = False
        for attempt in range(args.retries + 1):
            try:
                score = call_judge(args, JUDGE_SYSTEM, user_prompt)
                score["id"] = gid
                score["scenario"] = ann.get("scenario", "")
                score["generated_reply"] = gen.get("generated_reply", "")
                results.append(score)
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False, separators=(",", ":")) + "\n")
                done_ids.add(gid)
                status = "PASS" if score.get("通过") else "FAIL"
                print(f"[{status}] {gid} | {score.get('需求解决')}/{score.get('回答准确')} | {score.get('解释', '')[:60]}")
                success = True
                break
            except Exception as e:
                if attempt >= args.retries:
                    print(f"[ERR] {gid}: {e}")
                    failed += 1
                    audit_lines.append({"id": gid, "error": str(e), "prompt": user_prompt})
                time.sleep(min(30, 2 ** attempt))

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Re-read all results (including resumed)
    all_results = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    print(f"\nJudged: {len(all_results)}, failed API calls: {failed}")

    # Compute metrics
    overall = compute_metrics(all_results)
    by_scenario = compute_metrics_by_scenario(all_results)

    report = {
        "overall": overall,
        "by_scenario": by_scenario,
        "config": {
            "generations": str(gen_path),
            "annotations": str(ann_path),
            "model": args.model,
        },
    }

    # Save report
    report_path = output_dir / "judge_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Save markdown report
    md = [
        "# JDDC Rebuild V2 Judge Report",
        "",
        "## 总体指标",
        "",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| 总样本 | {overall.get('total', 0)} |",
        f"| 通过率 | {overall.get('通过率', 0)}% |",
        f"| 自动解决率 | {overall.get('自动解决率', 0)}% |",
        f"| 准确率 | {overall.get('准确率', 0)}% |",
        "",
        "## 各维度均分",
        "",
        f"| 维度 | 均分 |",
        f"|---|---|",
    ]
    for d in DIMENSIONS:
        md.append(f"| {d} | {overall.get('各维度均分', {}).get(d, '-')} |")
    md += [
        "",
        "## 失败标签分布",
        "",
        "| 标签 | 数量 |",
        "|---|---|",
    ]
    for tag, count in overall.get("失败标签分布", {}).items():
        md.append(f"| {tag} | {count} |")
    md += [
        "",
        "## 各类别指标",
        "",
        "| 类别 | 总数 | 通过率 | 解决率 | 准确率 |",
        "|---|---|---|---|---|",
    ]
    for scenario, m in by_scenario.items():
        md.append(f"| {scenario} | {m.get('total', 0)} | {m.get('通过率', 0)}% | {m.get('自动解决率', 0)}% | {m.get('准确率', 0)}% |")

    md_path = output_dir / "judge_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n=== Overall Metrics ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\nReport: {report_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
