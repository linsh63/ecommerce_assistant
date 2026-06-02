"""Augment DPO preference pairs with multi-objective error types.

Adds pairs targeting wrong_intent, missing_action, and history_ignored
on top of the existing too_generic pairs. chosen = SFT training response,
rejected = DeepSeek-generated response with a specific error injected.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/sft_train.jsonl"
DEFAULT_EXISTING_DPO = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/dpo_preference_pairs.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/dpo_preference_pairs_v2.jsonl"

# ── Prompt templates per error type ──────────────────────────────────────

SYSTEM_WRONG_INTENT = """你是电商客服错误回复构造专家。你的任务是针对给定的客户问题和原始正确回复，构造一个"错误意图"的回复。

错误规则（只犯这一种错误）：
1. 把用户的问题当成另一个不同场景来回答。比如用户问物流，你回答退款流程；用户问商品参数，你回答优惠券问题。
2. 回复仍然要像正常客服——不能胡说八道，只是"答非所问"。
3. 字数与原始回复接近（相差不超过25字）。

输出严格 JSON：
{{"rejected": "故意答非所问的回复", "pretend_scenario": "你假装在回答什么场景"}}"""

SYSTEM_MISSING_ACTION = """你是电商客服错误回复构造专家。你的任务是针对给定的客户问题和原始正确回复，构造一个"缺少操作路径"的回复。

错误规则（只犯这一种错误）：
1. 回复方向正确、信息准确，但只说"做什么"不说"怎么做"——不给具体入口、不给操作步骤。
2. 例如：正确回复说"订单详情页→申请售后→选择退货"，你的回复说"建议您申请售后"但不给路径。
3. 字数与原始回复接近（相差不超过25字）。

输出严格 JSON：
{{"rejected": "方向正确但缺少路径的回复"}}"""

SYSTEM_HISTORY_IGNORED = """你是电商客服错误回复构造专家。你的任务是针对给定的多轮对话和原始正确回复，构造一个"忽略历史信息"的回复。

错误规则（只犯这一种错误）：
1. 只看用户最后一条消息，完全忽略历史对话中用户提供过的关键信息（如订单号、之前说过的需求、已确认的事项）。
2. 导致回复虽然看起来没问题，但实际上没有利用历史中已有的重要信息。
3. 字数与原始回复接近（相差不超过25字）。

输出严格 JSON：
{{"rejected": "忽略历史信息的回复"}}"""

# ── API call ──────────────────────────────────────────────────────────────

def call_api(args: argparse.Namespace, system: str, user_prompt: str) -> dict[str, Any]:
    endpoint = args.api_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.model,
        "temperature": 0.7,
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
    return _parse_json(body["choices"][0]["message"]["content"])


def _parse_json(text: str) -> dict[str, Any]:
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


# ── Build user prompts ────────────────────────────────────────────────────

def build_user_prompt(row: dict[str, Any], error_type: str) -> str:
    msgs = row.get("messages") or []
    last_user = msgs[-2].get("content", "") if len(msgs) >= 2 else ""
    original_response = msgs[-1].get("content", "") if msgs else ""
    scenario = row.get("meta", {}).get("scenario", "电商客服")

    parts = [f"场景类别：{scenario}"]

    if error_type == "history_ignored" and len(msgs) > 2:
        history = msgs[:-2]
        parts.append("")
        parts.append("历史对话：")
        for m in history:
            role = "用户" if m.get("role") == "user" else "客服"
            parts.append(f"{role}：{m.get('content', '')}")

    parts.extend([
        "",
        f"用户最后提问：{last_user}",
        f"原始正确回复（{len(original_response)}字）：{original_response}",
        "",
    ])

    if error_type == "wrong_intent":
        # List other scenarios as options for the wrong answer
        all_scenarios = [
            "物流-查件催发", "物流-配送调整", "退款-取消价保",
            "售后-退换维修", "售后-质量异常", "发票-资质服务",
            "价格活动-优惠赠品", "商品咨询-使用方法", "商品咨询-参数",
            "购买决策-推荐对比", "投诉-安抚升级", "闲聊-礼貌收尾",
        ]
        others = [s for s in all_scenarios if s != scenario]
        random.shuffle(others)
        parts.append(f"请假装这是'{others[0]}'场景的问题来回答，故意答非所问。")
    elif error_type == "missing_action":
        parts.append("请保留正确回复的信息和方向，但把所有具体操作入口和步骤去掉，只留空洞的建议。")
    elif error_type == "history_ignored":
        parts.append("请只根据用户的最后一句话回答，故意忽略历史对话中的所有关键信息。")

    return "\n".join(parts)


# ── Validation ────────────────────────────────────────────────────────────

def validate_pair(chosen: str, rejected: str, error_type: str, max_len_gap: int = 25) -> tuple[dict | None, str | None]:
    chosen = str(chosen or "").strip()
    rejected = str(rejected or "").strip()

    if len(chosen) < 15:
        return None, "chosen_too_short"
    if len(rejected) < 15:
        return None, "rejected_too_short"
    if chosen == rejected:
        return None, "chosen_equals_rejected"
    if abs(len(chosen) - len(rejected)) > max_len_gap:
        return None, f"length_gap_{abs(len(chosen)-len(rejected))}"

    return {
        "chosen": chosen,
        "rejected": rejected,
        "error_type": error_type,
    }, None


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", default=str(DEFAULT_TRAIN))
    p.add_argument("--existing-dpo", default=str(DEFAULT_EXISTING_DPO))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--target-per-type", type=int, default=300, help="Target pairs per error type")
    p.add_argument("--max-len-gap", type=int, default=25, help="Max char diff between chosen/rejected")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--no-response-format", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api-base-url", default=os.environ.get("DEEPSEEK_API_BASE_URL") or "https://api.deepseek.com")
    p.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    p.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--error-types", default="wrong_intent,missing_action,history_ignored",
                   help="Comma-separated error types to generate")
    return p.parse_args()


def main():
    args = parse_args()
    error_types = [t.strip() for t in args.error_types.split(",")]
    rng = random.Random(args.seed)

    # ── Load training data ─────────────────────────────────────────────
    with open(args.train_file, encoding="utf-8") as f:
        train_rows = [json.loads(l) for l in f if l.strip()]
    print(f"训练数据: {len(train_rows)} 条")

    # ── Load existing DPO pairs ────────────────────────────────────────
    existing = []
    existing_path = Path(args.existing_dpo)
    if existing_path.exists():
        with existing_path.open(encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    existing.append(json.loads(l))
        # Keep track of existing prompts to avoid duplicates
        existing_prompts = {p.get("prompt", "")[:80] for p in existing}
        print(f"现有 DPO 对: {len(existing)} 条")
    else:
        existing_prompts = set()
        print("无现有 DPO 数据，从头生成")

    # ── Filter out eval set ────────────────────────────────────────────
    eval_path = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/eval_test.jsonl"
    eval_ids = set()
    if eval_path.exists():
        with eval_path.open(encoding="utf-8") as f:
            eval_ids = {json.loads(l)["id"] for l in f if l.strip()}
    train_rows = [r for r in train_rows if r.get("id") not in eval_ids]
    print(f"排除评测集后: {len(train_rows)} 条")

    # ── Prepare by error type ──────────────────────────────────────────
    systems = {
        "wrong_intent": SYSTEM_WRONG_INTENT,
        "missing_action": SYSTEM_MISSING_ACTION,
        "history_ignored": SYSTEM_HISTORY_IGNORED,
    }

    # For history_ignored, only use multi-turn samples
    multi_turn = [r for r in train_rows if len(r.get("messages", [])) > 2]
    print(f"多轮对话候选: {len(multi_turn)} 条")

    all_new_pairs: list[dict] = []
    stats: dict[str, dict] = {}

    for error_type in error_types:
        print(f"\n{'='*60}")
        print(f"生成 {error_type} 偏好对 (目标: {args.target_per_type})")

        # Select candidates
        if error_type == "history_ignored":
            candidates = multi_turn
        else:
            candidates = train_rows

        rng.shuffle(candidates)
        accepted = 0
        rejected_count = Counter()

        for row in candidates:
            if accepted >= args.target_per_type:
                break

            rid = row.get("id", "")
            msgs = row.get("messages", [])
            chosen_text = msgs[-1].get("content", "") if msgs else ""
            prompt_text = msgs[-2].get("content", "") if len(msgs) >= 2 else ""

            # Skip if prompt already used
            if prompt_text[:80] in existing_prompts:
                continue

            user_prompt = build_user_prompt(row, error_type)
            system = systems[error_type]

            for attempt in range(args.retries + 1):
                try:
                    result = call_api(args, system, user_prompt)
                    rejected_text = result.get("rejected", "")
                    pair, reason = validate_pair(chosen_text, rejected_text, error_type, args.max_len_gap)

                    if pair:
                        pair["id"] = f"aug_{error_type}_{accepted:04d}"
                        pair["scenario"] = row.get("meta", {}).get("scenario", "")
                        pair["messages"] = row.get("messages", [])
                        pair["prompt"] = prompt_text
                        all_new_pairs.append(pair)
                        existing_prompts.add(prompt_text[:80])
                        accepted += 1
                        print(f"  [{error_type} {accepted}/{args.target_per_type}] {rid} | {pair['scenario']}")
                        break
                    else:
                        rejected_count[str(reason)] += 1
                except Exception as e:
                    if attempt >= args.retries:
                        print(f"  [ERR {error_type}] {rid}: {e}")
                    time.sleep(min(30, 2 ** attempt))

            if args.sleep > 0:
                time.sleep(args.sleep)

        stats[error_type] = {"accepted": accepted, "rejected": dict(rejected_count.most_common())}
        print(f"  {error_type}: {accepted} 对, 拒绝: {dict(rejected_count.most_common())}")

    # ── Merge and save ─────────────────────────────────────────────────
    all_pairs = existing + all_new_pairs
    rng.shuffle(all_pairs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"输出: {output_path}")
    print(f"总对数: {len(all_pairs)}")
    print(f"  现有: {len(existing)}")
    for et, s in stats.items():
        print(f"  {et}: {s['accepted']}")
    print(f"  字数差阈值: {args.max_len_gap} 字")

    # Scenario distribution
    scenarios = Counter(p.get("scenario", "?") for p in all_pairs)
    print(f"\n场景分布:")
    for s, c in scenarios.most_common():
        print(f"  {s}: {c}")
    error_counts = Counter(p.get("error_type", "too_generic") for p in all_pairs)
    print(f"\n错误类型分布:")
    for e, c in error_counts.most_common():
        print(f"  {e}: {c}")


if __name__ == "__main__":
    main()
