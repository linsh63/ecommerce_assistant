"""Build DPO preference pairs by asking DeepSeek to generate good-vs-bad response pairs."""

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
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/05_final/sft_train.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/jddc_rebuild_v2/dpo_preference_pairs.jsonl"

# ── Prompt ────────────────────────────────────────────────────────────────

def build_system_prompt(style_examples: list[dict]) -> str:
    """Build system prompt with real training data examples as style reference."""
    example_text = ""
    for i, ex in enumerate(style_examples, 1):
        u = ex.get("user", "")
        a = ex.get("assistant", "")
        example_text += f"\n风格参考{i}：\n客户：{u}\n客服：{a}\n"

    return f"""你是电商客服 SFT 偏好数据构造专家。针对同一客户问题，生成两条客服回复：一条好（chosen）、一条差（rejected）。

以下是你必须模仿的客服回复风格（来自真实训练数据）：{example_text}

chosen 规则（优先级从高到低）：
1. 【意图优先】先准确判断用户最后一句话的真实意图——他在问什么？想要什么？如果用户意图明确，直接回应；如果模糊，先确认意图再给建议。绝不要答非所问。
2. 【可执行路径】给出通用电商平台都有的操作入口——订单详情页、申请售后、联系在线客服、取消订单、查看物流。不要编造具体的按钮名称（如"点击修改尺码"）、不要编造平台特定规则（如"套装不支持价保"）、不要编造具体时效（如"3-5个工作日"）。
3. 【多轮上下文】如对话有历史，必须利用历史关键信息回答，不能忽略用户前面说过的重要细节。
4. 礼貌、简洁、自然。

rejected 规则（故意违反以制造对比）：
1. 【意图偏离】不仔细读用户最后一句话，按大致场景猜测回应，出现答非所问或方向偏差。
2. 【缺少路径】只说方向不说具体入口——用户问怎么做，只回"建议您申请售后"而不说"订单详情页→申请售后"。
3. 仍需看起来像客服回复，不能胡言乱语。

【重要】字数与质量约束：
- 先决定两条回复的字数（50-100字之间），chosen 和 rejected 写接近的字数，差不超过10字。
- 差距只在语义上：chosen = 意图准确 + 有操作入口；rejected = 意图偏离或缺少入口。
- 不要为了"具体"而编造——通用电商路径就够了。

输出严格 JSON：
{{"chosen": "意图准确、有操作入口的好回复", "rejected": "意图偏离或缺少入口的差回复", "gap_reason": "一句话"}}"""



def build_user_prompt(row: dict[str, Any]) -> str:
    msgs = row.get("messages") or []
    if len(msgs) >= 2:
        last_user = msgs[-2].get("content", "")
        history_msgs = msgs[:-2]
    else:
        last_user = ""
        history_msgs = []

    parts = []
    if history_msgs:
        parts.append("历史对话：")
        for m in history_msgs:
            role = "用户" if m.get("role") == "user" else "客服"
            parts.append(f"{role}：{m.get('content', '')}")
        parts.append("")

    scenario = row.get("meta", {}).get("scenario", "电商客服")
    parts.append(f"场景类别：{scenario}")
    parts.append(f"客户最后提问：{last_user}")
    parts.append("")
    parts.append("请针对以上客户的提问，生成两条质量有明显差距的客服回复对比。")

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


def call_api(args: argparse.Namespace, user_prompt: str, system: str) -> dict[str, Any]:
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
    return parse_api_json(body["choices"][0]["message"]["content"])


# ── Validation ────────────────────────────────────────────────────────────

def validate_pair(result: dict[str, Any], row: dict[str, Any]) -> tuple[dict | None, str | None]:
    chosen = str(result.get("chosen") or "").strip()
    rejected = str(result.get("rejected") or "").strip()

    if len(chosen) < 20:
        return None, "chosen_too_short"
    if len(rejected) < 10:
        return None, "rejected_too_short"
    if chosen == rejected:
        return None, "chosen_equals_rejected"
    if abs(len(chosen) - len(rejected)) > 15:
        return None, f"length_gap_too_large:{abs(len(chosen)-len(rejected))}"

    msgs = row.get("messages") or []
    last_user = msgs[-2]["content"] if len(msgs) >= 2 else ""
    scenario = row.get("meta", {}).get("scenario", "")

    return {
        "id": row.get("id"),
        "scenario": scenario,
        "messages": msgs,
        "prompt": last_user,
        "chosen": chosen,
        "rejected": rejected,
        "gap_reason": result.get("gap_reason", ""),
    }, None


# ── Style reference sampling ─────────────────────────────────────────────

def sample_style_examples(train_rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Sample real assistant replies from training set as style reference."""
    examples = []
    for row in rng.sample(train_rows, min(count * 3, len(train_rows))):
        msgs = row.get("messages") or []
        if len(msgs) < 2:
            continue
        last_user = msgs[-2].get("content", "")
        last_assistant = msgs[-1].get("content", "")
        if len(last_user) > 5 and 20 <= len(last_assistant) <= 200:
            examples.append({"user": last_user, "assistant": last_assistant})
        if len(examples) >= count:
            break
    return examples


# ── Sampling ──────────────────────────────────────────────────────────────

def sample_from_train(train_path: Path, target: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    with open(train_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # Reserve a few as style examples
    style_examples = sample_style_examples(rows, 3, rng)

    # Group by scenario for balanced sampling
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("meta", {}).get("scenario", "unknown")].append(row)

    # Sample proportionally, min 50 per category
    sampled = []
    for scenario, group in sorted(grouped.items()):
        n = max(50, round(target * len(group) / len(rows)))
        n = min(n, len(group))
        sampled.extend(rng.sample(group, k=n))

    rng.shuffle(sampled)
    return sampled[:target], style_examples


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--target", type=int, default=2000, help="Target number of preference pairs")
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
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample from training set
    sampled, style_examples = sample_from_train(Path(args.train_file), args.target, args.seed)
    print(f"Sampled {len(sampled)} scenarios from training set")
    print(f"Style references: {len(style_examples)} samples")

    # Build system prompt once with style references
    system_prompt = build_system_prompt(style_examples)

    # Resume
    done_ids = set()
    if not args.no_resume and output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line).get("id", ""))
        print(f"Resuming: {len(done_ids)} already done")

    accepted = 0
    rejected_count = Counter()
    for row in sampled:
        if row.get("id") in done_ids:
            accepted += 1
            continue

        user_prompt = build_user_prompt(row)
        for attempt in range(args.retries + 1):
            try:
                result = call_api(args, user_prompt, system_prompt)
                pair, reason = validate_pair(result, row)
                if pair:
                    with output_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
                    done_ids.add(row["id"])
                    accepted += 1
                    print(f"[OK {accepted}/{len(sampled)}] {row['id']} | {pair['scenario']} | gap: {pair.get('gap_reason', '')[:60]}")
                    break
                else:
                    rejected_count[str(reason)] += 1
            except Exception as e:
                if attempt >= args.retries:
                    print(f"[ERR] {row['id']}: {e}")
                time.sleep(min(30, 2 ** attempt))

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Report
    print(f"\nAccepted: {accepted}")
    print(f"Rejected: {dict(rejected_count.most_common())}")
    print(f"Output: {output_path}")

    # Count scenarios
    pairs = []
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))

    counts = Counter(p.get("scenario", "?") for p in pairs)
    print(f"\nFinal pairs: {len(pairs)}")
    for s, c in counts.most_common():
        print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
