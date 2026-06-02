"""Rewrite JDDC rebuild-v2 source samples with a DeepSeek-compatible chat API."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2" / "03_split" / "train_source.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2" / "04_deepseek_rewritten"

VALID_INTENTS = {
    "物流-查件催发",
    "物流-配送调整",
    "售后-退换维修",
    "售后-质量异常",
    "退款-取消价保",
    "商品咨询-参数",
    "商品咨询-使用方法",
    "购买决策-推荐对比",
    "价格活动-优惠赠品",
    "发票-资质服务",
    "投诉-安抚升级",
    "闲聊-礼貌收尾",
}

CATEGORY_GUIDE = {
    "物流-查件催发": "用户关心发货、物流轨迹、快递进展、催促配送。回复要给查看轨迹、催促或联系平台核实的路径，不能编造具体到达时间。",
    "物流-配送调整": "用户想改地址、改配送时间、拒收、自提、转寄等。回复要区分出库前后，不能承诺一定能改。",
    "售后-退换维修": "用户咨询退货、换货、维修、保修、取件、寄回。回复要说明订单详情页售后入口、商品状态和凭证要求。",
    "售后-质量异常": "用户反馈破损、错发、漏发、异味、不能用、质量问题。回复要安抚，并要求照片视频等凭证提交售后核实。",
    "退款-取消价保": "用户咨询退款、取消订单、价保、差价、到账。回复要说明申请入口、规则、原支付渠道返回，不承诺具体金额或到账时间。",
    "商品咨询-参数": "用户询问尺寸、材质、型号、容量、保质期、适配范围等。回复要围绕参数和详情页判断点回答。",
    "商品咨询-使用方法": "用户询问安装、连接、清洗、设置、使用步骤。回复要给简洁步骤或排查路径。",
    "购买决策-推荐对比": "用户需要推荐、对比、怎么选。回复要按场景、预算、核心参数给建议。",
    "价格活动-优惠赠品": "用户咨询优惠券、满减、赠品、包邮、活动价。回复要直接说明看结算页/活动规则及处理路径。",
    "发票-资质服务": "用户咨询发票、抬头、税号、资质、客服电话、服务保障。回复要说明订单详情开票或联系平台客服。",
    "投诉-安抚升级": "用户表达不满、投诉、催处理、认为被敷衍。回复要先道歉理解，再给反馈或平台介入路径。",
    "闲聊-礼貌收尾": "用户表达感谢、确认、结束对话。回复要简短礼貌收尾，不要展开复杂流程。",
}

PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")
LONG_DIGIT_RE = re.compile(r"\d{8,}")
OVERPROMISE_RE = re.compile(r"(已为您查询|我这边看到|我已经|立即处理|马上处理|一定|务必|专员跟进|优先处理|财务正在|拦截成功)")


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


# 功能：追加 JSONL。
def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


# 功能：写出缩进 JSON。
def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 功能：从模型返回文本中解析 JSON。
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


# 功能：构造系统提示词。
def build_system_prompt() -> str:
    return (
        "你是电商客服 SFT 数据改写专家。你的任务是把真实客服原始对话改写成高质量训练样本。\n"
        "必须遵守：\n"
        "1. 保留用户真实诉求和场景类别，不要把问题改成别的问题。\n"
        "2. 去除具体品牌、具体商品专名、订单号、手机号、地址、姓名等隐私和强实体信息。\n"
        "3. 不要输出方括号占位符，例如 [数字x]、[ORDERID]、[商品信息]；需要数字时改成自然数字或泛化说法。\n"
        "4. 客服回复要直接回答问题，不能只说稍等、帮您查询、请耐心等待。\n"
        "5. 可以说明以订单详情页、活动页、售后页为准，但必须给出可执行路径。\n"
        "6. 不要编造已查询到的订单状态、库存、物流节点、退款金额、具体时效。\n"
        "7. 多轮对话中，如果最后回答依赖历史信息，必须使用历史里的关键信息。\n"
        "8. 回复风格要礼貌、简洁、自然，通常 30-160 个中文字符。\n"
        "9. 场景分类只写在 JSON 的 scenario 字段里，不要写进 assistant content。\n"
        "10. 最后一条 assistant content 只写客服回复正文，不要包含“分类：”或“回复：”前缀。\n"
        "11. 输出必须是严格 JSON，不要 Markdown，不要代码块。"
    )


# 功能：构造单条改写任务提示。
def build_user_prompt(record: dict[str, Any]) -> str:
    intent = str(record.get("meta", {}).get("intent") or record.get("meta", {}).get("scenario") or "")
    target_scenario = intent if intent in VALID_INTENTS else "请根据原始对话从 allowed_scenarios 中选择一个最合适类别"
    history = record.get("history") or []
    payload = {
        "task": "rewrite_jddc_customer_service_sample",
        "target_scenario": target_scenario,
        "allowed_scenarios": sorted(VALID_INTENTS),
        "scenario_definition": CATEGORY_GUIDE.get(intent, "按类别名称理解客服场景。"),
        "source_quality_flags": record.get("meta", {}).get("quality_flags", []),
        "history_required": bool(record.get("meta", {}).get("history_required")),
        "raw_dialogue": {
            "history": [{"user": pair[0], "assistant": pair[1]} for pair in history if isinstance(pair, list) and len(pair) >= 2],
            "current_user": record.get("prompt", ""),
            "current_assistant": record.get("response", ""),
        },
        "output_schema": {
            "id": record.get("id"),
            "scenario": "必须是 allowed_scenarios 中的一个类别",
            "messages": [
                {"role": "user", "content": "改写后的用户历史或当前问题"},
                {"role": "assistant", "content": "改写后的客服历史或最终回答"},
            ],
            "rewrite_notes": "一句话说明主要改写点，尤其说明是否使用了历史信息",
        },
        "requirements": [
            "messages 必须从用户开始，user/assistant 交替，最后一条必须是 assistant。",
            "可以保留多轮历史，但要更可读；无价值历史可以删减。",
            "最后一条 assistant 只写客服回复正文，不要写“分类：”和“回复：”。",
            "如果 target_scenario 是具体类别，scenario 必须严格等于 target_scenario；如果 target_scenario 要求你选择类别，scenario 必须是 allowed_scenarios 中最合适的一类。",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# 功能：调用 OpenAI-compatible chat completions API。
def call_chat_api(args: argparse.Namespace, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    endpoint = args.api_base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if not args.no_response_format:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return parse_api_json(body["choices"][0]["message"]["content"])


# 功能：兼容模型误带“分类/回复”包装时，只保留客服回复正文。
def clean_assistant_reply(text: str) -> str:
    reply = str(text or "").strip()
    match = re.search(r"回复[:：]\s*(.+)$", reply, flags=re.DOTALL)
    if match:
        reply = match.group(1).strip()
    reply = re.sub(r"^分类[:：].*?(?:\n|$)", "", reply, flags=re.DOTALL).strip()
    reply = re.sub(r"^回复[:：]\s*", "", reply).strip()
    return reply.strip(" \n\t\"'“”")


# 功能：校验改写结果是否能进入训练。
def validate_rewritten(raw_result: dict[str, Any], source: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    expected_intent = str(source.get("meta", {}).get("intent") or "")
    scenario = str(raw_result.get("scenario") or "").strip()
    if scenario not in VALID_INTENTS:
        return None, f"invalid_scenario:{scenario}"
    messages = raw_result.get("messages")
    if not isinstance(messages, list):
        return None, "bad_messages_type"
    if not messages:
        return None, "empty_final_reply"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None, "bad_message_item"
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            return None, f"bad_role_at_{index}"
        content = str(message.get("content") or "").strip()
        if not content:
            return None, f"empty_content_at_{index}"
        message["content"] = content
    if messages[-1].get("role") != "assistant":
        return None, f"bad_role_at_{len(messages) - 1}"
    final = clean_assistant_reply(str(messages[-1].get("content") or ""))
    if not final:
        return None, "empty_final_reply"
    messages[-1]["content"] = final
    text = "\n".join(str(message.get("content") or "") for message in messages)
    if PLACEHOLDER_RE.search(text):
        return None, "placeholder_leak"
    if LONG_DIGIT_RE.search(text):
        return None, "long_digit_leak"
    result = {
        "id": source.get("id"),
        "messages": messages,
        "meta": {
            **(source.get("meta") or {}),
            "intent": scenario,
            "scenario": scenario,
            "source_rule_intent": expected_intent,
            "source_id": source.get("source_id") or source.get("id"),
            "source_prompt": source.get("prompt"),
            "source_response": source.get("response"),
            "rewrite_model": raw_result.get("rewrite_model"),
            "rewrite_notes": raw_result.get("rewrite_notes", ""),
            "rewritten": True,
        },
    }
    return result, None


# 功能：失败重试。
def rewrite_one_with_retry(args: argparse.Namespace, source: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(source)
    audit = {"id": source.get("id"), "system_prompt": system_prompt, "user_prompt": user_prompt}
    last_error: str | None = None
    for attempt in range(args.retries + 1):
        try:
            raw_result = call_chat_api(args, system_prompt, user_prompt)
            raw_result.setdefault("rewrite_model", args.model)
            rewritten, reason = validate_rewritten(raw_result, source)
            audit["raw_result"] = raw_result
            audit["reject_reason"] = reason
            return rewritten, reason, audit
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt >= args.retries:
                break
            time.sleep(min(30, 2**attempt))
    audit["error"] = last_error
    return None, "api_error", audit


# 功能：写出 dry-run prompt 预览。
def write_prompt_preview(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# JDDC Rebuild V2 DeepSeek Prompt Preview", "", "## System Prompt", "", "```text", build_system_prompt(), "```", ""]
    for index, row in enumerate(rows, start=1):
        lines.extend([f"## Sample {index}: {row.get('id')}", "", "```json", build_user_prompt(row), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# 功能：解析参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=None, help="Defaults to <input stem>_rewritten.jsonl.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sample", action="store_true", help="Randomly sample instead of taking the first N.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--api-base-url", default=os.environ.get("DEEPSEEK_API_BASE_URL") or "https://api.deepseek.com")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat")
    return parser.parse_args()


# 功能：主流程，支持 dry-run prompt 审阅和小样本真实改写。
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    if args.sample:
        selected = rows if len(rows) <= args.limit else rng.sample(rows, k=args.limit)
    else:
        selected = rows[: args.limit]

    preview_path = output_dir / "prompt_preview.md"
    write_prompt_preview(selected[: min(5, len(selected))], preview_path)
    if args.dry_run:
        print(f"prompt_preview={preview_path}")
        return
    if not args.api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY. Use --dry-run to inspect prompts without API calls.")

    output_name = args.output_name or f"{input_path.stem}_rewritten.jsonl"
    output_path = output_dir / output_name
    audit_path = output_dir / "api_prompt_audit.jsonl"
    failed_path = output_dir / "failed_rewrites.jsonl"
    if args.no_resume:
        for path in [output_path, audit_path, failed_path]:
            if path.exists():
                path.unlink()

    existing_ids = {row.get("id") for row in read_jsonl(output_path)}
    accepted = 0
    rejected = Counter()
    for row in selected:
        if row.get("id") in existing_ids:
            continue
        rewritten, reason, audit = rewrite_one_with_retry(args, row)
        append_jsonl(audit_path, [audit])
        if rewritten:
            append_jsonl(output_path, [rewritten])
            accepted += 1
            print(f"[OK] {row.get('id')}")
        else:
            rejected[str(reason)] += 1
            append_jsonl(failed_path, [{"id": row.get("id"), "reason": reason, "source": row}])
            print(f"[WARN] {row.get('id')}: {reason}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    report = {
        "input": str(input_path),
        "selected_total": len(selected),
        "accepted": accepted,
        "rejected": dict(rejected.most_common()),
        "output_path": str(output_path),
        "audit_path": str(audit_path),
        "failed_path": str(failed_path),
        "prompt_preview": str(preview_path),
    }
    write_json(report, output_dir / "rewrite_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
