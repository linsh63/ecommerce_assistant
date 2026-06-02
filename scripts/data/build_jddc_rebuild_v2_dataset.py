"""Build classified, lightly cleaned, and split JDDC source data for rebuild v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "extract_train.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2"

INTENTS = [
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
]

INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "物流-查件催发": (
        "发货",
        "发出",
        "到货",
        "多久到",
        "什么时候到",
        "物流",
        "快递",
        "配送",
        "派送",
        "运单",
        "催",
        "查件",
        "没更新",
        "到哪",
        "揽收",
    ),
    "物流-配送调整": (
        "改地址",
        "地址错",
        "换地址",
        "改配送",
        "配送时间",
        "派送时间",
        "收货时间",
        "修改收货",
        "修改配送",
        "自提",
        "驿站",
        "拒收",
        "转寄",
        "拦截快递",
        "送货上门",
        "不上门",
    ),
    "售后-退换维修": (
        "退货",
        "换货",
        "维修",
        "返修",
        "售后",
        "保修",
        "取件",
        "上门取",
        "寄回",
        "七天无理由",
        "拆封",
        "退换",
    ),
    "售后-质量异常": (
        "坏了",
        "破损",
        "碎了",
        "裂",
        "漏发",
        "少发",
        "错发",
        "质量",
        "异味",
        "不能用",
        "不亮",
        "没反应",
        "瑕疵",
        "划痕",
    ),
    "退款-取消价保": (
        "退款",
        "取消",
        "价保",
        "差价",
        "退差",
        "到账",
        "原路退",
        "白条",
        "审核退款",
        "退钱",
        "撤销",
        "降价",
    ),
    "商品咨询-参数": (
        "尺寸",
        "大小",
        "材质",
        "型号",
        "规格",
        "容量",
        "保质期",
        "生产日期",
        "适配",
        "颜色",
        "重量",
        "参数",
        "成分",
        "含量",
        "正品",
        "原装",
        "真假",
        "授权",
    ),
    "商品咨询-使用方法": (
        "怎么用",
        "如何使用",
        "安装",
        "连接",
        "设置",
        "清洗",
        "保养",
        "配对",
        "充电",
        "说明书",
        "操作",
        "使用方法",
    ),
    "购买决策-推荐对比": (
        "推荐",
        "哪个好",
        "怎么选",
        "区别",
        "对比",
        "适合",
        "划算",
        "买哪",
        "选择",
        "家用",
        "老人用",
        "孩子用",
    ),
    "价格活动-优惠赠品": (
        "优惠",
        "优惠券",
        "满减",
        "活动",
        "赠品",
        "赠送",
        "包邮",
        "会员价",
        "折扣",
        "促销",
        "京豆",
        "价格",
    ),
    "发票-资质服务": (
        "发票",
        "开票",
        "抬头",
        "税号",
        "资质",
        "客服",
        "电话",
        "证明",
        "电子票",
        "服务",
        "保障",
    ),
    "投诉-安抚升级": (
        "投诉",
        "差评",
        "生气",
        "不满意",
        "赔偿",
        "补偿",
        "态度",
        "欺骗",
        "太慢",
        "敷衍",
        "没人管",
        "升级",
    ),
    "闲聊-礼貌收尾": (
        "谢谢",
        "感谢",
        "好的",
        "好吧",
        "知道了",
        "没事",
        "不用了",
        "再见",
        "嗯嗯",
        "可以了",
    ),
}

NOISE_RE = re.compile(r"[\s，。,.!?！？、:：;；~～'\"“”‘’（）()《》<>【】\[\]\-_]+")
PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")
LONG_DIGIT_RE = re.compile(r"\d{8,}")
REPEATED_SPAN_RE = re.compile(r"(.{3,24})\1{2,}")
HTML_RE = re.compile(r"<[^>]+>")
CONCRETE_PRIVATE_RE = re.compile(r"(订单号|手机号|收货地址|身份证|银行卡|姓名|电话)")
OVERPROMISE_RE = re.compile(
    r"(已为您查询|马上为您查询|正在为您核实|我这边看到|我已经|已记录|"
    r"一定|务必|立即|马上处理|专员跟进|优先处理|给您申请|帮您申请|"
    r"仓库会|快递会|财务正在|拦截成功|免费更换|补偿您)"
)
LOW_VALUE_RE = re.compile(
    r"(有什么问题我可以帮您处理或解决呢|请问还有其他还可以帮到您的吗|"
    r"请问您是咨询之前的问题还是有其他的问题需要处理呢|"
    r"感谢您对京东的支持|请您稍等|马上为您查询|正在查看|您好[，,]?请问有什么可以帮)"
)
GENERIC_PREVIOUS_OR_OTHER_RE = re.compile(r"请问您是咨询之前的问题还是有其他的问题需要处理呢")
GENERIC_HELP_ONLY_RE = re.compile(
    r"^(您好[，,]?)?(亲爱的?客户[，,]?)?"
    r"(有什么问题我可以帮您处理或解决呢|请问有什么可以帮|您好请问有什么可以帮)"
    r"[~～。!！?？\\s]*$"
)
GENERIC_HELP_PHRASE_RE = re.compile(r"(有什么问题我可以帮您处理或解决呢|请问有什么可以帮|您好[，,]?请问有什么可以帮)")
WAITING_PHRASE_RE = re.compile(
    r"(请您稍等一下，正在为您核实处理中哦|马上核实情况，请您稍等哈|"
    r"还请您稍等，马上为您查询|小妹为您看看，您稍等哦|"
    r"请您稍等，小妹正在查询中|请稍等|稍等|正在查看|在查询|"
    r"我找下哈|我查看下|帮您查看一下|马上为您查询|正在为您查询|"
    r"还辛苦您再等待)"
)
PRODUCT_SNAPSHOT_RE = re.compile(r"(\[商品快照\]|商品快照)")
CONFIRM_PRODUCT_RE = re.compile(r"(请问是这个商品吗|核实一下商品的信息|为了更好的解决您的问题)")
ACTION_HINT_RE = re.compile(r"(可以|无法|不能|需要|建议|预计|显示|已经|等待|退款|配送|发货|拒收|修改|取消|售后|订单|物流)")
CONTACT_DUMP_RE = re.compile(r"(地址[:：].*){2,}|(电话[:：].*){2,}")
ORDER_ONLY_RE = re.compile(r"^\s*(\[?ORDERID[_\d]*\]?|订单号[:：]?\s*\[?ORDERID[_\d]*\]?|[0-9\s]{8,})\s*$", re.I)
JD_ITEM_URL_RE = re.compile(r"https?://item\.jd\.com/\d+\.html")
PRODUCT_ID_NOISE_RE = re.compile(r"\b\d{5,}\b")
ASK_ORDER_ONLY_RE = re.compile(r"(请|麻烦|还麻烦).{0,8}(提供|发|给).{0,8}订单号|订单编号.*是吗|这个订单是吗")
ORDER_REQUEST_RE = re.compile(
    r"(提供下?订单号|发下?订单号|复制下?订单|选择一下.*订单|点击.*订单|"
    r"订单编号.*发给我|需要.*订单号|看下.*订单)"
)
ORDER_REQUEST_ANSWER_HINT_RE = re.compile(
    r"(无法|不能|支持|不支持|无需|不需要|需要您先|建议您|"
    r"预计|等待|退款|退货|换货|维修|保修|配送|发货|拒收|修改|取消|售后|价保|发票|寄回|上门)"
)
NO_SUBSTANTIVE_RESPONSE_RE = re.compile(
    r"^(好的|好|嗯|恩|是的|不是|可以|不可以|稍等|您好|亲|亲亲|请稍等|"
    r"正在查看|我为您看看|我帮您看一下|马上核实情况|请问有什么可以帮助您的么)[~～。!！?？\\s#E-s\\[数字x\\]]*$"
)
QUESTION_ONLY_RESPONSE_RE = re.compile(r"^[^。！？!?]{0,32}(吗|么|呢|是吗|对吗|可以吗)[？?]?$")
CONFLICT_RE = re.compile(
    r"(不能退.*可以退|可以退.*不能退|不支持.*支持|支持.*不支持|"
    r"无法取消.*可以取消|可以取消.*无法取消|没有货.*有货|有货.*没有货)"
)


# 功能：去掉客服过场话后，判断回复是否还剩下可学习的实质处理信息。
def has_substantive_answer(response: str) -> bool:
    cleaned = GENERIC_PREVIOUS_OR_OTHER_RE.sub("", response)
    cleaned = WAITING_PHRASE_RE.sub("", cleaned)
    cleaned = LOW_VALUE_RE.sub("", cleaned)
    cleaned = re.sub(r"(您好|亲亲|亲爱的客户|亲爱的|亲|哈|哦|呢|呀|~|～)", "", cleaned)
    return compact_len(cleaned) >= 18 and bool(ACTION_HINT_RE.search(cleaned))


# 功能：读取原始 JSON 列表。
def read_raw_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError(f"Raw data must be a JSON list: {path}")
    return [row for row in value if isinstance(row, dict)]


# 功能：写出紧凑 JSONL。
def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


# 功能：写出缩进 JSON。
def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 功能：生成宽松去重 key。
def normalize_key(text: Any) -> str:
    return NOISE_RE.sub("", str(text or "")).lower()


# 功能：计算不含标点空白的长度。
def compact_len(text: Any) -> int:
    return len(normalize_key(text))


# 功能：稳定短哈希，用于 source_id 和分组。
def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


# 功能：清理轻量文本噪声，但不做深度改写。
def normalize_text(text: Any) -> str:
    value = str(text or "")
    value = HTML_RE.sub("", value)
    value = value.replace("小妹", "我").replace("妹子", "我")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


# 功能：规范化历史对话结构。
def normalize_history(history: Any, max_turns: int) -> list[list[str]]:
    normalized: list[list[str]] = []
    if not isinstance(history, list):
        return normalized
    for item in history[-max_turns:]:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        user = normalize_text(item[0])
        assistant = normalize_text(item[1])
        if user and assistant:
            normalized.append([user, assistant])
    return normalized


# 功能：将 prompt/response/history 转为 messages 训练格式，分类只保存在 meta 中，不进入模型输出目标。
def build_messages(history: list[list[str]], prompt: str, response: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return messages


# 功能：判断文本内部是否存在明显复读。
def has_stutter(text: str) -> bool:
    compact = normalize_key(text)
    if REPEATED_SPAN_RE.search(compact):
        return True
    for width in range(1, 7):
        min_repeat = 6 if width <= 2 else 4
        if re.search(rf"(.{{{width}}})\1{{{min_repeat - 1},}}", compact):
            return True
    return False


# 功能：根据关键词规则给样本打场景类别。
def classify_intent(prompt: str, response: str, history: list[list[str]]) -> tuple[str, dict[str, int]]:
    history_text = " ".join(f"{u} {a}" for u, a in history[-2:])
    text = f"{prompt} {response} {history_text}"
    if re.search(r"(退款|退钱|到账|价保|差价|退差|取消订单|订单取消)", prompt):
        return "退款-取消价保", {"退款-取消价保": 100}
    if re.search(r"(换.{0,8}(尺寸|颜色|型号|尺码)|小一号|大一号)", prompt):
        return "售后-退换维修", {"售后-退换维修": 101}
    if re.search(r"(退货|换货|返修|维修|保修|售后|上门取件|寄回)", prompt):
        return "售后-退换维修", {"售后-退换维修": 100}
    if re.search(r"(坏了|破损|碎了|漏发|少发|错发|质量|异味|不能用|不亮|没反应|瑕疵|划痕)", prompt):
        return "售后-质量异常", {"售后-质量异常": 100}
    if re.search(r"(正品|原装|真假|授权)", prompt):
        return "商品咨询-参数", {"商品咨询-参数": 100}
    if re.search(r"(改|修改|换).{0,6}(地址|收货|配送|派送|时间|电话|手机号)|拒收|自提|驿站", prompt):
        return "物流-配送调整", {"物流-配送调整": 99}
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 3 if keyword in prompt else 1
        if score:
            scores[intent] = score
    if not scores:
        return "unknown", {}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], INTENTS.index(item[0]) if item[0] in INTENTS else 99))
    return ranked[0][0], dict(ranked)


# 功能：返回硬删除原因和可改写质量标签。
def judge_quality(
    prompt: str,
    response: str,
    history: list[list[str]],
    intent: str,
    args: argparse.Namespace,
) -> tuple[str | None, list[str]]:
    flags: list[str] = []
    prompt_len = compact_len(prompt)
    response_len = compact_len(response)

    if not prompt or not response:
        return "empty_prompt_or_response", flags
    if intent == "unknown":
        flags.append("unknown_intent")
    if prompt_len < args.hard_min_prompt_chars:
        return "too_short_prompt", flags
    if response_len < args.hard_min_reply_chars:
        return "too_short_reply_hard", flags
    if response_len < args.soft_min_reply_chars:
        flags.append("short_reply_rewrite")
    if response_len > args.hard_max_reply_chars:
        return "too_long_reply_hard", flags
    if response_len > args.soft_max_reply_chars:
        flags.append("long_reply_rewrite")
    if len(history) > args.max_history_turns:
        flags.append("history_truncated")
    if PLACEHOLDER_RE.search(prompt) or PLACEHOLDER_RE.search(response):
        flags.append("placeholder_rewrite")
    if LONG_DIGIT_RE.search(prompt) or LONG_DIGIT_RE.search(response):
        flags.append("long_digit_rewrite")
    if LOW_VALUE_RE.search(response):
        flags.append("boilerplate_rewrite")
    if OVERPROMISE_RE.search(response):
        flags.append("overpromise_rewrite")
    if CONCRETE_PRIVATE_RE.search(response):
        flags.append("privacy_or_private_info_rewrite")
    if CONFLICT_RE.search(normalize_key(response)):
        flags.append("possible_conflict_rewrite")
    if has_stutter(prompt) or has_stutter(response):
        flags.append("stutter_rewrite")
    if history:
        flags.append("has_history")
    return None, flags


# 功能：处理一条原始样本为标准 source 记录。
def normalize_record(raw: dict[str, Any], index: int, args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    prompt = normalize_text(raw.get("prompt"))
    response = normalize_text(raw.get("response"))
    history = normalize_history(raw.get("history"), args.max_history_turns)
    intent, intent_scores = classify_intent(prompt, response, history)
    drop_reason, quality_flags = judge_quality(prompt, response, history, intent, args)

    source_key = json.dumps(
        {"prompt": prompt, "response": response, "history": history},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    source_id = f"raw_{index:06d}_{stable_hash(source_key, 8)}"
    record = {
        "id": source_id,
        "source_id": source_id,
        "prompt": prompt,
        "response": response,
        "history": history,
        "messages": build_messages(history, prompt, response),
        "meta": {
            "scenario": intent,
            "intent": intent,
            "intent_scores": intent_scores,
            "source_index": index,
            "source": "jddc_raw_extract_train",
            "source_prompt_key": normalize_key(prompt),
            "source_pair_key": normalize_key(prompt + response),
            "history_turns": len(history),
            "history_required": bool(history),
            "quality_flags": quality_flags,
            "pipeline_version": "jddc_rebuild_v2",
            "needs_rewrite": bool(quality_flags),
        },
    }
    return record, drop_reason


# 功能：按类别和源问题分组切分，降低 train/test 泄漏。
def stratified_group_split(
    rows: list[dict[str, Any]],
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups_by_intent: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        intent = str(row.get("meta", {}).get("intent") or "unknown")
        prompt_key = str(row.get("meta", {}).get("source_prompt_key") or row.get("id"))
        group_key = stable_hash(prompt_key, 10)
        groups_by_intent[intent][group_key].append(row)

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for intent in sorted(groups_by_intent):
        groups = list(groups_by_intent[intent].values())
        rng.shuffle(groups)
        total = sum(len(group) for group in groups)
        target_test = round(total * test_ratio)
        target_dev = round(total * dev_ratio)
        current_test = 0
        current_dev = 0
        for group in groups:
            if current_test < target_test:
                test_rows.extend(group)
                current_test += len(group)
            elif current_dev < target_dev:
                dev_rows.extend(group)
                current_dev += len(group)
            else:
                train_rows.extend(group)

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    rng.shuffle(test_rows)
    return train_rows, dev_rows, test_rows


# 功能：给候选样本打分，用于从宽松清洗池中挑选更适合改写的 4k-6k 样本。
def candidate_score(row: dict[str, Any]) -> float:
    flags = set(row.get("meta", {}).get("quality_flags") or [])
    prompt_len = compact_len(row.get("prompt", ""))
    reply_len = compact_len(row.get("response", ""))
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
        "unknown_intent": 1.0,
    }
    for flag, penalty in penalties.items():
        if flag in flags:
            score -= penalty
    return score


# 功能：判断样本是否适合进入 6000 条改写源；只影响抽样，不从宽松池删除。
def eligible_for_rewrite_selection(row: dict[str, Any]) -> tuple[bool, str | None]:
    prompt = str(row.get("prompt") or "")
    response = str(row.get("response") or "")
    prompt_len = compact_len(prompt)
    response_len = compact_len(response)
    placeholder_count = len(PLACEHOLDER_RE.findall(prompt + response))
    prompt_without_url = JD_ITEM_URL_RE.sub("", prompt)
    prompt_without_url = PRODUCT_ID_NOISE_RE.sub("", prompt_without_url)

    if str(row.get("meta", {}).get("intent") or "") == "unknown":
        return False, "unknown_intent_for_selection"
    if JD_ITEM_URL_RE.search(prompt) and compact_len(prompt_without_url) < 6:
        return False, "url_or_product_id_only_prompt"
    if ORDER_ONLY_RE.search(prompt):
        return False, "order_only_prompt"
    if GENERIC_PREVIOUS_OR_OTHER_RE.search(response):
        return False, "generic_previous_or_other_question"
    if GENERIC_HELP_PHRASE_RE.search(response):
        return False, "generic_help_phrase_response"
    if GENERIC_HELP_ONLY_RE.search(response):
        return False, "generic_help_only_response"
    if ASK_ORDER_ONLY_RE.search(response) and response_len < 45:
        return False, "ask_order_only_response"
    if ORDER_REQUEST_RE.search(response) and not ORDER_REQUEST_ANSWER_HINT_RE.search(response):
        return False, "order_request_without_answer"
    if NO_SUBSTANTIVE_RESPONSE_RE.search(response):
        return False, "no_substantive_response"
    if QUESTION_ONLY_RESPONSE_RE.search(response) and response_len < 34:
        return False, "question_without_answer"
    if WAITING_PHRASE_RE.search(response) and not has_substantive_answer(response):
        return False, "wait_without_answer"
    if PRODUCT_SNAPSHOT_RE.search(response) and response_len < 70:
        return False, "product_snapshot_without_answer"
    if CONFIRM_PRODUCT_RE.search(response):
        cleaned = CONFIRM_PRODUCT_RE.sub("", response)
        cleaned = PRODUCT_SNAPSHOT_RE.sub("", cleaned)
        if compact_len(cleaned) < 18 or not ACTION_HINT_RE.search(cleaned):
            return False, "product_confirm_without_answer"
    if CONTACT_DUMP_RE.search(response):
        return False, "contact_or_address_dump"
    if LOW_VALUE_RE.search(response) and not has_substantive_answer(response):
        return False, "pure_boilerplate_response"
    if placeholder_count >= 8:
        return False, "too_many_placeholders"
    if prompt_len < 4 or response_len < 12:
        return False, "too_short_for_rewrite_selection"
    if has_stutter(prompt) and prompt_len < 20:
        return False, "stutter_prompt_low_value"
    return True, None


# 功能：计算每个类别进入 DeepSeek 改写池的目标数量。
def allocate_selected_counts(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, int]:
    available = Counter(str(row.get("meta", {}).get("intent") or "unknown") for row in rows)
    caps: dict[str, int] = {}
    for intent in INTENTS:
        caps[intent] = min(available[intent], args.max_per_intent)
    if available["unknown"]:
        caps["unknown"] = min(available["unknown"], args.unknown_cap)

    target_total = min(args.target_total, sum(caps.values()))
    counts: dict[str, int] = {}
    for intent in INTENTS:
        counts[intent] = min(caps[intent], args.min_per_intent)
    if "unknown" in caps:
        counts["unknown"] = 0

    while sum(counts.values()) < target_total:
        progressed = False
        for intent in sorted(caps, key=lambda name: (counts.get(name, 0) / max(caps[name], 1), name)):
            if counts.get(intent, 0) < caps[intent]:
                counts[intent] = counts.get(intent, 0) + 1
                progressed = True
                if sum(counts.values()) >= target_total:
                    break
        if not progressed:
            break
    return {intent: count for intent, count in counts.items() if count > 0}


# 功能：从宽松清洗池中分层抽出最终改写源，避免头部类别和多轮样本过度占比。
def select_rewrite_sources(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    eligible_rows: list[dict[str, Any]] = []
    ineligible_reasons: Counter[str] = Counter()
    for row in rows:
        eligible, reason = eligible_for_rewrite_selection(row)
        if eligible:
            eligible_rows.append(row)
        else:
            row.setdefault("meta", {})["rewrite_selection_reject_reason"] = reason
            ineligible_reasons[str(reason)] += 1

    target_counts = allocate_selected_counts(eligible_rows, args)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        groups[str(row.get("meta", {}).get("intent") or "unknown")].append(row)

    selected: list[dict[str, Any]] = []
    selection_report: dict[str, Any] = {
        "eligible_total": len(eligible_rows),
        "ineligible_total": len(rows) - len(eligible_rows),
        "ineligible_reasons": dict(ineligible_reasons.most_common()),
        "target_counts": target_counts,
        "selected_counts": {},
    }
    for intent, target in target_counts.items():
        group = groups[intent][:]
        for row in group:
            row.setdefault("meta", {})["selection_score"] = round(candidate_score(row), 3)
        history_rows = [row for row in group if row.get("history")]
        single_rows = [row for row in group if not row.get("history")]
        history_target = round(target * args.target_history_ratio)

        def sort_key(row: dict[str, Any]) -> tuple[float, float]:
            return (candidate_score(row), rng.random())

        history_rows.sort(key=sort_key, reverse=True)
        single_rows.sort(key=sort_key, reverse=True)
        chosen = history_rows[:history_target] + single_rows[: target - min(history_target, len(history_rows))]
        if len(chosen) < target:
            chosen_ids = {row["id"] for row in chosen}
            rest = [row for row in group if row["id"] not in chosen_ids]
            rest.sort(key=sort_key, reverse=True)
            chosen.extend(rest[: target - len(chosen)])
        for row in chosen:
            row.setdefault("meta", {})["selected_for_rewrite"] = True
        selected.extend(chosen)
        selection_report["selected_counts"][intent] = {
            "total": len(chosen),
            "history": sum(1 for row in chosen if row.get("history")),
            "single": sum(1 for row in chosen if not row.get("history")),
            "available": len(group),
        }
    rng.shuffle(selected)
    return selected, selection_report


# 功能：写出按类别抽样的 Markdown 审阅文件。
def write_review_markdown(rows: list[dict[str, Any]], path: Path, title: str, per_intent: int, seed: int) -> None:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("meta", {}).get("intent") or "unknown")].append(row)

    lines = [f"# {title}", "", f"- total: {len(rows)}", f"- per_intent_limit: {per_intent}", ""]
    for intent in sorted(groups, key=lambda value: INTENTS.index(value) if value in INTENTS else 99):
        group = groups[intent]
        sampled = group if len(group) <= per_intent else rng.sample(group, k=per_intent)
        lines.extend([f"## {intent}", "", f"- total: {len(group)}", f"- shown: {len(sampled)}", ""])
        for row in sampled:
            flags = ", ".join(row.get("meta", {}).get("quality_flags") or []) or "clean"
            lines.extend(
                [
                    f"### {row['id']} | flags: {flags}",
                    "",
                    f"history_turns: {row.get('meta', {}).get('history_turns', 0)}",
                    "",
                    "```text",
                ]
            )
            for idx, (user, assistant) in enumerate(row.get("history") or [], start=1):
                lines.append(f"[历史{idx}-用户] {user}")
                lines.append(f"[历史{idx}-客服] {assistant}")
            lines.append(f"[当前用户] {row.get('prompt', '')}")
            lines.append(f"[原始客服] {row.get('response', '')}")
            lines.extend(["```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# 功能：写出被删除样本的抽样 Markdown。
def write_removed_markdown(rows: list[dict[str, Any]], path: Path, limit: int, seed: int) -> None:
    rng = random.Random(seed)
    sampled = rows if len(rows) <= limit else rng.sample(rows, k=limit)
    lines = ["# Removed Samples Review", "", f"- total_removed: {len(rows)}", f"- shown: {len(sampled)}", ""]
    for row in sampled:
        lines.extend(
            [
                f"## {row.get('id')} | {row.get('drop_reason')} | {row.get('intent')}",
                "",
                "```text",
                f"[用户] {row.get('prompt', '')}",
                f"[客服] {row.get('response', '')}",
                "```",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# 功能：统计样本集合。
def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intents = Counter(str(row.get("meta", {}).get("intent") or "unknown") for row in rows)
    flags = Counter(flag for row in rows for flag in row.get("meta", {}).get("quality_flags", []))
    history_turns = Counter(str(row.get("meta", {}).get("history_turns", 0)) for row in rows)
    return {
        "total": len(rows),
        "by_intent": dict(intents.most_common()),
        "quality_flags": dict(flags.most_common()),
        "history_turns": dict(history_turns.most_common()),
    }


# 功能：解析参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="Path to raw extract_train.json.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Optional local smoke-test limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--target-total", type=int, default=6000, help="Target source rows selected for DeepSeek rewrite.")
    parser.add_argument("--min-per-intent", type=int, default=120)
    parser.add_argument("--max-per-intent", type=int, default=650)
    parser.add_argument("--unknown-cap", type=int, default=0)
    parser.add_argument("--target-history-ratio", type=float, default=0.3)
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--hard-min-prompt-chars", type=int, default=2)
    parser.add_argument("--hard-min-reply-chars", type=int, default=6)
    parser.add_argument("--soft-min-reply-chars", type=int, default=15)
    parser.add_argument("--soft-max-reply-chars", type=int, default=220)
    parser.add_argument("--hard-max-reply-chars", type=int, default=600)
    parser.add_argument("--review-per-intent", type=int, default=8)
    parser.add_argument("--removed-review-limit", type=int, default=120)
    return parser.parse_args()


# 功能：主流程，生成 source 级别的分类、温和清洗和切分文件。
def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    split_dir = output_dir / "03_split"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_raw_records(raw_path)
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]

    accepted: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    classified_rows: list[dict[str, Any]] = []
    seen_pair_keys: set[str] = set()
    duplicate_count = 0

    for index, raw in enumerate(raw_rows, start=1):
        record, drop_reason = normalize_record(raw, index, args)
        classified_rows.append(record)
        pair_key = str(record.get("meta", {}).get("source_pair_key"))
        if pair_key in seen_pair_keys:
            duplicate_count += 1
            drop_reason = drop_reason or "exact_duplicate_prompt_response"
        if drop_reason:
            removed.append(
                {
                    "id": record["id"],
                    "drop_reason": drop_reason,
                    "intent": record.get("meta", {}).get("intent"),
                    "quality_flags": record.get("meta", {}).get("quality_flags", []),
                    "prompt": record.get("prompt"),
                    "response": record.get("response"),
                    "history_turns": record.get("meta", {}).get("history_turns", 0),
                }
            )
            continue
        seen_pair_keys.add(pair_key)
        accepted.append(record)

    selected_rows, selection_report = select_rewrite_sources(accepted, args)
    train_rows, dev_rows, test_rows = stratified_group_split(selected_rows, args.dev_ratio, args.test_ratio, args.seed)

    write_jsonl(classified_rows, output_dir / "01_classified.jsonl")
    write_jsonl(accepted, output_dir / "02_cleaned.jsonl")
    write_jsonl(selected_rows, output_dir / "02_selected_for_rewrite.jsonl")
    write_jsonl(removed, output_dir / "02_removed.jsonl")
    write_jsonl(train_rows, split_dir / "train_source.jsonl")
    write_jsonl(dev_rows, split_dir / "dev_source.jsonl")
    write_jsonl(test_rows, split_dir / "test_source.jsonl")

    write_review_markdown(classified_rows, output_dir / "review_classified_samples.md", "Classified Source Samples", args.review_per_intent, args.seed)
    write_review_markdown(accepted, output_dir / "review_cleaned_samples.md", "Cleaned Source Samples", args.review_per_intent, args.seed + 1)
    write_review_markdown(selected_rows, output_dir / "review_selected_for_rewrite.md", "Selected Rewrite Source Samples", args.review_per_intent, args.seed + 3)
    rewrite_candidates = [row for row in accepted if row.get("meta", {}).get("needs_rewrite")]
    write_review_markdown(rewrite_candidates, output_dir / "review_rewrite_candidates.md", "Rewrite Candidate Samples", args.review_per_intent, args.seed + 2)
    write_removed_markdown(removed, output_dir / "review_removed_samples.md", args.removed_review_limit, args.seed)

    report = {
        "raw_path": str(raw_path),
        "output_dir": str(output_dir),
        "input_total": len(raw_rows),
        "classified_total": len(classified_rows),
        "accepted_total": len(accepted),
        "selected_for_rewrite_total": len(selected_rows),
        "removed_total": len(removed),
        "duplicate_removed_total": duplicate_count,
        "removed_reasons": dict(Counter(row["drop_reason"] for row in removed).most_common()),
        "accepted_summary": summarize_rows(accepted),
        "selected_for_rewrite_summary": summarize_rows(selected_rows),
        "selection_report": selection_report,
        "rewrite_candidate_summary": summarize_rows(rewrite_candidates),
        "split_summary": {
            "train": summarize_rows(train_rows),
            "dev": summarize_rows(dev_rows),
            "test": summarize_rows(test_rows),
        },
        "files": {
            "classified": str(output_dir / "01_classified.jsonl"),
            "cleaned": str(output_dir / "02_cleaned.jsonl"),
            "selected_for_rewrite": str(output_dir / "02_selected_for_rewrite.jsonl"),
            "removed": str(output_dir / "02_removed.jsonl"),
            "train_source": str(split_dir / "train_source.jsonl"),
            "dev_source": str(split_dir / "dev_source.jsonl"),
            "test_source": str(split_dir / "test_source.jsonl"),
            "classified_review": str(output_dir / "review_classified_samples.md"),
            "cleaned_review": str(output_dir / "review_cleaned_samples.md"),
            "selected_for_rewrite_review": str(output_dir / "review_selected_for_rewrite.md"),
            "rewrite_candidates_review": str(output_dir / "review_rewrite_candidates.md"),
            "removed_review": str(output_dir / "review_removed_samples.md"),
        },
        "args": vars(args),
    }
    write_json({"raw_path": str(raw_path), "input_total": len(raw_rows)}, output_dir / "00_raw_manifest.json")
    write_json(report, output_dir / "build_report.json")

    print(json.dumps({k: report[k] for k in ["input_total", "accepted_total", "selected_for_rewrite_total", "removed_total", "removed_reasons"]}, ensure_ascii=False, indent=2))
    print(f"review_cleaned_samples={output_dir / 'review_cleaned_samples.md'}")
    print(f"review_selected_for_rewrite={output_dir / 'review_selected_for_rewrite.md'}")
    print(f"review_rewrite_candidates={output_dir / 'review_rewrite_candidates.md'}")


if __name__ == "__main__":
    main()
