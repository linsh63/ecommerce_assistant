"""Generate initial annotations for all eval samples and write annotation_sheet.md."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2" / "05_final" / "eval_test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "jddc_rebuild_v2" / "05_final" / "annotation_sheet.md"

# ── Annotation rules per intent ──────────────────────────────────────────
# Each intent has:
#   must:   list of common must_cover items (pick 1-3)
#   should: list of common should_cover items (pick 0-2)
#   avoid:  list of common avoid items (pick 1-3)
#
# Key design choice (per user): 编造具体状态/时间/金额 is ALLOWED.
# In a real agent system the model would query real data; we care about
# the behavioral pattern, not factual accuracy of made-up details.
# So "avoid" focuses on: off-topic, empty replies, ignoring history,
# contradictory advice, and genuinely harmful patterns.

INTENT_RULES: dict[str, dict[str, list[str]]] = {
    "物流-查件催发": {
        "must": [
            "给出查看物流或订单状态的可操作路径",
            "说明催促发货或联系平台核实的方式",
            "引导用户到订单详情页查看物流轨迹",
        ],
        "should": [
            "安抚用户等待的焦急情绪",
            "提醒物流可能因大促/天气等延迟",
        ],
        "avoid": [
            "只道歉或安抚但不给任何可执行路径",
            "建议用户取消订单重新下单（查件场景下不合适）",
        ],
    },
    "物流-配送调整": {
        "must": [
            "给出修改地址/配送时间的操作入口或路径",
            "区分已出库和未出库两种情况的处理方式",
        ],
        "should": [
            "提醒修改可能受限于物流节点进度",
            "说明拒收或转寄的条件",
        ],
        "avoid": [
            "承诺一定能修改成功",
            "建议用户联系快递员个人手机号",
        ],
    },
    "售后-退换维修": {
        "must": [
            "指出申请售后的入口（订单详情页/客服通道）",
            "说明退货/换货/维修的基本条件和凭证要求",
        ],
        "should": [
            "区分七天无理由和质保期内维修的不同流程",
            "提醒保留包装和凭证",
        ],
        "avoid": [
            "承诺一定可以退换而不提需审核",
            "给出与用户问题矛盾的结论（如用户要退货却说不能退）",
        ],
    },
    "售后-质量异常": {
        "must": [
            "先表达歉意或理解用户的不满情绪",
            "要求用户提供照片/视频等凭证以便核实",
            "给出提交凭证后的售后处理路径",
        ],
        "should": [
            "说明核实后可能的处理方案（补发/退款/换货）",
        ],
        "avoid": [
            "暗示是用户自己弄坏的",
            "只说抱歉但不给任何处理路径",
        ],
    },
    "退款-取消价保": {
        "must": [
            "说明退款/取消/价保的申请入口和操作方式",
            "提及款项原路返还的规则",
        ],
        "should": [
            "说明审核周期或到账时间的大致范围",
            "区分已发货和未发货的不同处理方式",
        ],
        "avoid": [
            "承诺具体到账时间或金额",
            "说已经退款成功但无上下文支持",
        ],
    },
    "商品咨询-参数": {
        "must": [
            "围绕商品参数或详情页信息进行回答",
            "给出用户自行核实信息的路径（如查看详情页/联系品牌客服）",
        ],
        "should": [
            "说明同类商品参数差异的对比维度",
        ],
        "avoid": [
            "编造不存在的具体规格参数值",
            "建议用户直接购买而不回答参数问题",
        ],
    },
    "商品咨询-使用方法": {
        "must": [
            "给出简洁可执行的操作步骤或排查路径",
            "针对用户具体使用场景进行回答",
        ],
        "should": [
            "说明常见问题或注意事项",
        ],
        "avoid": [
            "只说参考说明书而不给任何具体指导",
            "推荐用户购买其他配件或商品（偏离使用方法咨询）",
        ],
    },
    "购买决策-推荐对比": {
        "must": [
            "根据用户场景或预算给出选择建议",
            "说明对比的核心维度（功能/价格/适用场景）",
        ],
        "should": [
            "提醒以商品详情页实际参数为准",
        ],
        "avoid": [
            "硬推某一款而不给理由",
            "贬低竞品或使用绝对化用语",
        ],
    },
    "价格活动-优惠赠品": {
        "must": [
            "说明优惠活动或赠品的基本规则",
            "引导查看活动页面或结算页确认最终优惠",
        ],
        "should": [
            "说明优惠可能存在的时间限制或数量限制",
        ],
        "avoid": [
            "承诺一定可以享受优惠或获得赠品",
            "编造不存在的优惠力度",
        ],
    },
    "发票-资质服务": {
        "must": [
            "说明开票入口和操作路径",
            "区分电子发票和纸质发票的开具方式",
        ],
        "should": [
            "提及开票时效或限制",
        ],
        "avoid": [
            "说不需要发票或建议用户放弃开票",
        ],
    },
    "投诉-安抚升级": {
        "must": [
            "先表达歉意或理解用户的不满",
            "给出投诉反馈或平台介入的具体路径",
        ],
        "should": [
            "说明问题会记录并升级处理",
            "给出用户可跟踪处理进度的方式",
        ],
        "avoid": [
            "只道歉不给出任何处理路径",
            "暗示用户小题大做或推卸责任",
            "要求用户自己联系快递/商家而不提供平台协助",
        ],
    },
    "闲聊-礼貌收尾": {
        "must": [
            "简短礼貌地回应",
            "确认用户当前问题是否已解决",
        ],
        "should": [
            "表达感谢或服务意愿",
        ],
        "avoid": [
            "展开新的话题或推销商品",
            "长篇大论",
        ],
    },
}

# ── Sample-specific overrides ───────────────────────────────────────────
# Key: eval index (1-based). Override or supplement the intent-level rules.

SAMPLE_OVERRIDES: dict[int, dict[str, list[str]]] = {
    # ── 物流-查件催发 ──
    2: {  # multi-turn, user has been waiting, dialect
        "must": [
            "利用历史中用户已确认在家的信息",
            "给出催促配送或联系平台核实的路径",
        ],
        "should": ["安抚用户方言表达的急切情绪"],
    },
    3: {  # multi-turn, user needs fast delivery for trip
        "must": [
            "利用历史中用户要回老家的紧急背景",
            "给出查看物流进展或联系平台催促的路径",
        ],
        "should": ["说明紧急情况下可尝试联系当地站点"],
    },
    5: {  # user asking about international shipping time
        "must": [
            "给出查看跨境物流轨迹的路径",
            "说明跨境物流可能涉及清关等环节",
        ],
        "should": ["说明不同国家/地区的时效差异"],
        "avoid": [
            "只道歉或安抚但不给任何可执行路径",
            "建议用户取消订单重新下单",
        ],
    },
    6: {  # multi, address uncertainty in delivery area
        "must": [
            "利用历史中用户提供的收货地址信息",
            "说明核实地址后可能的配送方案",
        ],
    },
    # ── 物流-配送调整 ──
    21: {  # user wants to change address before delivery
        "must": [
            "给出修改收货地址的操作入口",
            "区分已出库和未出库的修改方式",
        ],
    },
    22: {  # user worried about delivery arrangement
        "must": [
            "利用历史中用户提到的家人代收不便的背景",
            "给出发货前修改配送方式的路径",
        ],
    },
    # ── 售后-退换维修 ──
    39: {  # user asks about warranty repair time
        "must": [
            "说明维修/换货的大致流程和时间",
            "指出申请售后维修的入口",
        ],
        "should": ["区分质保期内外的不同处理方式"],
    },
    41: {  # multi, garment size exchange
        "must": [
            "利用历史中用户想换尺码的信息",
            "给出换货申请的入口和条件",
        ],
    },
    # ── 退款-取消价保 ──
    60: {  # multi, user wants refund
        "must": [
            "利用历史中用户想退款的诉求",
            "给出退款申请的入口和流程",
        ],
        "should": ["说明退款到账的大致周期"],
    },
    61: {  # price protection
        "must": [
            "说明申请价保的条件和入口",
            "提醒价保申请的时效限制",
        ],
    },
    # ── 商品咨询-参数 ──
    78: {  # authenticity check
        "must": [
            "给出验证正品的方法（防伪码/品牌官网等）",
            "说明商品详情页的正品保障信息位置",
        ],
    },
    # ── 购买决策-推荐对比 ──
    97: {  # which one to buy
        "must": [
            "根据用户使用场景给出选择维度",
            "说明各选项的核心差异",
        ],
    },
    # ── 投诉-安抚升级 ──
    125: {  # user angry about delays
        "must": [
            "先道歉并理解用户的不满",
            "给出投诉或升级处理的路径",
            "说明问题会被记录并跟进",
        ],
        "avoid": [
            "只道歉不给出任何处理路径",
            "暗示用户小题大做",
            "要求用户自己联系快递/商家而不提供平台协助",
        ],
    },
    # ── 闲聊-礼貌收尾 ──
    140: {  # simple thank you
        "must": [
            "简短礼貌回应感谢",
            "确认对话可以结束",
        ],
        "should": ["表达后续服务意愿"],
        "avoid": [
            "展开新话题或推销",
            "长篇大论",
        ],
    },
}

# ── Markdown generation ──────────────────────────────────────────────────

CATEGORY_HINTS: dict[str, str] = {
    "物流-查件催发": "关注：回复是否引导查看物流轨迹、是否给出催促路径",
    "物流-配送调整": "关注：是否给出修改地址/时间的可操作入口、是否区分出库前后",
    "售后-退换维修": "关注：是否指向售后入口、是否说明条件和凭证要求",
    "售后-质量异常": "关注：是否先安抚、是否要求凭证、是否给出处理路径",
    "退款-取消价保": "关注：是否说明申请入口、是否提及原路返回、是否给出审核周期",
    "商品咨询-参数": "关注：是否围绕参数/详情页回答、是否给出核实路径",
    "商品咨询-使用方法": "关注：是否给出可执行步骤、是否针对具体场景",
    "购买决策-推荐对比": "关注：是否按场景/预算给建议、是否说明对比维度",
    "价格活动-优惠赠品": "关注：是否引导查看活动页、是否说明规则和限制",
    "发票-资质服务": "关注：是否说明开票入口和时效",
    "投诉-安抚升级": "关注：是否先道歉、是否给出反馈或升级路径",
    "闲聊-礼貌收尾": "关注：是否简洁礼貌、是否不展开新话题",
}

ANNOTATION_GUIDE = """# 评测标注指南

## 标注内容

每条样本需要填写三类关键点：

| 类型 | 含义 | 评判标准 |
|---|---|---|
| **必须覆盖 (must_cover)** | 合格回复必须包含的信息或行为 | 缺了 → 不合格 |
| **加分项 (should_cover)** | 有了更好的点 | 缺了不扣分，有了加分 |
| **禁止出现 (avoid)** | 绝对不能出现的内容 | 出现了 → 不合格 |

## 填写原则

1. 写"关键点"而非完整答案。如写"引导到订单详情页查看物流轨迹"，不写"您可以到订单详情页查看物流轨迹"
2. 每条 1-2 句话
3. must_cover 通常 1-3 条，avoid 通常 1-3 条，should_cover 通常 0-2 条
4. 关注模型的最后一轮回复

## 关于"编造内容"

本项目中模型可以编造具体状态/时间/金额。在实际场景中 Agent 会查询真实数据替换这些内容。所以 avoid 不包含"编造了物流状态/退款金额"等，只关注行为模式层面真正有问题的回答。

## 示例

### 示例 1：物流-查件催发（单轮）

用户：我的快递三天没更新了，怎么办？

- [x] must: 引导查看订单物流轨迹
- [x] must: 给出可执行路径（联系平台/发起催促等）
- [x] should: 安抚用户焦急情绪
- [x] avoid: 只道歉或安抚但不给任何可执行路径
- [ ] avoid:

### 示例 2：售后-质量异常（多轮）

历史：[用户] 收到的杯子碎了 → [客服] 很抱歉，请提供照片我们核实
用户：照片发你了，怎么处理？

- [x] must: 引用历史中用户已提供照片的事实
- [x] must: 给出后续处理路径（补发/退款/售后申请）
- [x] avoid: 让用户重新提供已发过的照片
- [ ] avoid:

---
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_intent(row: dict[str, Any]) -> str:
    return str(row.get("meta", {}).get("scenario") or row.get("meta", {}).get("intent") or "unknown")


def pick_annotation(intent: str, question: str, history_msgs: list, index: int) -> dict[str, list[str]]:
    """Pick must/should/avoid for a sample, with overrides applied."""
    rules = INTENT_RULES.get(intent, INTENT_RULES["闲聊-礼貌收尾"])

    result: dict[str, list[str]] = {"must": [], "should": [], "avoid": []}

    # Apply sample-specific overrides first
    override = SAMPLE_OVERRIDES.get(index, {})
    if "must" in override:
        result["must"] = list(override["must"])
    else:
        result["must"] = list(rules["must"])

    if "should" in override:
        result["should"] = list(override["should"])
    else:
        result["should"] = list(rules["should"])

    if "avoid" in override:
        result["avoid"] = list(override["avoid"])
    else:
        result["avoid"] = list(rules["avoid"])

    return result


def build_sample_section(row: dict[str, Any], index: int) -> str:
    messages = row.get("messages") or []
    intent = get_intent(row)
    history_required = row.get("meta", {}).get("history_required", False)
    history_turns = row.get("meta", {}).get("history_turns", 0)
    hint = CATEGORY_HINTS.get(intent, "")

    if len(messages) >= 4:
        history_msgs = messages[:-2]
        last_user = messages[-2]
        last_assistant = messages[-1]
    else:
        history_msgs = []
        last_user = messages[-2] if len(messages) >= 2 else {"role": "user", "content": ""}
        last_assistant = messages[-1] if len(messages) >= 2 else {"role": "assistant", "content": ""}

    question = last_user.get("content", "")
    multi_label = f"多轮({history_turns})" if history_required else "单轮"

    ann = pick_annotation(intent, question, history_msgs, index)

    lines = [
        f"## eval_{index:03d} | {intent} | {multi_label}",
        "",
        f"> {hint}",
        "",
    ]

    if history_msgs:
        lines.append("<details>")
        lines.append("<summary>历史对话（点击展开）</summary>")
        lines.append("")
        for msg in history_msgs:
            role_label = "用户" if msg.get("role") == "user" else "客服"
            lines.append(f"- **{role_label}：** {msg.get('content', '')}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append(f"**用户最后问题：** {question}")
    lines.append("")

    lines.append("<details>")
    lines.append("<summary>改写后的参考回答（点击展开）</summary>")
    lines.append("")
    lines.append(f"{last_assistant.get('content', '')}")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    lines.append("### 标注")
    lines.append("")

    lines.append("**必须覆盖 (must_cover)：**")
    for item in ann["must"]:
        lines.append(f"- [x] {item}")
    for _ in range(3 - len(ann["must"])):
        lines.append("- [ ] ")
    lines.append("")

    lines.append("**加分项 (should_cover)：**")
    for item in ann["should"]:
        lines.append(f"- [x] {item}")
    if not ann["should"]:
        lines.append("- [ ] ")
    lines.append("")

    lines.append("**禁止出现 (avoid)：**")
    for item in ann["avoid"]:
        lines.append(f"- [x] {item}")
    for _ in range(3 - len(ann["avoid"])):
        lines.append("- [ ] ")
    lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    eval_rows = read_jsonl(Path(DEFAULT_EVAL))
    intent_order = [
        "物流-查件催发", "物流-配送调整", "售后-退换维修", "售后-质量异常",
        "退款-取消价保", "商品咨询-参数", "商品咨询-使用方法",
        "购买决策-推荐对比", "价格活动-优惠赠品", "发票-资质服务",
        "投诉-安抚升级", "闲聊-礼貌收尾",
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        grouped[get_intent(row)].append(row)

    lines = [
        "# JDDC Rebuild V2 评测标注表（已预填充）",
        "",
        f"- 总样本数：{len(eval_rows)}",
        f"- 类别数：{len(grouped)}",
        "- 标注状态：**已预填充初始版本，请在每类基础上修改**",
        "- 修改方式：修改 `[x]` 后的文字，或新增 `[x]` 行，不需要的改为 `[ ]`",
        "",
        ANNOTATION_GUIDE,
    ]

    global_idx = 0
    for intent in intent_order:
        rows = grouped.get(intent, [])
        if not rows:
            continue
        lines.append(f"# {intent}（{len(rows)}条）")
        lines.append("")
        lines.append(f"类别提示：{CATEGORY_HINTS.get(intent, '')}")
        lines.append("")
        for row in rows:
            global_idx += 1
            lines.append(build_sample_section(row, global_idx))

    output_path = Path(DEFAULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    # Count filled vs empty
    filled = sum(1 for line in lines if line.startswith("- [x]"))
    empty = sum(1 for line in lines if line.startswith("- [ ] "))
    print(f"Generated: {output_path}")
    print(f"Samples: {global_idx}")
    print(f"Filled annotation lines: {filled}")
    print(f"Empty annotation lines: {empty}")


if __name__ == "__main__":
    main()
