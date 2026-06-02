"""LoRA SFT training utilities for the e-commerce customer service project.

重构后的结构（对标 dpo_training.py）：
  SFTDataProcessor      —— 数据加载、清洗、ChatML 模板、tokenize、Dataset 构建
  SFTModelBuilder        —— 模型加载、LoRA 配置、量化、tokenizer
  SFTPostTrainGenerator  —— 训练后评测集生成
  SFTTrainingPipeline    —— 训练编排，串联上述组件并保存产物

模块级函数（配置、YAML、工具）保持不变，原有公开函数名通过 wrapper 向后兼容。
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import (
    IGNORE_INDEX,
    find_project_root,
    read_yaml_config,
    resolve_project_path,
    set_train_seed,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 配置 dataclass —— 类型安全、可校验、可被命令行覆盖
# ═══════════════════════════════════════════════════════════════════════════════

# 功能：保存基础模型加载相关配置。
@dataclass
class SFTModelConfig:
    base_model: str
    adapter_model: str | None = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    quantization_method: str = "bnb"
    quantization_type: str = "nf4"
    double_quantization: bool = True
    device_map: str | None = "ddp"


# LoRA 配置。面试必问：rank 怎么选的？——消融实验对比了 rank=4/16/64，rank=16 是效果和显存的最优平衡点。
# rank=64 效果提升不到 1%，但显存多 40%，不划算。
# alpha 通常是 rank 的 2 倍，控制 LoRA 增量的缩放幅度。
# target_modules 只选 q_proj 和 v_proj —— 注意力的 query 和 value 投影，这是 LoRA 论文推荐的默认配置。
# 只加在 attention 上（不加 FFN）是因为：(1) 参数效率最高，(2) 大部分微调适配发生在 attention 模式上。
@dataclass
class SFTLoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # 功能：为未显式配置 target_modules 的情况提供项目默认值。
    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


# 功能：保存 Trainer 和优化器相关配置。
@dataclass
class SFTTrainingConfig:
    output_dir: str = "outputs/sft_lora_qwen3_8b"
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    optim: str = "paged_adamw_8bit"
    ddp_find_unused_parameters: bool = False
    report_to: str = "none"
    seed: int = 42
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    resume_from_checkpoint: str | None = None
    log_sample_count: int = 3


# 数据配置。核心设计选择：
# - messages 格式：直接用 user/assistant 交替的 OpenAI 格式，兼容训练和推理。
# - mask_history=true：多轮对话中只监督最后一轮 assistant 回复，历史轮次不参与 loss。
#   原理：客服场景的"正确回答"只在最后一轮，历史回答只是上下文，不应被当成训练目标。
# - chat_template=qwen3_nothink：用 Qwen3 的 ChatML 模板但不生成 <think> 标签（推理时不需要思考过程）。
# - max_history_turns=6：保留最近 6 轮历史 + 当前轮，超出截断，防止超长对话撑爆 max_seq_length。
@dataclass
class SFTDataConfig:
    train_file: str | None = None
    eval_file: str | None = None
    hf_dataset_name: str | None = None
    hf_dataset_config: str | None = None
    hf_split: str = "train"
    dialogue_column: str = "dialogue"
    role_field: str = "role"
    text_field: str = "text"
    max_samples: int = 3000
    max_history_turns: int = 6
    hf_eval_size: int = 200
    chat_template: str = "qwen3_nothink"
    default_system: str = ""
    train_on_prompt: bool = False
    mask_history: bool = False
    validation_split_ratio: float = 0.1
    validation_split_size: int | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    shuffle_seed: int = 42


# 功能：保存训练结束后自动回答测试集问题的生成配置。
@dataclass
class SFTPostTrainGenerationConfig:
    enabled: bool = False
    test_file: str | None = None
    output_jsonl: str = "post_train_test_generations.jsonl"
    output_markdown: str = "post_train_test_generations.md"
    max_samples: int | None = None
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    print_every: int = 10


# 功能：聚合一次 SFT 实验需要的全部配置。
@dataclass
class SFTExperimentConfig:
    model: SFTModelConfig
    lora: SFTLoraConfig
    training: SFTTrainingConfig
    data: SFTDataConfig
    post_train_generation: SFTPostTrainGenerationConfig


# 功能：记录数据预处理后的样本统计，便于排查清洗和截断问题。
@dataclass
class SFTDatasetStats:
    total_records: int
    usable_records: int
    skipped_records: int
    truncated_records: int
    max_length: int


# 功能：按 causal LM 训练要求对 input_ids、attention_mask 和 labels 做 padding。
class AssistantOnlyDataCollator:
    # 功能：初始化 padding token 和 label padding 的取值。
    def __init__(self, pad_token_id: int, label_pad_token_id: int = IGNORE_INDEX) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    # 功能：把一批不同长度的 tokenized 样本整理成 Trainer 可消费的 tensor batch。
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        # input_ids、attention_mask、labels 必须补到同一长度，才能组成一个 batch tensor。
        input_ids = self._pad_2d([item["input_ids"] for item in features], self.pad_token_id)
        attention_mask = self._pad_2d([item["attention_mask"] for item in features], 0)

        # labels 的 padding 用 -100，这样 padding token 不会参与 cross entropy loss。
        labels = self._pad_2d([item["labels"] for item in features], self.label_pad_token_id)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # 功能：把二维变长列表右侧补齐到同一长度。
    def _pad_2d(self, values: list[list[int]], pad_value: int) -> list[list[int]]:
        max_len = max(len(item) for item in values)
        return [item + [pad_value] * (max_len - len(item)) for item in values]


# ═══════════════════════════════════════════════════════════════════════════════
# SFTDataProcessor —— 数据加载、清洗、ChatML 模板、tokenize、Dataset 构建
# ═══════════════════════════════════════════════════════════════════════════════

class SFTDataProcessor:
    """SFT 数据处理器。

    负责：
      1. 从 jsonl / HuggingFace 加载原始样本
      2. 清洗并归一化为 user/assistant 交替的 messages 格式
      3. 套用 Qwen ChatML 模板
      4. tokenize 并构建 assistant-only labels
      5. 输出 HuggingFace Dataset（可直接喂给 TRL SFTTrainer）

    SFT 的核心 trick 在这个类的 tokenize 方法里：
      prompt 部分 label=-100，只有 assistant 回复的 token 参与 loss。
    """

    def __init__(self, tokenizer: Any, config: SFTExperimentConfig):
        self.tokenizer = tokenizer
        self.config = config
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ── 角色归一化 ──────────────────────────────────────────────────────

    # 功能：把常见数据集角色名归一化到 user/assistant/system。
    @staticmethod
    def normalize_role(role: Any) -> str | None:
        role_text = "" if role is None else str(role).strip()
        role_text_lower = role_text.lower()
        role_map = {
            "human": "user",
            "user": "user",
            "customer": "user",
            "buyer": "user",
            "顾客": "user",
            "用户": "user",
            "客户": "user",
            "买家": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "agent": "assistant",
            "service": "assistant",
            "客服": "assistant",
            "商家": "assistant",
            "seller": "assistant",
            "卖家": "assistant",
            "system": "system",
        }
        return role_map.get(role_text) or role_map.get(role_text_lower)

    # ── 数据格式转换 ────────────────────────────────────────────────────

    # 功能：把本项目 prompt/response/history 格式转换成 user/assistant 交替 messages。
    @staticmethod
    def prompt_response_to_messages(example: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
        if "prompt" not in example or "response" not in example:
            return [], ""

        prompt = str(example.get("prompt") or "").strip()
        response = str(example.get("response") or "").strip()
        if not prompt or not response:
            return [], str(example.get("system") or "").strip()

        history = example.get("history") or []
        if not isinstance(history, list):
            history = []

        messages: list[dict[str, str]] = []
        for pair in history:
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            user_text = str(pair[0] or "").strip()
            assistant_text = str(pair[1] or "").strip()
            if user_text and assistant_text:
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})

        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        return messages, str(example.get("system") or "").strip()

    # 功能：清洗并校验 messages，使其符合 user/assistant 交替的 SFT 输入形态。
    @staticmethod
    def normalize_sft_messages(example: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
        raw_messages = example.get("messages") or example.get("conversations")
        if not isinstance(raw_messages, list):
            return SFTDataProcessor.prompt_response_to_messages(example)

        system = str(example.get("system") or "").strip()
        messages: list[dict[str, str]] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                continue

            # 兼容 OpenAI messages(role/content) 和 ShareGPT(from/value) 两种常见字段名。
            role = SFTDataProcessor.normalize_role(
                raw_message.get("role", raw_message.get("from"))
            )
            content = raw_message.get("content", raw_message.get("value", ""))
            content = "" if content is None else str(content).strip()
            if not role or not content:
                continue

            # system 不进入 user/assistant 交替序列，而是后面放到首轮 prompt 前。
            if role == "system":
                system = content
                continue
            if role not in {"user", "assistant"}:
                continue

            # 同角色连续出现时合并，修复少量脏数据中的重复 user 或 assistant turn。
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = (messages[-1]["content"] + "\n" + content).strip()
            else:
                messages.append({"role": role, "content": content})

        # SFT 样本必须是 user 开头、assistant 结尾；否则无法形成 prompt → response 监督。
        while messages and messages[0]["role"] == "assistant":
            messages.pop(0)
        while messages and messages[-1]["role"] == "user":
            messages.pop()
        if len(messages) < 2 or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
            return [], system
        return messages, system

    # 功能：从 jsonl 文件读取原始训练样本。
    @staticmethod
    def read_jsonl_records(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Each jsonl row must be an object at {path}:{line_no}")
                records.append(record)
                if max_samples is not None and len(records) >= max_samples:
                    break
        return records

    # ── Qwen ChatML 模板 ────────────────────────────────────────────────
    # Qwen 的对话模板格式：
    #   <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
    # assistant 的回复不加 <|im_end|> 在 source 中——因为 target 部分会单独拼接 <|im_end|>。
    # 这样 tokenization 后 source 部分是"模型看到的输入"，target 部分是"模型要生成的输出"。

    # 功能：按照 Qwen/ChatML 模板构造一轮用户提示片段。
    @staticmethod
    def format_user_segment(content: str, system: str = "") -> str:
        system_segment = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        return f"{system_segment}<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    # 功能：按照 Qwen/ChatML 模板构造一轮助手回答片段。
    @staticmethod
    def format_assistant_segment(content: str) -> str:
        return f"{content}<|im_end|>\n"

    # 功能：把多轮 messages 转成 source/target 文本对，贴近 LlamaFactory 的 multiturn 编码方式。
    @staticmethod
    def build_chatml_turns(messages: list[dict[str, str]], system: str = "") -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        pending_user: str | None = None
        first_user = True
        for message in messages:
            if message["role"] == "user":
                pending_user = message["content"]
            elif message["role"] == "assistant" and pending_user is not None:
                # 只在第一轮前放 system，和 Qwen/ChatML 多轮模板保持一致。
                turn_system = system if first_user else ""

                # source 是模型输入中不计算 loss 的部分，target 是 assistant 要学习的回答。
                source = SFTDataProcessor.format_user_segment(pending_user, turn_system)
                target = SFTDataProcessor.format_assistant_segment(message["content"])
                turns.append((source, target))
                pending_user = None
                first_user = False
        return turns

    # 功能：在长度超限时优先保留 assistant target，避免把回答监督信号截没。
    @staticmethod
    def infer_source_target_lengths(source_len: int, target_len: int, max_len: int) -> tuple[int, int]:
        if source_len + target_len <= max_len:
            return source_len, target_len
        if max_len <= 0:
            return 0, 0
        if target_len >= max_len:
            return 0, max_len
        return max_len - target_len, target_len

    # ── Tokenization ────────────────────────────────────────────────────

    # SFT 的核心：tokenize + assistant-only labels
    #
    # 这是理解"微调 loss 和预训练 loss 有什么区别"的关键。
    #
    # 预训练：每个 token 都参与 loss 计算（input = labels = 完整文本）。
    # SFT 微调：只有 assistant 回复的 token 参与 loss，user 和 system 部分 label = -100（被忽略）。
    #
    # 举个例子，一条"用户：我要退货 → 客服：您可以..."的对话：
    #   input_ids:  [user_tokens] [assistant_tokens]
    #   labels:     [-100 x N]    [assistant_tokens_copy]
    #
    # CrossEntropyLoss(ignore_index=-100) 会自动把 label=-100 的位置的 loss 置零。
    # 所以模型只学习"如何生成好的客服回复"，不会学习"如何复读用户问题"。
    #
    # mask_history=true 时的额外行为：
    #   多轮对话中，历史 assistant 的 label 也设 -100，只保留最后一轮 assistant 参与 loss。
    #   这是本项目的设计选择——"历史回答是上下文，最后一轮才是训练目标"。
    # 功能：把一个样本编码为 input_ids、labels 和 attention_mask。
    def tokenize_sft_example(self, example: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        # 第一步：把原始 messages 清成合法的 user/assistant 交替序列。
        messages, system = SFTDataProcessor.normalize_sft_messages(example)
        if not messages:
            return None, False
        messages = SFTDataProcessor.trim_messages_to_max_history_turns(
            messages, self.config.data.max_history_turns
        )

        # 第二步：套用 Qwen/ChatML 模板，得到每轮的 source(prompt) 和 target(answer)。
        system = system or self.config.data.default_system
        turns = SFTDataProcessor.build_chatml_turns(messages, system)
        if not turns:
            return None, False

        input_ids: list[int] = []
        labels: list[int] = []
        truncated = False

        # mask_history=True 时优先保留最后一轮，这符合"只训练最新回答"的场景。
        turn_iterable = list(reversed(turns)) if self.config.data.mask_history else turns
        for turn_index, (source, target) in enumerate(turn_iterable):
            remaining = self.config.training.max_seq_length - len(input_ids)
            if remaining <= 0:
                truncated = True
                break

            # source 和 target 分开 tokenize，后面才能对 prompt 和 answer 设置不同 label。
            source_ids = self.tokenizer.encode(source, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)

            # 超长时优先保留 target，因为 target 才是监督学习信号。
            source_len, target_len = SFTDataProcessor.infer_source_target_lengths(
                len(source_ids), len(target_ids), remaining
            )
            if source_len < len(source_ids) or target_len < len(target_ids):
                truncated = True

            source_ids = source_ids[-source_len:] if source_len > 0 else []
            target_ids = target_ids[:target_len]

            # 默认不训练 prompt：source label 全部置为 -100，Transformers 会自动忽略。
            source_labels = source_ids if self.config.data.train_on_prompt else [IGNORE_INDEX] * len(source_ids)
            if self.config.data.mask_history and turn_index != 0:
                # 只训练最后一轮时，历史 assistant 的 target 也要 mask 掉。
                target_labels = [IGNORE_INDEX] * len(target_ids)
            else:
                target_labels = target_ids

            if self.config.data.mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_labels + target_labels + labels
            else:
                input_ids.extend(source_ids + target_ids)
                labels.extend(source_labels + target_labels)

        if not input_ids or all(label == IGNORE_INDEX for label in labels):
            return None, truncated

        # text 只用于保存样本预览，真正训练使用 input_ids/labels。
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "text": text,
        }, truncated

    # 功能：控制多轮样本长度，保留最后若干轮历史以及当前监督回答。
    @staticmethod
    def trim_messages_to_max_history_turns(messages: list[dict[str, str]], max_history_turns: int) -> list[dict[str, str]]:
        if max_history_turns < 0 or len(messages) <= 2:
            return messages
        max_pairs = max_history_turns + 1
        return messages[-2 * max_pairs :]

    # ── Dataset 构建 ────────────────────────────────────────────────────

    # 功能：把原始 jsonl 文件转换成 tokenized Hugging Face Dataset。
    def build_tokenized_dataset(
        self, path: Path, max_samples: int | None = None
    ) -> tuple[Any, SFTDatasetStats]:
        from datasets import Dataset

        records = SFTDataProcessor.read_jsonl_records(path, max_samples=max_samples)
        rows: list[dict[str, Any]] = []
        truncated_count = 0
        for record in records:
            # 每条样本都会变成已经带 assistant-only labels 的 tokenized row。
            row, truncated = self.tokenize_sft_example(record)
            if row is not None:
                rows.append(row)
            if truncated:
                truncated_count += 1

        if not rows:
            raise RuntimeError(f"No usable SFT samples were produced from {path}.")

        # 这些统计会在训练启动前打印，帮助判断数据是否大量被跳过或截断。
        stats = SFTDatasetStats(
            total_records=len(records),
            usable_records=len(rows),
            skipped_records=len(records) - len(rows),
            truncated_records=truncated_count,
            max_length=max(len(row["input_ids"]) for row in rows),
        )
        return Dataset.from_list(rows), stats

    # 功能：把内存中的 messages 样本转换成 tokenized Hugging Face Dataset。
    def build_tokenized_dataset_from_records(
        self, records: list[dict[str, Any]], max_samples: int | None = None
    ) -> tuple[Any, SFTDatasetStats]:
        from datasets import Dataset

        selected_records = records[:max_samples] if max_samples is not None else records
        rows: list[dict[str, Any]] = []
        truncated_count = 0
        for record in selected_records:
            row, truncated = self.tokenize_sft_example(record)
            if row is not None:
                rows.append(row)
            if truncated:
                truncated_count += 1

        if not rows:
            raise RuntimeError("No usable SFT samples were produced from in-memory records.")

        stats = SFTDatasetStats(
            total_records=len(selected_records),
            usable_records=len(rows),
            skipped_records=len(selected_records) - len(rows),
            truncated_records=truncated_count,
            max_length=max(len(row["input_ids"]) for row in rows),
        )
        return Dataset.from_list(rows), stats

    # 功能：把训练文件划分为 train/eval 原始样本，eval_file 为空时按比例从训练集切出验证集。
    def build_train_eval_datasets_from_files(
        self, train_file: Path, eval_file: Path | None
    ) -> tuple[Any, SFTDatasetStats, Any | None, SFTDatasetStats | None, dict[str, Any]]:
        train_records = SFTDataProcessor.read_jsonl_records(train_file)
        split_report: dict[str, Any] = {
            "train_file": str(train_file),
            "eval_file": str(eval_file) if eval_file else None,
            "validation_split_ratio": self.config.data.validation_split_ratio,
            "validation_split_size": self.config.data.validation_split_size,
        }

        if eval_file is not None and eval_file.exists():
            eval_records = SFTDataProcessor.read_jsonl_records(eval_file)
            split_report["split_source"] = "explicit_eval_file"
        else:
            train_records, eval_records = SFTDataProcessor.split_records_for_validation(
                train_records,
                ratio=self.config.data.validation_split_ratio,
                split_size=self.config.data.validation_split_size,
                seed=self.config.data.shuffle_seed,
            )
            split_report["split_source"] = "train_file_holdout"

        split_report["raw_train_records"] = len(train_records)
        split_report["raw_eval_records"] = len(eval_records)
        train_dataset, train_stats = self.build_tokenized_dataset_from_records(
            train_records, max_samples=self.config.data.max_train_samples
        )
        eval_dataset = None
        eval_stats = None
        if eval_records:
            eval_dataset, eval_stats = self.build_tokenized_dataset_from_records(
                eval_records, max_samples=self.config.data.max_eval_samples
            )
        return train_dataset, train_stats, eval_dataset, eval_stats, split_report

    # 功能：按固定 seed 打乱并从训练集切出验证集。
    @staticmethod
    def split_records_for_validation(
        records: list[dict[str, Any]],
        ratio: float,
        split_size: int | None,
        seed: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if len(records) < 2 or (not split_size and ratio <= 0):
            return records, []

        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        if split_size is not None:
            eval_size = min(split_size, len(shuffled) - 1)
        else:
            eval_size = int(round(len(shuffled) * ratio))
            eval_size = min(max(1, eval_size), len(shuffled) - 1)
        eval_records = shuffled[:eval_size]
        train_records = shuffled[eval_size:]
        return train_records, eval_records

    # 功能：按参考项目字段约定读取 JDDC 的 dialogue turn。
    def normalize_jddc_turns(self, example: dict[str, Any]) -> list[dict[str, str]]:
        dialogue = example.get(self.config.data.dialogue_column)
        if not isinstance(dialogue, list):
            return []

        messages: list[dict[str, str]] = []
        for turn in dialogue:
            if not isinstance(turn, dict):
                continue
            role = SFTDataProcessor.normalize_role(turn.get(self.config.data.role_field))
            content = turn.get(self.config.data.text_field, "")
            content = "" if content is None else str(content).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        return messages

    # 功能：复刻参考项目 prepare-sft，把每个 assistant 回复展开成一条 SFT messages 样本。
    def build_reference_jddc_sft_records(self, examples: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for example in examples:
            messages = self.normalize_jddc_turns(dict(example))
            history = []
            if self.config.data.default_system:
                history.append({"role": "system", "content": self.config.data.default_system})
            for message in messages:
                if message["role"] == "assistant" and any(item["role"] == "user" for item in history):
                    prompt = history[-self.config.data.max_history_turns * 2 - 1 :]
                    rows.append({"messages": prompt + [message]})
                history.append(message)
            if len(rows) >= self.config.data.max_samples:
                break
        return rows[: self.config.data.max_samples]

    # 功能：直接加载和参考项目一致的 Hugging Face JDDC 数据，并构造 train/eval tokenized dataset。
    def build_hf_jddc_tokenized_datasets(self) -> tuple[Any, SFTDatasetStats, Any, SFTDatasetStats]:
        from datasets import Dataset, load_dataset

        if self.config.data.hf_dataset_config:
            raw_dataset = load_dataset(
                self.config.data.hf_dataset_name,
                self.config.data.hf_dataset_config,
                split=self.config.data.hf_split,
            )
        else:
            raw_dataset = load_dataset(self.config.data.hf_dataset_name, split=self.config.data.hf_split)
        rows = self.build_reference_jddc_sft_records(raw_dataset)
        if not rows:
            raise RuntimeError(f"No SFT rows were produced from {self.config.data.hf_dataset_name}.")

        dataset = Dataset.from_list(rows)
        split = dataset.train_test_split(
            test_size=min(self.config.data.hf_eval_size, max(1, len(dataset) // 10)),
            seed=self.config.training.seed,
        )
        train_dataset, train_stats = self.build_tokenized_dataset_from_records(
            [dict(row) for row in split["train"]],
            max_samples=self.config.data.max_train_samples,
        )
        eval_dataset, eval_stats = self.build_tokenized_dataset_from_records(
            [dict(row) for row in split["test"]],
            max_samples=self.config.data.max_eval_samples,
        )
        return train_dataset, train_stats, eval_dataset, eval_stats


# ═══════════════════════════════════════════════════════════════════════════════
# SFTModelBuilder —— 模型加载、LoRA 配置、量化、tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

class SFTModelBuilder:
    """SFT 模型构建器。

    负责：
      1. 加载 tokenizer（Qwen 系列需要 trust_remote_code）
      2. 构建量化配置（QLoRA: 4bit NF4 + 双重量化）
      3. 加载基础模型（支持 QLoRA / FP16）
      4. 创建 LoRA adapter 配置
      5. 对齐可训练参数 dtype（FP16 AMP 兼容）

    QLoRA 原理（面试常问）：
      - LoRA:  基础模型 FP16 + LoRA FP16 adapter → 显存约 16GB (Qwen3-8B)
      - QLoRA: 基础模型 4bit + LoRA FP16 adapter → 显存约 6GB
      精度损失通常 < 1%，但显存省了 ~60%，单卡 3090 就能跑 8B 模型。
    """

    def __init__(self, config: SFTExperimentConfig):
        self.config = config

    # 功能：把配置中的 dtype 字符串转换为 torch dtype。
    @staticmethod
    def resolve_torch_dtype(dtype_name: str) -> Any:
        import torch

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        return dtype_map[dtype_name]

    # 功能：根据配置创建 bitsandbytes 4bit/8bit 量化配置。
    def build_quantization_config(self) -> Any:
        if not self.config.model.load_in_4bit:
            return None
        if self.config.model.quantization_method != "bnb":
            raise ValueError("This lightweight trainer currently supports only bitsandbytes QLoRA.")

        from transformers import BitsAndBytesConfig

        # 这是 QLoRA 的关键：基础模型 4bit 加载，LoRA adapter 仍以可训练参数形式更新。
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=SFTModelBuilder.resolve_torch_dtype(self.config.model.torch_dtype),
            bnb_4bit_use_double_quant=self.config.model.double_quantization,
            bnb_4bit_quant_type=self.config.model.quantization_type,
        )

    # 功能：为单机多卡 DDP 或单进程推理/训练选择合适的 device_map。
    def resolve_device_map(self) -> Any:
        device_map = self.config.model.device_map
        if device_map is None or str(device_map).lower() in {"none", "null", ""}:
            return None
        if str(device_map).lower() != "ddp":
            return device_map

        # torchrun/accelerate 多进程训练会设置 LOCAL_RANK；每个进程只绑定自己的 GPU。
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return {"": int(local_rank)}

        # 非 torchrun 启动时退回 auto，方便单卡 smoke test。
        return "auto"

    # 功能：根据配置加载 tokenizer，并补齐 pad token。
    def load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        # Qwen 系列通常需要 trust_remote_code；pad_token 缺失时用 eos 兜底。
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # ── 模型加载：QLoRA 原理 ──
    # QLoRA = 4bit 量化基础模型 + LoRA adapter 微调。
    # 基础模型被冻结（4bit 存储），只有 LoRA adapter 是可训练参数（float32/float16）。
    # 这样显存占用大幅下降：Qwen3-8B 原本需要 ~16GB（FP16），QLoRA 只需 ~6GB。
    #
    # 流程：
    # 1. BitsAndBytesConfig 把模型加载为 4bit（NF4 量化，双重量化）
    # 2. prepare_model_for_kbit_training 处理 LoRA 适配（layernorm 转 FP32 等）
    # 3. 后续 build_lora_config 创建的 LoRA adapter 注入到模型中
    #
    # 这里的关键区别（面试常问）：
    # - LoRA:  基础模型 FP16 + LoRA FP16 adapter → 显存约 16GB (Qwen3-8B)
    # - QLoRA: 基础模型 4bit + LoRA FP16 adapter → 显存约 6GB
    # 精度损失通常 < 1%，但显存省了 ~60%，单卡 3090 就能跑 8B 模型。
    # 功能：加载基础模型，并在 QLoRA 场景下准备 k-bit 训练；如配置 adapter_model，则继续训练已有 LoRA。
    def load_model(self) -> Any:
        from transformers import AutoModelForCausalLM

        # 如果开启 load_in_4bit，这里会把 BitsAndBytesConfig 传给 from_pretrained。
        quantization_config = self.build_quantization_config()
        device_map = self.resolve_device_map()
        model_kwargs = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "torch_dtype": SFTModelBuilder.resolve_torch_dtype(self.config.model.torch_dtype),
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if device_map:
            model_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(self.config.model.base_model, **model_kwargs)

        # 训练时关掉 cache，否则 gradient checkpointing 场景容易冲突并额外占显存。
        model.config.use_cache = False

        if self.config.training.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        if self.config.model.load_in_4bit:
            from peft import prepare_model_for_kbit_training

            # PEFT 官方推荐：k-bit 训练前处理 layer norm、输入梯度等细节。
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing,
            )
        if self.config.model.adapter_model:
            from peft import PeftModel

            # 用已有 LoRA adapter 初始化模型，并设为可训练，用于"接着上一版 adapter 继续训"的实验。
            model = PeftModel.from_pretrained(model, self.config.model.adapter_model, is_trainable=True)
        return model

    # 功能：根据项目配置创建 PEFT 的 LoRAConfig。
    def build_lora_config(self) -> Any:
        if self.config.model.adapter_model:
            return None

        from peft import LoraConfig, TaskType

        task_type = getattr(TaskType, self.config.lora.task_type)

        # SFTTrainer 会拿这个 peft_config 把 LoRA adapter 注入到基础模型里。
        return LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            task_type=task_type,
        )

    # 功能：在 FP16 AMP 训练前修正 LoRA 可训练参数 dtype，避免 GradScaler 处理低精度梯度。
    def align_trainable_parameter_dtypes(self, model: Any) -> None:
        if not self.config.training.fp16 or self.config.training.bf16:
            return

        import torch

        converted_count = 0
        for parameter in model.parameters():
            # 4bit QLoRA 中基础模型通常被冻结；这里只处理 LoRA adapter 等可训练参数。
            if parameter.requires_grad and parameter.dtype in {torch.float16, torch.bfloat16}:
                parameter.data = parameter.data.to(torch.float32)
                converted_count += 1

        rank = os.environ.get("RANK", "0")
        if converted_count and rank == "0":
            print(
                f"Converted {converted_count} low-precision trainable parameters to FP32 "
                "for FP16 AMP compatibility."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SFTPostTrainGenerator —— 训练后评测集生成
# ═══════════════════════════════════════════════════════════════════════════════

class SFTPostTrainGenerator:
    """训练后用模型回答测试集问题，不做评测、不打分，仅保存生成结果。"""

    def __init__(self, tokenizer: Any, config: SFTExperimentConfig):
        self.tokenizer = tokenizer
        self.config = config

    # 功能：推断生成输入应该放在哪张卡上。
    @staticmethod
    def infer_model_input_device(model: Any) -> Any:
        import torch

        try:
            return model.get_input_embeddings().weight.device
        except Exception:
            return next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

    # 功能：清理生成文本中的 ChatML 结束符和 Qwen thinking 片段。
    @staticmethod
    def clean_generated_text(text: str) -> str:
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        for marker in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            if marker in text:
                text = text.split(marker, 1)[0]
        return text.strip()

    # 功能：如果模型按"分类/回复"格式输出，就拆出便于统计和人工查看的字段。
    @staticmethod
    def parse_intent_and_reply(text: str) -> tuple[str | None, str | None]:
        intent_match = re.search(r"分类[:：]\s*(.+?)(?:\n|$)", text)
        reply_match = re.search(r"回复[:：]\s*(.+)", text, flags=re.DOTALL)
        intent = intent_match.group(1).strip() if intent_match else None
        reply = reply_match.group(1).strip() if reply_match else None
        return intent or None, reply or None

    # 功能：清洗测试集 messages，使最后停在 user turn，方便模型继续生成 assistant。
    @staticmethod
    def normalize_generation_messages(example: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
        raw_messages = example.get("messages") or example.get("conversations") or []
        system = str(example.get("system") or "").strip()
        messages: list[dict[str, str]] = []
        if not isinstance(raw_messages, list):
            return messages, system
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                continue
            role = SFTDataProcessor.normalize_role(
                raw_message.get("role", raw_message.get("from"))
            )
            content = raw_message.get("content", raw_message.get("value", ""))
            content = "" if content is None else str(content).strip()
            if not role or not content:
                continue
            if role == "system":
                system = content
                continue
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": content})
        while messages and messages[-1]["role"] == "assistant":
            messages.pop()
        while messages and messages[0]["role"] == "assistant":
            messages.pop(0)
        return messages, system

    # 功能：把 user/assistant 历史消息渲染为以 assistant 开头待生成的 ChatML prompt。
    @staticmethod
    def messages_to_generation_prompt(messages: list[dict[str, str]], system: str = "") -> str:
        prompt_parts: list[str] = []
        first_user = True
        pending_user: str | None = None
        for message in messages:
            if message["role"] == "user":
                if pending_user is not None:
                    prompt_parts.append(
                        SFTDataProcessor.format_user_segment(pending_user, system if first_user else "")
                    )
                    first_user = False
                pending_user = message["content"]
            elif message["role"] == "assistant" and pending_user is not None:
                prompt_parts.append(
                    SFTDataProcessor.format_user_segment(pending_user, system if first_user else "")
                )
                prompt_parts.append(SFTDataProcessor.format_assistant_segment(message["content"]))
                first_user = False
                pending_user = None
        if pending_user is not None:
            prompt_parts.append(
                SFTDataProcessor.format_user_segment(pending_user, system if first_user else "")
            )
        return "".join(prompt_parts)

    # 功能：把测试集 prompt/response/history 或 query/context 记录转换成推理 prompt。
    def build_generation_prompt(self, record: dict[str, Any]) -> str:
        if record.get("messages") or record.get("conversations"):
            messages, system = SFTPostTrainGenerator.normalize_generation_messages(record)
            return SFTPostTrainGenerator.messages_to_generation_prompt(
                messages, system or self.config.data.default_system
            )

        system = str(record.get("system") or self.config.data.default_system or "").strip()
        if "query" in record:
            query = str(record.get("query") or "").strip()
            context = str(record.get("context") or "").strip()
            prompt = f"用户问题：{query}\n业务背景：{context}" if context else query
            return SFTDataProcessor.format_user_segment(prompt, system)

        history = record.get("history") or []
        messages: list[dict[str, str]] = []
        if isinstance(history, list):
            for pair in history:
                if not (isinstance(pair, list) and len(pair) == 2):
                    continue
                user_text = str(pair[0] or "").strip()
                assistant_text = str(pair[1] or "").strip()
                if user_text and assistant_text:
                    messages.append({"role": "user", "content": user_text})
                    messages.append({"role": "assistant", "content": assistant_text})
        prompt = str(record.get("prompt") or "").strip()
        if prompt:
            messages.append({"role": "user", "content": prompt})
        return SFTPostTrainGenerator.messages_to_generation_prompt(messages, system)

    # 功能：用训练后的模型生成一条测试集回答，不计算任何评测指标。
    def generate_answer(self, model: Any, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = SFTPostTrainGenerator.infer_model_input_device(model)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        eos_token_ids = [
            token_id
            for token_id in [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ]
            if token_id is not None
        ]
        gen_cfg = self.config.post_train_generation
        generation_kwargs = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "do_sample": gen_cfg.do_sample,
            "repetition_penalty": gen_cfg.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if eos_token_ids:
            generation_kwargs["eos_token_id"] = eos_token_ids
        if gen_cfg.do_sample:
            generation_kwargs["temperature"] = gen_cfg.temperature
            generation_kwargs["top_p"] = gen_cfg.top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return SFTPostTrainGenerator.clean_generated_text(decoded)

    # 功能：训练结束后回答测试集问题并保存，不做评测、不打分。
    def generate_and_save(
        self,
        model: Any,
        output_dir: Path,
        project_root: Path,
    ) -> None:
        gen_cfg = self.config.post_train_generation
        if not gen_cfg.enabled:
            return
        if not gen_cfg.test_file:
            print("post_train_generation.enabled=true but test_file is empty; skipped generation.")
            return

        test_file = resolve_project_path(gen_cfg.test_file, project_root)
        records = SFTDataProcessor.read_jsonl_records(test_file, max_samples=gen_cfg.max_samples)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / gen_cfg.output_jsonl
        markdown_path = output_dir / gen_cfg.output_markdown

        generation_model = getattr(model, "module", model)
        if hasattr(generation_model, "gradient_checkpointing_disable"):
            generation_model.gradient_checkpointing_disable()
        if hasattr(generation_model, "config"):
            generation_model.config.use_cache = True
        generation_model.eval()

        rows: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            prompt = self.build_generation_prompt(record)
            generated_response = self.generate_answer(generation_model, prompt)
            generated_intent, generated_reply = SFTPostTrainGenerator.parse_intent_and_reply(generated_response)
            rows.append(
                {
                    "id": record.get("id", f"case_{index:03d}"),
                    "prompt": record.get("prompt", record.get("query", "")),
                    "history": record.get("history", []),
                    "generated_response": generated_response,
                    "generated_intent": generated_intent,
                    "generated_reply": generated_reply,
                    "source_record": record,
                }
            )
            if gen_cfg.print_every > 0 and index % gen_cfg.print_every == 0:
                print(f"post-train generation: {index}/{len(records)}")

        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        lines = [
            "# Post-Train Test Generations", "",
            f"- 测试集：`{gen_cfg.test_file}`", f"- 数量：`{len(rows)}`", ""
        ]
        for row in rows:
            lines.extend([
                f"## {row['id']}", "",
                f"用户问题：{row['prompt']}", "",
                f"分类结果：{row['generated_intent'] or ''}", "",
                f"拆分回复：{row['generated_reply'] or ''}", "",
                f"模型回答：{row['generated_response']}", "",
            ])
        markdown_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved post-train test generations to {jsonl_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SFTTrainingPipeline —— 训练编排
# ═══════════════════════════════════════════════════════════════════════════════

class SFTTrainingPipeline:
    """完整的 SFT 训练流水线。

    流程概览：
      Step 1: 加载 YAML 配置 + CLI 覆盖 → 解析为强类型 dataclass
      Step 2: 加载 tokenizer
      Step 3: 读取 jsonl → normalize messages → ChatML 模板 → tokenize + labels
              → 这一步实现了"SFT loss 只算 assistant 部分"
      Step 4: 保存预处理样本，方便排查模板或 labels 错误
      Step 5: 加载 QLoRA 模型 + 创建 LoRA adapter
      Step 6: 创建 TRL SFTTrainer，注入 tokenized dataset
      Step 7: 训练循环（TRL/Trainer 自动处理 DDP、梯度累积、checkpoint）
      Step 8: 保存 adapter、metrics、loss 曲线、评测集生成结果
    """

    def __init__(self, config: SFTExperimentConfig, project_root: str | Path | None = None):
        self.config = config
        self.project_root = Path(project_root).resolve() if project_root else find_project_root()

    # ── Trainer 参数构建 ────────────────────────────────────────────────

    # 功能：优先使用 TRL 的 SFTConfig，缺失时回退到 Transformers TrainingArguments。
    @staticmethod
    def _resolve_training_args_class() -> type:
        try:
            from trl import SFTConfig
            return SFTConfig
        except ImportError:
            from transformers import TrainingArguments
            return TrainingArguments

    # 功能：兼容 Transformers 新旧版本中 evaluation_strategy/eval_strategy 的命名差异。
    @staticmethod
    def _add_strategy_kwargs(args_class: type, kwargs: dict[str, Any], config: SFTExperimentConfig) -> dict[str, Any]:
        signature = inspect.signature(args_class.__init__)
        if "eval_strategy" in signature.parameters:
            kwargs["eval_strategy"] = config.training.eval_strategy
        else:
            kwargs["evaluation_strategy"] = config.training.eval_strategy
        kwargs["save_strategy"] = config.training.save_strategy
        if config.training.metric_for_best_model:
            kwargs["metric_for_best_model"] = config.training.metric_for_best_model
        if config.training.greater_is_better is not None:
            kwargs["greater_is_better"] = config.training.greater_is_better
        return kwargs

    # 功能：在 TRL SFTConfig 可用时加入跳过二次数据预处理的参数。
    @staticmethod
    def _add_sft_config_kwargs(args_class: type, kwargs: dict[str, Any], config: SFTExperimentConfig) -> dict[str, Any]:
        signature = inspect.signature(args_class.__init__)

        # 新版 TRL 的 SFTConfig 支持这些参数；旧版不支持时会在后面自动过滤。
        if "max_seq_length" in signature.parameters:
            kwargs["max_seq_length"] = config.training.max_seq_length
        if "max_length" in signature.parameters:
            kwargs["max_length"] = config.training.max_seq_length
        if "packing" in signature.parameters:
            kwargs["packing"] = False
        if "dataset_kwargs" in signature.parameters:
            # 我们已经自己完成 tokenize 和 labels，要求 TRL 不要再次 prepare dataset。
            kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
        if "remove_unused_columns" in signature.parameters:
            kwargs["remove_unused_columns"] = False
        return kwargs

    # 功能：过滤当前库版本不支持的参数，提升脚本跨版本可用性。
    @staticmethod
    def _filter_supported_kwargs(args_class: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(args_class.__init__)
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    # 功能：按当前 Transformers/TRL 版本创建训练参数对象。
    def _build_training_arguments(self) -> Any:
        args_class = SFTTrainingPipeline._resolve_training_args_class()

        # 先按通用 Transformers TrainingArguments 写参数，再根据实际版本过滤。
        kwargs = {
            "output_dir": self.config.training.output_dir,
            "learning_rate": self.config.training.learning_rate,
            "num_train_epochs": self.config.training.num_train_epochs,
            "per_device_train_batch_size": self.config.training.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.training.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
            "warmup_ratio": self.config.training.warmup_ratio,
            "lr_scheduler_type": self.config.training.lr_scheduler_type,
            "logging_steps": self.config.training.logging_steps,
            "save_steps": self.config.training.save_steps,
            "eval_steps": self.config.training.eval_steps,
            "save_total_limit": self.config.training.save_total_limit,
            "max_grad_norm": self.config.training.max_grad_norm,
            "gradient_checkpointing": self.config.training.gradient_checkpointing,
            "bf16": self.config.training.bf16,
            "fp16": self.config.training.fp16,
            "optim": self.config.training.optim,
            "ddp_find_unused_parameters": self.config.training.ddp_find_unused_parameters,
            "report_to": [] if self.config.training.report_to == "none" else [self.config.training.report_to],
            "seed": self.config.training.seed,
            "load_best_model_at_end": self.config.training.load_best_model_at_end,
        }
        kwargs = SFTTrainingPipeline._add_strategy_kwargs(args_class, kwargs, self.config)
        kwargs = SFTTrainingPipeline._add_sft_config_kwargs(args_class, kwargs, self.config)
        return args_class(**SFTTrainingPipeline._filter_supported_kwargs(args_class, kwargs))

    # 功能：按 TRL 版本差异构造 SFTTrainer。
    def _build_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Any | None,
        data_collator: AssistantOnlyDataCollator,
        peft_config: Any,
        training_args: Any,
    ) -> Any:
        try:
            from trl import SFTTrainer
        except ImportError as exc:
            raise RuntimeError("TRL is required for SFT training. Please install `trl`.") from exc

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
        }
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config
        signature = inspect.signature(SFTTrainer.__init__)

        # TRL 不同版本用 tokenizer 或 processing_class，靠签名判断更稳。
        if "processing_class" in signature.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in signature.parameters:
            trainer_kwargs["tokenizer"] = tokenizer
        if "dataset_kwargs" in signature.parameters:
            # 这里再次传入 skip_prepare_dataset，防止 SFTTrainer 覆盖我们做好的 labels。
            trainer_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

        try:
            return SFTTrainer(**trainer_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "Failed to initialize TRL SFTTrainer with pre-tokenized assistant-only labels. "
                "Please use a TRL version that supports `dataset_kwargs={'skip_prepare_dataset': True}` "
                "or adjust the trainer compatibility layer."
            ) from exc

    # ── 产物保存 ────────────────────────────────────────────────────────

    # 功能：判断当前进程是否负责写输出文件，避免 DDP 多进程重复写。
    @staticmethod
    def _is_world_process_zero(trainer: Any) -> bool:
        checker = getattr(trainer, "is_world_process_zero", None)
        if callable(checker):
            return bool(checker())
        return os.environ.get("RANK", "0") == "0"

    # 功能：在 DDP 下让非主进程等待主进程完成训练后生成，避免提前退出。
    @staticmethod
    def _barrier_if_distributed() -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # 功能：把最终配置保存到输出目录，保证服务器训练可追溯。
    @staticmethod
    def _save_resolved_config(config: SFTExperimentConfig, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_sft_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    # 功能：保存少量 tokenized 样本，方便复查模板和 label mask 是否正确。
    @staticmethod
    def _save_preprocessed_samples(dataset: Any, output_dir: Path, sample_count: int) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_path = output_dir / "preprocessed_samples.jsonl"
        with sample_path.open("w", encoding="utf-8") as f:
            for index in range(min(sample_count, len(dataset))):
                item = dataset[index]
                label_count = sum(1 for value in item["labels"] if value != IGNORE_INDEX)
                payload = {
                    "index": index,
                    "length": len(item["input_ids"]),
                    "label_token_count": label_count,
                    "text": item.get("text", ""),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 功能：保存本次训练/验证划分信息，便于复现实验。
    @staticmethod
    def _save_dataset_split_report(report: dict[str, Any], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "dataset_split_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Loss 可视化 ─────────────────────────────────────────────────────

    # 功能：从日志中抽取某个指标的 step-value 点。
    @staticmethod
    def _metric_points(log_history: list[dict[str, Any]], metric_name: str) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        for row in log_history:
            if row.get(metric_name) is None or row.get("step") is None:
                continue
            try:
                points.append((int(row["step"]), float(row[metric_name])))
            except (TypeError, ValueError):
                continue
        return points

    # 功能：把 eval loss 转成 perplexity 曲线点，过大 loss 跳过以避免数值溢出。
    @staticmethod
    def _perplexity_points(eval_points: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return [(step, math.exp(loss)) for step, loss in eval_points if loss < 20]

    # 功能：保存一张单指标折线图。
    @staticmethod
    def _save_line_plot(
        plt: Any, points: list[tuple[int, float]], path: Path,
        title: str, ylabel: str, label: str,
    ) -> None:
        if not points:
            return
        steps, values = zip(*points)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, values, marker="o", linewidth=1.6, label=label)
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    # 功能：从 Trainer state 中导出 train/eval loss 历史和可视化曲线。
    def _save_loss_artifacts(self, trainer: Any, output_dir: Path) -> None:
        if not SFTTrainingPipeline._is_world_process_zero(trainer):
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        log_history = list(getattr(trainer.state, "log_history", []) or [])
        (output_dir / "loss_history.json").write_text(
            json.dumps(log_history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        fields = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
        with (output_dir / "loss_history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in log_history:
                writer.writerow({field: row.get(field, "") for field in fields})

        train_points = SFTTrainingPipeline._metric_points(log_history, "loss")
        eval_points = SFTTrainingPipeline._metric_points(log_history, "eval_loss")
        lr_points = SFTTrainingPipeline._metric_points(log_history, "learning_rate")
        grad_norm_points = SFTTrainingPipeline._metric_points(log_history, "grad_norm")
        ppl_points = SFTTrainingPipeline._perplexity_points(eval_points)
        if not any([train_points, eval_points, lr_points, grad_norm_points, ppl_points]):
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipped training curve image generation.")
            return

        self._save_loss_curve(plt, train_points, eval_points, output_dir / "loss_curve.png")
        SFTTrainingPipeline._save_line_plot(
            plt, ppl_points, output_dir / "eval_perplexity_curve.png",
            "SFT Eval Perplexity Curve", "perplexity", "eval perplexity",
        )
        SFTTrainingPipeline._save_line_plot(
            plt, lr_points, output_dir / "learning_rate_curve.png",
            "SFT Learning Rate Curve", "learning rate", "learning rate",
        )
        SFTTrainingPipeline._save_line_plot(
            plt, grad_norm_points, output_dir / "grad_norm_curve.png",
            "SFT Grad Norm Curve", "grad norm", "grad norm",
        )
        self._save_training_diagnostics_plot(
            plt, output_dir / "training_diagnostics.png",
            train_points=train_points, eval_points=eval_points,
            ppl_points=ppl_points, lr_points=lr_points, grad_norm_points=grad_norm_points,
        )

    # 功能：保存 train/eval loss 同图。
    @staticmethod
    def _save_loss_curve(
        plt: Any, train_points: list[tuple[int, float]],
        eval_points: list[tuple[int, float]], path: Path,
    ) -> None:
        if not train_points and not eval_points:
            return
        plt.figure(figsize=(8, 5))
        if train_points:
            train_steps, train_losses = zip(*train_points)
            plt.plot(train_steps, train_losses, label="train loss", linewidth=1.6)
        if eval_points:
            eval_steps, eval_losses = zip(*eval_points)
            plt.plot(eval_steps, eval_losses, label="eval loss", marker="o", linewidth=1.6)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("SFT Loss Curve")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    # 功能：在子图上画一条指标曲线。
    @staticmethod
    def _plot_points_on_axis(
        axis: Any, points: list[tuple[int, float]],
        label: str, xlabel: str, ylabel: str,
    ) -> None:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if not points:
            axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)
            return
        steps, values = zip(*points)
        axis.plot(steps, values, marker="o", linewidth=1.5, label=label)

    # 功能：保存四宫格训练诊断图，便于快速查看 loss、困惑度、学习率和梯度范数。
    @staticmethod
    def _save_training_diagnostics_plot(
        plt: Any, path: Path,
        train_points: list[tuple[int, float]],
        eval_points: list[tuple[int, float]],
        ppl_points: list[tuple[int, float]],
        lr_points: list[tuple[int, float]],
        grad_norm_points: list[tuple[int, float]],
    ) -> None:
        if not any([train_points, eval_points, ppl_points, lr_points, grad_norm_points]):
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        SFTTrainingPipeline._plot_points_on_axis(axes[0][0], train_points, "train loss", "step", "loss")
        SFTTrainingPipeline._plot_points_on_axis(axes[0][0], eval_points, "eval loss", "step", "loss")
        axes[0][0].set_title("Loss")
        axes[0][0].legend()

        SFTTrainingPipeline._plot_points_on_axis(axes[0][1], ppl_points, "eval perplexity", "step", "perplexity")
        axes[0][1].set_title("Eval Perplexity")

        SFTTrainingPipeline._plot_points_on_axis(axes[1][0], lr_points, "learning rate", "step", "learning rate")
        axes[1][0].set_title("Learning Rate")

        SFTTrainingPipeline._plot_points_on_axis(axes[1][1], grad_norm_points, "grad norm", "step", "grad norm")
        axes[1][1].set_title("Gradient Norm")

        for axis in axes.flat:
            axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    # 功能：输出数据预处理统计，帮助定位跳样本和截断情况。
    @staticmethod
    def _log_dataset_stats(name: str, stats: SFTDatasetStats) -> None:
        print(
            f"{name}: total={stats.total_records}, usable={stats.usable_records}, "
            f"skipped={stats.skipped_records}, truncated={stats.truncated_records}, "
            f"max_length={stats.max_length}"
        )

    # ── 主编排 ──────────────────────────────────────────────────────────

    # 功能：执行完整的 LoRA SFT 训练流程。
    def run(self) -> None:
        # Step 1: 固定随机种子
        set_train_seed(self.config.training.seed)

        # Step 2: 解析路径
        train_file = (
            resolve_project_path(self.config.data.train_file, self.project_root)
            if self.config.data.train_file else None
        )
        eval_file = (
            resolve_project_path(self.config.data.eval_file, self.project_root)
            if self.config.data.eval_file else None
        )
        output_dir = resolve_project_path(self.config.training.output_dir, self.project_root)

        # Step 3: 加载 tokenizer + 数据处理器
        model_builder = SFTModelBuilder(self.config)
        tokenizer = model_builder.load_tokenizer()
        data_processor = SFTDataProcessor(tokenizer, self.config)

        # Step 4: 构建训练/验证 Dataset
        if self.config.data.hf_dataset_name:
            train_dataset, train_stats, eval_dataset, eval_stats = (
                data_processor.build_hf_jddc_tokenized_datasets()
            )
            split_report = {
                "split_source": "hf_dataset_train_test_split",
                "hf_dataset_name": self.config.data.hf_dataset_name,
                "hf_eval_size": self.config.data.hf_eval_size,
            }
            SFTTrainingPipeline._log_dataset_stats("eval", eval_stats)
        else:
            train_dataset, train_stats, eval_dataset, eval_stats, split_report = (
                data_processor.build_train_eval_datasets_from_files(train_file, eval_file)
            )
            if eval_stats is not None:
                SFTTrainingPipeline._log_dataset_stats("eval", eval_stats)

        # Step 5: 保存配置和样本预览
        SFTTrainingPipeline._log_dataset_stats("train", train_stats)
        SFTTrainingPipeline._save_resolved_config(self.config, output_dir)
        SFTTrainingPipeline._save_dataset_split_report(split_report, output_dir)
        SFTTrainingPipeline._save_preprocessed_samples(
            train_dataset, output_dir, self.config.training.log_sample_count
        )

        # Step 6: 加载模型 + LoRA
        model = model_builder.load_model()
        peft_config = model_builder.build_lora_config()
        training_args = self._build_training_arguments()
        data_collator = AssistantOnlyDataCollator(pad_token_id=tokenizer.pad_token_id)

        # Step 7: 构建 Trainer
        trainer = self._build_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            peft_config=peft_config,
            training_args=training_args,
        )
        model_builder.align_trainable_parameter_dtypes(trainer.model)

        # Step 8: 训练
        train_result = trainer.train(
            resume_from_checkpoint=self.config.training.resume_from_checkpoint
        )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        trainer.save_state()
        self._save_loss_artifacts(trainer, output_dir)

        # Step 9: 训练后生成评测集回答
        if self.config.post_train_generation.enabled:
            if not SFTTrainingPipeline._is_world_process_zero(trainer):
                SFTTrainingPipeline._barrier_if_distributed()
                return
            generator = SFTPostTrainGenerator(tokenizer, self.config)
            generator.generate_and_save(
                model=getattr(trainer.model, "module", trainer.model),
                output_dir=output_dir,
                project_root=self.project_root,
            )
            SFTTrainingPipeline._barrier_if_distributed()


# ═══════════════════════════════════════════════════════════════════════════════
# 配置加载与校验
# ═══════════════════════════════════════════════════════════════════════════════

# 功能：把 YAML 字典转换成强类型 SFT 配置对象。
def load_sft_config(config_path: str | Path) -> SFTExperimentConfig:
    path = Path(config_path).expanduser().resolve()
    raw = read_yaml_config(path)

    # YAML 顶层按 model/lora/training/data 分区，对应下面四个 dataclass。
    model = SFTModelConfig(**raw.get("model", {}))
    lora = SFTLoraConfig(**raw.get("lora", {}))
    training = SFTTrainingConfig(**raw.get("training", {}))
    data = SFTDataConfig(**raw.get("data", {}))
    post_train_generation = SFTPostTrainGenerationConfig(**raw.get("post_train_generation", {}))

    # 聚合后做一次统一校验，避免训练跑到一半才发现配置缺失。
    config = SFTExperimentConfig(
        model=model,
        lora=lora,
        training=training,
        data=data,
        post_train_generation=post_train_generation,
    )
    validate_sft_config(config)
    return config


# 功能：检查配置中的关键字段，尽早暴露路径或超参错误。
def validate_sft_config(config: SFTExperimentConfig) -> None:
    if not config.model.base_model:
        raise ValueError("`model.base_model` is required.")
    if not config.data.train_file and not config.data.hf_dataset_name:
        raise ValueError("Either `data.train_file` or `data.hf_dataset_name` is required.")
    if config.training.max_seq_length <= 0:
        raise ValueError("`training.max_seq_length` must be positive.")
    if config.lora.rank <= 0:
        raise ValueError("`lora.rank` must be positive.")
    if not 0 <= config.data.validation_split_ratio < 1:
        raise ValueError("`data.validation_split_ratio` must be in [0, 1).")
    if config.data.validation_split_size is not None and config.data.validation_split_size < 0:
        raise ValueError("`data.validation_split_size` must be non-negative.")
    if config.data.chat_template not in {"qwen3_nothink", "chatml"}:
        raise ValueError("Only `qwen3_nothink` and `chatml` templates are supported in this project.")


# 功能：允许命令行覆盖服务器上最常变化的训练路径。
def apply_train_cli_overrides(
    config: SFTExperimentConfig,
    base_model: str | None = None,
    output_dir: str | None = None,
    train_file: str | None = None,
    eval_file: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> SFTExperimentConfig:
    if base_model:
        config.model.base_model = base_model
    if output_dir:
        config.training.output_dir = output_dir
    if train_file:
        config.data.train_file = train_file
    if eval_file:
        config.data.eval_file = eval_file
    if resume_from_checkpoint:
        config.training.resume_from_checkpoint = resume_from_checkpoint
    validate_sft_config(config)
    return config



# ═══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════════════════

# 功能：执行完整的 LoRA SFT 训练流程。
def train_sft(
    config_path: str | Path,
    project_root: str | Path | None = None,
    base_model: str | None = None,
    output_dir: str | None = None,
    train_file: str | None = None,
    eval_file: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    # 1. 读取 YAML 配置，并固定随机种子，保证数据处理和训练尽量可复现。
    config = load_sft_config(config_path)
    config = apply_train_cli_overrides(
        config,
        base_model=base_model,
        output_dir=output_dir,
        train_file=train_file,
        eval_file=eval_file,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    pipeline = SFTTrainingPipeline(config, project_root)
    pipeline.run()


# 功能：构造命令行参数解析器。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a LoRA SFT model for e-commerce customer service.")
    parser.add_argument("--config", default="configs/sft_lora.yaml", help="Path to the SFT YAML config.")
    parser.add_argument("--project_root", default=None, help="Project root for resolving relative paths.")
    parser.add_argument("--base_model", default=None, help="Override model.base_model, useful for server-local models.")
    parser.add_argument("--output_dir", default=None, help="Override training.output_dir.")
    parser.add_argument("--train_file", default=None, help="Override data.train_file.")
    parser.add_argument("--eval_file", default=None, help="Override data.eval_file.")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Override training.resume_from_checkpoint.")
    return parser


# 功能：命令行入口，解析参数并启动训练。
def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train_sft(
        args.config,
        project_root=args.project_root,
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_file=args.train_file,
        eval_file=args.eval_file,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
