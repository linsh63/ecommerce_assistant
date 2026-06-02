"""DPO（Direct Preference Optimization）偏好对齐训练模块。

参考：project_guide_project1.md 第 3 周。
核心思路：DPO 直接在偏好数据上优化模型，让它偏好 chosen 而非 rejected，
不需要额外训练奖励模型。DPO 损失函数为：

    L_DPO = -E[ log σ( β * log(π_θ(chosen)/π_ref(chosen))
                      - β * log(π_θ(rejected)/π_ref(rejected)) ) ]

其中 β 控制 π_θ（策略模型）可以偏离 π_ref（参考模型）多远。
β 越大 → 越保守、越不敢偏离 SFT 模型；β 越小 → 偏好信号越强、但过拟合风险越大。
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import read_yaml_config


# 与 SFT 模块保持一致：padding token 的 label 用 -100，CrossEntropyLoss 会自动跳过。
IGNORE_INDEX = -100


# ═══════════════════════════════════════════════════════════════════════════
# 配置 dataclass —— 类型安全、可校验、可被命令行覆盖
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DPOModelConfig:
    """模型加载配置。

    SFT adapter 作为 DPO 训练的起点（策略模型 π_θ），
    参考模型 π_ref 用冻结的基座模型提供 KL 惩罚的基线概率。
    """

    # SFT 阶段训练出的最优 LoRA adapter 路径
    sft_adapter_path: str = ""
    # 基座模型（用于参考模型和加载 adapter 的 backbone）
    base_model: str = "Qwen/Qwen3-8B"
    trust_remote_code: bool = True
    torch_dtype: str = "float16"
    load_in_4bit: bool = True


@dataclass
class DPOTrainingConfig:
    """DPO 训练超参。

    beta —— 最关键的 DPO 超参，控制 KL 惩罚强度：
      - beta=0.1（默认）：适度偏离参考模型，大多数情况下的安全选择。
      - beta=0.05：偏好信号更强，模型更激进地学 chosen 风格，但可能过拟合。
      - beta=0.3：更保守，更贴近 SFT 模型，适合偏好数据噪声较大时。

    面试答法："beta 调大更保守、不敢偏离参考模型；调小偏好更强但容易过拟合。
    我一般从 0.1 起步，再跑 0.05 和 0.3 做消融对比。"
    """

    output_dir: str = "outputs/dpo_jddc_rebuild_v2"
    beta: float = 0.1
    # DPO 学习率通常比 SFT 低一个数量级（5e-5 vs 2e-4），
    # 因为起点已经是微调好的模型，不需要大幅更新。
    learning_rate: float = 5e-5
    # DPO 通常 1 个 epoch 就够了——偏好信号强，多 epoch 容易过拟合。
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 1024  # prompt（用户问题+历史）的最大 token 数
    max_length: int = 2048  # prompt + response 的总最大 token 数
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = False
    fp16: bool = True
    optim: str = "paged_adamw_8bit"
    seed: int = 42
    save_total_limit: int = 2
    report_to: str = "none"
    resume_from_checkpoint: str | None = None


@dataclass
class DPODataConfig:
    """DPO 偏好数据路径与处理配置。"""

    train_file: str = ""
    eval_file: str = ""
    max_samples: int | None = None
    max_eval_samples: int = 200  # DPO 验证集不需要太大
    shuffle_seed: int = 42


@dataclass
class DPOExperimentConfig:
    """一次完整 DPO 实验的聚合配置。"""

    model: DPOModelConfig = field(default_factory=DPOModelConfig)
    dpo: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)
    data: DPODataConfig = field(default_factory=DPODataConfig)


# ═══════════════════════════════════════════════════════════════════════════
# 数据处理器 —— 偏好对 → Dataset
# ═══════════════════════════════════════════════════════════════════════════

class DPODataProcessor:
    """加载、校验偏好对，构建 DPO 训练所需的 Dataset。

    DPO 数据格式（jsonl，每行一个偏好对）：
      {"prompt": "用户问题文本", "chosen": "好回复", "rejected": "差回复"}

    TRL 的 DPOTrainer 期望 Dataset 包含三个字符串列：
      - prompt: 用户问题
      - chosen: 被偏好的回复
      - rejected: 被拒绝的回复
    tokenization 由 DPOTrainer 内部自动完成。
    """

    def __init__(self, tokenizer: Any, config: DPOExperimentConfig):
        self.tokenizer = tokenizer
        self.config = config
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    # 功能：从 jsonl 读取偏好对，校验 chosen/rejected 字段合法。
    def load_pairs(self, path: Path, max_samples: int | None = None) -> list[dict]:
        pairs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)

                chosen = str(row.get("chosen") or "").strip()
                rejected = str(row.get("rejected") or "").strip()
                # 必须同时有 chosen 和 rejected，且不能相同
                if not chosen or not rejected:
                    continue
                if chosen == rejected:
                    continue

                # 优先用已有的 prompt 字段；没有则从 messages 中取最后一条 user 消息
                prompt = row.get("prompt", "")
                if not prompt and row.get("messages"):
                    msgs = row["messages"]
                    for m in reversed(msgs):
                        if m.get("role") == "user":
                            prompt = m["content"]
                            break

                if not prompt:
                    continue

                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })
                if max_samples and len(pairs) >= max_samples:
                    break
        return pairs

    # 功能：构建训练集和可选的验证集。
    def build_datasets(
        self, train_path: Path, eval_path: Path | None = None
    ) -> tuple[Any, Any | None]:
        from datasets import Dataset

        train_pairs = self.load_pairs(train_path, self.config.data.max_samples)
        if not train_pairs:
            raise RuntimeError(f"No valid DPO pairs found in {train_path}")

        train_dataset = Dataset.from_list(train_pairs)

        eval_dataset = None
        if eval_path and eval_path.exists():
            eval_pairs = self.load_pairs(eval_path, self.config.data.max_eval_samples)
            if eval_pairs:
                eval_dataset = Dataset.from_list(eval_pairs)

        return train_dataset, eval_dataset


# ═══════════════════════════════════════════════════════════════════════════
# 模型构建器 —— 加载 SFT adapter 作为 DPO 起点
# ═══════════════════════════════════════════════════════════════════════════

class DPOModelBuilder:
    """加载 DPO 所需的两个模型。

    DPO 需要两份模型：
      1. 策略模型 π_θ —— 从 SFT adapter 初始化，训练中更新参数。
      2. 参考模型 π_ref —— 冻结的 SFT 模型副本，提供 KL 惩罚的基线概率。
         不计算梯度，只用于前向传播得到参考 log-prob。

    TRL 的 DPOTrainer 如果收到 ref_model=None，会自动克隆一份策略模型
    作为参考模型。这里提供显式加载方法，方便排查问题时单独控制。
    """

    def __init__(self, config: DPOExperimentConfig):
        self.config = config

    # 功能：加载 SFT adapter 作为 DPO 策略模型（可训练）。
    def load_policy_model(self) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel, prepare_model_for_kbit_training

        model_kwargs: dict = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "torch_dtype": torch.float16,
        }

        # QLoRA: 4bit 量化加载基座模型，节省显存
        if self.config.model.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # DDP 时每个进程绑定到自己的 GPU，避免两个 rank 挤在同一张卡上
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            model_kwargs["device_map"] = {"": int(local_rank)}

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model, **model_kwargs
        )

        if self.config.model.load_in_4bit:
            base_model = prepare_model_for_kbit_training(base_model)

        # 把 SFT 训好的 LoRA adapter 挂到基座上，作为 DPO 的起点
        if self.config.model.sft_adapter_path:
            model = PeftModel.from_pretrained(
                base_model,
                self.config.model.sft_adapter_path,
                is_trainable=True,
            )
        else:
            model = base_model

        model.config.use_cache = False  # 训练时必须关闭 KV cache
        return model

    # 功能：FP16 AMP 训练前把 bfloat16 可训练参数转为 FP32，避免 GradScaler 报错。
    def align_trainable_dtypes(self, model: Any) -> None:
        import torch

        if not self.config.dpo.fp16 or self.config.dpo.bf16:
            return
        converted = 0
        for p in model.parameters():
            if p.requires_grad and p.dtype in {torch.float16, torch.bfloat16}:
                p.data = p.data.to(torch.float32)
                converted += 1
        if converted:
            rank = os.environ.get("RANK", "0")
            if rank == "0":
                print(f"DPO: converted {converted} low-precision params to FP32 for AMP compat")

    # 功能：返回参考模型。返回 None 让 DPOTrainer 自动从策略模型克隆。
    def load_reference_model(self) -> Any:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 训练流水线 —— 串联数据、模型、训练、保存
# ═══════════════════════════════════════════════════════════════════════════

class DPOTrainingPipeline:
    """完整的 DPO 训练流水线。

    流程概览：
      Step 1: 加载 tokenizer
      Step 2: 从 jsonl 加载偏好对 → Dataset
      Step 3: 加载 SFT adapter 作为策略模型
      Step 4: 配置 DPOConfig（含 beta 等关键超参）
      Step 5: 创建 DPOTrainer 并训练
      Step 6: 保存 adapter + 训练指标
    """

    def __init__(self, config: DPOExperimentConfig):
        self.config = config
        self._validate_config()

    # 功能：启动前校验关键配置。
    def _validate_config(self) -> None:
        if not self.config.data.train_file:
            raise ValueError("data.train_file is required for DPO training")
        if self.config.dpo.beta <= 0:
            raise ValueError("dpo.beta must be positive")

    # 功能：执行完整 DPO 训练流程。
    def run(self) -> None:
        import torch
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer

        # ── Step 1: 加载 tokenizer ───────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Step 2: 加载偏好数据 ─────────────────────────────────────
        processor = DPODataProcessor(tokenizer, self.config)
        train_path = Path(self.config.data.train_file)
        eval_path = (
            Path(self.config.data.eval_file)
            if self.config.data.eval_file
            else None
        )
        train_dataset, eval_dataset = processor.build_datasets(train_path, eval_path)
        print(f"DPO 训练偏好对数量: {len(train_dataset)}")
        if eval_dataset:
            print(f"DPO 验证偏好对数量: {len(eval_dataset)}")

        # ── Step 3: 加载策略模型（从 SFT adapter 初始化） ────────────
        builder = DPOModelBuilder(self.config)
        policy_model = builder.load_policy_model()
        builder.align_trainable_dtypes(policy_model)
        ref_model = builder.load_reference_model()

        # ── Step 4: 配置 DPO 训练参数 ─────────────────────────────────
        # DPOConfig 继承自 Transformers 的 TrainingArguments，增加了
        # beta、max_prompt_length、max_length 等 DPO 专属参数。
        # 不同 TRL 版本参数名可能不同，用签名过滤保持兼容。
        dpo_kwargs = {
            "output_dir": self.config.dpo.output_dir,
            "beta": self.config.dpo.beta,
            "max_prompt_length": self.config.dpo.max_prompt_length,
            "max_length": self.config.dpo.max_length,
            "learning_rate": self.config.dpo.learning_rate,
            "num_train_epochs": self.config.dpo.num_train_epochs,
            "per_device_train_batch_size": self.config.dpo.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.dpo.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.config.dpo.gradient_accumulation_steps,
            "warmup_ratio": self.config.dpo.warmup_ratio,
            "lr_scheduler_type": self.config.dpo.lr_scheduler_type,
            "logging_steps": self.config.dpo.logging_steps,
            "save_steps": self.config.dpo.save_steps,
            "eval_steps": self.config.dpo.eval_steps,
            "max_grad_norm": self.config.dpo.max_grad_norm,
            "gradient_checkpointing": self.config.dpo.gradient_checkpointing,
            "bf16": self.config.dpo.bf16,
            "fp16": self.config.dpo.fp16,
            "optim": self.config.dpo.optim,
            "seed": self.config.dpo.seed,
            "save_total_limit": self.config.dpo.save_total_limit,
            "report_to": (
                []
                if self.config.dpo.report_to == "none"
                else [self.config.dpo.report_to]
            ),
            "remove_unused_columns": False,
        }
        import inspect
        sig = inspect.signature(DPOConfig.__init__)
        dpo_kwargs = {k: v for k, v in dpo_kwargs.items() if k in sig.parameters}
        dpo_args = DPOConfig(**dpo_kwargs)

        # ── Step 5: 创建 DPOTrainer 并开始训练 ────────────────────────
        # DPOTrainer 内部对每个 (prompt, chosen, rejected) 三元组：
        #   1. 在策略模型和参考模型下分别计算 chosen 和 rejected 的 log-prob
        #   2. 代入 DPO loss 公式，用 beta 控制 KL 惩罚强度
        #   3. 反向传播只更新策略模型，参考模型保持冻结
        trainer = DPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        train_result = trainer.train(
            resume_from_checkpoint=self.config.dpo.resume_from_checkpoint
        )

        # ── Step 6: 保存 adapter 和训练指标 ───────────────────────────
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        if eval_dataset:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        print(f"DPO 训练完成。Adapter 已保存到 {self.config.dpo.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# 配置加载 —— YAML → 强类型 dataclass
# ═══════════════════════════════════════════════════════════════════════════

# 功能：从 YAML 文件加载 DPO 实验配置。
def load_dpo_config(config_path: str | Path) -> DPOExperimentConfig:
    path = Path(config_path).expanduser().resolve()
    raw = read_yaml_config(path)

    model = DPOModelConfig(**raw.get("model", {}))
    dpo = DPOTrainingConfig(**raw.get("dpo", {}))
    data = DPODataConfig(**raw.get("data", {}))
    return DPOExperimentConfig(model=model, dpo=dpo, data=data)

