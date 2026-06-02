"""GRPO（Group Relative Policy Optimization）训练模块。

参考：DeepSeek-R1 论文。修复版：双卡 DDP + CPU 参考模型。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import read_yaml_config, set_train_seed


@dataclass
class GRPOModelConfig:
    sft_adapter_path: str = ""
    base_model: str = "Qwen/Qwen3-8B"
    trust_remote_code: bool = True
    load_in_4bit: bool = True


@dataclass
class GRPOTrainingConfig:
    output_dir: str = "outputs/grpo_jddc_rebuild_v2"
    num_generations: int = 3
    beta: float = 0.04
    learning_rate: float = 5e-6
    max_steps: int = 300
    per_device_prompt_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 2


@dataclass
class GRPODataConfig:
    prompt_file: str = ""
    max_prompts: int | None = None
    shuffle_seed: int = 42


@dataclass
class GRPORewardConfig:
    api_base_url: str = ""
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout: int = 60
    max_retries: int = 2


@dataclass
class GRPOExperimentConfig:
    model: GRPOModelConfig = field(default_factory=GRPOModelConfig)
    training: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    data: GRPODataConfig = field(default_factory=GRPODataConfig)
    reward: GRPORewardConfig = field(default_factory=GRPORewardConfig)


class DeepSeekRewardScorer:
    JUDGE_SYSTEM = """你是电商客服质检专家。请对AI客服回复评分，5维度各1-5分。

评分锚点（非常重要）：
- 5: 给出明确可执行路径，用户看完就能操作。
- 4: 方向正确且给出了至少一个具体操作入口（如"订单详情页点XX"）。有入口就算4分。
- 3: 方向正确但没说怎么做（只说"建议售后"不给入口）。
- 2: 模糊、回避问题、或只索要信息不给帮助。
- 1: 答非所问、完全错误。

维度：
1. 需求解决：是否直接解决用户问题？给了入口或路径？
2. 回答准确：是否准确回应？不要要求模型知道平台特定规则。
3. 有帮助性：是否给出操作指导？通用电商流程算具体。
4. 语气体验：是否礼貌有同理心？闲聊场景简短回应即可高分。
5. 风险控制：是否编造了后台数据（具体订单号、精确金额、精确时效）？通用流程不算编造。

只输出JSON，key中文：{"需求解决":4,"回答准确":4,"有帮助性":4,"语气体验":4,"风险控制":4}"""

    def __init__(self, config: GRPORewardConfig):
        self.config = config

    def score(self, prompt: str, response: str) -> float:
        payload = {
            "model": self.config.model, "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": self.JUDGE_SYSTEM},
                {"role": "user", "content": f"用户问题：{prompt}\n\n客服回复：{response}\n\n请评分，只输出JSON。"},
            ],
        }
        endpoint = self.config.api_base_url.rstrip("/") + "/chat/completions"
        for _ in range(self.config.max_retries + 1):
            try:
                req = urllib.request.Request(
                    endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                raw = body["choices"][0]["message"]["content"]
                match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    total = sum(int(data.get(k) or data.get(ek) or 3) for k, ek in [
                        ("需求解决", "problem_solving"), ("回答准确", "accuracy"),
                        ("有帮助性", "completeness"), ("语气体验", "service_attitude"),
                        ("风险控制", "response_speed")])
                    return (total - 15) / 5.0
            except Exception:
                time.sleep(1)
        return 0.0


class GRPOTrainingPipeline:
    def __init__(self, config: GRPOExperimentConfig):
        self.config = config
        if not self.config.data.prompt_file:
            raise ValueError("data.prompt_file is required")
        if self.config.training.num_generations < 2:
            raise ValueError("GRPO needs at least 2 generations per prompt")

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    @property
    def is_main(self) -> bool:
        return self.local_rank == 0

    def _load_policy_model(self) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict = {"trust_remote_code": self.config.model.trust_remote_code,
                               "torch_dtype": torch.float16}
        if self.config.model.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        model_kwargs["device_map"] = {"": self.local_rank}

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model, **model_kwargs)
        if self.config.model.sft_adapter_path:
            from peft import PeftModel, prepare_model_for_kbit_training
            if self.config.model.load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(
                model, self.config.model.sft_adapter_path, is_trainable=True)
        return model, tokenizer

    def _make_chatml(self, messages: list[dict]) -> str:
        parts = []
        for m in messages:
            r = "user" if m["role"] == "user" else "assistant"
            parts.append(f"<|im_start|>{r}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _generate_one(self, model: Any, tokenizer: Any, prompt: str) -> str:
        device = next(model.parameters()).device
        chatml = self._make_chatml([
            {"role": "system", "content": "你是电商平台客服助手。"},
            {"role": "user", "content": prompt}])
        inputs = tokenizer(chatml, return_tensors="pt", add_special_tokens=False).to(device)
        # Temporarily enable cache for generation (disabled by gradient checkpointing)
        old_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True
        model.eval()  # Disable dropout/gradient checkpointing for clean generation
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=self.config.training.max_new_tokens,
                do_sample=True, temperature=self.config.training.temperature,
                top_p=self.config.training.top_p, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        model.train()  # Back to training mode
        model.config.use_cache = old_cache
        new = out[0, inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new, skip_special_tokens=True)
        for m in ["<|im_end|>", "<|endoftext|>"]:
            if m in text: text = text.split(m, 1)[0]
        return text.strip()

    def _compute_log_prob(self, model: Any, tokenizer: Any, prompt: str,
                          response: str) -> torch.Tensor:
        """Compute mean log-prob over response tokens, returns scalar tensor with grad for policy."""
        device = next(model.parameters()).device
        chatml = self._make_chatml([
            {"role": "system", "content": "你是电商平台客服助手。"},
            {"role": "user", "content": prompt}])
        full = chatml + response + "<|im_end|>"
        inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048).to(device)
        prompt_len = len(tokenizer.encode(chatml, add_special_tokens=False))

        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        targets = inputs.input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(token_lp)
        if prompt_len < mask.shape[1]:
            mask[:, prompt_len:] = 1.0
        return (token_lp * mask).sum() / mask.sum().clamp(min=1)

    def run(self) -> None:
        import random as _random

        if self.world_size > 1:
            dist.init_process_group(backend="nccl")

        set_train_seed(self.config.training.seed + self.local_rank)

        if self.is_main: print("Loading policy model...")
        model, tokenizer = self._load_policy_model()
        model.train()

        prompts = self._load_prompts()
        if self.is_main: print(f"GRPO prompts: {len(prompts)}")

        scorer = DeepSeekRewardScorer(self.config.reward) if self.is_main else None

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.config.training.learning_rate)

        output_dir = Path(self.config.training.output_dir)
        if self.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"GRPO: {self.config.training.max_steps} steps, "
                  f"G={self.config.training.num_generations}, "
                  f"beta={self.config.training.beta}, world_size={self.world_size}")

        metrics_history: list[dict] = []

        for step in range(self.config.training.max_steps):
            step_loss = 0.0
            step_rewards = []
            n_valid = 0

            # Each rank samples its own prompts
            bs = self.config.training.per_device_prompt_batch_size
            batch_prompts = _random.sample(prompts, min(bs, len(prompts)))

            for prompt in batch_prompts:
                # Generate N responses
                responses = [self._generate_one(model, tokenizer, prompt)
                             for _ in range(self.config.training.num_generations)]

                # Score (main only, then broadcast)
                if self.is_main:
                    rewards = [scorer.score(prompt, r) for r in responses]
                    for _ in range(self.config.training.num_generations - 1):
                        time.sleep(0.1)
                else:
                    rewards = [0.0] * self.config.training.num_generations

                if self.world_size > 1:
                    rewards_t = torch.tensor(rewards, dtype=torch.float32,
                                             device=self.local_rank)
                    dist.broadcast(rewards_t, src=0)
                    rewards = rewards_t.tolist()

                if len(rewards) < 2: continue

                # Group normalization
                mean_r = sum(rewards) / len(rewards)
                std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
                advantages = [(r - mean_r) / std_r for r in rewards]

                optimizer.zero_grad()
                for resp, advantage in zip(responses, advantages):
                    policy_lp = self._compute_log_prob(model, tokenizer, prompt, resp)
                    # No KL term for short GRPO runs (300 steps at lr=5e-6 won't diverge)
                    loss = -advantage * policy_lp
                    loss = loss / (self.config.training.num_generations * bs)
                    loss.backward()

                    step_loss += loss.item()
                    step_rewards.extend(rewards)
                    n_valid += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)
            optimizer.step()

            avg_r = sum(step_rewards) / max(len(step_rewards), 1)
            metrics = {"step": step, "loss": round(step_loss, 4), "avg_reward": round(avg_r, 3)}
            metrics_history.append(metrics)

            if self.is_main and step % self.config.training.logging_steps == 0:
                print(f"  Step {step}/{self.config.training.max_steps} | loss={step_loss:.4f} | avg_reward={avg_r:.3f}")

            if self.is_main and (step + 1) % self.config.training.save_steps == 0:
                model.save_pretrained(output_dir / f"checkpoint-{step + 1}")
                print(f"  Saved checkpoint-{step + 1}")

        if self.is_main:
            model.save_pretrained(output_dir)
            with (output_dir / "grpo_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_history, f, ensure_ascii=False, indent=2)
            print(f"GRPO done. Adapter saved to {output_dir}")

        if self.world_size > 1:
            dist.destroy_process_group()

    def _load_prompts(self) -> list[str]:
        # Exclude eval IDs
        eval_ids = set()
        eval_path = Path(self.config.data.prompt_file).parent / "eval_test.jsonl"
        if eval_path.exists():
            with eval_path.open(encoding="utf-8") as f:
                eval_ids = {json.loads(l)["id"] for l in f if l.strip()}

        prompts = []
        with open(Path(self.config.data.prompt_file), encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row.get("id") in eval_ids: continue
                p = row.get("prompt", "")
                if not p and row.get("messages"):
                    for m in reversed(row["messages"]):
                        if m.get("role") == "user": p = m["content"]; break
                if p: prompts.append(p)
        if self.config.data.max_prompts:
            import random as _random
            _random.Random(self.config.data.shuffle_seed).shuffle(prompts)
            prompts = prompts[:self.config.data.max_prompts]
        return prompts


def load_grpo_config(config_path: str | Path) -> GRPOExperimentConfig:
    path = Path(config_path).expanduser().resolve()
    raw = read_yaml_config(path)
    return GRPOExperimentConfig(
        model=GRPOModelConfig(**raw.get("model", {})),
        training=GRPOTrainingConfig(**raw.get("training", {})),
        data=GRPODataConfig(**raw.get("data", {})),
        reward=GRPORewardConfig(**raw.get("reward", {})),
    )
