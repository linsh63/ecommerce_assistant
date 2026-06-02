"""PPO（Proximal Policy Optimization）RLHF 训练模块。

修复版：不再 deepcopy 参考模型、修正双前向 bug、梯度累积。
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .utils import read_yaml_config, set_train_seed


@dataclass
class PPOModelConfig:
    sft_adapter_path: str = ""
    base_model: str = "Qwen/Qwen3-8B"
    trust_remote_code: bool = True
    load_in_4bit: bool = True


@dataclass
class PPOTrainingConfig:
    output_dir: str = "outputs/ppo_jddc_rebuild_v2"
    learning_rate: float = 5e-6
    beta: float = 0.1
    clip_epsilon: float = 0.2
    max_steps: int = 200
    prompts_per_step: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    logging_steps: int = 5
    save_steps: int = 50
    save_total_limit: int = 2


@dataclass
class PPODataConfig:
    prompt_file: str = ""
    max_prompts: int | None = 500
    shuffle_seed: int = 42


@dataclass
class PPORewardConfig:
    api_base_url: str = ""
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout: int = 60
    max_retries: int = 2


@dataclass
class PPOExperimentConfig:
    model: PPOModelConfig = field(default_factory=PPOModelConfig)
    training: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)
    data: PPODataConfig = field(default_factory=PPODataConfig)
    reward: PPORewardConfig = field(default_factory=PPORewardConfig)


class DeepSeekRewardScorer:
    SYSTEM = """你是电商客服质检专家。请对AI客服回复评分，5维度各1-5分。

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

    def __init__(self, config: PPORewardConfig):
        self.config = config

    def score(self, prompt: str, response: str) -> float:
        payload = {
            "model": self.config.model, "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
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


class PPOTrainingPipeline:
    def __init__(self, config: PPOExperimentConfig):
        self.config = config
        if not self.config.data.prompt_file:
            raise ValueError("data.prompt_file is required")

    def _load_policy_model(self) -> tuple[Any, Any]:
        """Load trainable policy model + tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"trust_remote_code": self.config.model.trust_remote_code, "torch_dtype": torch.float16}
        if self.config.model.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(self.config.model.base_model, **model_kwargs)
        if self.config.model.sft_adapter_path:
            from peft import PeftModel, prepare_model_for_kbit_training
            if self.config.model.load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(model, self.config.model.sft_adapter_path, is_trainable=True)
        return model, tokenizer

    def _make_chatml(self, messages: list[dict]) -> str:
        """Build ChatML prompt from messages."""
        parts = []
        for m in messages:
            r = "user" if m["role"] == "user" else "assistant"
            parts.append(f"<|im_start|>{r}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _generate(self, model: Any, tokenizer: Any, prompt: str,
                  history_msgs: list[dict] = None) -> str:
        """Generate one response."""
        device = next(model.parameters()).device
        msgs = (history_msgs or []) + [{"role": "user", "content": prompt}]
        if not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": "你是电商平台客服助手。"}] + msgs
        chatml = self._make_chatml(msgs)
        inputs = tokenizer(chatml, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.config.training.max_new_tokens,
                                 do_sample=True, temperature=self.config.training.temperature,
                                 top_p=self.config.training.top_p, pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id)
        new = out[0, inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new, skip_special_tokens=True)
        for m in ["<|im_end|>", "<|endoftext|>"]:
            if m in text: text = text.split(m, 1)[0]
        return text.strip()

    def _compute_log_prob(self, model: Any, tokenizer: Any, prompt: str, response: str,
                          history_msgs: list[dict] = None) -> float:
        """Compute mean log-prob over response tokens (returns detached float)."""
        device = next(model.parameters()).device
        msgs = (history_msgs or []) + [{"role": "user", "content": prompt}]
        if not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": "你是电商平台客服助手。"}] + msgs
        chatml = self._make_chatml(msgs)
        full = chatml + response + "<|im_end|>"
        inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048).to(device)
        prompt_len = len(tokenizer.encode(chatml, add_special_tokens=False))

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        targets = inputs.input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(token_lp)
        if prompt_len < mask.shape[1]:
            mask[:, prompt_len:] = 1.0
        return (token_lp * mask).sum().item() / mask.sum().clamp(min=1).item()

    def _compute_loss(self, model: Any, tokenizer: Any,
                      prompt: str, response: str, advantage: float,
                      old_log_prob: float, history_msgs: list[dict]) -> torch.Tensor:
        """Single forward pass → PPO loss. No ref model needed for short runs."""
        device = next(model.parameters()).device
        msgs = (history_msgs or []) + [{"role": "user", "content": prompt}]
        if not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": "你是电商平台客服助手。"}] + msgs
        chatml = self._make_chatml(msgs)
        full = chatml + response + "<|im_end|>"
        inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048).to(device)
        prompt_len = len(tokenizer.encode(chatml, add_special_tokens=False))

        # Single forward
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        targets = inputs.input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(token_lp)
        if prompt_len < mask.shape[1]:
            mask[:, prompt_len:] = 1.0
        new_log_prob = (token_lp * mask).sum() / mask.sum().clamp(min=1)

        # PPO clip loss（无 KL 项，短训练不需要）
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped = torch.clamp(ratio, 1 - self.config.training.clip_epsilon,
                              1 + self.config.training.clip_epsilon)
        loss = -torch.min(ratio * advantage, clipped * advantage)
        return loss

    def run(self) -> None:
        set_train_seed(self.config.training.seed)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_main = local_rank == 0

        if is_main: print("Loading policy model...")
        model, tokenizer = self._load_policy_model()
        model.train()

        prompts = self._load_prompts(Path(self.config.data.prompt_file),
                                     self.config.data.max_prompts)
        if is_main: print(f"PPO prompts: {len(prompts)}")

        scorer = DeepSeekRewardScorer(self.config.reward)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.config.training.learning_rate)

        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = []
        ga = self.config.training.gradient_accumulation_steps

        if is_main:
            print(f"PPO: {self.config.training.max_steps} steps, clip={self.config.training.clip_epsilon}, beta={self.config.training.beta}, grad_accum={ga}")

        for step in range(self.config.training.max_steps):
            optimizer.zero_grad()
            step_loss = 0.0
            step_rewards = []
            n_valid = 0

            for ga_i in range(ga):
                batch_prompts = random.sample(prompts, min(self.config.training.prompts_per_step, len(prompts)))

                for prompt_text in batch_prompts:
                    # Generate
                    response = self._generate(model, tokenizer, prompt_text)
                    old_lp = self._compute_log_prob(model, tokenizer, prompt_text, response)
                    reward = scorer.score(prompt_text, response)
                    advantage = reward

                    # Compute loss in single forward
                    loss = self._compute_loss(model, tokenizer, prompt_text, response,
                                              advantage, old_lp, None)
                    loss = loss / ga
                    loss.backward()

                    step_loss += loss.item()
                    step_rewards.append(reward)
                    n_valid += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)
            optimizer.step()

            avg_r = sum(step_rewards) / max(len(step_rewards), 1)
            step_loss /= max(n_valid, 1)
            metrics.append({"step": step, "loss": round(step_loss, 4), "avg_reward": round(avg_r, 3)})

            if is_main and step % self.config.training.logging_steps == 0:
                print(f"  Step {step}/{self.config.training.max_steps} | loss={step_loss:.4f} | avg_reward={avg_r:.3f}")

            if is_main and (step + 1) % self.config.training.save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{step + 1}"
                model.save_pretrained(ckpt_dir)
                print(f"  Saved {ckpt_dir}")

        if is_main:
            model.save_pretrained(output_dir)
            with (output_dir / "ppo_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"PPO done. Adapter saved to {output_dir}")

    @staticmethod
    def _load_prompts(path: Path, max_prompts: int | None = None) -> list[str]:
        # Exclude eval IDs to prevent contamination
        eval_ids = set()
        eval_path = path.parent / "eval_test.jsonl"
        if eval_path.exists():
            with eval_path.open(encoding="utf-8") as f:
                eval_ids = {json.loads(l)["id"] for l in f if l.strip()}

        prompts = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row.get("id") in eval_ids: continue
                p = row.get("prompt", "")
                if not p and row.get("messages"):
                    for m in reversed(row["messages"]):
                        if m.get("role") == "user": p = m["content"]; break
                if p: prompts.append(p)
        if max_prompts:
            random.shuffle(prompts)
            prompts = prompts[:max_prompts]
        return prompts


def load_ppo_config(config_path: str | Path) -> PPOExperimentConfig:
    path = Path(config_path).expanduser().resolve()
    raw = read_yaml_config(path)
    return PPOExperimentConfig(
        model=PPOModelConfig(**raw.get("model", {})),
        training=PPOTrainingConfig(**raw.get("training", {})),
        data=PPODataConfig(**raw.get("data", {})),
        reward=PPORewardConfig(**raw.get("reward", {})),
    )
