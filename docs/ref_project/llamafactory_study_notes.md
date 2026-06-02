# LlamaFactory 学习笔记

本文档基于本地参考仓库 `project/ProjectSet/ref_project/LLM/LlamaFactory/` 整理，目标是帮助我们理解：数据格式模板、LoRA/QLoRA 配置、对齐算法实现，以及后续如何把电商客服数据接入 LlamaFactory。

## 1. 推荐阅读路线

先按下面顺序读，会比从入口文件硬啃更顺：

| 主题 | 重点文件 | 读什么 |
| --- | --- | --- |
| 数据格式规范 | `data/README_zh.md` | Alpaca、ShareGPT、偏好数据、KTO 数据的字段约定 |
| 数据集注册 | `data/dataset_info.json`、`src/llamafactory/data/parser.py` | 自定义数据如何通过 `dataset_info.json` 映射字段 |
| 数据格式转换 | `src/llamafactory/data/converter.py` | 原始字段如何统一成 `_prompt`、`_response`、`_system` |
| 数据加载与分发 | `src/llamafactory/data/loader.py` | 不同 `stage` 用哪个 dataset processor |
| 模板系统 | `src/llamafactory/data/template.py` | ChatML/Qwen3 模板、thinking 逻辑、tokenizer 修正 |
| SFT 预处理 | `src/llamafactory/data/processor/supervised.py` | prompt 部分如何 mask，只训练 assistant |
| 偏好数据预处理 | `src/llamafactory/data/processor/pairwise.py` | chosen/rejected 如何编码 |
| KTO 预处理 | `src/llamafactory/data/processor/feedback.py` | bool 反馈如何转成 desirable/undesirable |
| LoRA 参数 | `src/llamafactory/hparams/finetuning_args.py` | `lora_rank`、`lora_alpha`、`lora_target` 等 |
| LoRA 注入 | `src/llamafactory/model/adapter.py` | PEFT `LoraConfig` 如何创建、resume、merge |
| 量化/QLoRA | `src/llamafactory/hparams/model_args.py`、`src/llamafactory/model/model_utils/quantization.py` | `quantization_bit: 4` 如何触发 bitsandbytes |
| 训练入口 | `src/llamafactory/train/tuner.py` | `stage` 如何路由到 sft/rm/ppo/dpo/kto |
| 对齐算法 | `src/llamafactory/train/{rm,dpo,ppo,kto}/` | Reward Model、DPO/ORPO/SimPO、PPO、KTO 的核心实现 |

## 2. 数据格式规范

LlamaFactory 的核心规则是：训练配置里的 `dataset: xxx` 不直接指向文件，而是先在 `dataset_dir/dataset_info.json` 中查找 `xxx` 的描述。默认 `dataset_dir` 是 `data`，也可以在 YAML 里改成自己的目录。

`dataset_info.json` 支持三类来源：

- 本地文件：`file_name`
- Hugging Face / ModelScope / OpenMind 数据集：`hf_hub_url`、`ms_hub_url`、`om_hub_url`
- 本地加载脚本：`script_url`

支持的文件类型包括 `json`、`jsonl`、`csv`、`parquet`、`arrow`。数据格式主要是 `alpaca`、`sharegpt`，代码里还支持 `openai` 风格 converter。

### 2.1 Alpaca SFT 格式

适合单轮或带 `history` 的指令微调：

```json
{
  "instruction": "用户指令",
  "input": "补充输入，可为空",
  "output": "模型回答",
  "system": "系统提示，可选",
  "history": [
    ["上一轮用户", "上一轮助手"]
  ]
}
```

`instruction` 和 `input` 会拼成 `instruction\ninput`，作为 user prompt；`output` 是 assistant response。注意：如果使用 `history`，历史里的 assistant 回复也会参与训练。

对应 `dataset_info.json`：

```json
"my_sft_data": {
  "file_name": "my_sft_data.jsonl",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

### 2.2 ShareGPT SFT 格式

适合多轮对话，也是我们电商客服数据更自然的格式：

```json
{
  "conversations": [
    {"from": "human", "value": "用户问题"},
    {"from": "gpt", "value": "客服回答"}
  ],
  "system": "系统提示，可选"
}
```

默认角色规则：

- `human` 和 `observation` 必须在奇数位置，也就是第 1、3、5... 条消息。
- `gpt` 和 `function_call` 必须在偶数位置，也就是第 2、4、6... 条消息。
- SFT 时，默认所有 `gpt` / `function_call` 消息都会作为学习目标。

对应 `dataset_info.json`：

```json
"my_sharegpt_sft": {
  "file_name": "my_sharegpt_sft.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system"
  }
}
```

如果数据字段是 OpenAI messages 风格，例如 `role/content`，可以用 `tags` 重映射：

```json
"my_messages_sft": {
  "file_name": "my_messages_sft.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

### 2.3 偏好数据格式：DPO / ORPO / SimPO / RM

偏好数据需要一个 prompt，加一好一坏两个回答。

Alpaca 偏好格式：

```json
{
  "instruction": "用户问题",
  "input": "",
  "chosen": "更好的回答",
  "rejected": "更差的回答"
}
```

ShareGPT 偏好格式：

```json
{
  "conversations": [
    {"from": "human", "value": "用户问题"}
  ],
  "chosen": {"from": "gpt", "value": "更好的回答"},
  "rejected": {"from": "gpt", "value": "更差的回答"}
}
```

`dataset_info.json` 必须设置 `"ranking": true`：

```json
"my_dpo_data": {
  "file_name": "my_dpo_data.jsonl",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### 2.4 KTO 数据格式

KTO 不要求成对的 chosen/rejected，而是对单条回答打一个 bool 标签：

```json
{
  "conversations": [
    {"from": "human", "value": "用户问题"},
    {"from": "gpt", "value": "模型回答"}
  ],
  "kto_tag": true
}
```

`true` 表示 desirable，`false` 表示 undesirable。对应注册：

```json
"my_kto_data": {
  "file_name": "my_kto_data.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "kto_tag": "kto_tag"
  }
}
```

## 3. 数据在代码里如何流动

数据流可以理解成四步：

1. `parser.py` 读取 `dataset_info.json`，把 dataset 名称解析成 `DatasetAttr`。
2. `loader.py` 根据 `DatasetAttr` 从本地文件或远程仓库加载数据。
3. `converter.py` 把 Alpaca / ShareGPT / OpenAI 格式统一成内部格式：
   - `_prompt`: user/history/tool prompt 消息列表
   - `_response`: assistant response，SFT 是 1 条，偏好数据是 2 条
   - `_system`: system prompt
   - `_tools`、`_images`、`_videos`、`_audios`: 工具和多模态字段
4. `loader.py` 按 `stage` 选择 processor：
   - `pt`: `PretrainDatasetProcessor`
   - `sft`: `SupervisedDatasetProcessor` 或 packing 版本
   - `rm`: `PairwiseDatasetProcessor`
   - `kto`: `FeedbackDatasetProcessor`
   - 其他如 `ppo`: `UnsupervisedDatasetProcessor`

一个容易踩的点：`run_dpo()` 里调用 `get_dataset(..., stage="rm")`，因为 DPO 和 RM 都消费 pairwise ranking 数据。也就是说 DPO 配置里虽然写 `stage: dpo`，但数据检查和 processor 走的是 RM/pairwise 那套逻辑。

## 4. 模板系统与 Qwen3

模板定义集中在 `src/llamafactory/data/template.py`。

一个模板主要定义：

- `format_user`: 用户消息如何包 prompt
- `format_assistant`: 助手消息如何结尾
- `format_system`: system prompt 如何插入
- `stop_words`: 额外停止符
- `replace_eos`: 是否替换 tokenizer 的 eos
- `template_class`: 普通 `Template` 或 `ReasoningTemplate`

Qwen3 相关模板在 `template.py` 约 1960 行附近：

- `qwen3`: 使用 `ReasoningTemplate`，会处理 `<think>...</think>`。
- `qwen3_nothink`: 普通模板，不自动插入 thinking 逻辑。

Qwen3 的消息格式大致是：

```text
<|im_start|>user
用户内容<|im_end|>
<|im_start|>assistant
助手内容<|im_end|>
```

对于我们的电商客服 SFT 数据，当前没有 CoT/思维链，推荐先使用：

```yaml
template: qwen3_nothink
```

原因是项目示例里的 `qwen3_lora_sft.yaml`、`qwen3_lora_dpo.yaml` 也都使用 `qwen3_nothink`。如果改用 `qwen3`，要额外理解 `enable_thinking`：

- `enable_thinking: true`：没有思维链时，会把空 thinking 加到 assistant response 中并计算 loss。
- `enable_thinking: false`：没有思维链时，会把空 thinking 加到 prompt 中，不计算这部分 loss。
- 训练和推理时要保持一致，否则输出风格容易偏。

## 5. LoRA 与 QLoRA 配置

LoRA 参数定义在 `FinetuningArguments` 里，主要字段：

```yaml
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
lora_target: all
```

含义：

- `lora_rank`: LoRA 低秩维度，默认 8。越大容量越强，也更吃显存。
- `lora_alpha`: 缩放因子，默认是 `lora_rank * 2`。
- `lora_dropout`: LoRA dropout，默认 0。
- `lora_target`: 注入哪些模块。`all` 表示所有 linear module。
- `additional_target`: 除 LoRA 层外，额外训练并保存的模块，例如扩 vocab 后的 embedding。
- `use_rslora`、`use_dora`、`pissa_init`: LoRA 变体。

真正创建 LoRA adapter 的地方是 `model/adapter.py`：

- 如果 `adapter_name_or_path` 存在，会先加载已有 adapter；可 merge 多个 adapter，也可 resume 最后一个 adapter。
- 如果是新训练，`lora_target: all` 会通过 `find_all_linear_modules()` 找所有线性层。
- 最终用 PEFT 的 `LoraConfig(task_type=TaskType.CAUSAL_LM, ...)` 创建 adapter，再 `get_peft_model()` 注入模型。

QLoRA 只是在 LoRA 训练基础上增加加载时量化：

```yaml
quantization_bit: 4
quantization_method: bnb
finetuning_type: lora
```

`model_utils/quantization.py` 中，`quantization_bit: 4` + `bnb` 会创建 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, ...)`。这就是典型 QLoRA。

合并 LoRA 时注意：`examples/merge_lora/qwen3_lora_sft.yaml` 明确提示不要在 merge 时使用 quantized model 或 `quantization_bit`。

## 6. 常用 YAML 配置结构

SFT LoRA 示例：

```yaml
### model
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
max_samples: 1000
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3-4b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

DPO LoRA 只需把 method 和 dataset 换成偏好数据：

```yaml
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid
dataset: my_dpo_data
template: qwen3_nothink
learning_rate: 5.0e-6
```

`pref_loss` 常见选择：

- `sigmoid`: 标准 DPO
- `orpo`: 不需要 reference model
- `simpo`: 不需要 reference model
- `hinge`、`ipo`、`kto_pair`: 也在 TRL/DPO loss 家族中使用

`FinetuningArguments.__post_init__()` 里有一条关键逻辑：

```python
use_ref_model = stage == "dpo" and pref_loss not in ["orpo", "simpo"]
```

所以标准 DPO 会用 reference model；ORPO/SimPO 不用。

## 7. 对齐算法实现

训练总入口是 `src/llamafactory/train/tuner.py`。它根据 `stage` 分发：

- `stage: sft` -> `train/sft/workflow.py`
- `stage: rm` -> `train/rm/workflow.py`
- `stage: dpo` -> `train/dpo/workflow.py`
- `stage: ppo` -> `train/ppo/workflow.py`
- `stage: kto` -> `train/kto/workflow.py`

### 7.1 SFT

关键文件：

- `train/sft/workflow.py`
- `train/sft/trainer.py`
- `data/processor/supervised.py`

SFT processor 会把每轮对话编码成 `source_ids + target_ids`：

- prompt/source 部分 label 是 `IGNORE_INDEX`，不参与 loss。
- assistant/target 部分 label 是原 token，参与 loss。
- 如果 `train_on_prompt: true`，prompt 也参与训练。
- 如果 `mask_history: true`，只训练最后一轮回答。

### 7.2 Reward Model

关键文件：

- `train/rm/workflow.py`
- `train/rm/trainer.py`
- `data/processor/pairwise.py`

Reward Model 使用 pairwise 数据。processor 会构造：

- `chosen_input_ids` / `chosen_labels`
- `rejected_input_ids` / `rejected_labels`

`PairwiseTrainer.compute_loss()` 从模型 value head 取 chosen/rejected 最后一个有效 token 的分数，用：

```text
-logsigmoid(chosen_score - rejected_score)
```

训练目标是让 chosen 分数高于 rejected。

### 7.3 DPO / ORPO / SimPO

关键文件：

- `train/dpo/workflow.py`
- `train/dpo/trainer.py`

DPO workflow 仍然使用 pairwise 数据。`CustomDPOTrainer` 继承 TRL 的 `DPOTrainer`，但重写了关键逻辑：

- `concatenated_forward()`: 一次 forward 同时算 chosen/rejected log probability。
- `compute_reference_log_probs()`: 标准 DPO 下计算 reference model 的 log probability；LoRA 情况下也可以临时 disable adapter，把 base model 当 reference。
- `compute_preference_loss()`: 根据 `pref_loss` 选择损失。
- `get_batch_loss_metrics()`: 记录 `rewards/chosen`、`rewards/rejected`、`rewards/accuracies`、`rewards/margins` 等指标。

不同偏好算法的直观区别：

- 标准 DPO：比较 policy 相对 reference 的 chosen/rejected 概率差。
- ORPO：把 SFT loss 和 odds ratio loss 合在一起，不依赖 reference model。
- SimPO：只比较 chosen/rejected 的平均 log probability，并加入 reward margin `simpo_gamma`，也不依赖 reference model。
- BCO：通过 `pref_bco_weight` 作为附加项混入 DPO loss。
- `pref_ftx`: 可以额外混入 chosen response 的 SFT loss，缓解偏好训练后语言质量漂移。

### 7.4 PPO

关键文件：

- `train/ppo/workflow.py`
- `train/ppo/trainer.py`
- `train/trainer_utils.py`

PPO 需要 reward model：

```yaml
stage: ppo
reward_model: path/to/reward_model
reward_model_type: lora  # 或 full/api
```

`run_ppo()` 会：

1. 加载带 value head 的 policy model。
2. 创建 reference model。
3. 创建 reward model。
4. 生成 response。
5. 用 reward model 打分。
6. 调用 TRL PPO step 更新 policy。

PPO 训练复杂度和不稳定性都更高。对于我们的项目，优先级建议是：先 SFT，再 DPO；PPO 可以作为了解项，不作为第一版主线。

### 7.5 KTO

关键文件：

- `train/kto/workflow.py`
- `train/kto/trainer.py`
- `data/processor/feedback.py`

KTO 数据不是 pairwise，而是单条回答 + `kto_tag`。processor 会把 `true/false` 转成 desirable/undesirable，并额外构造 KL 用的样本。`CustomKTOTrainer` 继承 TRL 的 `KTOTrainer`，主要设置：

- `beta = pref_beta`
- `desirable_weight = kto_chosen_weight`
- `undesirable_weight = kto_rejected_weight`
- `ftx_gamma = pref_ftx`

适用场景：我们只有“这条回答好/不好”的反馈，而不是成对偏好。

## 8. 我们电商客服项目如何接入

当前我们的 SFT 数据位于：

```text
project/ProjectSet/ecommerce_assistant/data/processed/sft_v1/train.jsonl
project/ProjectSet/ecommerce_assistant/data/processed/sft_v1/eval.jsonl
```

数据是 OpenAI messages 风格：

```json
{
  "messages": [
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "客服回答"}
  ],
  "meta": {}
}
```

如果放进 LlamaFactory，推荐在 LlamaFactory 的 `data/dataset_info.json` 里加类似条目，或者复制一份专用 `dataset_info.json` 到我们的数据目录：

```json
"ecommerce_sft_v1_train": {
  "file_name": "sft_v1/train.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

然后训练配置里写：

```yaml
dataset_dir: project/ProjectSet/ecommerce_assistant/data/processed
dataset: ecommerce_sft_v1_train
template: qwen3_nothink
stage: sft
finetuning_type: lora
```

如果后面构造 DPO 数据，建议格式：

```json
{
  "messages": [
    {"role": "user", "content": "用户问题"}
  ],
  "chosen": {"role": "assistant", "content": "更好的客服回答"},
  "rejected": {"role": "assistant", "content": "更差的客服回答"},
  "meta": {}
}
```

对应注册：

```json
"ecommerce_dpo_v1_train": {
  "file_name": "dpo_v1/train.jsonl",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "messages",
    "chosen": "chosen",
    "rejected": "rejected"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

## 9. 对我们项目的建议

第一阶段建议只学并落地这条最短闭环：

1. 用当前 `sft_v1/train.jsonl` 跑 Qwen3 LoRA SFT。
2. 使用 `template: qwen3_nothink`，避免 thinking 模式干扰客服口吻。
3. 用小 `max_samples` 先 smoke test，确认数据字段、模板、loss 都正常。
4. 再跑完整 6000 条 SFT。
5. 人工抽样评测后，再构造 DPO 数据。
6. DPO 先用 `pref_loss: sigmoid` 标准 DPO；显存紧张或想简化 reference model 时，再试 `orpo` 或 `simpo`。

本项目不建议一开始上 PPO。PPO 需要 reward model 或 reward API，调参成本明显更高；面试项目里 SFT + DPO 已经能覆盖“监督微调 + 偏好对齐”的核心能力展示。

