# ecommerce-jddc-sft-dpo 服务器跑通手册

更新时间：2026-05-24

本文档用于在服务器上跑通开源项目 `rick0673/ecommerce-jddc-sft-dpo` 的 SFT + DPO 全流程。目标不是先追求最优效果，而是快速获得正反馈：准备数据、跑通 QLoRA SFT、构建偏好数据、跑通 DPO、完成一次 judge 评估。

已知条件：

- 服务器已有 Qwen3-8B 本地模型：`/data2/songxinshuai/hf_models/Qwen3-8B`
- 服务器已有 conda 环境能力。
- 服务器 GPU 可按实际空闲情况选择，例如 `CUDA_VISIBLE_DEVICES=6` 或 `CUDA_VISIBLE_DEVICES=6,7`。

注意：该开源项目 README 默认写的是 `Qwen/Qwen3-7B` 和 `Qwen/Qwen3-14B`。本手册统一改成已经下载好的 Qwen3-8B，避免再次下载大模型。

## 0. 项目特点和注意事项

源码位置：

```text
project/ProjectSet/ref_project/LLM/ecommerce-jddc-sft-dpo/
```

上游仓库：

```text
https://github.com/rick0673/ecommerce-jddc-sft-dpo.git
```

核心脚本：

```text
ecommerce_jddc_sft_dpo.py
```

该脚本支持以下命令：

- `prepare-sft`：把 JDDC 对话转成 SFT 数据。
- `train-sft`：使用 TRL `SFTTrainer` 做 LoRA/QLoRA SFT。
- `generate-preferences`：用强模型生成 DPO chosen，默认用原始客服答案作为 rejected。
- `import-preferences`：导入人工筛选后的偏好数据。
- `train-dpo`：使用 TRL `DPOTrainer` 做 DPO。
- `evaluate`：用 judge 模型输出质检 JSONL。

重要注意：

- 这个脚本内部 `load_model()` 使用 `device_map="auto"`，不建议一开始用 `torchrun`。先用普通 `python` 单进程跑通。
- 如果单卡显存够，优先 `CUDA_VISIBLE_DEVICES=6`；如果偏好生成或评估同时加载多个模型导致 OOM，再用 `CUDA_VISIBLE_DEVICES=6,7` 让 `device_map=auto` 自动分配。
- 训练阶段建议加 `--use_4bit`，否则 8B FP16 训练显存压力较大。
- `generate-preferences` 和 `evaluate` 没有 `--use_4bit` 参数，推理默认按 FP16/auto 加载；OOM 时用两张卡可见。

## 1. 获取项目代码

推荐在服务器上单独放一个目录：

```bash
cd /data2/songxinshuai/linsihan
git clone https://github.com/rick0673/ecommerce-jddc-sft-dpo.git
cd ecommerce-jddc-sft-dpo
```

如果服务器无法访问 GitHub，可以从本地打包上传：

```bash
cd /Users/linsh/Documents/Recommendation/project/ProjectSet/ref_project/LLM
tar -czf ecommerce-jddc-sft-dpo.tar.gz ecommerce-jddc-sft-dpo
```

上传到服务器后解压：

```bash
cd /data2/songxinshuai/linsihan
tar -xzf ecommerce-jddc-sft-dpo.tar.gz
cd ecommerce-jddc-sft-dpo
```

## 2. 准备 conda 环境

如果已有可用的 `ecommerce-sft` 环境，也可以复用。但为了避免和当前项目依赖互相影响，更推荐新建环境：

```bash
conda create -n jddc-sft-dpo python=3.10 -y
conda activate jddc-sft-dpo
python -m pip install -U pip
```

安装 PyTorch。服务器 CUDA Driver 显示 CUDA 12.4 时，可以优先安装 cu124 轮子：

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

安装项目依赖：

```bash
pip install -U transformers datasets accelerate peft bitsandbytes sentencepiece safetensors protobuf
pip install -e .
```

说明：

- `pip install -e .` 会使用该仓库自带的本地 `trl/` 源码。
- `bitsandbytes` 是 `--use_4bit` 必需依赖，但该项目 `pyproject.toml` 没有显式列出，所以需要手动安装。

检查环境：

```bash
python - <<'PY'
import torch
import transformers
import datasets
import peft
import trl

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("trl local:", trl.__file__)
PY
```

确认 `trl local` 指向当前仓库下的 `trl/__init__.py` 或 editable install 路径。

## 3. 设置缓存和基础变量

建议把 Hugging Face 缓存放在 `/data2`，不要写到 home：

```bash
cd /data2/songxinshuai/linsihan/ecommerce-jddc-sft-dpo
conda activate jddc-sft-dpo

export BASE_MODEL=/data2/songxinshuai/hf_models/Qwen3-8B
export HF_HOME=/data2/songxinshuai/hf_cache
export HF_DATASETS_CACHE=/data2/songxinshuai/hf_cache/datasets
export TRANSFORMERS_CACHE=/data2/songxinshuai/hf_cache/transformers
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
```

如果服务器访问 Hugging Face 不稳定，可以临时使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

注意：模型已经在本地，`BASE_MODEL` 不需要走 Hugging Face 下载；只有在线 JDDC 数据集需要联网。

## 4. 最小 smoke test

先用仓库自带的 `ecommerce_jddc_smoke.jsonl` 跑通数据处理和极小步数训练。

### 4.1 构建 smoke SFT 数据

```bash
python ecommerce_jddc_sft_dpo.py prepare-sft \
  --local_json ecommerce_jddc_smoke.jsonl \
  --split train \
  --dialogue_column dialogue \
  --role_field role \
  --text_field text \
  --max_samples 20 \
  --output_dir data/ecommerce_jddc_sft_smoke
```

成功时会看到类似：

```text
Saved N train and M eval SFT rows to data/ecommerce_jddc_sft_smoke.
```

### 4.2 极小步数 SFT

先用一张卡：

```bash
mkdir -p runs/qwen3_8b_jddc_sft_smoke

CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py train-sft \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_dir data/ecommerce_jddc_sft_smoke \
  --output_dir runs/qwen3_8b_jddc_sft_smoke \
  --lora_r 4 \
  --lora_alpha 8 \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_length 1024 \
  --dtype float16 \
  --use_4bit \
  2>&1 | tee runs/qwen3_8b_jddc_sft_smoke/train.log
```

这个 smoke test 只验证环境、模型加载、TRL Trainer 和 LoRA 注入是否正常。跑通后再做正式数据。

## 5. 构建真实 SFT 数据

### 5.1 在线读取 JDDC 数据

默认使用 `FourHan/jddc2.0`：

```bash
python ecommerce_jddc_sft_dpo.py prepare-sft \
  --dataset_name FourHan/jddc2.0 \
  --split train \
  --dialogue_column dialogue \
  --role_field role \
  --text_field text \
  --max_samples 3000 \
  --output_dir data/ecommerce_jddc_sft
```

如果字段名不匹配，先检查一条数据：

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("FourHan/jddc2.0", split="train[:1]")
print(ds.column_names)
print(ds[0])
PY
```

然后根据实际字段调整：

- `--dialogue_column`
- `--role_field`
- `--text_field`

### 5.2 无法联网时的替代方案

如果服务器无法下载 `FourHan/jddc2.0`，先继续用 smoke 数据跑完全流程，确认 SFT+DPO 命令链路没问题。之后再考虑把 JDDC 数据集提前下载到本地，或使用自己的 JSONL。

本地 JSONL 格式要求：

```json
{"dialogue":[{"role":"用户","text":"我的订单什么时候发货？"},{"role":"客服","text":"您好，请提供订单号，我帮您查询发货进度。"}]}
```

命令：

```bash
python ecommerce_jddc_sft_dpo.py prepare-sft \
  --local_json path/to/jddc.jsonl \
  --split train \
  --dialogue_column dialogue \
  --role_field role \
  --text_field text \
  --max_samples 3000 \
  --output_dir data/ecommerce_jddc_sft
```

## 6. 正式 QLoRA SFT

建议先跑 100 steps 获得正反馈，再决定是否加到 500 或更多。

```bash
mkdir -p runs/qwen3_8b_ecommerce_sft_r16

CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py train-sft \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_dir data/ecommerce_jddc_sft \
  --output_dir runs/qwen3_8b_ecommerce_sft_r16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 2048 \
  --learning_rate 2e-4 \
  --dtype float16 \
  --use_4bit \
  2>&1 | tee runs/qwen3_8b_ecommerce_sft_r16/train.log
```

如果单卡 OOM：

```bash
CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py train-sft \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_dir data/ecommerce_jddc_sft \
  --output_dir runs/qwen3_8b_ecommerce_sft_r16_len1024 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-4 \
  --dtype float16 \
  --use_4bit
```

不建议一开始使用：

```bash
torchrun --nproc_per_node=2 ...
```

原因是该脚本内部已经用 `device_map="auto"` 加载模型，和 DDP/torchrun 容易冲突。先用普通 `python` 跑通。

## 7. 构建 DPO 偏好数据

这个项目的默认偏好构造方式：

- `chosen`：由强模型生成。
- `rejected`：默认使用原始客服答案。

由于我们已经下载的是 Qwen3-8B，没有 Qwen3-14B，所以先用 Qwen3-8B 作为 strong model 跑通流程。

```bash
CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py generate-preferences \
  --sft_data_dir data/ecommerce_jddc_sft \
  --strong_model_name_or_path "$BASE_MODEL" \
  --max_samples 200 \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --dtype float16 \
  --output_dir data/ecommerce_jddc_dpo \
  --review_file data/ecommerce_jddc_preference_review.jsonl
```

如果生成偏好时 OOM，改用两张卡让 `device_map=auto` 分配模型：

```bash
CUDA_VISIBLE_DEVICES=6,7 python ecommerce_jddc_sft_dpo.py generate-preferences \
  --sft_data_dir data/ecommerce_jddc_sft \
  --strong_model_name_or_path "$BASE_MODEL" \
  --max_samples 200 \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --dtype float16 \
  --output_dir data/ecommerce_jddc_dpo \
  --review_file data/ecommerce_jddc_preference_review.jsonl
```

说明：

- `data/ecommerce_jddc_preference_review.jsonl` 会带 `keep=true` 字段，便于人工筛选。
- 如果只是跑通流程，可以先不人工改，直接导入。
- 如果要更可信，抽样检查 `chosen` 是否真的优于 `rejected`。

导入偏好数据：

```bash
python ecommerce_jddc_sft_dpo.py import-preferences \
  --screened_json data/ecommerce_jddc_preference_review.jsonl \
  --output_dir data/ecommerce_jddc_dpo_screened
```

## 8. 跑通 DPO

从 SFT adapter 继续做 DPO：

```bash
mkdir -p runs/qwen3_8b_ecommerce_dpo_r16

CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py train-dpo \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_name_or_path runs/qwen3_8b_ecommerce_sft_r16 \
  --dataset_dir data/ecommerce_jddc_dpo_screened \
  --output_dir runs/qwen3_8b_ecommerce_dpo_r16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_steps 50 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 5e-5 \
  --dtype float16 \
  --use_4bit \
  2>&1 | tee runs/qwen3_8b_ecommerce_dpo_r16/train.log
```

说明：

- `--adapter_name_or_path` 指向 SFT 输出目录。
- 这里先用 `max_length=1024`，降低 DPO 显存压力。
- `--lora_r` 和 `--lora_alpha` 在加载已有 adapter 时基本不再决定新建 LoRA，但保留参数不影响流程。
- DPO 比 SFT 更容易 OOM，如果报显存不足，先降 `--max_length`，再减少 `--max_steps` 或改用两卡可见。

## 9. 评估

该项目评估会同时加载：

- base model + adapter，即待评估模型。
- judge model。

如果都用 Qwen3-8B，单卡可能吃紧，建议两张卡可见：

```bash
mkdir -p runs/eval

CUDA_VISIBLE_DEVICES=6,7 python ecommerce_jddc_sft_dpo.py evaluate \
  --dataset_dir data/ecommerce_jddc_sft \
  --base_model_name_or_path "$BASE_MODEL" \
  --model_name_or_path runs/qwen3_8b_ecommerce_dpo_r16 \
  --judge_model_name_or_path "$BASE_MODEL" \
  --max_samples 50 \
  --max_new_tokens 256 \
  --dtype float16 \
  --output_file runs/eval/ecommerce_eval_qwen3_8b_judge.jsonl \
  2>&1 | tee runs/eval/evaluate.log
```

输出文件：

```text
runs/eval/ecommerce_eval_qwen3_8b_judge.jsonl
```

每行包含：

- `prompt`
- `answer`
- `judge`

注意：这里用同一个 Qwen3-8B 当 judge，只适合流程验证，不适合作为严肃最终指标。正式评估建议换更强模型，或引入人工抽检。

## 10. 推荐的完整 tmux 流程

开 tmux：

```bash
tmux new -s jddc
```

进入环境：

```bash
cd /data2/songxinshuai/linsihan/ecommerce-jddc-sft-dpo
conda activate jddc-sft-dpo
export BASE_MODEL=/data2/songxinshuai/hf_models/Qwen3-8B
export HF_HOME=/data2/songxinshuai/hf_cache
export HF_DATASETS_CACHE=/data2/songxinshuai/hf_cache/datasets
export TRANSFORMERS_CACHE=/data2/songxinshuai/hf_cache/transformers
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
```

按顺序执行：

```bash
# 1. SFT 数据
python ecommerce_jddc_sft_dpo.py prepare-sft \
  --dataset_name FourHan/jddc2.0 \
  --split train \
  --dialogue_column dialogue \
  --role_field role \
  --text_field text \
  --max_samples 3000 \
  --output_dir data/ecommerce_jddc_sft

# 2. SFT
mkdir -p runs/qwen3_8b_ecommerce_sft_r16
CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py train-sft \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_dir data/ecommerce_jddc_sft \
  --output_dir runs/qwen3_8b_ecommerce_sft_r16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 2048 \
  --learning_rate 2e-4 \
  --dtype float16 \
  --use_4bit \
  2>&1 | tee runs/qwen3_8b_ecommerce_sft_r16/train.log

# 3. 偏好数据
CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py generate-preferences \
  --sft_data_dir data/ecommerce_jddc_sft \
  --strong_model_name_or_path "$BASE_MODEL" \
  --max_samples 200 \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --dtype float16 \
  --output_dir data/ecommerce_jddc_dpo \
  --review_file data/ecommerce_jddc_preference_review.jsonl

# 4. 导入偏好数据
python ecommerce_jddc_sft_dpo.py import-preferences \
  --screened_json data/ecommerce_jddc_preference_review.jsonl \
  --output_dir data/ecommerce_jddc_dpo_screened

# 5. DPO
mkdir -p runs/qwen3_8b_ecommerce_dpo_r16
CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py train-dpo \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_name_or_path runs/qwen3_8b_ecommerce_sft_r16 \
  --dataset_dir data/ecommerce_jddc_dpo_screened \
  --output_dir runs/qwen3_8b_ecommerce_dpo_r16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_steps 50 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 5e-5 \
  --dtype float16 \
  --use_4bit \
  2>&1 | tee runs/qwen3_8b_ecommerce_dpo_r16/train.log

# 6. 评估
mkdir -p runs/eval
CUDA_VISIBLE_DEVICES=6,7 python ecommerce_jddc_sft_dpo.py evaluate \
  --dataset_dir data/ecommerce_jddc_sft \
  --base_model_name_or_path "$BASE_MODEL" \
  --model_name_or_path runs/qwen3_8b_ecommerce_dpo_r16 \
  --judge_model_name_or_path "$BASE_MODEL" \
  --max_samples 50 \
  --max_new_tokens 256 \
  --dtype float16 \
  --output_file runs/eval/ecommerce_eval_qwen3_8b_judge.jsonl \
  2>&1 | tee runs/eval/evaluate.log
```

## 11. 跑完后保存结果

这个开源项目 `.gitignore` 会忽略 `data/` 和 `runs/`，所以如果要回传结果，建议单独打包：

```bash
cd /data2/songxinshuai/linsihan/ecommerce-jddc-sft-dpo
tar -czf jddc_sft_dpo_results_$(date +%Y%m%d_%H%M).tar.gz \
  data/ecommerce_jddc_sft \
  data/ecommerce_jddc_dpo_screened \
  data/ecommerce_jddc_preference_review.jsonl \
  runs/qwen3_8b_ecommerce_sft_r16 \
  runs/qwen3_8b_ecommerce_dpo_r16 \
  runs/eval
```

如果只想回传轻量结果，不传 adapter 权重：

```bash
mkdir -p result_summary
cp runs/qwen3_8b_ecommerce_sft_r16/train.log result_summary/sft_train.log
cp runs/qwen3_8b_ecommerce_dpo_r16/train.log result_summary/dpo_train.log
cp runs/eval/evaluate.log result_summary/evaluate.log
cp runs/eval/ecommerce_eval_qwen3_8b_judge.jsonl result_summary/
cp data/ecommerce_jddc_preference_review.jsonl result_summary/preference_review_sample.jsonl
tar -czf jddc_sft_dpo_summary_$(date +%Y%m%d_%H%M).tar.gz result_summary
```

## 12. 常见问题

### 12.1 `ModuleNotFoundError: No module named 'trl'`

在项目根目录执行：

```bash
pip install -e .
```

或确认当前目录下有 `trl/`，并从项目根目录启动脚本。

### 12.2 `bitsandbytes` 相关错误

确认安装：

```bash
python - <<'PY'
import bitsandbytes as bnb
print(bnb.__version__)
PY
```

如果导入失败：

```bash
pip install -U bitsandbytes
```

### 12.3 Hugging Face 数据集下载失败

先设置缓存和镜像：

```bash
export HF_HOME=/data2/songxinshuai/hf_cache
export HF_DATASETS_CACHE=/data2/songxinshuai/hf_cache/datasets
export HF_ENDPOINT=https://hf-mirror.com
```

如果仍失败，先使用 `ecommerce_jddc_smoke.jsonl` 跑通流程。

### 12.4 SFT 或 DPO OOM

优先按顺序处理：

1. 确认加了 `--use_4bit`。
2. 把 `--max_length 2048` 改成 `1024`。
3. 保持 `--per_device_train_batch_size 1`。
4. 降低 `--max_steps` 先验证流程。
5. 用 `CUDA_VISIBLE_DEVICES=6,7` 单进程可见两张卡，让 `device_map=auto` 分配。

### 12.5 不要一开始使用 torchrun

该项目脚本内部用 `device_map="auto"`。`torchrun` 多进程训练很容易和自动模型切分冲突。先用：

```bash
CUDA_VISIBLE_DEVICES=6 python ecommerce_jddc_sft_dpo.py train-sft ...
```

而不是：

```bash
torchrun --nproc_per_node=2 ecommerce_jddc_sft_dpo.py train-sft ...
```

## 13. 跑通验收标准

最小验收：

- `data/ecommerce_jddc_sft/` 存在，能 `load_from_disk`。
- `runs/qwen3_8b_ecommerce_sft_r16/` 存在 LoRA adapter。
- `data/ecommerce_jddc_dpo_screened/` 存在 DPO 数据。
- `runs/qwen3_8b_ecommerce_dpo_r16/` 存在 DPO 后 adapter。
- `runs/eval/ecommerce_eval_qwen3_8b_judge.jsonl` 有至少 20 条 judge 输出。

学习验收：

- 能解释 SFT 数据里的 `messages/prompt/answer`。
- 能解释 DPO 数据里的 `prompt/chosen/rejected`。
- 能解释 LoRA adapter 如何从 SFT 继续用于 DPO。
- 能指出本流程中 Qwen3-8B judge 只是流程验证，不是严格评测。

