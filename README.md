# 电商客服大模型微调与偏好对齐

基于 Qwen3-8B 完成 SFT + DPO 全流程，面向退换货、物流、商品咨询等 12 类电商客服场景。最终 LLM judge 通过率 81.3%，用户偏好胜率 56.1%。

## 核心结果

| 模型 | 通过率 | 自动解决率 | 偏好胜率 |
|------|:--:|:--:|:--:|
| 裸 Qwen3-8B | 53.3% | 78.7% | — |
| SFT v1 | 72.7% | 95.3% | — |
| **DPO β=0.05** | **81.3%** | **95.3%** | **56.1%** |

详细实验记录见 `docs/brain/interview_talking_points.md`。

## 代码结构

```
src/ecommerce_cs_sft_dpo/
├── utils.py            # 共享工具（YAML解析、随机种子）
├── sft_training.py     # SFT 训练（4个类）
├── dpo_training.py     # DPO 训练（3个类）
├── ppo_training.py     # PPO RLHF（自实现）
├── grpo_training.py    # GRPO（DeepSeek-R1 同款，DDP）
scripts/
├── train/             # 训练入口
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── train_ppo.py
│   ├── train_grpo.py
│   └── run_baseline_lora_ablation.py
├── data/              # 数据构造
│   ├── build_jddc_rebuild_v2_dataset.py
│   ├── rewrite_jddc_rebuild_v2_with_deepseek.py
│   ├── shrink_eval_set.py
│   ├── finalize_jddc_rebuild_v2_dataset.py
│   ├── augment_undersampled_categories.py
│   ├── merge_augmented_to_train.py
│   ├── build_dpo_preference_pairs.py
│   ├── augment_dpo_pairs.py
│   ├── generate_annotation_sheet.py
│   └── sync_annotation_sheet.py
└── eval/              # 评测
    ├── judge_jddc_rebuild_v2_with_deepseek.py
    └── generate_dpo_responses.py
configs/
├── sft_lora_jddc_rebuild_v2.yaml
└── dpo_jddc_rebuild_v2.yaml
data/processed/jddc_rebuild_v2/
├── 05_final/                    # 训练集 + 评测集 + 标注
└── dpo_preference_pairs_clean.jsonl  # DPO 偏好数据
```

## 快速开始

```bash
# SFT 训练
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29500 \
  scripts/train/train_sft.py --config configs/sft_lora_jddc_rebuild_v2.yaml \
  --base_model /path/to/Qwen3-8B

# DPO 训练
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29500 \
  scripts/train/train_dpo.py --config configs/dpo_jddc_rebuild_v2.yaml \
  --base_model /path/to/Qwen3-8B \
  --sft_adapter_path outputs/sft_lora_jddc_rebuild_v2

# 评测
python scripts/eval/generate_dpo_responses.py \
  --base_model /path/to/Qwen3-8B --adapter_path outputs/dpo_jddc_rebuild_v2 \
  --eval_file data/processed/jddc_rebuild_v2/05_final/eval_test.jsonl \
  --output_file outputs/eval_generations.jsonl

python scripts/eval/judge_jddc_rebuild_v2_with_deepseek.py \
  --generations outputs/eval_generations.jsonl --output-dir outputs/judge_results
```
