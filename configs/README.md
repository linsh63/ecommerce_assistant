# Config Layout

当前配置目录只保留主线实验和后续会继续用到的配置。

## Main SFT

- `sft_lora_intent_v1_classified_augmented.yaml`
  - v1 SFT base。
- `sft_lora_intent_v1_answer_patch.yaml`
  - v1 base + 小规模答非所问补丁，用于人工复评。
- `sft_lora_plan_a_multiturn_continue.yaml`
  - planA：基于 v1 adapter 继续训练多轮补丁。
- `sft_lora_plan_c_single_plus_multiturn_from_base.yaml`
  - planC：单轮 + 多轮混合数据，从 base 重新训练。

## Ablation

- `lora_ablation_baseline/`
  - baseline 数据集上的 LoRA 消融实验配置。

## Next Stage

- `dpo.yaml`
  - 后续 DPO/RLHF 阶段的入口配置。
- `sft_eval.yaml`
  - 通用推理评测配置，保留作辅助工具。

旧探索配置已移出项目目录，归档在：

`project/ProjectSet/_local_data_archive/ecommerce_assistant_cleanup_20260527_144914/`
