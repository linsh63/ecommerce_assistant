# LoRA 消融实验对比（JDDC Rebuild V2 增广数据集）

## 资源与效果

| experiment | variable | rank | target | precision | train_loss | eval_loss | ppl | runtime(min) | mem/gpu(MiB) | mem_total(MiB) | adapter(MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rank4_qv_qlora | rank | 4 | q_proj+v_proj | QLoRA-4bit | 1.6181 | 1.4574 | 4.2900 | 87.0647 | 14598 | 29104 |  |
| rank16_qv_qlora | rank | 16 | q_proj+v_proj | QLoRA-4bit | 1.5181 | 1.3912 | 4.0200 | 125.4209 | 14636 | 29228 |  |
| rank64_qv_qlora | rank | 64 | q_proj+v_proj | QLoRA-4bit | 1.3841 | 1.3219 | 3.7500 | 80.9192 | 15150 | 30300 |  |
| rank16_all_linear_qlora | target_modules | 16 | all_linear | QLoRA-4bit | 1.2507 | 1.2581 | 3.5200 | 105.1185 | 15348 | 30696 |  |
| rank16_qv_lora_fp16 | quantization | 16 | q_proj+v_proj | LoRA-FP16 | 1.5169 | 1.3888 | 4.0100 | 47.3735 | 21872 | 43740 |  |

## 评测指标（绝对评分 LLM judge，严格标准：需求>=4 且 准确>=4 通过）

| experiment | 通过率 | 解决率 | 准确率 | 需求 | 准确 | 帮助 | 语气 | 风险 | 历史利用 | 主要失败标签 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank4_qv_qlora | 4400.00% | 7930.00% | 8800.00% | 3.4200 | 3.7500 | 3.3900 | 4.0700 | 4.9400 | 3.4300 | too_generic:38, history_ignored:21, missing_action:18 |
| rank16_qv_qlora | 5070.00% | 8400.00% | 9130.00% | 3.5400 | 3.8500 | 3.5000 | 4.1000 | 4.9500 | 3.5000 | too_generic:39, history_ignored:18, missing_action:13 |
| rank64_qv_qlora | 5070.00% | 8730.00% | 9470.00% | 3.5600 | 3.9100 | 3.5500 | 4.1500 | 4.9500 | 3.6400 | too_generic:40, history_ignored:15, missing_action:14 |
| rank16_all_linear_qlora | 4930.00% | 8530.00% | 9200.00% | 3.5300 | 3.9100 | 3.4700 | 4.1500 | 4.9700 | 3.5300 | too_generic:48, missing_action:14, history_ignored:11 |
| rank16_qv_lora_fp16 | 4930.00% | 8070.00% | 8930.00% | 3.5000 | 3.8700 | 3.4900 | 4.0700 | 4.9600 | 3.5600 | too_generic:38, history_ignored:17, missing_action:15 |
