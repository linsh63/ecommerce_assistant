"""Run baseline-dataset LoRA ablations, generation, LLM judge, and summary table."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "lora_ablation_baseline"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "docs" / "results" / "lora_ablation_jddc_rebuild_v2"
DEFAULT_TRAIN_FILE = "data/processed/jddc_rebuild_v2/05_final/sft_train.jsonl"
DEFAULT_EVAL_FILE = "data/processed/jddc_rebuild_v2/05_final/sft_dev.jsonl"
DEFAULT_TEST_FILE = "data/processed/jddc_rebuild_v2/05_final/eval_test.jsonl"
DEFAULT_ANNOTATION_FILE = "data/processed/jddc_rebuild_v2/05_final/eval_test_annotation.json"
DEFAULT_SYSTEM = (
    "你是电商平台客服助手。请先判断用户咨询类型，再给出直接、具体、礼貌、可执行的中文回复。"
    "分类只能从以下12类中选择：物流-查件催发、物流-配送调整、售后-退换维修、售后-质量异常、"
    "退款-取消价保、商品咨询-参数、商品咨询-使用方法、购买决策-推荐对比、价格活动-优惠赠品、"
    "发票-资质服务、投诉-安抚升级、闲聊-礼貌收尾。输出格式必须严格为两行：第一行【分类：<上述类别之一>】，"
    "第二行【回复：<客服回复>】。不要空泛套模板，不要输出方括号占位符。"
)

ARTIFACTS = (
    "train.log",
    "train_results.json",
    "eval_results.json",
    "trainer_state.json",
    "resolved_sft_config.json",
    "dataset_split_report.json",
    "preprocessed_samples.jsonl",
    "loss_history.json",
    "loss_history.csv",
    "loss_curve.png",
    "eval_perplexity_curve.png",
    "learning_rate_curve.png",
    "grad_norm_curve.png",
    "training_diagnostics.png",
    "post_train_test_generations.jsonl",
    "post_train_test_generations.md",
    "runtime_metrics.json",
)


# 功能：保存单个 LoRA 消融实验的变量配置。
@dataclass
class AblationExperiment:
    name: str
    group: str
    description: str
    rank: int
    alpha: int
    target_modules: list[str]
    load_in_4bit: bool
    torch_dtype: str = "float16"
    optim: str = "paged_adamw_8bit"

    # 功能：返回训练输出目录。
    @property
    def output_dir(self) -> str:
        return f"outputs/lora_ablation_jddc_rebuild_v2/{self.name}"

    # 功能：返回文档结果目录。
    @property
    def result_dir(self) -> str:
        return f"docs/results/lora_ablation_baseline/{self.name}"

    # 功能：返回人类可读的精度/量化标签。
    @property
    def precision_label(self) -> str:
        return "QLoRA-4bit" if self.load_in_4bit else "LoRA-FP16"

    # 功能：返回 target_modules 的压缩展示。
    @property
    def target_label(self) -> str:
        if self.target_modules == ["q_proj", "v_proj"]:
            return "q_proj+v_proj"
        return "all_linear"


EXPERIMENTS = [
    AblationExperiment(
        name="rank4_qv_qlora",
        group="rank",
        description="rank=4, q_proj+v_proj, QLoRA 4bit",
        rank=4,
        alpha=8,
        target_modules=["q_proj", "v_proj"],
        load_in_4bit=True,
    ),
    AblationExperiment(
        name="rank16_qv_qlora",
        group="rank",
        description="rank=16, q_proj+v_proj, QLoRA 4bit; baseline LoRA setting",
        rank=16,
        alpha=32,
        target_modules=["q_proj", "v_proj"],
        load_in_4bit=True,
    ),
    AblationExperiment(
        name="rank64_qv_qlora",
        group="rank",
        description="rank=64, q_proj+v_proj, QLoRA 4bit",
        rank=64,
        alpha=128,
        target_modules=["q_proj", "v_proj"],
        load_in_4bit=True,
    ),
    AblationExperiment(
        name="rank16_all_linear_qlora",
        group="target_modules",
        description="rank=16, all linear projection modules, QLoRA 4bit",
        rank=16,
        alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        load_in_4bit=True,
    ),
    AblationExperiment(
        name="rank16_qv_lora_fp16",
        group="quantization",
        description="rank=16, q_proj+v_proj, FP16 LoRA without 4bit quantization",
        rank=16,
        alpha=32,
        target_modules=["q_proj", "v_proj"],
        load_in_4bit=False,
        optim="adamw_torch",
    ),
]


# 功能：读取 JSON 文件，文件不存在时返回默认值。
def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# 功能：写出缩进 JSON 文件。
def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 功能：读取 JSONL 文件。
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


# 功能：从生成输出中拆出分类和回复，兼容少量缺字段的结果。
def split_generated_response(row: dict[str, Any]) -> dict[str, str | None]:
    raw = str(row.get("generated_response") or "").strip()
    intent = row.get("generated_intent")
    reply = row.get("generated_reply")
    if not intent:
        match = re.search(r"分类[:：]\s*(.+?)(?:\n|$)", raw)
        intent = match.group(1).strip() if match else None
    if not reply:
        match = re.search(r"回复[:：]\s*(.+)", raw, flags=re.DOTALL)
        if match:
            reply = match.group(1).strip()
    return {
        "intent": str(intent).strip() if intent else None,
        "reply": str(reply).strip() if reply else None,
        "raw": raw,
    }


# 功能：把 history pair 或 role/content 列表统一成 role/content 列表。
def flatten_history(history: Any) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    if not isinstance(history, list):
        return flattened
    for item in history:
        if isinstance(item, dict):
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role and content:
                flattened.append({"role": role, "content": content})
        elif isinstance(item, list) and len(item) == 2:
            user_text = str(item[0] or "").strip()
            assistant_text = str(item[1] or "").strip()
            if user_text:
                flattened.append({"role": "user", "content": user_text})
            if assistant_text:
                flattened.append({"role": "assistant", "content": assistant_text})
    return flattened


# 功能：基于同一评测集生成多模型横向比较 JSON。
def build_ablation_comparison(experiments: list[AblationExperiment], result_root: Path) -> list[dict[str, Any]]:
    by_experiment: dict[str, dict[str, dict[str, Any]]] = {}
    for experiment in experiments:
        path = result_root / experiment.name / "post_train_test_generations.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing generation file for {experiment.name}: {path}")
        by_experiment[experiment.name] = {str(row["id"]): row for row in read_jsonl(path)}

    baseline_name = experiments[0].name
    case_ids = sorted(by_experiment[baseline_name])
    records: list[dict[str, Any]] = []
    for case_id in case_ids:
        baseline = by_experiment[baseline_name][case_id]
        source_record = baseline.get("source_record", {})
        record = {
            "id": case_id,
            "prompt": baseline.get("prompt", ""),
            "expected_intent": source_record.get("meta", {}).get("intent"),
            "history": flatten_history(baseline.get("history") or source_record.get("history")),
            "responses": {},
        }
        for experiment in experiments:
            row = by_experiment[experiment.name].get(case_id)
            if row is None:
                raise ValueError(f"Missing {case_id} in {experiment.name}")
            record["responses"][experiment.name] = split_generated_response(row)
        records.append(record)
    return records


# 功能：把对比 JSON 渲染成简洁 Markdown，便于抽查原始回答。
def render_comparison_markdown(records: list[dict[str, Any]], experiments: list[AblationExperiment]) -> str:
    lines = ["# Baseline LoRA Ablation Comparison", "", f"- cases: {len(records)}", ""]
    for record in records:
        lines.append(f"## {record.get('id')} | {record.get('expected_intent')}")
        if record.get("history"):
            lines.append("")
            lines.append("history:")
            for item in record["history"]:
                lines.append(f"- {item['role']}: {item['content']}")
        lines.append("")
        lines.append(f"user: {record.get('prompt')}")
        for experiment in experiments:
            response = record["responses"][experiment.name]
            lines.append("")
            lines.append(f"### {experiment.name}")
            lines.append(f"- intent: {response.get('intent')}")
            lines.append(f"- reply: {response.get('reply')}")
        lines.append("")
    return "\n".join(lines)


# 功能：写出单个实验的 SFT YAML 配置。
def write_training_config(experiment: AblationExperiment, args: argparse.Namespace) -> Path:
    config = {
        "model": {
            # 服务器本地模型路径通过 train_sft.py --base_model 覆盖，配置文件保留可移植的 Hub 名称。
            "base_model": "Qwen/Qwen3-8B",
            "adapter_model": None,
            "trust_remote_code": True,
            "torch_dtype": experiment.torch_dtype,
            "load_in_4bit": experiment.load_in_4bit,
            "quantization_method": "bnb",
            "quantization_type": "nf4",
            "double_quantization": True,
            "device_map": "ddp",
        },
        "lora": {
            "rank": experiment.rank,
            "alpha": experiment.alpha,
            "dropout": args.lora_dropout,
            "target_modules": experiment.target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "training": {
            "output_dir": experiment.output_dir,
            "max_seq_length": args.max_seq_length,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "bf16": False,
            "fp16": True,
            "optim": experiment.optim,
            "ddp_find_unused_parameters": False,
            "report_to": "none",
            "seed": args.seed,
            "save_total_limit": 2,
            "load_best_model_at_end": False,
            "resume_from_checkpoint": None,
            "log_sample_count": 3,
        },
        "data": {
            "train_file": DEFAULT_TRAIN_FILE,
            "eval_file": DEFAULT_EVAL_FILE,
            "hf_dataset_name": None,
            "hf_dataset_config": None,
            "hf_split": "train",
            "dialogue_column": "dialogue",
            "role_field": "role",
            "text_field": "text",
            "max_samples": None,
            "max_history_turns": 6,
            "hf_eval_size": 200,
            "chat_template": "qwen3_nothink",
            "default_system": DEFAULT_SYSTEM,
            "train_on_prompt": False,
            "mask_history": False,
            "validation_split_ratio": 0.1,
            "validation_split_size": None,
            "max_train_samples": None,
            "max_eval_samples": None,
            "shuffle_seed": args.seed,
        },
        "post_train_generation": {
            "enabled": True,
            "test_file": DEFAULT_TEST_FILE,
            "output_jsonl": "post_train_test_generations.jsonl",
            "output_markdown": "post_train_test_generations.md",
            "max_samples": None,
            "max_new_tokens": 320,
            "do_sample": False,
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "print_every": 10,
        },
    }
    path = args.config_dir / f"{experiment.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_simple_yaml(config), encoding="utf-8")
    return path


# 功能：把简单 Python dict/list/scalar 转成 YAML，避免流水线额外依赖 PyYAML。
def dump_simple_yaml(value: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(dump_simple_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{prefix}{key}: {format_yaml_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(dump_simple_yaml(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{prefix}- {format_yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{format_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


# 功能：格式化 YAML 标量；字符串统一使用 JSON 引号，避免冒号和中文标点歧义。
def format_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


# 功能：查询指定物理 GPU 的显存占用，供训练时峰值统计。
def query_gpu_memory_mib(devices: str) -> list[int]:
    command = [
        "nvidia-smi",
        f"--id={devices}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    values: list[int] = []
    for line in output.splitlines():
        match = re.search(r"\d+", line)
        if match:
            values.append(int(match.group(0)))
    return values


# 功能：启动后台线程采样 GPU 显存峰值。
def start_memory_monitor(devices: str, interval: float) -> tuple[threading.Event, threading.Thread, dict[str, Any]]:
    stop_event = threading.Event()
    stats: dict[str, Any] = {
        "devices": devices,
        "peak_gpu_memory_per_gpu_mib": [],
        "peak_gpu_memory_total_mib": 0,
        "samples": 0,
        "available": True,
        "warning": None,
    }

    def monitor() -> None:
        while not stop_event.is_set():
            try:
                values = query_gpu_memory_mib(devices)
                if values:
                    if not stats["peak_gpu_memory_per_gpu_mib"]:
                        stats["peak_gpu_memory_per_gpu_mib"] = [0 for _ in values]
                    stats["peak_gpu_memory_per_gpu_mib"] = [
                        max(old, new) for old, new in zip(stats["peak_gpu_memory_per_gpu_mib"], values)
                    ]
                    stats["peak_gpu_memory_total_mib"] = max(stats["peak_gpu_memory_total_mib"], sum(values))
                    stats["samples"] += 1
            except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
                stats["available"] = False
                stats["warning"] = str(exc)
                return
            stop_event.wait(interval)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return stop_event, thread, stats


# 功能：运行子进程，并同时写日志和回显到终端。
def run_streamed_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return process.wait()


# 功能：训练单个消融实验，并记录耗时和显存。
def run_training_experiment(
    experiment: AblationExperiment,
    config_path: Path,
    args: argparse.Namespace,
    index: int,
) -> None:
    output_dir = PROJECT_ROOT / experiment.output_dir
    done_file = output_dir / "post_train_test_generations.jsonl"
    if args.skip_existing and done_file.exists() and not args.force:
        print(f"[SKIP] {experiment.name}: found {done_file}")
        return

    command = [
        args.torchrun,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(args.master_port + index),
        "scripts/train_sft.py",
        "--config",
        str(config_path.relative_to(PROJECT_ROOT)),
    ]
    if args.base_model:
        command.extend(["--base_model", args.base_model])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"\n===== TRAIN {experiment.name} =====")
    print(" ".join(command))
    if args.dry_run:
        return

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.perf_counter()
    stop_event, monitor_thread, memory_stats = start_memory_monitor(args.devices, args.memory_interval)
    return_code = run_streamed_command(command, output_dir / "train.log", env)
    stop_event.set()
    monitor_thread.join(timeout=5)
    runtime_seconds = round(time.perf_counter() - start_time, 3)
    ended_at = datetime.now().isoformat(timespec="seconds")
    runtime_payload = {
        "experiment": asdict(experiment),
        "command": command,
        "cuda_visible_devices": args.devices,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": runtime_seconds,
        "return_code": return_code,
        **memory_stats,
    }
    write_json(runtime_payload, output_dir / "runtime_metrics.json")
    if return_code != 0:
        raise RuntimeError(f"{experiment.name} failed with return code {return_code}")


# 功能：把 outputs 中的实验产物复制到 docs/results，便于提交和复盘。
def collect_experiment_artifacts(experiment: AblationExperiment, config_path: Path, result_root: Path) -> None:
    source_dir = PROJECT_ROOT / experiment.output_dir
    target_dir = result_root / experiment.name
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    warnings: list[str] = []
    for filename in ARTIFACTS:
        source = source_dir / filename
        if not source.exists():
            warnings.append(f"missing: {source}")
            continue
        shutil.copy2(source, target_dir / filename)
        copied.append(filename)
    shutil.copy2(config_path, target_dir / "config.yaml")
    copied.append("config.yaml")
    write_json(
        {
            "name": experiment.name,
            "experiment": asdict(experiment),
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "copied": copied,
            "warnings": warnings,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        target_dir / "manifest.json",
    )


# 功能：安全地从字典读取数字字段。
def number_or_none(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# 功能：估算 adapter 文件总大小（MB），用于对比部署成本。
def estimate_adapter_size_mb(result_dir: Path) -> float | None:
    adapter_dir = result_dir / "adapter_model"
    if not adapter_dir.exists():
        # 训练产物 adapter_model.safetensors 在 output_dir 下
        adapter_dir = result_dir
    total = 0
    for f in adapter_dir.glob("adapter_model*.safetensors"):
        total += f.stat().st_size
    for f in adapter_dir.glob("adapter_model*.bin"):
        total += f.stat().st_size
    return round(total / (1024 * 1024), 2) if total > 0 else None


# 功能：收集训练、显存、judge 指标，形成最终对比表的数据源（使用绝对评分 judge 输出）。
def collect_summary_rows(experiments: list[AblationExperiment], result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        result_dir = result_root / experiment.name
        train_metrics = read_json(result_dir / "train_results.json", default={}) or {}
        eval_metrics = read_json(result_dir / "eval_results.json", default={}) or {}
        runtime_metrics = read_json(result_dir / "runtime_metrics.json", default={}) or {}
        peak_per_gpu = runtime_metrics.get("peak_gpu_memory_per_gpu_mib") or []
        peak_memory_max_gpu = max(peak_per_gpu) if peak_per_gpu else None
        peak_memory_total_mib = runtime_metrics.get("peak_gpu_memory_total_mib")

        # Read our absolute judge report
        judge_report = read_json(result_dir / "judge_results" / "judge_report.json", default={}) or {}
        overall = judge_report.get("overall", {})
        dim_scores = overall.get("各维度均分", {})

        # Derived metrics
        runtime_seconds = runtime_metrics.get("runtime_seconds") or (
            (train_metrics.get("train_runtime") or 0)
        )
        runtime_min = number_or_none(runtime_seconds / 60) if runtime_seconds else None
        eval_loss = number_or_none(eval_metrics.get("eval_loss"))
        perplexity = round(math.exp(eval_loss), 2) if eval_loss and eval_loss < 20 else None
        adapter_mb = estimate_adapter_size_mb(result_dir)

        rows.append(
            {
                "name": experiment.name,
                "group": experiment.group,
                "description": experiment.description,
                "rank": experiment.rank,
                "alpha": experiment.alpha,
                "target_modules": experiment.target_label,
                "precision": experiment.precision_label,
                "train_loss": number_or_none(train_metrics.get("train_loss")),
                "eval_loss": eval_loss,
                "eval_perplexity": perplexity,
                "train_runtime_min": runtime_min,
                "peak_memory_max_gpu_mib": peak_memory_max_gpu,
                "peak_memory_total_mib": peak_memory_total_mib,
                "adapter_size_mb": adapter_mb,
                "pass_rate": overall.get("通过率"),
                "auto_resolution_rate": overall.get("自动解决率"),
                "accuracy_rate": overall.get("准确率"),
                "dim_need": dim_scores.get("需求解决"),
                "dim_accurate": dim_scores.get("回答准确"),
                "dim_helpful": dim_scores.get("有帮助性"),
                "dim_tone": dim_scores.get("语气体验"),
                "dim_risk": dim_scores.get("风险控制"),
                "dim_history": dim_scores.get("历史利用"),
                "fail_tags": overall.get("失败标签分布", {}),
            }
        )
    return rows


# 功能：渲染最终 LoRA 消融对比表（使用绝对评分指标）。
def render_ablation_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LoRA 消融实验对比（JDDC Rebuild V2 增广数据集）",
        "",
        "## 资源与效果",
        "",
        "| experiment | variable | rank | target | precision | train_loss | eval_loss | ppl | runtime(min) | mem/gpu(MiB) | mem_total(MiB) | adapter(MB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {group} | {rank} | {target} | {precision} | {train_loss} | {eval_loss} | {ppl} | {runtime} | {mem} | {mem_total} | {adapter} |".format(
                name=row["name"], group=row["group"], rank=row["rank"],
                target=row["target_modules"], precision=row["precision"],
                train_loss=format_cell(row.get("train_loss")),
                eval_loss=format_cell(row.get("eval_loss")),
                ppl=format_cell(row.get("eval_perplexity")),
                runtime=format_cell(row.get("train_runtime_min")),
                mem=format_cell(row.get("peak_memory_max_gpu_mib")),
                mem_total=format_cell(row.get("peak_memory_total_mib")),
                adapter=format_cell(row.get("adapter_size_mb")),
            )
        )
    lines.append("")
    lines.append("## 评测指标（绝对评分 LLM judge，严格标准：需求>=4 且 准确>=4 通过）")
    lines.append("")
    lines.append("| experiment | 通过率 | 解决率 | 准确率 | 需求 | 准确 | 帮助 | 语气 | 风险 | 历史利用 | 主要失败标签 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        fail_summary = ", ".join("%s:%d" % (k, v) for k, v in sorted(
            (row.get("fail_tags") or {}).items(), key=lambda x: -x[1]
        )[:3])
        lines.append(
            "| {name} | {pass_} | {auto} | {acc} | {need} | {dim_acc} | {help} | {tone} | {risk} | {history} | {fail} |".format(
                name=row["name"],
                pass_=format_percent_cell(row.get("pass_rate")),
                auto=format_percent_cell(row.get("auto_resolution_rate")),
                acc=format_percent_cell(row.get("accuracy_rate")),
                need=format_cell(row.get("dim_need")),
                dim_acc=format_cell(row.get("dim_accurate")),
                help=format_cell(row.get("dim_helpful")),
                tone=format_cell(row.get("dim_tone")),
                risk=format_cell(row.get("dim_risk")),
                history=format_cell(row.get("dim_history")),
                fail=fail_summary,
            )
        )
    lines.append("")
    return "\n".join(lines)


# 功能：格式化普通数值单元格。
def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# 功能：格式化比例单元格。
def format_percent_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


# 功能：调用绝对评分 DeepSeek judge（judge_jddc_rebuild_v2_with_deepseek.py），每个实验独立评测。
def run_judge_for_experiment(experiment: AblationExperiment, args: argparse.Namespace, result_root: Path) -> None:
    gen_file = result_root / experiment.name / "post_train_test_generations.jsonl"
    if not gen_file.exists():
        print(f"[WARN] No generation file for {experiment.name}, skipping judge")
        return

    judge_out = result_root / experiment.name / "judge_results"
    judge_out.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "scripts/judge_jddc_rebuild_v2_with_deepseek.py",
        "--generations", str(gen_file),
        "--annotations", str(PROJECT_ROOT / DEFAULT_ANNOTATION_FILE),
        "--output-dir", str(judge_out),
        "--sleep", str(args.judge_sleep),
        "--retries", str(args.judge_retries),
    ]
    if args.judge_limit is not None:
        command.extend(["--limit", str(args.judge_limit)])
    if args.no_response_format:
        command.append("--no-response-format")

    print("\n===== JUDGE %s =====" % experiment.name)
    print(" ".join(command))
    if args.dry_run:
        return
    return_code = run_streamed_command(command, result_root / experiment.name / "judge.log", os.environ.copy())
    if return_code != 0:
        raise RuntimeError("%s judge failed with return code %d" % (experiment.name, return_code))


# 功能：对所有实验跑绝对评分 judge。
def run_all_judges(args: argparse.Namespace, experiments: list[AblationExperiment], result_root: Path) -> None:
    for experiment in experiments:
        run_judge_for_experiment(experiment, args, result_root)


# 功能：选择本次要跑的实验列表。
def select_experiments(only: str | None) -> list[AblationExperiment]:
    if not only:
        return EXPERIMENTS
    names = {item.strip() for item in only.split(",") if item.strip()}
    selected = [experiment for experiment in EXPERIMENTS if experiment.name in names or experiment.group in names]
    missing = names - {experiment.name for experiment in EXPERIMENTS} - {experiment.group for experiment in EXPERIMENTS}
    if missing:
        raise ValueError(f"Unknown experiment/group in --only: {sorted(missing)}")
    return selected


# 功能：解析命令行参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=None, help="Server-local base model path, e.g. /data2/.../Qwen3-8B.")
    parser.add_argument("--devices", default="4,5", help="Physical GPU ids for CUDA_VISIBLE_DEVICES and nvidia-smi.")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29610)
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--only", default=None, help="Comma-separated experiment names or groups: rank,target_modules,quantization.")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-interval", type=float, default=5.0)
    parser.add_argument("--judge-limit", type=int, default=None)
    parser.add_argument("--judge-sleep", type=float, default=0.5)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Rerun training even if generation file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print commands without running training/judge.")
    return parser.parse_args()


# 功能：脚本入口，串起配置生成、训练、回收、LLM judge 和最终表格。
def main() -> None:
    args = parse_args()
    args.config_dir = args.config_dir.resolve()
    args.result_dir = args.result_dir.resolve()
    experiments = select_experiments(args.only)

    if not args.skip_judge and not args.dry_run and not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")):
        raise ValueError("Set DEEPSEEK_API_KEY before running judge, or pass --skip-judge.")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    config_paths = {experiment.name: write_training_config(experiment, args) for experiment in experiments}
    write_json(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "experiments": [asdict(experiment) for experiment in experiments],
            "config_paths": {name: str(path) for name, path in config_paths.items()},
            "train_file": DEFAULT_TRAIN_FILE,
            "eval_file": DEFAULT_EVAL_FILE,
            "test_file": DEFAULT_TEST_FILE,
        },
        args.result_dir / "ablation_manifest.json",
    )

    if not args.skip_train:
        for index, experiment in enumerate(experiments):
            run_training_experiment(experiment, config_paths[experiment.name], args, index)

    if not args.dry_run:
        for experiment in experiments:
            collect_experiment_artifacts(experiment, config_paths[experiment.name], args.result_dir)
        comparison = build_ablation_comparison(experiments, args.result_dir)
        comparison_path = args.result_dir / "ablation_comparison.json"
        write_json(comparison, comparison_path)
        (args.result_dir / "ablation_comparison.md").write_text(
            render_comparison_markdown(comparison, experiments),
            encoding="utf-8",
        )
    else:
        comparison_path = args.result_dir / "ablation_comparison.json"

    if not args.skip_judge:
        run_all_judges(args, experiments, args.result_dir)

    if not args.dry_run:
        rows = collect_summary_rows(experiments, args.result_dir)
        write_json(rows, args.result_dir / "ablation_summary.json")
        (args.result_dir / "ablation_summary.md").write_text(render_ablation_summary_markdown(rows), encoding="utf-8")
        print(f"\nsummary_table={args.result_dir / 'ablation_summary.md'}")


if __name__ == "__main__":
    main()
