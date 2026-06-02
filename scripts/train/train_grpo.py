"""Entry point for GRPO training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_cs_sft_dpo.grpo_training import GRPOTrainingPipeline, load_grpo_config


def main():
    parser = argparse.ArgumentParser(description="GRPO preference optimization for e-commerce customer service.")
    parser.add_argument("--config", default="configs/grpo_jddc.yaml")
    parser.add_argument("--sft_adapter_path", default=None, help="Override model.sft_adapter_path")
    parser.add_argument("--base_model", default=None, help="Override model.base_model")
    parser.add_argument("--output_dir", default=None, help="Override training.output_dir")
    parser.add_argument("--prompt_file", default=None, help="Override data.prompt_file")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--api_key", default=None, help="DeepSeek API key")
    args = parser.parse_args()

    config = load_grpo_config(args.config)
    if args.sft_adapter_path:
        config.model.sft_adapter_path = args.sft_adapter_path
    if args.base_model:
        config.model.base_model = args.base_model
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.prompt_file:
        config.data.prompt_file = args.prompt_file
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.api_key:
        config.reward.api_key = args.api_key

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    pipeline = GRPOTrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
