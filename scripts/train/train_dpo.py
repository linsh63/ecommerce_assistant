"""Entry point for DPO training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_cs_sft_dpo.dpo_training import DPOTrainingPipeline, load_dpo_config


def main():
    parser = argparse.ArgumentParser(description="DPO preference optimization for e-commerce customer service.")
    parser.add_argument("--config", default="configs/dpo_jddc_rebuild_v2.yaml")
    parser.add_argument("--sft_adapter_path", default=None, help="Override model.sft_adapter_path")
    parser.add_argument("--base_model", default=None, help="Override model.base_model")
    parser.add_argument("--output_dir", default=None, help="Override dpo.output_dir")
    parser.add_argument("--train_file", default=None, help="Override data.train_file")
    parser.add_argument("--beta", type=float, default=None, help="Override dpo.beta (try 0.05/0.1/0.3)")
    parser.add_argument("--epochs", type=float, default=None, help="Override dpo.num_train_epochs")
    args = parser.parse_args()

    config = load_dpo_config(args.config)

    # CLI overrides
    if args.sft_adapter_path:
        config.model.sft_adapter_path = args.sft_adapter_path
    if args.base_model:
        config.model.base_model = args.base_model
    if args.output_dir:
        config.dpo.output_dir = args.output_dir
    if args.train_file:
        config.data.train_file = args.train_file
    if args.beta is not None:
        config.dpo.beta = args.beta
    if args.epochs is not None:
        config.dpo.num_train_epochs = args.epochs

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    pipeline = DPOTrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
