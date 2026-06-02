"""Command-line entry point for LoRA SFT training."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_cs_sft_dpo.sft_training import main


# 功能：从命令行启动 SFT 训练。
def run() -> None:
    main()


if __name__ == "__main__":
    run()
