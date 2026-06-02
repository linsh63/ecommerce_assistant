"""项目共享工具：YAML 解析、路径解析、随机种子。

sft_training.py 和 dpo_training.py 共用此模块，避免 YAML 解析逻辑重复。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any


# SFT 的核心 trick：把 prompt 部分的 label 设为 -100，Transformers 的 CrossEntropyLoss 会自动跳过这些 token。
# 为什么是 -100？因为 PyTorch 的 CrossEntropyLoss 默认 ignore_index=-100。
IGNORE_INDEX = -100


# ═══════════════════════════════════════════════════════════════════════════════
# 路径与随机种子
# ═══════════════════════════════════════════════════════════════════════════════

# 功能：根据当前文件位置找到 ecommerce_assistant 项目根目录。
def find_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 功能：把相对路径解析为项目根目录下的绝对路径。
def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root = Path(project_root).expanduser().resolve() if project_root else find_project_root()
    return (root / candidate).resolve()


# 功能：设置 Python、NumPy 和 Torch 的随机种子。
def set_train_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# YAML 解析 —— 优先用 PyYAML，不可用时回退到简易解析器
# ═══════════════════════════════════════════════════════════════════════════════

# 功能：从 YAML 文件读取原始配置字典（优先 PyYAML）。
def read_yaml_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        text = config_path.read_text(encoding="utf-8")
        return _parse_simple_yaml(text, source=str(config_path))

    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return payload


# 功能：在 PyYAML 不可用时解析本项目配置用到的简单 YAML 子集。
def _parse_simple_yaml(text: str, source: str) -> dict[str, Any]:
    lines = text.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, raw_line in enumerate(lines):
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unexpected YAML list item in {source}:{index + 1}")
            parent.append(_parse_yaml_scalar(content[2:].strip()))
            continue
        if ":" not in content:
            raise ValueError(f"Invalid YAML line in {source}:{index + 1}: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not isinstance(parent, dict):
            raise ValueError(f"Unexpected YAML mapping in {source}:{index + 1}")
        if raw_value:
            parent[key] = _parse_yaml_scalar(raw_value)
            continue
        container: Any = [] if _next_yaml_child_is_list(lines, index, indent) else {}
        parent[key] = container
        stack.append((indent, container))
    return root


# 功能：去掉简单 YAML 行尾注释。
def _strip_yaml_comment(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return ""
    marker = " #"
    if marker in line:
        return line.split(marker, 1)[0]
    return line.rstrip()


# 功能：判断当前空值键下面的第一个有效子节点是不是列表。
def _next_yaml_child_is_list(lines: list[str], current_index: int, parent_indent: int) -> bool:
    for raw_line in lines[current_index + 1 :]:
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= parent_indent:
            return False
        return line.strip().startswith("- ")
    return False


# 功能：把简单 YAML 标量转换成 Python 类型。
def _parse_yaml_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(token in value for token in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value
