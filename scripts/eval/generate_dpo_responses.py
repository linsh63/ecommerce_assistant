"""Generate responses from DPO adapter on eval set, output in judge-compatible format."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_cs_sft_dpo.utils import set_train_seed
from ecommerce_cs_sft_dpo.sft_training import (
    SFTDataProcessor,
    SFTModelBuilder,
    SFTPostTrainGenerator,
    SFTExperimentConfig,
    SFTModelConfig,
    SFTLoraConfig,
    SFTTrainingConfig,
    SFTDataConfig,
    SFTPostTrainGenerationConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_train_seed(args.seed)

    # Build minimal config for model loading
    config = SFTExperimentConfig(
        model=SFTModelConfig(
            base_model=args.base_model,
            torch_dtype="float16",
            load_in_4bit=True,
            device_map="auto",
        ),
        lora=SFTLoraConfig(),
        training=SFTTrainingConfig(),
        data=SFTDataConfig(),
        post_train_generation=SFTPostTrainGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        ),
    )

    # Load model with DPO adapter
    builder = SFTModelBuilder(config)
    tokenizer = builder.load_tokenizer()
    model = builder.load_model()

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # Generate
    eval_path = Path(args.eval_file)
    records = SFTDataProcessor.read_jsonl_records(eval_path)
    print(f"Loaded {len(records)} eval samples from {eval_path}")

    generator = SFTPostTrainGenerator(tokenizer, config)
    results = []
    for i, record in enumerate(records, 1):
        prompt = generator.build_generation_prompt(record)
        response = generator.generate_answer(model, prompt)
        results.append({
            "id": record.get("id", f"case_{i:03d}"),
            "messages": record.get("messages", []),
            "prompt": record.get("prompt", record.get("query", "")),
            "generated_response": response,
            "scenario": record.get("meta", {}).get("scenario", ""),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(records)}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} generations to {output_path}")


if __name__ == "__main__":
    main()
