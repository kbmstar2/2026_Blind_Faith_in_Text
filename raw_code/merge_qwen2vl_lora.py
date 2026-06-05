import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration


def get_model_class(model_name):
    lowered = model_name.lower()
    if "qwen3" in lowered:
        return Qwen3VLForConditionalGeneration
    if "qwen2" in lowered:
        return Qwen2VLForConditionalGeneration
    raise ValueError(f"Unsupported model for this script: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_cls = get_model_class(args.base_model)
    base = model_cls.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(output_dir)
    print(f"saved merged model to {output_dir}")


if __name__ == "__main__":
    main()
