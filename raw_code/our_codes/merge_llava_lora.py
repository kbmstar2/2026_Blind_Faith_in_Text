import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForPreTraining, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForPreTraining.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.merge_and_unload()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(out)
    print(f"saved merged model to {out}")


if __name__ == "__main__":
    main()
