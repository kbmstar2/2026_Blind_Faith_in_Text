import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from itertools import cycle

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration


def get_model_class(model_name):
    lowered = model_name.lower()
    if "qwen3" in lowered:
        return Qwen3VLForConditionalGeneration
    if "qwen2" in lowered:
        return Qwen2VLForConditionalGeneration
    raise ValueError(f"Unsupported model for this script: {model_name}")


def build_prompt(processor, prompt, image):
    user = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)


def encode_batch(processor, examples, answers, device, max_length=0):
    texts = []
    prompt_lens = []
    images = []
    eos = processor.tokenizer.eos_token or ""

    for ex, answer in zip(examples, answers):
        prompt_text = build_prompt(processor, ex["prompt"], ex["image"])
        full_text = prompt_text + answer + eos
        prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False).input_ids
        texts.append(full_text)
        prompt_lens.append(len(prompt_ids))
        images.append(ex["image"])

    processor_kwargs = {"text": texts, "images": images, "padding": True, "return_tensors": "pt"}
    if max_length:
        processor_kwargs.update({"truncation": True, "max_length": max_length})
    batch = processor(**processor_kwargs)
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for i, prompt_len in enumerate(prompt_lens):
        labels[i, :prompt_len] = -100
    return batch, labels


def sequence_logps(model, processor, examples, answers, device, max_length=0):
    batch, labels = encode_batch(processor, examples, answers, device, max_length=max_length)
    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, weights=None):
    policy_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = beta * (policy_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    if weights is not None:
        weights = weights.to(losses.device, dtype=losses.dtype)
        losses = losses * weights
    rewards_chosen = beta * (policy_chosen - ref_chosen).detach()
    rewards_rejected = beta * (policy_rejected - ref_rejected).detach()
    return losses.mean(), rewards_chosen.mean(), rewards_rejected.mean()


def sequence_nll(model, processor, examples, answers, device, max_length=0):
    batch, labels = encode_batch(processor, examples, answers, device, max_length=max_length)
    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_nll = -F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    denom = mask.sum(dim=-1).clamp_min(1)
    return (token_nll * mask).sum(dim=-1) / denom


def collate(rows):
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--min_pixels", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=0)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--sft_dataset_dir", default="", help="Optional preservation SFT dataset to mix with DPO.")
    parser.add_argument("--sft_weight", type=float, default=0.0, help="Weight for preservation SFT loss.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    ds = load_from_disk(args.dataset_dir)["train"]
    sft_ds = None
    if args.sft_dataset_dir:
        sft_ds = load_from_disk(args.sft_dataset_dir)["train"]
        print(f"loaded preservation SFT rows: {len(sft_ds)}")
    model_cls = get_model_class(args.model_name)
    processor_kwargs = {"trust_remote_code": True}
    if args.min_pixels:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    policy = model_cls.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": device},
    )
    ref = model_cls.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": device},
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if not args.no_gradient_checkpointing:
        policy.config.use_cache = False
        if hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    policy = get_peft_model(policy, lora_config)
    policy.print_trainable_parameters()
    policy.train()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    sft_iter = None
    if sft_ds is not None and args.sft_weight > 0:
        sft_loader = DataLoader(sft_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        sft_iter = cycle(sft_loader)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for examples in tqdm(loader, desc=f"train step {step}/{args.max_steps}"):
            chosen = [ex["chosen"] for ex in examples]
            rejected = [ex["rejected"] for ex in examples]
            weights = torch.tensor(
                [float(ex.get("sample_weight", 1.0) or 1.0) for ex in examples],
                device=device,
            )

            policy_chosen = sequence_logps(policy, processor, examples, chosen, device, args.max_length)
            policy_rejected = sequence_logps(policy, processor, examples, rejected, device, args.max_length)
            with torch.no_grad():
                ref_chosen = sequence_logps(ref, processor, examples, chosen, device, args.max_length)
                ref_rejected = sequence_logps(ref, processor, examples, rejected, device, args.max_length)

            dpo, reward_chosen, reward_rejected = dpo_loss(
                policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta, weights
            )
            sft = torch.zeros((), device=device, dtype=dpo.dtype)
            if sft_iter is not None:
                sft_examples = next(sft_iter)
                sft_answers = [ex["chosen"] for ex in sft_examples]
                sft_weights = torch.tensor(
                    [float(ex.get("sample_weight", 1.0) or 1.0) for ex in sft_examples],
                    device=device,
                )
                sft_losses = sequence_nll(policy, processor, sft_examples, sft_answers, device, args.max_length)
                sft = (sft_losses * sft_weights).mean()
            loss = dpo + args.sft_weight * sft
            (loss / args.grad_accum).backward()
            running_loss += float(loss.detach())

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 10 == 0:
                avg = running_loss / 10
                running_loss = 0.0
                print(
                    f"step={step} loss={avg:.4f} dpo={float(dpo):.4f} sft={float(sft):.4f} "
                    f"reward_chosen={float(reward_chosen):.4f} reward_rejected={float(reward_rejected):.4f}"
                )
            if args.save_every and step % args.save_every == 0:
                policy.save_pretrained(output_dir / f"checkpoint-{step}")
                processor.save_pretrained(output_dir / f"checkpoint-{step}")
            if step >= args.max_steps:
                break

    policy.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
