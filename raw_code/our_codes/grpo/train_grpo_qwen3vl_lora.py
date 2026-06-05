import argparse
import re
import string
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
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


def normalize(text):
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def answer_match(pred, answers):
    pred_norm = normalize(pred)
    if not pred_norm:
        return False
    for ans in answers:
        ans_norm = normalize(ans)
        if ans_norm and (pred_norm == ans_norm or ans_norm in pred_norm or pred_norm in ans_norm):
            return True
    return False


def extract_aux_text(prompt):
    marker = "please use it with caution:"
    sep = "\n----------\n"
    lower = (prompt or "").lower()
    start = lower.find(marker)
    rest = prompt[start + len(marker) :] if start >= 0 else prompt
    end = rest.find(sep)
    return rest[:end].strip() if end >= 0 else rest.strip()


def answer_in_text(answer, text):
    ans = normalize(answer)
    ctx = normalize(text)
    if not ans or not ctx:
        return False
    if ans in ctx:
        return True
    toks = ans.split()
    if len(toks) <= 1:
        return False
    ctx_toks = set(ctx.split())
    return sum(tok in ctx_toks for tok in toks) / len(toks) >= 0.8


def clean_candidate(text):
    return re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n.,;:!?()[]{}\"'")


def is_bad_candidate(candidate, chosen_answers, question):
    cand = normalize(candidate)
    if not cand or len(cand) < 2:
        return True
    if len(cand.split()) > 8:
        return True
    if any(answer_match(candidate, [ans]) for ans in chosen_answers):
        return True
    q = normalize(question)
    if cand in q and not re.search(r"\d", cand):
        return True
    common = {"yes", "no", "none", "unknown", "n/a", "not applicable"}
    return cand in common


def mine_misleading_candidates(aux_text, chosen_answers, question, rejected="", max_candidates=6):
    candidates = []
    seen = set()

    def add(candidate):
        candidate = clean_candidate(candidate)
        key = normalize(candidate)
        if key in seen or is_bad_candidate(candidate, chosen_answers, question):
            return
        seen.add(key)
        candidates.append(candidate)

    if rejected:
        add(rejected)

    patterns = [
        r"\b\d{1,4}(?:[./:%-]\d{1,4}){0,4}%?\b",
        r"\b[A-Z]{1,5}\d{1,5}(?:[-/][A-Z0-9]{1,8})+\b",
        r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z.]+){1,5}\b",
        r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){0,5}\b",
        r"\$?\b\d+(?:,\d{3})*(?:\.\d+)?\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, aux_text):
            add(match.group(0))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def high_confidence_candidate(candidate):
    candidate = clean_candidate(candidate)
    if not candidate:
        return False
    has_digit = bool(re.search(r"\d", candidate))
    looks_structured = bool(re.search(r"[./:%-]", candidate))
    looks_code = bool(re.fullmatch(r"[A-Z0-9]{2,}(?:[-/][A-Z0-9]{1,12})+", candidate))
    short_number = bool(re.fullmatch(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?", candidate))
    return has_digit and (looks_structured or looks_code or short_number)


def build_prompt(processor, prompt, image):
    user = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)


def encode_batch(processor, examples, answers, device, max_length=0):
    texts, prompt_lens, images = [], [], []
    eos = processor.tokenizer.eos_token or ""
    for ex, answer in zip(examples, answers):
        prompt_text = build_prompt(processor, ex["prompt"], ex["image"])
        full_text = prompt_text + answer + eos
        prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False).input_ids
        texts.append(full_text)
        prompt_lens.append(len(prompt_ids))
        images.append(ex["image"])

    kwargs = {"text": texts, "images": images, "padding": True, "return_tensors": "pt"}
    if max_length:
        kwargs.update({"truncation": True, "max_length": max_length})
    batch = processor(**kwargs)
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
    lengths = mask.sum(dim=-1).clamp_min(1)
    return (token_logps * mask).sum(dim=-1), lengths


@torch.no_grad()
def generate_completions(model, processor, ex, device, num_generations, max_new_tokens, temperature, top_p):
    prompt_text = build_prompt(processor, ex["prompt"], ex["image"])
    inputs = processor(
        text=[prompt_text] * num_generations,
        images=[ex["image"]] * num_generations,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    completion_ids = output_ids[:, input_len:]
    texts = processor.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [clean_completion(t) for t in texts]


def clean_completion(text):
    text = str(text or "").strip()
    text = text.split("\n")[0].strip()
    text = re.sub(r"^(answer\\s*:\\s*)", "", text, flags=re.I).strip()
    return text


def compute_reward(ex, completion):
    chosen = [ex["chosen"]]
    rejected = ex.get("rejected", "")
    question = ex.get("prompt", "").split("----------")[-1]
    aux_text = extract_aux_text(ex.get("prompt", ""))
    candidates = mine_misleading_candidates(aux_text, chosen, question, rejected=rejected)

    reward = 0.0
    correct = answer_match(completion, chosen)
    rejected_match = bool(rejected and answer_match(completion, [rejected]))
    high_conf_candidates = [cand for cand in candidates if high_confidence_candidate(cand)]
    high_conf_match = any(answer_match(completion, [cand]) for cand in high_conf_candidates)

    if correct:
        reward += 1.5
    else:
        reward -= 0.05

    if not correct and rejected_match:
        reward -= 0.8
    elif not correct and high_conf_match:
        reward -= 0.3

    if not completion:
        reward -= 0.5
    word_count = len(completion.split())
    if 1 <= word_count <= 6:
        reward += 0.1
    elif word_count > 8:
        reward -= 0.1
    return reward


def collate(rows):
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta_kl", type=float, default=0.02)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--min_pixels", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    ds = load_from_disk(args.dataset_dir)["train"]
    model_cls = get_model_class(args.model_name)
    processor_kwargs = {"trust_remote_code": True}
    if args.min_pixels:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    policy_base = model_cls.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True, device_map={"": device}
    )
    ref = model_cls.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True, device_map={"": device}
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    policy_base.config.use_cache = False
    if hasattr(policy_base, "gradient_checkpointing_enable"):
        policy_base.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    policy = get_peft_model(policy_base, lora_config)
    policy.print_trainable_parameters()
    policy.train()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    running = {"loss": 0.0, "reward": 0.0, "correct": 0.0}
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch_examples in tqdm(loader, desc=f"grpo step {step}/{args.max_steps}"):
            all_examples, completions, rewards = [], [], []
            for ex in batch_examples:
                gens = generate_completions(
                    policy, processor, ex, device, args.num_generations, args.max_new_tokens, args.temperature, args.top_p
                )
                for gen in gens:
                    all_examples.append(ex)
                    completions.append(gen)
                    rewards.append(compute_reward(ex, gen))

            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            grouped = rewards_t.view(len(batch_examples), args.num_generations)
            mean = grouped.mean(dim=1, keepdim=True)
            std = grouped.std(dim=1, keepdim=True).clamp_min(1e-4)
            advantages = ((grouped - mean) / std).view(-1).detach()

            policy_logps, lengths = sequence_logps(policy, processor, all_examples, completions, device, args.max_length)
            with torch.no_grad():
                ref_logps, _ = sequence_logps(ref, processor, all_examples, completions, device, args.max_length)

            avg_policy_logps = policy_logps / lengths
            avg_ref_logps = ref_logps / lengths
            kl_proxy = avg_policy_logps - avg_ref_logps
            loss = -(advantages * avg_policy_logps).mean() + args.beta_kl * kl_proxy.pow(2).mean()
            (loss / args.grad_accum).backward()

            running["loss"] += float(loss.detach())
            running["reward"] += float(rewards_t.mean())
            running["correct"] += float(sum(answer_match(c, [ex["chosen"]]) for c, ex in zip(completions, all_examples)) / len(completions))

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 10 == 0:
                print(
                    f"step={step} loss={running['loss']/10:.4f} "
                    f"reward={running['reward']/10:.4f} correct_gen={running['correct']/10:.4f}"
                )
                running = {"loss": 0.0, "reward": 0.0, "correct": 0.0}
            if args.save_every and step % args.save_every == 0:
                policy.save_pretrained(output_dir / f"checkpoint-{step}")
                processor.save_pretrained(output_dir / f"checkpoint-{step}")
            if step >= args.max_steps:
                break

    policy.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"saved GRPO LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
