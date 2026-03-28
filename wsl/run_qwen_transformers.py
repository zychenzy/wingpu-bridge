#!/usr/bin/env python3
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local Qwen model with Transformers")
    p.add_argument("--model-path", required=True, help="Local model directory path")
    p.add_argument("--prompt", default="Give me 3 concise ideas for testing a local LLM server.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Computation dtype when not using 4-bit quantization",
    )
    return p.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This runner expects GPU execution.")

    t0 = time.time()
    print(f"[info] loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model_kwargs = {"device_map": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = resolve_dtype(args.dtype)

    print(
        "[info] loading model with "
        f"4bit={args.load_in_4bit}, dtype={args.dtype}, device_map=auto"
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    load_seconds = time.time() - t0
    print(f"[info] model loaded in {load_seconds:.1f}s")

    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_len = encoded["input_ids"].shape[-1]

    gen_start = time.time()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_seconds = time.time() - gen_start

    response_ids = output[0][input_len:]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print("[result]")
    print(text.strip())
    print(f"[metrics] generation_time_s={gen_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
