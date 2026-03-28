#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive chat with local Qwen model via Transformers")
    p.add_argument("--model-path", required=True)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--system", default="")
    p.add_argument("--max-history-turns", type=int, default=10)
    return p.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def load_model(args: argparse.Namespace):
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
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def trim_history(messages, max_history_turns: int, has_system: bool):
    # Keep recent user/assistant pairs while preserving optional system prompt.
    if has_system and messages:
        system = messages[0]
        body = messages[1:]
    else:
        system = None
        body = messages

    max_msgs = max_history_turns * 2
    if len(body) > max_msgs:
        body = body[-max_msgs:]

    return ([system] + body) if system else body


def generate(model, tokenizer, messages, args):
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_len = encoded["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = output[0][input_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. This chat script expects GPU.")

    print(f"[chat] loading model: {args.model_path}")
    model, tokenizer = load_model(args)
    print("[chat] ready. commands: /exit /clear")

    messages = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] bye")
            break

        if not user:
            continue
        if user == "/exit":
            print("[chat] bye")
            break
        if user == "/clear":
            messages = []
            if args.system.strip():
                messages.append({"role": "system", "content": args.system.strip()})
            print("[chat] history cleared")
            continue

        messages.append({"role": "user", "content": user})
        messages = trim_history(
            messages,
            max_history_turns=args.max_history_turns,
            has_system=bool(args.system.strip()),
        )

        answer = generate(model, tokenizer, messages, args)
        print(f"Model> {answer}")
        messages.append({"role": "assistant", "content": answer})
        messages = trim_history(
            messages,
            max_history_turns=args.max_history_turns,
            has_system=bool(args.system.strip()),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
