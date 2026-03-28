#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_PROMPTS = [
    "Explain in 3 bullet points how to evaluate a local LLM.",
    "Summarize this in one sentence: Transformers process sequences with attention.",
    "Return JSON only with keys task and priority for: 'set up nightly benchmark'.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small benchmark for local Qwen models in Transformers")
    p.add_argument("--model-path", required=True)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--runs-per-prompt", type=int, default=1)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--prompts-json", help="JSON array of prompts; if omitted, built-in prompts are used")
    p.add_argument("--output-json", help="Optional path to write benchmark JSON")
    return p.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def maybe_sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model_and_tokenizer(args: argparse.Namespace):
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

    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    maybe_sync_cuda()
    load_seconds = time.perf_counter() - start
    return model, tokenizer, load_seconds


def run_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_tokens = int(encoded["input_ids"].shape[-1])

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    maybe_sync_cuda()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, **gen_kwargs)
    maybe_sync_cuda()
    elapsed = time.perf_counter() - t0

    gen_ids = output[0][input_tokens:]
    output_tokens = int(gen_ids.numel())
    tps = (output_tokens / elapsed) if elapsed > 0 else 0.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed,
        "tokens_per_s": tps,
    }


def summarize(results: List[dict]) -> dict:
    latencies = [x["elapsed_s"] for x in results]
    throughputs = [x["tokens_per_s"] for x in results if x["output_tokens"] > 0]
    out_tokens = [x["output_tokens"] for x in results]
    return {
        "n": len(results),
        "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_s": statistics.median(latencies) if latencies else 0.0,
        "avg_tokens_per_s": statistics.mean(throughputs) if throughputs else 0.0,
        "avg_output_tokens": statistics.mean(out_tokens) if out_tokens else 0.0,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Benchmark expects GPU.")

    prompts = json.loads(args.prompts_json) if args.prompts_json else DEFAULT_PROMPTS
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a non-empty JSON list")

    print(f"[bench] loading model from {args.model_path}")
    model, tokenizer, load_seconds = load_model_and_tokenizer(args)
    print(f"[bench] load_seconds={load_seconds:.2f}")

    print(f"[bench] warmup_runs={args.warmup_runs}")
    for _ in range(args.warmup_runs):
        _ = run_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompts[0],
            max_new_tokens=min(args.max_new_tokens, 32),
            temperature=args.temperature,
            top_p=args.top_p,
        )

    runs = []
    for prompt_idx, prompt in enumerate(prompts):
        for run_idx in range(args.runs_per_prompt):
            r = run_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            r["prompt_index"] = prompt_idx
            r["run_index"] = run_idx
            runs.append(r)
            print(
                "[bench] "
                f"p{prompt_idx} r{run_idx} "
                f"lat={r['elapsed_s']:.2f}s out={r['output_tokens']} tok "
                f"tps={r['tokens_per_s']:.2f}"
            )

    summary = summarize(runs)
    report = {
        "model_path": args.model_path,
        "load_in_4bit": args.load_in_4bit,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "load_seconds": load_seconds,
        "summary": summary,
        "runs": runs,
    }

    print("[bench] summary")
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(f"[bench] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
