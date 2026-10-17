# llama.cpp upgrade / tuning notes, 2026-10-17 (HKT)

- Old build: b10454, commit 4df29be4f (2026-08-16), ~/src/llama.cpp/build-cuda89, nvcc 12.0.
- New build: b11706, commit f74e2fa6a. First built in ~/src/llama.cpp-new/build-cuda89-b11706
  (benchmarked; llama-current points here). Rebuilding at ~/src/llama.cpp/.worktrees/b11706/build-cuda89
  so the passwordless /usr/local/sbin/wingpu-install-llamacpp-system wrapper accepts it.
- CUDA 12.8 not installed: needs sudo password (no passwordless sudo). Kept nvcc 12.0.
- Benchmarks: ~/bench/results/* in WSL (harness ~/bench/harness.py, summary ~/bench/sum.py).
- Config backup: ~/.config/wingpu/wingpu.local.toml.bak-20261017-tada
- Branch llama-versioned-upgrade (not pushed): update_llama.sh rewrite + README section.
- 220 W test done, reset to 285 W.
- Pending: switch llama-current to .worktrees build, install /usr/local/bin copies, verify,
  delete ~/src/llama.cpp/build-cuda89 and ~/src/llama.cpp-new worktree.

## Done (15:40 HKT)
- llama-current -> ~/src/llama.cpp/.worktrees/b11706/build-cuda89 (b11706, f74e2fa6a); /usr/local/bin copies refreshed.
- Verified via gateway: 9/9 + 5/5 + 9/9 requests OK (thinking on/off, streaming), speeds = baseline.
- Deleted: ~/src/llama.cpp/build-cuda89 (b10454), temp worktree ~/src/llama.cpp-new, rollback lane upstream-mtp-previous.
- Kept active: Q3_K_XL, q4_0 KV, 65K ctx, MTP n=3, threads 6, 285 W.
- Not adopted: q8_0 KV at 65K (15.96 GB, 32K decode 30 tok/s = VRAM spill), IQ4_XS models (slower; option only).
