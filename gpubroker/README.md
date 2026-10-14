# gpubroker

Shared GPU lease broker for wingpu and the sibling bridges. No third-party dependencies.

- `broker.py` is the stdlib program wingpu uploads to `~/.gpu-bridge/gpu_broker.py`.
- `client.py` talks to that program over a runner. `install` sends the file on stdin.
- `ssh.py` is the SSH runner and the loopback `nc` relay (not `ssh -L`).
- `config.py` loads a `[broker]` table.
- `lease.py` is acquire, release, and touch for a sibling CLI.
- `runtimes.py` holds the runtime ids. Whisper is `live-asr`. Qwen ASR is `live-asr-qwen`.

Only wingpu should call `ensure_installed(..., upgrade=True)` or `LeaseSession(..., allow_install=True)`.
Everyone else leaves a missing or mismatched broker in place and tells the user to run
`wingpu broker install`.

`gpu-broker version` prints `{"ok", "version", "protocol"}`.

Migration steps, the exact uv dependency line, and which vendored files to delete:
[docs/gpubroker-migration.md](../docs/gpubroker-migration.md).
