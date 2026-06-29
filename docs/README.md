# GPU Bridge Guide

This guide is the main reference for the published `bridge/` project.

It describes the current working system:

- `wingpu` runs on macOS
- a lightweight local gateway stays up on the Mac
- the actual model server runs inside WSL Ubuntu on a Windows host
- the runtime uses `llama.cpp`-style servers on an NVIDIA GPU
- apps on the Mac consume one stable local endpoint: `http://127.0.0.1:8000/v1`

## 1. Core Design

The system is intentionally split into a stable outer contract and an experimental inner runtime.

Stable outer contract:

- local base URL: `http://127.0.0.1:8000/v1`
- served model name: `qwen-local`
- consumer app: `the app`
- local gateway process stays alive even when the model is offloaded

Experimental inner runtime:

- model file can change
- runtime lane can change
- KV cache type can change
- benchmark settings can change
- the remote model can be automatically unloaded when idle

That design lets us experiment with Qwen, KV compression, and TurboQuant without having to keep reconfiguring the Mac app layer.

## 2. Architecture

### Request path

```text
+-------------------+      local API       +-------------------+      SSH tunnel     +-------------------+
| the app on Mac   | -------------------> | wingpu gateway    | ------------------> | Windows OpenSSH   |
+---------+---------+                      +---------+---------+                     +---------+---------+
          ^                                          |                                         |
          | OpenAI-compatible response               |                                         |
          |                                          | on demand                               |
          |                                          v                                         v
          |                                +---------+---------+      WSL entry      +---------+---------+
          +------------------------------- | backend local port | <------------------ | WSL Ubuntu        |
                                           | 127.0.0.1:18000    |    prompt/decode    | llama-server       |
                                           +-------------------+                     +-------------------+
```

### Component map

```text
+----------------------------- macOS -----------------------------+
|                                                                |
|  +-------------------+     +-------------------------------+   |
|  | wingpu CLI        |     | the app                      |   |
|  | Python + uv       |     | uses local OpenAI endpoint    |   |
|  +---------+---------+     +---------------+---------------+   |
|            |                                 |                   |
|            | reads                           | calls             |
|            v                                 v                   |
|  +-------------------+     +-------------------------------+   |
|  | ~/.config/wingpu/ |     | 127.0.0.1:8000/v1            |   |
|  | wingpu.local.toml |     | wingpu gateway               |   |
|  +-------------------+     +---------------+---------------+   |
+------------------------------------------------|---------------+
                                                 |
                                                 | backend tunnel
                                                 v
+--------------------------- Windows ---------------------------+
|  +-------------------+      +------------------------------+ |
|  | OpenSSH Server    | ---> | WSL launcher                 | |
|  +-------------------+      +---------------+--------------+ |
+----------------------------------------------|----------------+
                                               |
                                               v
+-------------------------- WSL Ubuntu -------------------------+
|  +-------------------+      +------------------------------+ |
|  | runtime lane      | ---> | llama-server / llama-bench   | |
|  | upstream or       |      | loads GGUF from model store  | |
|  | turboquant-cuda   |      +---------------+--------------+ |
|  +-------------------+                      |                |
+---------------------------------------------|----------------+
                                              |
                                              v
                                   +----------+-----------+
                                   | NVIDIA GPU           |
                                   +----------------------+
```

## 3. Key Parts Of The Repository

### `bridge/mac/`

This is the live controller project.

Important files:

- `bridge/mac/pyproject.toml`: Python package and `uv` project
- `bridge/mac/src/wingpu_cli/main.py`: main control-plane implementation
- `bridge/mac/mirror_remote_wsl_llama.sh`: optional remote mirror helper

### `bridge/config/`

This directory separates publish-safe defaults from machine-specific overrides.

- `wingpu.defaults.toml`: safe defaults that can be committed
- `qwen_gguf_catalog.json`: committed model catalog template
- `wingpu.local.toml`: local machine settings, gitignored

### `bridge/windows/`

PowerShell helpers for Windows host preparation:

- SSH/firewall hardening
- startup task setup
- native WSL runtime setup helpers

### `bridge/wsl/`

WSL-side scripts. The live one is `gpu_broker.py`, the shared GPU lease arbiter (see section 11);
wingpu uploads it to the GPU host and drives it over SSH. The remaining files
(`bridge_ctl.py`, `bridge_db.py`, `worker.py`, `boot.sh`, `install.sh`) are an older Docker job
queue kept only as reference; they are not part of the daily control path.

## 4. Configuration Model

The controller starts from committed defaults, then applies one machine-specific local config.

Config sources:

1. committed defaults from `bridge/config/wingpu.defaults.toml` or the packaged copy
2. active local override, selected in this order:
   - `WINGPU_CONFIG_FILE`, when set
   - `~/.config/wingpu/wingpu.local.toml`, or `$XDG_CONFIG_HOME/wingpu/wingpu.local.toml` when `XDG_CONFIG_HOME` is set
   - `bridge/config/wingpu.local.toml`, when run from this repo or `WINGPU_PROJECT_DIR`
3. explicit CLI or environment overrides

The home-directory config is the central machine config for both installed and repo-local `wingpu` commands. The project-local config is a fallback for isolated checkouts or one-off experiments.

Typical local fields to set:

- `connection.host`
- `connection.distro`
- `connection.api_key`
- `gateway.backend_local_port`
- `gateway.idle_timeout_seconds`
- `paths.remote_home`
- `paths.remote_src_root`
- `paths.remote_models_root`

Create or inspect the home-directory config with:

```bash
wingpu config init --global
wingpu config path --global
```

## 5. First-Time Setup

### Windows host

Required:

- OpenSSH Server enabled
- WSL2 with Ubuntu
- NVIDIA driver and WSL GPU support working
- sleep and hibernate disabled for unattended runs on AC power

### WSL Ubuntu

Required:

- build tools
- CUDA toolkit availability for native builds
- source trees under the configured remote source root
- GGUF model files inside the WSL filesystem

From the Mac, after config is in place:

```bash
wingpu admin build-prereqs
wingpu admin experiment-prereqs
wingpu admin cuda-toolkit
```

### Mac

Install the CLI (from the repo root, now `~/projects/bridge`):

```bash
uv tool install --from ./mac wingpu
```

Create the central user config:

```bash
wingpu config init --global
$EDITOR ~/.config/wingpu/wingpu.local.toml
```

### Model catalog vs selected model

This is the main behavior that tends to surprise people:

- `bridge/config/qwen_gguf_catalog.json` is the catalog of known models
- `default_model` inside that catalog is only the fallback
- `wingpu model set ...` persists the selected model and takes precedence over the catalog default

That means:

- if you add a new model entry manually, it should appear on the next `wingpu model list`
- if you change `default_model`, that does not override an already-selected model
- if a runtime is already warm, restart it to actually serve the newly selected model

Useful commands:

```bash
wingpu model list
wingpu model current
wingpu status
```

To make a manual catalog change take effect immediately:

```bash
wingpu model set YOUR_MODEL_NAME
wingpu stop
wingpu start
wingpu status
```

## 6. Daily Workflow

### Build runtime lanes

```bash
wingpu build upstream
wingpu build turboquant-cuda
```

### Select model, runtime, and KV settings

```bash
wingpu model list
wingpu model current
wingpu model set Qwen3.6-27B-MTP-UD-IQ2_M
wingpu runtime list
wingpu runtime set upstream-mtp
wingpu kv show
wingpu kv set --k q4_0 --v q4_0
```

### Start and verify

```bash
wingpu gateway start
wingpu start
wingpu status
wingpu models
```

Expected outcome:

- local API reachable on `127.0.0.1:8000`
- gateway remains available even if the remote model is later offloaded
- served model remains `qwen-local`
- `the app` does not need reconfiguration when you switch model or runtime lane
- `wingpu status` shows the selected GGUF entry, not just the stable served model id

### Benchmark

```bash
wingpu benchmark run
```

## 7. Client Contract

`the app` should point to:

- base URL: `http://127.0.0.1:8000/v1`
- provider type: OpenAI-compatible completions
- model id: `qwen-local`

The important design rule is that `the app` never needs to know the actual GGUF filename or the runtime lane.
The local gateway absorbs cold-start and idle-offload behavior.

## 8. Runtime Lanes

### `upstream`

Native CUDA build of upstream `llama.cpp`.

Use it when you want:

- baseline behavior
- clean comparison runs
- fewer experimental variables

This lane is not a drop-in replacement for `turboquant-cuda` while the active setup depends on TurboQuant-specific KV cache types such as `turbo3_0`.
Only switch the daily runtime back to official upstream once those cache types, or an equivalent long-context memory path, are supported there.

### `upstream-mtp`

Native upstream `llama.cpp` with Qwen3.6 MTP speculative decoding enabled through `--spec-type draft-mtp`.

Use it when you want:

- the default bridge runtime
- faster long-context decode throughput
- the current 27B MTP model path on the 16 GiB GPU

The current default uses `Qwen3.6-27B-MTP-UD-IQ2_M` with `q4_0` K/V cache and `--spec-draft-n-max 3`.

### `turboquant-cuda`

CUDA-focused TurboQuant fork used for KV-cache experiments.
This lane is a customized fork and should be updated conservatively: pull/rebuild its custom branch when that branch is available, but do not replace it with official upstream just because upstream is newer.

Use it when you want:

- longer-context KV-cache experiments
- memory-pressure comparisons
- TurboQuant vs baseline benchmarks

This lane is experimental and should be treated accordingly.

To switch back to TurboQuant from the MTP default:

```bash
wingpu runtime set turboquant-cuda
wingpu model set Qwen3.6-27B-UD-IQ2_M
wingpu kv set --k turbo3 --v turbo3
wingpu restart
```

## 9. Local-Only Working Areas

These areas are intentionally not part of the published surface:

- `bridge/config/wingpu.local.toml`
- `~/.config/wingpu/wingpu.local.toml`
- `bridge/profiles/*.json` except `template.json`
- `bridge/reports/`
- `bridge/remote-only/wsl-llama/`

This keeps the repo publishable while still leaving room for local notes, mirrored snapshots, and one-off profiles.

## 10. Troubleshooting Checklist

- `wingpu status` should show a healthy local gateway and, when warm, a backend tunnel and live remote runtime
- `wingpu gateway status` should tell you whether the model is loaded or offloaded
- `wingpu models` should return `qwen-local`
- if a build fails, rerun the relevant `wingpu admin ...` prerequisite command
- if the runtime starts but generation fails, compare the selected KV cache types against the active runtime lane's supported cache types
- if the installed `wingpu` command cannot find project config, run it from the repo or set `WINGPU_PROJECT_DIR`
- if `wingpu start` reports the GPU is held by another runtime, see section 11 (`wingpu broker status`, `--force-gpu`)

## 11. Shared GPU Broker

The Windows host has one NVIDIA GPU shared by several sibling bridges: this one (`wingpu`,
llama.cpp), plus `locatectl`, `ocrctl`, and `minerctl` in the neighboring repos. Each loads a
model that does not fit in VRAM alongside the others, but historically nothing coordinated them,
so two loaded at once meant an out-of-memory failure or an orphaned process pinning VRAM.

`wsl/gpu_broker.py` is the single authority on who holds the GPU. It is daemonless: state lives in
SQLite (`~/.gpu-bridge/broker.db`) guarded by an flock, and every command is a short-lived process.
There is no service to keep alive.

### How wingpu uses it

- Before starting `llama-server`, wingpu calls `acquire`. If another runtime holds the card, the
  broker evicts it (graceful POST shutdown if the runtime offers one, otherwise SIGTERM then
  SIGKILL on its pidfile) and waits for VRAM to actually drain (`nvidia-smi`) before granting.
- On idle-offload and on `wingpu stop`, wingpu calls `release`, so the holder is cleared.
- "Busy" is judged from real GPU utilization, so an actively running generation is not killed.
  If the holder is busy, `acquire` waits up to `broker.acquire_wait_seconds`, then fails the start
  with a clear message. Use `wingpu start --force-gpu` to preempt a busy holder immediately.

This is fail-open: if the broker is unreachable or not yet installed, wingpu prints a warning and
starts anyway, so a broker problem never blocks normal use. Set `broker.enabled = false` to opt out
entirely.

### Commands

```bash
wingpu broker status            # who holds the GPU, VRAM use, watchdog state
wingpu broker status --json     # same, machine-readable
wingpu broker install           # upload/refresh gpu_broker.py on the WSL host
wingpu broker evict             # manually free the GPU (evict the current holder)
wingpu broker watchdog start    # start the idle-eviction loop (see below)
wingpu broker watchdog stop
```

`wingpu status` also shows a "GPU broker" block (holder, VRAM, watchdog).

### Idle watchdog

`gpu_broker.py watchdog` is an optional background loop that evicts whichever runtime has gone idle,
freeing VRAM for the whole box. It is **off by default**. wingpu already has its own gateway
idle-offload; `locatectl` / `ocrctl` / `minerctl` (now broker-wired) do not, so the watchdog is the
cross-bridge idle-offload for them. It is left off by default because it will evict (kill) a warm,
idle model after `watchdog_idle_timeout_seconds`, which you may not want for a model you keep loaded.
Enable it when you want hands-off VRAM reclamation across all four bridges:

```bash
wingpu broker watchdog start    # idle timeout + poll come from [broker] config
wingpu broker watchdog stop
```

### Configuration (`[broker]`)

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `true` | Acquire/release around runtime start. Set `false` to disable. |
| `runtime_id` | `llama` | This bridge's identity to the broker. |
| `remote_dir` | `""` | Where `gpu_broker.py` + `broker.db` live on WSL. Empty uses `runtime_defaults.remote_state_dir` (`~/.gpu-bridge`). |
| `vram_floor_mb` | `2000` | VRAM (MB) at/below which the card counts as freed after an eviction. |
| `free_timeout_seconds` | `60` | Max wait for VRAM to drain after evicting a holder. |
| `acquire_wait_seconds` | `30` | Wait this long for a busy holder to go idle before failing (use `--force-gpu` to skip). |
| `busy_util_pct` | `15` | GPU utilization at/above which the holder counts as busy. |
| `watchdog_idle_timeout_seconds` | `900` | Watchdog evicts a holder idle this long. |
| `watchdog_poll_seconds` | `15` | Watchdog poll interval. |

### State on the WSL host

```text
~/.gpu-bridge/
  gpu_broker.py          uploaded broker program
  broker.db              SQLite: current holder + event log
  broker.lock            flock for mutual exclusion
  run/watchdog.pid       watchdog pid (when running)
  logs/watchdog.log      watchdog output
```

### Troubleshooting

- "GPU is held by 'locate' which is busy": another bridge is actively using the card. Wait, stop
  that bridge, or run `wingpu start --force-gpu`.
- Stuck holder after a crash: `wingpu broker status` shows `alive=false`; the next `acquire`
  (or `wingpu broker evict`) clears it.
- To see the raw event log on the host: `sqlite3 ~/.gpu-bridge/broker.db 'select * from events order by id desc limit 20'`.
