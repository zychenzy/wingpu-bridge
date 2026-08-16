# GPU Bridge

`bridge/` is a Mac-controlled remote GPU workspace for running Qwen with `llama.cpp` on a Windows host through WSL, then exposing a stable OpenAI-compatible endpoint back to macOS.

The published project keeps one stable app contract:

- Base URL: `http://127.0.0.1:8000/v1`
- Served model id: `qwen-local`
- App target: `the app` on macOS

## What This Project Covers

- Mac-side control with the `wingpu` CLI
- A lightweight always-on local gateway on macOS
- Windows + WSL + NVIDIA runtime management over SSH
- Native `llama.cpp` runtime lanes, with and without MTP speculative decoding
- GGUF model catalog and model switching without changing app-side config
- Automatic idle offload of the remote model runtime
- Local benchmark and experiment workflows for long-context Qwen use
- A shared GPU broker so this bridge and its siblings (`locatectl`, `ocrctl`, `minerctl`) never fight over VRAM

## System Shape

```text
+-----------------------+                                    +----------------------+
| macOS                 |                                    | Windows host         |
| - the app            | ---> 127.0.0.1:8000/v1 ---------> | - OpenSSH            |
| - wingpu gateway      |                                    | - WSL launcher       |
+-----------+-----------+                                    +----------+-----------+
            |                                                           |
            | auto-start / idle-offload                                 |
            v                                                           v
            |                                                +----------+-----------+
            +----------------------------------------------- | WSL Ubuntu           |
                                                              | - llama-server       |
                                                              | - runtime lane       |
                                                              +----------+-----------+
                                                                         |
                                                                         v
                                                              +----------+-----------+
                                                              | NVIDIA GPU           |
                                                              +----------------------+
```

## Quick Start

1. Install the CLI (run from the repo root, `~/projects/bridge`):

```bash
uv tool install --from ./mac wingpu
```

(This repo now lives at `~/projects/bridge`, a sibling of `~/projects/pdf-to-md` and
`~/projects/locateanything`. If you previously installed from `autoresearch/bridge/mac`,
reinstall with `uv tool install --force --from ./mac wingpu` from the new path.)

2. Create the central user config:

```bash
wingpu config init --global
$EDITOR ~/.config/wingpu/wingpu.local.toml
```

`wingpu` uses `~/.config/wingpu/wingpu.local.toml`, or `WINGPU_CONFIG_FILE` if set.
The repo-local `bridge/config/wingpu.local.toml` is only a fallback when the central config is missing.

3. Build or select a runtime and model:

```bash
wingpu build upstream
wingpu runtime set upstream
wingpu model set Qwen3.8-27B-UD-Q3_K_XL
wingpu kv set --k q4_0 --v q4_0
```

4. Start and verify the local endpoint:

```bash
wingpu gateway start
wingpu start
wingpu status
wingpu models
```

## Model Selection Rules

`wingpu` keeps two separate ideas:

- `bridge/config/qwen_gguf_catalog.json` defines the catalog and its `default_model`
- `wingpu model set ...` writes the persisted selected model used for normal starts

In practice:

- adding or editing catalog entries is picked up on the next `wingpu` command
- changing `default_model` only matters when no selected model is already pinned
- `wingpu start` uses the selected model first, then falls back to `default_model`

If you changed the catalog manually and want the new model active right away:

```bash
wingpu model set YOUR_MODEL_NAME
wingpu stop
wingpu start
wingpu status
```

Useful checks:

```bash
wingpu model list
wingpu model current
```

## Shared GPU Broker

The Windows host has one GPU shared by this bridge and its siblings. `wsl/gpu_broker.py` is a
daemonless lease arbiter (SQLite + flock on the WSL host) that lets only one runtime hold the GPU
at a time. Before starting `llama-server`, wingpu acquires the GPU, evicting whatever holds it and
waiting for VRAM to drain; it releases on idle-offload and `stop`. A busy holder (judged from GPU
utilization) is not killed unless you pass `--force-gpu`.

```bash
wingpu broker status            # who holds the GPU, VRAM, watchdog state
wingpu broker install           # upload/refresh the broker on the WSL host
wingpu broker evict             # manually free the GPU
wingpu start --force-gpu        # preempt a busy holder
```

It is fail-open (a broker problem never blocks a start) and can be turned off with
`broker.enabled = false`. Full details: [Project Guide, section 11](docs/README.md).

## Read This Next

- [Project Guide](docs/README.md)

## Switching Runtime Lanes

The default lane is plain `upstream`. To switch the bridge to the MTP speculative-decoding lane:

```bash
wingpu runtime set upstream-mtp
wingpu model set Qwen3.8-27B-UD-Q3_K_XL
wingpu kv set --k q4_0 --v q4_0
wingpu restart
```

Both lanes build from the same `llama.cpp` checkout. See
[Project Guide, section 8](docs/README.md) for the trade-offs.

## Local-Only Areas

These paths are intentionally kept out of git so the repo can stay publishable while your machine-specific work stays local:

- `bridge/config/wingpu.local.toml`
- `~/.config/wingpu/wingpu.local.toml`
- `bridge/profiles/*.json` except `template.json`
- `bridge/reports/`
- `bridge/remote-only/wsl-llama/`
- `bridge/mac/.venv/`

## Repository Layout

- `bridge/config/`: publish-safe defaults and model catalog
- `bridge/docs/`: durable project documentation
- `bridge/mac/`: Python `wingpu` controller and utilities
- `bridge/profiles/`: local profile templates
- `bridge/windows/`: Windows-side setup scripts
- `bridge/wsl/`: the GPU broker (`gpu_broker.py`) plus older reference utilities
