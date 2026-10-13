# Migrating a sibling CLI onto gpubroker

`gpubroker` is the shared, dependency-free package for the GPU lease. It lives in this repo
at `gpubroker/`. wingpu is the only program that installs or upgrades the broker on the WSL host.
The other bridges (locatectl, ocrctl, minerctl, remote-asr) should import the package and stop
shipping their own `gpu_broker.py`.

The public llama contract does not change: `http://127.0.0.1:8000/v1` and model id `qwen-local`.

## Dependency line

Add this to the sibling `pyproject.toml`. The path is for a checkout that sits next to this repo
when this repo's directory is named `bridge` (two levels up from that sibling's `pyproject.toml`):

```toml
dependencies = ["gpubroker"]

[tool.uv.sources]
gpubroker = { path = "../../bridge/gpubroker", editable = true }
```

unlimited-ocr, mineru, and live-transcription keep their Python project one directory deeper, under
`pdf-to-md`. From those `pyproject.toml` files the path is:

```toml
gpubroker = { path = "../../../bridge/gpubroker", editable = true }
```

Without a local checkout:

```toml
gpubroker = { git = "https://github.com/zychenzy/wingpu-bridge", subdirectory = "gpubroker" }
```

The package has no third-party dependencies. The file wingpu uploads is
`gpubroker/src/gpubroker/broker.py`. It is stdlib-only, so WSL does not need the package installed.

## Files to delete

locateanything, unlimited-ocr, and mineru:

- `wsl/gpu_broker.py`
- `mac/src/<cli>/broker_client.py`
- the hatch `force-include` line that copies `gpu_broker.py` into the wheel

live-transcription:

- the vendored broker (`live_transcription/resources/gpu_broker.py`, or whichever path the package data points at)
- `live_transcription/broker_client.py`
- the package-data entry for that broker file

Do not keep a runner that places the broker script on the SSH command line. `install` sends the
file on stdin (`runner(script, *, stdin=bytes)`). A `bash -lc "<script>"` runner will hit Windows'
command-line limit once the script is large, and siblings should not be installing the broker anyway.

Replace each project's `broker install` command with a message that tells the user to run
`wingpu broker install`. `LeaseSession` already does this when `allow_install` is left at its
default, `False`.

## What to call

```python
from gpubroker.config import BrokerConfig, load_section
from gpubroker.lease import ActivityTouch, LeaseSession, TouchWhileActive
from gpubroker.ssh import SshTarget, run_wsl_script, serve_tcp_relay
from gpubroker import runtimes

session = LeaseSession(runner, config, remote_dir=remote_dir)  # allow_install stays False
session.acquire(pidfile=pidfile, label="ocrctl", vram_mb=8000)
session.release()
```

`runner` is `runner(script, *, timeout=None, stdin=None) -> (returncode, stdout)`.
`gpubroker.ssh.run_wsl_script` honors `timeout` (exit 124 on expiry). Copy-pasted runners that
accept `timeout` and then ignore it should be deleted in favor of this one.

`load_section` reads a `[broker]` table and ignores unknown keys.

## Runtime ids

`acquire` with the same runtime id reasserts the lease and does not stop the process that already
holds it. Two servers that share an id both stay resident.

| Workload | Runtime id |
|---|---|
| llama.cpp (wingpu) | `llama` |
| LocateAnything | `locate` |
| Unlimited-OCR | `ocr` |
| MinerU | `mineru` |
| Whisper ASR | `live-asr` |
| Qwen ASR | `live-asr-qwen` |

The constants are `gpubroker.runtimes`. Give the Qwen ASR server `live-asr-qwen`. Do not keep both
engines on `live-asr`.

## Touch, or the watchdog will evict an in-use model

Call `touch` from the code that is doing work:

- one-shot requests: `ActivityTouch(...).ping()` at the start of the request
- a stream: `TouchWhileActive` started when the stream opens and stopped when it closes

A timer that touches for the whole life of the process keeps the idle clock fresh forever, and
the watchdog will never offload that model. wingpu touches from the gateway on each request, at
most once every `touch_interval_seconds` (30). It does not touch while the model is merely loaded.

## Relay instead of `ssh -L`

OCR and live transcription open a plain `ssh -L` tunnel. On this Windows host, TCP into WSL over
the Hyper-V NAT is dropped after about 15 seconds. wingpu and locatectl already avoid that by
spawning `ssh … wsl -- nc 127.0.0.1 PORT` per connection.

Use `gpubroker.ssh.serve_tcp_relay`. It listens on loopback only and refuses `0.0.0.0`. The backend
command is `relay_backend_command`, which dials `nc` inside WSL.

## After the switch

On the Mac:

```bash
wingpu broker install
wingpu broker version --json    # version 1.0.0, protocol 2
```

Then start one sibling and `wingpu broker status --json`. The holder should be that sibling's
runtime id, and `version` should be the package version. A sibling that still uploads its own
`gpu_broker.py` will overwrite this file. That is the failure this split is meant to stop.
