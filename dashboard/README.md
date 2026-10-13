# GPU dashboard

A local operations page for the wingpu GPU host. It shows the card, the lease, the next idle
eviction, llama.cpp model and lane choices, and the sibling workloads (OCR, LocateAnything,
MinerU, live transcription), and it can start and stop them.

It listens on `127.0.0.1:18081` only. Any other bind address is refused, and a request whose TCP
peer is not loopback is rejected. Actions are an allowlist of `wingpu`, `ocrctl`, `locatectl`,
`remote-asr`, and `minerctl` argument vectors. Nothing is passed to a shell. Evict needs a second
confirmation in the page.

## Run

From the repo root, with the sibling CLIs on `PATH` when you want live data:

```bash
PYTHONPATH=dashboard/src python3 -m gpu_dashboard
PYTHONPATH=dashboard/src python3 -m gpu_dashboard --mock
```

`--mock` serves the sample snapshot and does not call SSH. Open `http://127.0.0.1:18081/`.

Install the command with uv:

```bash
uv tool install --from ./dashboard gpu-dashboard
gpu-dashboard
```

`--project-dir` points at a checkout that contains `config/qwen_gguf_catalog.json` when the
dashboard is installed outside the repo. From a source tree it finds that file by walking parents.

The Python process serves the built page from `src/gpu_dashboard/static`. Running the dashboard
does not need Node. Dark is the default. Light is the header toggle, or `?theme=light`. The choice
is stored in `localStorage` under `gpu-dashboard-theme`.

## Frontend

The UI is a React app in `dashboard/web` (Vite, Tailwind, shadcn/ui). Rebuild it, from the repo
root, with:

```bash
cd dashboard/web && npm install && npm run build
```

That writes `dashboard/src/gpu_dashboard/static`, which is the tree the Python server already
serves. The build is committed, so a checkout can run the page without Node. Fonts, CSS, and
JavaScript are local files. Nothing is loaded from a CDN.

## Design

The layout follows the shadcn dashboard: a collapsible sidebar, overview cards, a workload table,
an event table, and log tabs. Cards use a 1px ring and no drop shadow. Radius is 8px. Spacing
steps are 4px and 8px. The typeface is Geist, with Geist Mono for times, process ids, and logs.
Figures use tabular numerals.

Colour stays neutral. The primary button is near-black in light mode and near-white in dark mode.
Emerald, amber, and red appear only on status (alive, busy, VRAM at or above 92%, destructive
actions). Motion is limited to colour transitions of about 150ms.

References: [Vercel Geist](https://vercel.com/geist) and the [Geist font](https://github.com/vercel/geist-font)
(the webfonts under `dashboard/web/src/fonts`, SIL OFL), the [shadcn/ui](https://ui.shadcn.com)
dashboard and sidebar blocks, [Tremor](https://tremor.so) for status colour on metrics, and the
dense neutral consoles of Linear, Railway, Fly.io, and Tailscale. When mock mode is on, the header
badge and the line under it say the numbers are example data.

## launchd

`launchd/com.zychenzy.gpu-dashboard.plist` is a template. Replace the binary path with the
`gpu-dashboard` uv installed (often `~/.local/bin/gpu-dashboard`), then:

```bash
cp dashboard/launchd/com.zychenzy.gpu-dashboard.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.zychenzy.gpu-dashboard.plist
```

The plist does not start the broker watchdog. The watchdog still kills a warm idle model, so it
stays off until you press Start on the page or run `wingpu broker watchdog start`.

## Container

`Containerfile` builds a loopback-only image. A container does not have the Mac's SSH agent or
ControlMaster sockets, so it cannot drive the Windows host unless you plumb those in yourself.
Prefer launchd. The image still refuses to bind `0.0.0.0`.

```bash
podman build -f dashboard/Containerfile -t gpu-dashboard dashboard
podman run --rm --network host gpu-dashboard
```

`--network host` is a Linux podman/docker feature. Docker Desktop on macOS does not share the
host's loopback that way.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{"ok", "mock", "bind"}` |
| GET | `/api/status` | GPU, lease, watchdog deadline, gateway, catalog, services, events |
| GET | `/api/events?limit=` | Broker event ring |
| GET | `/api/logs/{source}?n=` | `llama`, `gateway`, `watchdog`, `ocr`, `locate`, `asr`, `mineru` |
| POST | `/api/actions/{name}` | Allowlisted action. Body is JSON. |
| GET | `/api/jobs/{id}` | A live action that is still running |

`POST /api/actions/evict` requires `{"confirm": true}`. `llama-apply` takes `model`, `lane`,
`cache_k`, and `cache_v`, each checked against the catalog. `mineru-run` takes an absolute `.pdf`
path with no `..` segment. `asr-start` takes `engine`: `whisper`, `qwen3`, or `both`.
