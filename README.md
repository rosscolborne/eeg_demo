# EEG Demo

Proof-of-concept EEG web app for Muse Athena acquisition through a local BrainFlow service and a Vite React frontend.

## Requirements

- Linux
- Node.js and npm
- Python with `uv`
- Chrome/Chromium for local testing
- Muse Athena headset, or the BrainFlow Synthetic provider for testing without hardware

## Install

Install frontend dependencies:

```bash
npm install
```

Install Python dependencies:

```bash
uv sync --extra test
```

## Run

Start the BrainFlow service in one terminal:

```bash
npm run brainflow
```

Start the Vite frontend in a second terminal:

```bash
npm run dev
```

Open the URL shown by Vite, usually:

```text
http://127.0.0.1:5173/
```

The frontend expects the BrainFlow service at:

```text
http://127.0.0.1:8000
```

If you run the service on a different port, start Vite with:

```bash
VITE_BRAINFLOW_SERVICE_URL=http://127.0.0.1:8001 npm run dev
```

## Testing Without The Headset

Use the device selector in the app and choose:

```text
BrainFlow Synthetic
```

This uses BrainFlow's synthetic board and feeds the same normalized frontend pipeline as the live Muse provider.

## Recording And Replay

Recording is controlled in the browser.

1. Connect a live or synthetic provider.
2. Click `Record`.
3. Click `Stop Recording`.
4. Click `Download`.

Replay only supports JSON files downloaded by this app. Old raw BrainFlow streamer CSV files are not supported by the app replay provider.

To replay:

1. Select `Replay Recording`.
2. Click `Upload File`.
3. Choose a downloaded app JSON recording.

## Using This Service From Another Front-End

`brainflow_service` is a standalone Python package (own `pyproject.toml`, no
dependency on the React app) that computes four finished, display-ready
scores per window — `mindfulnessScore`, `restfulnessScore`, `focusScore`,
`relaxScore`, all 0-100 — so any front-end, in any language, can consume them
without reimplementing the scoring itself.

Most consumers don't need to install anything: run this service as a
standalone process and call it over HTTP/SSE from any language (see below).
If you're embedding the scoring code directly into another **Python**
backend instead, install it from GitHub with `uv` or `pip`, pinned to a
release tag:

```bash
uv add "git+https://github.com/rosscolborne/eeg_demo.git@v0.1.0#subdirectory=brainflow_service"
```

```bash
pip install "git+https://github.com/rosscolborne/eeg_demo.git@v0.1.0#subdirectory=brainflow_service"
```

### Releasing a new version

`pyproject.toml`'s `version` field is the source of truth. To cut a release:

1. Bump `version` in [pyproject.toml](pyproject.toml).
2. Commit, then tag to match and push the tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.

### Updating to a new version

A git dependency pinned to a tag (`@v0.1.0`) is a fixed, reproducible
target — pushing new commits or new tags to this repo does **not** affect
consumers already installed, and plain `uv sync` will never fetch a newer
tag on its own. To move to a new release, the consumer changes which tag
they depend on and re-resolves:

```bash
uv add "git+https://github.com/rosscolborne/eeg_demo.git@vX.Y.Z#subdirectory=brainflow_service"
```

(This is a deliberate action on their end, not something that happens
automatically — that's the point of pinning to a tag rather than `main`.)

Then run it as a service (it's a FastAPI app; run it with any ASGI server):

```bash
uvicorn brainflow_service.app:app --host 0.0.0.0 --port 8000
```

By default CORS only allows the bundled app's own `localhost:517x` dev ports.
To allow a different front-end's origin, set:

```bash
EEG_BRAINFLOW_CORS_ORIGINS=https://my-other-frontend.example.com uvicorn brainflow_service.app:app --port 8000
```

Two ways to get scores:

- `POST /analyze-window` — stateless, single-window scores (no smoothing,
  since there's no session to smooth across). Send `{ sampleRateHz, samples }`
  where `samples` is row-major EEG values; the response's `features` object
  carries the four scores plus raw band powers.
- `GET /sessions/{id}/stream` (after `POST /sessions`) — an SSE stream of
  normalized `signalFrame` events. Each frame's `features` object carries the
  same four scores, smoothed with the same slow EMA the bundled app displays,
  so a live session needs no client-side scoring logic at all.

The scoring logic itself (`brainflow_service/metrics.py`) is also usable
directly as a library — `compute_neurofeedback_scores`, `MindStateSmoother` —
if you want to embed it in your own Python backend instead of running this
one as a service.

## Run Checks

Build the frontend:

```bash
npm run build
```

Run Python tests:

```bash
npm run test:python
```

## Troubleshooting

If ports are already in use, stop the old dev server with `Ctrl+C` in the terminal that started it.

Check port listeners:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

If Chrome is showing stale behavior after code changes, restart the BrainFlow service and hard-refresh the browser tab with `Ctrl+Shift+R`.
