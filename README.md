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
