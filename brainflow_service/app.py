from __future__ import annotations

import os
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_PROCESSING, DEVICE_CONFIGS
from .dsp import (
    extract_band_power_features,
    extract_brainflow_mindfulness,
    extract_brainflow_restfulness,
    preprocess_eeg_window,
)
from .metrics import compute_neurofeedback_scores, normalize_brainflow_score
from .models import SignalFeatures
from .runtime import SessionStore, sse_event


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    device_id: str = Field(alias="deviceId")
    mac_address: str | None = Field(default=None, alias="macAddress")
    serial_number: str | None = Field(default=None, alias="serialNumber")


class StartSessionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    state: Literal["connected"]
    device_info: dict = Field(alias="deviceInfo")


class AnalyzeWindowRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sample_rate_hz: float = Field(alias="sampleRateHz")
    samples: list[list[float]]


class AnalyzeWindowResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    features: SignalFeatures | None


app = FastAPI(title="EEG Demo BrainFlow Service")
store = SessionStore()

# Additional origins (comma-separated) that may call this service, on top of
# the bundled app's own localhost dev ports. Set this so a different
# front-end -- served from another host/port -- can reach the API, e.g.:
#   EEG_BRAINFLOW_CORS_ORIGINS=https://my-other-frontend.example.com
_extra_cors_origins = [
    origin.strip()
    for origin in os.environ.get("EEG_BRAINFLOW_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", *_extra_cors_origins],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):517[0-9]",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/devices")
def devices() -> list[dict[str, str]]:
    return [
        {
            "id": config.id,
            "label": config.label,
            "mode": config.mode,
            "boardId": config.board_id_name,
        }
        for config in DEVICE_CONFIGS.values()
    ]


@app.post("/analyze-window")
def analyze_window(request: AnalyzeWindowRequest) -> AnalyzeWindowResponse:
    sample_rate = int(round(request.sample_rate_hz))
    if sample_rate <= 0:
        raise HTTPException(status_code=400, detail="sampleRateHz must be positive.")
    if not request.samples:
        raise HTTPException(status_code=400, detail="samples must not be empty.")

    window = np.asarray(request.samples, dtype=float).T
    if window.ndim != 2 or window.shape[0] == 0 or window.shape[1] == 0:
        raise HTTPException(status_code=400, detail="samples must be row-major EEG values.")
    if not np.isfinite(window).all():
        raise HTTPException(status_code=400, detail="samples contain non-finite values.")

    processed = preprocess_eeg_window(window, sample_rate, DEFAULT_PROCESSING)
    band_powers = extract_band_power_features(processed, sample_rate)
    brainflow_mindfulness = extract_brainflow_mindfulness(window, sample_rate)
    brainflow_restfulness = extract_brainflow_restfulness(window, sample_rate)
    features = None
    if band_powers or brainflow_mindfulness is not None or brainflow_restfulness is not None:
        # This endpoint is stateless (no session to smooth across), so these
        # are single-window scores -- callers streaming a session should use
        # `/sessions/{id}/stream` instead, which returns the same finished
        # scores with the bundled app's EMA smoothing applied.
        neurofeedback = compute_neurofeedback_scores(
            theta_power=band_powers.absolute.get("theta", 0.0) if band_powers else 0.0,
            alpha_power=band_powers.absolute.get("alpha", 0.0) if band_powers else 0.0,
            beta_power=band_powers.absolute.get("beta", 0.0) if band_powers else 0.0,
        )
        features = SignalFeatures(
            bandPowers=band_powers,
            brainflowConcentration=brainflow_mindfulness,
            brainflowRestfulness=brainflow_restfulness,
            mindfulnessScore=normalize_brainflow_score(brainflow_mindfulness),
            restfulnessScore=normalize_brainflow_score(brainflow_restfulness),
            focusScore=neurofeedback.focus_score,
            relaxScore=neurofeedback.relax_score,
        )

    return AnalyzeWindowResponse(features=features).model_dump(by_alias=True)


@app.post("/sessions")
def start_session(request: StartSessionRequest) -> StartSessionResponse:
    if request.device_id not in DEVICE_CONFIGS:
        raise HTTPException(status_code=404, detail="Unknown BrainFlow device.")
    config = DEVICE_CONFIGS[request.device_id]

    session = None
    device_info = None
    last_error: Exception | None = None
    for attempt in range(1, config.startup_attempts + 1):
        session = store.create(
            request.device_id,
            mac_address=request.mac_address,
            serial_number=request.serial_number,
        )
        try:
            device_info = session.prepare()
            session.start()
            break
        except Exception as exc:
            last_error = exc
            store.stop(session.id)
            session = None
            if attempt < config.startup_attempts:
                time.sleep(config.startup_retry_delay_seconds)

    if not session or not device_info:
        detail = str(last_error) if last_error else "Unable to start BrainFlow session."
        raise HTTPException(status_code=500, detail=detail)

    return StartSessionResponse(
        sessionId=session.id,
        state="connected",
        deviceInfo=device_info.model_dump(by_alias=True),
    )


@app.get("/sessions/{session_id}/stream")
async def stream_session(session_id: str) -> StreamingResponse:
    try:
        session = store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown BrainFlow session.") from exc

    async def events():
        yield sse_event("state", {"state": "streaming"})
        try:
            async for frame in session.frames():
                yield sse_event("signalFrame", frame)
        except Exception as exc:
            yield sse_event("error", {"message": str(exc)})
        finally:
            store.stop(session_id)
            yield sse_event("state", {"state": "disconnected"})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}")
def stop_session(session_id: str) -> dict[str, str]:
    store.stop(session_id)
    return {"state": "disconnected"}
