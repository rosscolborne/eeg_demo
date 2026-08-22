from __future__ import annotations

from dataclasses import asdict
import logging
import threading

import numpy as np

from .config import DEFAULT_BANDS, DEFAULT_PROCESSING, FrequencyBand, ProcessingConfig
from .models import BandPowerFeatures

logger = logging.getLogger(__name__)
_ml_models: dict[str, object] = {}
_ml_lock = threading.Lock()


def build_eeg_window(data: np.ndarray, eeg_channels: list[int], samples: int) -> np.ndarray | None:
    if data.size == 0 or samples <= 0:
        return None
    if data.shape[1] < samples:
        return None

    window = data[np.array(eeg_channels), -samples:].astype(float, copy=True)
    if not np.isfinite(window).all():
        return None
    return np.ascontiguousarray(window)


def preprocess_eeg_window(
    window: np.ndarray,
    sampling_rate: int,
    config: ProcessingConfig = DEFAULT_PROCESSING,
) -> np.ndarray:
    processed = np.ascontiguousarray(window.astype(float, copy=True))
    try:
        from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes

        for channel_index in range(processed.shape[0]):
            channel = processed[channel_index]
            DataFilter.detrend(channel, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                channel,
                sampling_rate,
                config.bandpass_low_hz,
                config.bandpass_high_hz,
                config.filter_order,
                FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                0.0,
            )
            DataFilter.perform_bandstop(
                channel,
                sampling_rate,
                config.notch_low_hz,
                config.notch_high_hz,
                config.filter_order,
                FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                0.0,
            )
    except Exception:
        processed -= processed.mean(axis=1, keepdims=True)
    return processed


def extract_band_power_features(
    window: np.ndarray,
    sampling_rate: int,
    bands: tuple[FrequencyBand, ...] = DEFAULT_BANDS,
    channel_ids: list[str] | None = None,
) -> BandPowerFeatures | None:
    if window.size == 0 or window.shape[1] < max(16, sampling_rate):
        return None
    if not np.isfinite(window).all():
        return None

    ids = resolve_channel_ids(window.shape[0], channel_ids)
    per_channel: dict[str, dict[str, float]] = {}

    try:
        from brainflow.data_filter import DataFilter, WindowOperations

        nfft = DataFilter.get_nearest_power_of_two(window.shape[1])
        if nfft >= window.shape[1]:
            nfft = max(16, nfft // 2)

        for channel_index, channel in enumerate(window):
            psd = DataFilter.get_psd_welch(
                np.ascontiguousarray(channel),
                nfft,
                nfft // 2,
                sampling_rate,
                WindowOperations.HANNING.value,
            )
            per_channel[ids[channel_index]] = {
                band.id: float(DataFilter.get_band_power(psd, band.low_hz, band.high_hz))
                for band in bands
            }
    except Exception:
        per_channel = fallback_band_powers(window, sampling_rate, bands, ids)

    absolute = mean_band_powers(per_channel, bands)
    relative = relative_band_powers(absolute)
    theta = relative.get("theta", 0.0)
    alpha = relative.get("alpha", 0.0)
    beta = relative.get("beta", 0.0)

    return BandPowerFeatures(
        absolute=absolute,
        relative=relative,
        ratios={
            "alphaTheta": log_ratio(alpha, theta),
            "betaTheta": log_ratio(beta, theta),
            "thetaBeta": log_ratio(theta, beta),
            "betaOverAlphaTheta": log_ratio(beta, alpha + theta),
        },
        per_channel=per_channel,
        window_seconds=window.shape[1] / sampling_rate,
        method="brainflow_welch_psd",
    )


def extract_brainflow_mental_state(
    window: np.ndarray,
    sampling_rate: int,
    metric: str,
) -> float | None:
    if window.size == 0 or window.shape[1] < max(16, sampling_rate):
        return None
    if not np.isfinite(window).all():
        return None

    try:
        from brainflow.data_filter import DataFilter
        from brainflow.ml_model import (
            BrainFlowClassifiers,
            BrainFlowMetrics,
            BrainFlowModelParams,
            MLModel,
        )

        metric_id = {
            "mindfulness": BrainFlowMetrics.MINDFULNESS.value,
            "restfulness": BrainFlowMetrics.RESTFULNESS.value,
        }.get(metric)
        if metric_id is None:
            return None

        eeg_rows = list(range(window.shape[0]))
        avg_band_powers, _ = DataFilter.get_avg_band_powers(
            np.ascontiguousarray(window),
            eeg_rows,
            sampling_rate,
            False,
        )
        with _ml_lock:
            model = _ml_models.get(metric)
            if model is None:
                model = MLModel(
                    BrainFlowModelParams(
                        metric_id,
                        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value,
                    ),
                )
                model.prepare()
                _ml_models[metric] = model
            prediction = model.predict(avg_band_powers)

        if len(prediction) == 0 or not np.isfinite(prediction[0]):
            return None
        return float(np.clip(prediction[0], 0.0, 1.0))
    except Exception as exc:
        logger.warning("BrainFlow %s metric extraction failed: %s", metric, exc)
        return None


def extract_brainflow_mindfulness(window: np.ndarray, sampling_rate: int) -> float | None:
    return extract_brainflow_mental_state(window, sampling_rate, "mindfulness")


def extract_brainflow_restfulness(window: np.ndarray, sampling_rate: int) -> float | None:
    return extract_brainflow_mental_state(window, sampling_rate, "restfulness")


def config_metadata(config: ProcessingConfig = DEFAULT_PROCESSING) -> dict[str, object]:
    return {
        "processing": asdict(config),
        "bands": [asdict(band) for band in DEFAULT_BANDS],
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(1e-9, denominator))


def log_ratio(numerator: float, denominator: float) -> float:
    return float(np.log(max(1e-9, numerator) / max(1e-9, denominator)))


def resolve_channel_ids(channel_count: int, channel_ids: list[str] | None) -> list[str]:
    if channel_ids and len(channel_ids) == channel_count:
        return [
            (channel_id.strip().lower() or f"ch{index}")
            for index, channel_id in enumerate(channel_ids)
        ]
    return [f"ch{index}" for index in range(channel_count)]


def mean_band_powers(
    per_channel: dict[str, dict[str, float]],
    bands: tuple[FrequencyBand, ...],
) -> dict[str, float]:
    absolute: dict[str, float] = {}
    for band in bands:
        values = [max(0.0, channel.get(band.id, 0.0)) for channel in per_channel.values()]
        absolute[band.id] = float(np.mean(values)) if values else 0.0
    return absolute


def relative_band_powers(absolute: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in absolute.values())
    return {
        key: (max(0.0, value) / total if total > 0 else 0.0)
        for key, value in absolute.items()
    }


def fallback_band_powers(
    window: np.ndarray,
    sampling_rate: int,
    bands: tuple[FrequencyBand, ...],
    channel_ids: list[str],
) -> dict[str, dict[str, float]]:
    freqs = np.fft.rfftfreq(window.shape[1], d=1.0 / sampling_rate)
    spectrum = np.abs(np.fft.rfft(window - window.mean(axis=1, keepdims=True), axis=1)) ** 2
    per_channel: dict[str, dict[str, float]] = {}
    for channel_index, channel_id in enumerate(channel_ids):
        per_channel[channel_id] = {}
        for band in bands:
            mask = (freqs >= band.low_hz) & (freqs <= band.high_hz)
            per_channel[channel_id][band.id] = (
                float(np.mean(spectrum[channel_index, mask])) if mask.any() else 0.0
            )
    return per_channel
