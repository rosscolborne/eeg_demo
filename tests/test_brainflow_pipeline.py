from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from brainflow_service.config import DEFAULT_PROCESSING, DEVICE_CONFIGS
from brainflow_service.dsp import (
    build_eeg_window,
    extract_band_power_features,
    extract_brainflow_mindfulness,
    preprocess_eeg_window,
)
from brainflow_service.models import SignalFeatures
from brainflow_service.runtime import BrainFlowSession


def sine_window(freq_hz: float = 10.0, sample_rate: int = 256, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    signal = np.sin(2 * math.pi * freq_hz * t) * 20.0
    return np.vstack([signal, signal * 0.9, signal * 1.1, signal])


def test_window_construction_extracts_eeg_rows() -> None:
    data = np.arange(6 * 100, dtype=float).reshape(6, 100)
    window = build_eeg_window(data, [1, 3], 20)

    assert window is not None
    assert window.shape == (2, 20)
    assert np.array_equal(window[0], data[1, -20:])
    assert np.array_equal(window[1], data[3, -20:])


def test_window_construction_rejects_insufficient_data() -> None:
    data = np.arange(6 * 10, dtype=float).reshape(6, 10)

    assert build_eeg_window(data, [1, 3], 20) is None


def test_window_construction_rejects_nan() -> None:
    data = np.arange(6 * 100, dtype=float).reshape(6, 100)
    data[1, 99] = np.nan

    assert build_eeg_window(data, [1], 20) is None


def test_preprocess_keeps_shape_and_finite_values() -> None:
    window = sine_window()
    processed = preprocess_eeg_window(window, 256, DEFAULT_PROCESSING)

    assert processed.shape == window.shape
    assert np.isfinite(processed).all()


def test_band_power_extracts_expected_bands() -> None:
    features = extract_band_power_features(sine_window(freq_hz=10), 256)

    assert features is not None
    assert features.method == "brainflow_welch_psd"
    assert features.absolute["alpha"] > features.absolute["theta"]
    assert features.absolute["alpha"] > features.absolute["beta"]
    assert "betaOverAlphaTheta" in features.ratios


def test_band_power_handles_exact_power_of_two_window() -> None:
    features = extract_band_power_features(sine_window(freq_hz=10, sample_rate=256, seconds=2), 256)

    assert features is not None
    assert features.window_seconds == 2


def test_invalid_feature_window_returns_none() -> None:
    assert extract_band_power_features(np.array([[1.0, float("inf")]]), 256) is None


def test_brainflow_mindfulness_extracts_bounded_score() -> None:
    pytest.importorskip("brainflow")

    score = extract_brainflow_mindfulness(sine_window(freq_hz=10, seconds=4), 256)

    assert score is not None
    assert 0 <= score <= 1


def test_signal_features_serialize_for_frontend() -> None:
    features = SignalFeatures(brainflowConcentration=0.42)

    assert features.model_dump(by_alias=True)["brainflowConcentration"] == 0.42


def test_brainflow_device_configs_include_live_and_synthetic() -> None:
    assert DEVICE_CONFIGS["brainflow-muse-athena"].board_id_name == "MUSE_S_ATHENA_BOARD"
    assert DEVICE_CONFIGS["brainflow-synthetic"].board_id_name == "SYNTHETIC_BOARD"


def test_synthetic_session_initializes_when_brainflow_is_available() -> None:
    pytest.importorskip("brainflow")

    session = BrainFlowSession(DEVICE_CONFIGS["brainflow-synthetic"])
    try:
        info = session.prepare()
        assert info.provider_name == "BrainFlow BoardShim"
        assert info.capabilities[0].kind == "eeg"
        assert len(info.capabilities[0].channels) > 0
    finally:
        session.stop()


def test_synthetic_board_to_features_end_to_end() -> None:
    pytest.importorskip("brainflow")

    script = """
import asyncio
from brainflow_service.config import DEVICE_CONFIGS
from brainflow_service.runtime import BrainFlowSession

async def collect_one_frame():
    session = BrainFlowSession(DEVICE_CONFIGS["brainflow-synthetic"])
    try:
        session.prepare()
        session.start()
        async for frame in session.frames():
            if frame.features and frame.features.band_powers:
                assert frame.sensor == "eeg"
                assert frame.features.band_powers.absolute["theta"] >= 0
                assert frame.features.band_powers.absolute["alpha"] >= 0
                assert frame.features.band_powers.absolute["beta"] >= 0
                assert frame.features.brainflow_concentration is not None
                assert 0 <= frame.features.brainflow_concentration <= 1
                return
    finally:
        session.stop()
    raise AssertionError("No feature-bearing frame emitted")

asyncio.run(asyncio.wait_for(collect_one_frame(), timeout=5.0))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
