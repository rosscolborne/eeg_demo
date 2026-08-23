from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator

import numpy as np

from .config import DEFAULT_PROCESSING, DEVICE_CONFIGS, BrainFlowDeviceConfig, ProcessingConfig
from .dsp import (
    build_eeg_window,
    config_metadata,
    extract_band_power_features,
    extract_brainflow_mindfulness,
    extract_brainflow_restfulness,
    preprocess_eeg_window,
)
from .affective_state import AffectiveCalibrationState, AffectiveStateProvider, FitQualityHint
from .headset_fit import HeuristicHeadsetFitProvider, to_signal_quality_metadata
from .models import DeviceInfo, SensorCapability, SignalChannel, SignalFeatures, SignalFrame


class BrainFlowSession:
    def __init__(
        self,
        config: BrainFlowDeviceConfig,
        *,
        mac_address: str | None = None,
        serial_number: str | None = None,
        processing: ProcessingConfig = DEFAULT_PROCESSING,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.config = config
        self.mac_address = mac_address
        self.serial_number = serial_number
        self.processing = processing
        self.sequence_id = 0
        self.board = None
        self.board_id = 0
        self.master_board_id = 0
        self.eeg_channels: list[int] = []
        self.timestamp_channel: int | None = None
        self.device_info: DeviceInfo | None = None
        self._running = False
        self._affective_state = AffectiveStateProvider()
        self._headset_fit = HeuristicHeadsetFitProvider()

    def start_calibration(self) -> None:
        self._affective_state.start_calibration()

    def reset_calibration(self) -> None:
        self._affective_state.reset_calibration()

    def get_calibration_state(self) -> AffectiveCalibrationState:
        return self._affective_state.get_calibration_state()

    def prepare(self) -> DeviceInfo:
        from brainflow.board_shim import BoardIds, BrainFlowInputParams, BrainFlowPresets, BoardShim

        params = BrainFlowInputParams()
        params.mac_address = self.mac_address or ""
        params.serial_number = self.serial_number or ""
        params.other_info = self.config.other_info

        self.board_id = getattr(BoardIds, self.config.board_id_name).value
        self.master_board_id = self.board_id
        BoardShim.release_all_sessions()
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()

        self.eeg_channels = list(
            BoardShim.get_eeg_channels(
                self.master_board_id,
                BrainFlowPresets.DEFAULT_PRESET.value,
            ),
        )
        self.timestamp_channel = BoardShim.get_timestamp_channel(
            self.master_board_id,
            BrainFlowPresets.DEFAULT_PRESET.value,
        )
        self.device_info = self._device_info(BoardShim, BrainFlowPresets)
        return self.device_info

    def start(self) -> None:
        if not self.board:
            self.prepare()
        try:
            self.board.start_stream(45000)
            self._running = True
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if not self.board:
            return
        try:
            if self._running:
                try:
                    self.board.stop_stream()
                except Exception:
                    pass
        finally:
            self._running = False
            try:
                if self.board and self.board.is_prepared():
                    self.board.release_session()
            finally:
                try:
                    from brainflow.board_shim import BoardShim

                    BoardShim.release_all_sessions()
                except Exception:
                    pass
            self.board = None

    async def frames(self) -> AsyncIterator[SignalFrame]:
        from brainflow.board_shim import BrainFlowPresets

        if not self.board:
            raise RuntimeError("BrainFlow session is not prepared.")

        sample_rate = self.device_info.capabilities[0].sample_rate_hz if self.device_info else 256
        sample_rate_int = int(sample_rate or 256)
        window_samples = max(16, int(self.processing.window_seconds * sample_rate_int))
        update_sleep = max(0.05, self.processing.update_interval_seconds)

        while self._running:
            await asyncio.sleep(update_sleep)
            data = self.board.get_current_board_data(
                window_samples,
                BrainFlowPresets.DEFAULT_PRESET.value,
            )
            if data.size == 0:
                continue
            frame = self._normalize_frame(np.asarray(data), sample_rate_int, window_samples)
            if frame:
                yield frame

    def _normalize_frame(
        self,
        data: np.ndarray,
        sample_rate: int,
        window_samples: int,
    ) -> SignalFrame | None:
        if not self.device_info:
            return None
        if data.shape[1] < 2:
            return None

        self.sequence_id += 1
        eeg_rows = data[np.array(self.eeg_channels), :].T
        eeg_samples = eeg_rows.astype(float).tolist()
        timestamps_ms = None
        if self.timestamp_channel is not None and self.timestamp_channel < data.shape[0]:
            timestamps_ms = (data[self.timestamp_channel, :] * 1000.0).astype(float).tolist()

        channels = self.device_info.capabilities[0].channels
        fit_snapshot = self._headset_fit.update(channels=channels, samples=eeg_samples)
        quality = to_signal_quality_metadata(fit_snapshot)

        window = build_eeg_window(data, self.eeg_channels, window_samples)
        features = None
        if window is not None:
            processed = preprocess_eeg_window(window, sample_rate, self.processing)
            band_powers = extract_band_power_features(processed, sample_rate)
            brainflow_mindfulness = extract_brainflow_mindfulness(window, sample_rate)
            brainflow_restfulness = extract_brainflow_restfulness(window, sample_rate)
            if band_powers or brainflow_mindfulness is not None or brainflow_restfulness is not None:
                # Mirrors AffectiveStateProvider's `if (quality?.excessiveArtifact)
                # return null` gate and its confidence quality factor, using
                # this session's real headset-fit assessment instead of
                # always assuming full reliability.
                sample = self._affective_state.push(
                    at_ms=time.time() * 1000.0,
                    theta_power=band_powers.absolute.get("theta", 0.0) if band_powers else 0.0,
                    alpha_power=band_powers.absolute.get("alpha", 0.0) if band_powers else 0.0,
                    beta_power=band_powers.absolute.get("beta", 0.0) if band_powers else 0.0,
                    gamma_power=band_powers.absolute.get("gamma", 0.0) if band_powers else 0.0,
                    raw_mindfulness=brainflow_mindfulness,
                    raw_restfulness=brainflow_restfulness,
                    reliable=not fit_snapshot.excessive_artifact,
                    fit=FitQualityHint(ready=fit_snapshot.ready, state=fit_snapshot.state),
                )
                features = SignalFeatures(
                    bandPowers=band_powers,
                    brainflowConcentration=brainflow_mindfulness,
                    brainflowRestfulness=brainflow_restfulness,
                    mindfulnessScore=sample.mindfulness_score if sample else None,
                    restfulnessScore=sample.restfulness_score if sample else None,
                    focusScore=sample.focus_score if sample else None,
                    relaxScore=sample.relax_score if sample else None,
                    valence=sample.valence if sample else None,
                    arousal=sample.arousal if sample else None,
                    rawValence=sample.raw_valence if sample else None,
                    rawArousal=sample.raw_arousal if sample else None,
                    stateLabel=sample.label if sample else None,
                    confidence=sample.confidence if sample else None,
                    calibrationActive=sample.calibration_active if sample else False,
                )

        return SignalFrame(
            sensor="eeg",
            sampleRateHz=sample_rate,
            channels=channels,
            samples=eeg_samples,
            timestampsMs=timestamps_ms,
            receivedAtMs=time.time() * 1000.0,
            sequenceId=self.sequence_id,
            quality=quality,
            features=features,
        )

    def _device_info(self, board_shim, presets) -> DeviceInfo:
        board_id = self.master_board_id
        preset = presets.DEFAULT_PRESET.value
        sample_rate = self.board.get_board_sampling_rate(preset)
        eeg_names_raw = board_shim.get_eeg_names(board_id, preset)
        if isinstance(eeg_names_raw, str):
            eeg_names = [name.strip() for name in eeg_names_raw.split(",") if name.strip()]
        else:
            eeg_names = list(eeg_names_raw)

        channels = [
            SignalChannel(
                id=(eeg_names[index] if index < len(eeg_names) else f"eeg_{index + 1}").lower(),
                label=eeg_names[index] if index < len(eeg_names) else f"EEG {index + 1}",
                unit="uV",
                index=channel_index,
            )
            for index, channel_index in enumerate(self.eeg_channels)
        ]

        capabilities = [
            SensorCapability(kind="eeg", sampleRateHz=sample_rate, channels=channels),
        ]
        metadata = {
            "brainflowBoardId": board_id,
            "brainflowSessionBoardId": self.board_id,
            "mode": self.config.mode,
            **config_metadata(self.processing),
        }
        for getter, kind, preset_name in (
            ("get_accel_channels", "accelerometer", "AUXILIARY_PRESET"),
            ("get_gyro_channels", "gyroscope", "AUXILIARY_PRESET"),
            ("get_optical_channels", "optical", "ANCILLARY_PRESET"),
            ("get_ppg_channels", "ppg", "ANCILLARY_PRESET"),
        ):
            try:
                preset_value = getattr(presets, preset_name).value
                indices = list(getattr(board_shim, getter)(board_id, preset_value))
                if indices:
                    capabilities.append(
                        SensorCapability(
                            kind=kind,
                            sampleRateHz=board_shim.get_sampling_rate(board_id, preset_value),
                            channels=[
                                SignalChannel(id=f"{kind}_{i + 1}", label=f"{kind.upper()} {i + 1}", unit="", index=value)
                                for i, value in enumerate(indices)
                            ],
                        ),
                    )
            except Exception:
                continue

        return DeviceInfo(
            label=self.config.label,
            model=self.config.board_id_name,
            providerName="BrainFlow BoardShim",
            capabilities=capabilities,
            metadata=metadata,
        )


class SessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, BrainFlowSession] = {}

    def create(self, device_id: str, **kwargs) -> BrainFlowSession:
        self.stop_all()
        config = DEVICE_CONFIGS[device_id]
        session = BrainFlowSession(config, **kwargs)
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> BrainFlowSession:
        return self.sessions[session_id]

    def stop(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            session.stop()

    def stop_all(self) -> None:
        for session_id in list(self.sessions):
            self.stop(session_id)
        try:
            from brainflow.board_shim import BoardShim

            BoardShim.release_all_sessions()
        except Exception:
            pass


def sse_event(event: str, payload: object) -> str:
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(by_alias=True)
    else:
        data = payload
    return f"event: {event}\ndata: {json.dumps(data, allow_nan=False)}\n\n"
