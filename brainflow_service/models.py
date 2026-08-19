from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SignalChannel(BaseModel):
    id: str
    label: str
    unit: str
    index: int | None = None


class BandPowerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    absolute: dict[str, float]
    relative: dict[str, float]
    ratios: dict[str, float]
    window_seconds: float = Field(alias="windowSeconds")
    method: Literal["brainflow_welch_psd"]


class SignalFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    band_powers: BandPowerFeatures | None = Field(default=None, alias="bandPowers")
    brainflow_concentration: float | None = Field(default=None, alias="brainflowConcentration")


class SignalQualityMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: Literal["device", "inferred"] = "inferred"
    excessive_artifact: bool = Field(default=False, alias="excessiveArtifact")
    motion_rms: float | None = Field(default=None, alias="motionRms")
    message: str | None = None


class SensorCapability(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: str
    sample_rate_hz: float | None = Field(alias="sampleRateHz")
    channels: list[SignalChannel]


class DeviceInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: str
    model: str
    provider_name: str = Field(alias="providerName")
    firmware_version: str | None = Field(default=None, alias="firmwareVersion")
    capabilities: list[SensorCapability]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalFrame(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sensor: str
    sample_rate_hz: float | None = Field(alias="sampleRateHz")
    channels: list[SignalChannel]
    samples: list[list[float]]
    timestamps_ms: list[float] | None = Field(default=None, alias="timestampsMs")
    received_at_ms: float = Field(alias="receivedAtMs")
    sequence_id: int = Field(alias="sequenceId")
    quality: SignalQualityMetadata | None = None
    features: SignalFeatures | None = None


class CalibrationProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    algorithm_version: str = Field(alias="algorithmVersion")
    created_at_ms: float = Field(alias="createdAtMs")
    baseline_ratio: float | None = Field(alias="baselineRatio")
    accepted_windows: int = Field(alias="acceptedWindows")
    rejected_windows: int = Field(alias="rejectedWindows")
    rejection_reasons: dict[str, int] = Field(alias="rejectionReasons")
    device_info: DeviceInfo | None = Field(default=None, alias="deviceInfo")
    metadata: dict[str, Any] = Field(default_factory=dict)
