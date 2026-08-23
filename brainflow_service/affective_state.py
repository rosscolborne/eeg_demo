"""Valence/arousal state, calibration, and smoothing.

Python port of `src/metrics/affectiveStateMetric.ts`'s `AffectiveStateProvider`.
Wraps `MindStateSmoother` (mindfulness/restfulness/focus/relax) and adds the
two-axis valence/arousal proxy, its baseline calibration, and nearest-label
classification, so a session carries the full set of scores the bundled
app's "Valence / Arousal" panel shows -- not just the four headline scores.

Deviation from the TS version: `estimate_confidence` there also factors in
a browser-only headset-fit "quality" signal (`headsetFitProvider.ts`) that
has no server-side equivalent yet, so confidence here is distance-based
only. `AffectiveStateProvider.push` accepts an optional `reliable` flag if
a caller wants to wire in their own artifact/contact gating later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .metrics import DEFAULT_SMOOTHING_ALPHA, MindStateSmoother

NEUTRAL_RADIUS = 0.18
DEFAULT_CALIBRATION_SAMPLE_COUNT = 24


@dataclass(frozen=True)
class EmotionRegion:
    label: str
    valence: float
    arousal: float


# Port of `affectiveEmotionRegions` in affectiveStateMetric.ts.
AFFECTIVE_EMOTION_REGIONS: tuple[EmotionRegion, ...] = (
    EmotionRegion("Tense", -0.25, 0.78),
    EmotionRegion("Angry", -0.68, 0.55),
    EmotionRegion("Frustrated", -0.72, 0.25),
    EmotionRegion("Depressed", -0.74, -0.25),
    EmotionRegion("Bored", -0.58, -0.58),
    EmotionRegion("Tired", -0.25, -0.82),
    EmotionRegion("Calm", 0.25, -0.82),
    EmotionRegion("Relaxed", 0.58, -0.58),
    EmotionRegion("Content", 0.72, -0.25),
    EmotionRegion("Happy", 0.72, 0.25),
    EmotionRegion("Delighted", 0.62, 0.55),
    EmotionRegion("Excited", 0.32, 0.78),
)


@dataclass(frozen=True)
class AffectiveCalibrationState:
    status: Literal["off", "collecting", "active"]
    progress: int
    required: int


@dataclass(frozen=True)
class AffectiveStateSample:
    at_ms: float
    valence: float
    arousal: float
    raw_valence: float
    raw_arousal: float
    calibration_active: bool
    label: str
    confidence: float
    theta_power: float
    alpha_power: float
    beta_power: float
    gamma_power: float
    mindfulness_score: float | None
    restfulness_score: float | None
    focus_score: int
    relax_score: int
    reliable: bool


@dataclass(frozen=True)
class RawAffectiveSample:
    """Single-window valence/arousal with no smoothing or calibration --
    used by the stateless `/analyze-window` endpoint."""

    valence: float
    arousal: float
    label: str
    confidence: float


def map_ratio_to_axis(ratio: float) -> float:
    """Port of `mapRatioToAxis`."""
    if not math.isfinite(ratio) or ratio <= 0:
        return 0.0
    return _clamp(math.tanh(math.log2(ratio) / 2.5), -1.0, 1.0)


def classify_affective_state(valence: float, arousal: float) -> str:
    """Port of `classifyAffectiveState`: nearest named region, or "Neutral"
    within `NEUTRAL_RADIUS` of the origin."""
    if math.hypot(valence, arousal) < NEUTRAL_RADIUS:
        return "Neutral"

    nearest = AFFECTIVE_EMOTION_REGIONS[0]
    nearest_distance = math.inf
    for region in AFFECTIVE_EMOTION_REGIONS:
        distance = math.hypot(valence - region.valence, arousal - region.arousal)
        if distance < nearest_distance:
            nearest = region
            nearest_distance = distance
    return nearest.label


def estimate_confidence(valence: float, arousal: float) -> float:
    """Distance-based port of `estimateConfidence`. See module docstring for
    why the quality-factor multiplier from the TS version isn't included."""
    distance = min(1.0, math.hypot(valence, arousal))
    return _clamp(distance, 0.0, 1.0)


def compute_raw_affective_sample(
    theta_power: float,
    alpha_power: float,
    beta_power: float,
    gamma_power: float,
) -> RawAffectiveSample | None:
    theta = _finite_power(theta_power)
    alpha = _finite_power(alpha_power)
    beta = _finite_power(beta_power)
    gamma = _finite_power(gamma_power)
    if theta + alpha + beta + gamma <= 0:
        return None

    raw_arousal = map_ratio_to_axis((beta + gamma) / (alpha + theta + 1e-9))
    raw_valence = map_ratio_to_axis(alpha / (theta + beta + 1e-9))
    return RawAffectiveSample(
        valence=raw_valence,
        arousal=raw_arousal,
        label=classify_affective_state(raw_valence, raw_arousal),
        confidence=estimate_confidence(raw_valence, raw_arousal),
    )


@dataclass
class _CalibrationProfile:
    valence: float
    arousal: float


class AffectiveStateProvider:
    """Stateful, per-session port of `AffectiveStateProvider`
    (affectiveStateMetric.ts): valence/arousal with slow-EMA smoothing and
    optional median-baseline calibration, plus the four headline scores via
    an internal `MindStateSmoother`."""

    def __init__(
        self,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        calibration_sample_count: int = DEFAULT_CALIBRATION_SAMPLE_COUNT,
    ) -> None:
        self._alpha = smoothing_alpha
        self._calibration_sample_count = calibration_sample_count
        self._mind_state = MindStateSmoother(smoothing_alpha)
        self._smoothed_valence: float | None = None
        self._smoothed_arousal: float | None = None
        self._calibration_status: Literal["off", "collecting", "active"] = "off"
        self._calibration_valence_values: list[float] = []
        self._calibration_arousal_values: list[float] = []
        self._calibration_profile: _CalibrationProfile | None = None

    def reset(self) -> None:
        self._smoothed_valence = None
        self._smoothed_arousal = None
        self._mind_state.reset()
        self.reset_calibration()

    def start_calibration(self) -> None:
        self._calibration_status = "collecting"
        self._calibration_valence_values = []
        self._calibration_arousal_values = []
        self._calibration_profile = None
        self._smoothed_valence = None
        self._smoothed_arousal = None

    def reset_calibration(self) -> None:
        self._calibration_status = "off"
        self._calibration_valence_values = []
        self._calibration_arousal_values = []
        self._calibration_profile = None
        self._smoothed_valence = None
        self._smoothed_arousal = None

    def get_calibration_state(self) -> AffectiveCalibrationState:
        if self._calibration_status == "collecting":
            progress = min(len(self._calibration_valence_values), self._calibration_sample_count)
        elif self._calibration_status == "active":
            progress = self._calibration_sample_count
        else:
            progress = 0
        return AffectiveCalibrationState(
            status=self._calibration_status,
            progress=progress,
            required=self._calibration_sample_count,
        )

    def push(
        self,
        *,
        at_ms: float,
        theta_power: float,
        alpha_power: float,
        beta_power: float,
        gamma_power: float,
        raw_mindfulness: float | None,
        raw_restfulness: float | None,
        reliable: bool = True,
    ) -> AffectiveStateSample | None:
        if not reliable:
            return None

        theta = _finite_power(theta_power)
        alpha = _finite_power(alpha_power)
        beta = _finite_power(beta_power)
        gamma = _finite_power(gamma_power)
        if theta + alpha + beta + gamma <= 0:
            return None

        raw_arousal = map_ratio_to_axis((beta + gamma) / (alpha + theta + 1e-9))
        raw_valence = map_ratio_to_axis(alpha / (theta + beta + 1e-9))
        self._accept_calibration_sample(raw_valence, raw_arousal)

        calibrated_valence = (
            raw_valence
            if self._calibration_profile is None
            else _clamp(raw_valence - self._calibration_profile.valence, -1.0, 1.0)
        )
        calibrated_arousal = (
            raw_arousal
            if self._calibration_profile is None
            else _clamp(raw_arousal - self._calibration_profile.arousal, -1.0, 1.0)
        )

        mind_state = self._mind_state.push(
            theta_power=theta,
            alpha_power=alpha,
            beta_power=beta,
            raw_mindfulness=raw_mindfulness,
            raw_restfulness=raw_restfulness,
        )

        self._smoothed_valence = (
            calibrated_valence
            if self._smoothed_valence is None
            else _smooth(self._smoothed_valence, calibrated_valence, self._alpha)
        )
        self._smoothed_arousal = (
            calibrated_arousal
            if self._smoothed_arousal is None
            else _smooth(self._smoothed_arousal, calibrated_arousal, self._alpha)
        )

        valence = _clamp(self._smoothed_valence, -1.0, 1.0)
        arousal = _clamp(self._smoothed_arousal, -1.0, 1.0)

        return AffectiveStateSample(
            at_ms=at_ms,
            valence=valence,
            arousal=arousal,
            raw_valence=raw_valence,
            raw_arousal=raw_arousal,
            calibration_active=self._calibration_profile is not None,
            label=classify_affective_state(valence, arousal),
            confidence=estimate_confidence(valence, arousal),
            theta_power=theta,
            alpha_power=alpha,
            beta_power=beta,
            gamma_power=gamma,
            mindfulness_score=mind_state.mindfulness_score,
            restfulness_score=mind_state.restfulness_score,
            focus_score=mind_state.focus_score,
            relax_score=mind_state.relax_score,
            reliable=reliable,
        )

    def _accept_calibration_sample(self, raw_valence: float, raw_arousal: float) -> None:
        if self._calibration_status != "collecting":
            return

        self._calibration_valence_values.append(raw_valence)
        self._calibration_arousal_values.append(raw_arousal)
        if len(self._calibration_valence_values) < self._calibration_sample_count:
            return

        self._calibration_profile = _CalibrationProfile(
            valence=_median(self._calibration_valence_values),
            arousal=_median(self._calibration_arousal_values),
        )
        self._calibration_status = "active"
        self._smoothed_valence = None
        self._smoothed_arousal = None


def _finite_power(value: float) -> float:
    return max(0.0, value) if math.isfinite(value) else 0.0


def _smooth(current: float, target: float, weight: float) -> float:
    return current * (1 - weight) + target * weight


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2
    return ordered[mid]


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
