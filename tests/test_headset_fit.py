from __future__ import annotations

import math

from brainflow_service.headset_fit import (
    DEFAULT_HEADSET_FIT_THRESHOLDS,
    HeuristicHeadsetFitProvider,
    to_signal_quality_metadata,
)
from brainflow_service.models import SignalChannel

FOUR_MUSE_CHANNELS = [
    SignalChannel(id="tp9", label="TP9", unit="uV", index=0),
    SignalChannel(id="af7", label="AF7", unit="uV", index=1),
    SignalChannel(id="af8", label="AF8", unit="uV", index=2),
    SignalChannel(id="tp10", label="TP10", unit="uV", index=3),
]


def _repeat_sine(*, amplitude: float, freq_hz: float, sample_rate: int, seconds: float, channels: int) -> list[list[float]]:
    count = int(sample_rate * seconds)
    return [
        [amplitude * math.sin(2 * math.pi * freq_hz * (i / sample_rate))] * channels
        for i in range(count)
    ]


def test_clean_signal_is_reported_good() -> None:
    samples = _repeat_sine(amplitude=50, freq_hz=10, sample_rate=256, seconds=1, channels=4)

    snapshot = HeuristicHeadsetFitProvider().update(channels=FOUR_MUSE_CHANNELS, samples=samples)

    assert snapshot.state == "good"
    assert snapshot.worn is True
    assert snapshot.excessive_artifact is False
    assert all(c.state == "good" for c in snapshot.channels)


def test_flat_signal_is_reported_poor() -> None:
    samples = [[0.0, 0.0, 0.0, 0.0]] * 100

    snapshot = HeuristicHeadsetFitProvider().update(channels=FOUR_MUSE_CHANNELS, samples=samples)

    assert snapshot.state == "poor"
    assert all(c.state == "poor" for c in snapshot.channels)
    assert all(c.message == "Signal is too flat" for c in snapshot.channels)


def test_clipped_signal_is_reported_poor() -> None:
    samples = [[150000.0] * 4 if i % 2 == 0 else [10.0] * 4 for i in range(100)]

    snapshot = HeuristicHeadsetFitProvider().update(channels=FOUR_MUSE_CHANNELS, samples=samples)

    assert snapshot.state == "poor"
    assert all(c.message == "Signal is clipped or saturated" for c in snapshot.channels)


def test_large_steps_are_reported_as_excessive_noise() -> None:
    samples = [[5000.0] * 4 if i % 2 == 0 else [-5000.0] * 4 for i in range(100)]

    snapshot = HeuristicHeadsetFitProvider().update(channels=FOUR_MUSE_CHANNELS, samples=samples)

    assert snapshot.state == "adjusting"
    assert all(c.message == "Excessive noise or movement" for c in snapshot.channels)
    # Steps this large also trip the overall excessive-artifact gate used to
    # decide `reliable` for the headline scores.
    assert snapshot.excessive_artifact is True


def test_moderately_unstable_signal_is_reported_adjusting() -> None:
    samples = _repeat_sine(amplitude=2000, freq_hz=1, sample_rate=256, seconds=1, channels=4)

    snapshot = HeuristicHeadsetFitProvider().update(channels=FOUR_MUSE_CHANNELS, samples=samples)

    assert snapshot.state == "adjusting"
    assert all(c.message == "Signal is unstable; check headset fit" for c in snapshot.channels)
    assert snapshot.excessive_artifact is False


def test_too_few_samples_is_reported_adjusting() -> None:
    samples = [[1.0, 2.0, 3.0, 4.0]] * 5  # fewer than min_samples_per_frame

    snapshot = HeuristicHeadsetFitProvider().update(channels=FOUR_MUSE_CHANNELS, samples=samples)

    assert all(c.state == "adjusting" for c in snapshot.channels)
    assert all(c.message == "Waiting for enough samples" for c in snapshot.channels)


def test_side_adjustment_blocker_points_at_the_worse_side() -> None:
    # tp9/af7 (left) flat, tp10/af8 (right) clean.
    samples = [
        [0.0, 0.0, 50 * math.sin(2 * math.pi * 10 * i / 256), 50 * math.sin(2 * math.pi * 10 * i / 256)]
        for i in range(256)
    ]

    snapshot = HeuristicHeadsetFitProvider().update(channels=FOUR_MUSE_CHANNELS, samples=samples)

    assert "Adjust left side." in snapshot.blockers


def test_empty_channels_does_not_crash_and_is_reported_poor() -> None:
    snapshot = HeuristicHeadsetFitProvider().update(channels=[], samples=[])

    assert snapshot.state == "poor"
    assert snapshot.worn is False
    assert snapshot.channels == []


def test_stability_timer_reaches_ready_after_required_duration() -> None:
    provider = HeuristicHeadsetFitProvider()
    samples = _repeat_sine(amplitude=50, freq_hz=10, sample_rate=256, seconds=1, channels=4)

    first = provider.update(channels=FOUR_MUSE_CHANNELS, samples=samples, now_ms=0)
    assert first.state == "good"
    assert first.ready is False
    assert first.stable_for_ms == 0

    just_before = provider.update(
        channels=FOUR_MUSE_CHANNELS, samples=samples,
        now_ms=DEFAULT_HEADSET_FIT_THRESHOLDS.stable_ready_ms - 1,
    )
    assert just_before.ready is False

    at_threshold = provider.update(
        channels=FOUR_MUSE_CHANNELS, samples=samples,
        now_ms=DEFAULT_HEADSET_FIT_THRESHOLDS.stable_ready_ms,
    )
    assert at_threshold.ready is True
    assert at_threshold.state == "ready"


def test_stability_timer_resets_once_signal_becomes_unacceptable() -> None:
    provider = HeuristicHeadsetFitProvider()
    good_samples = _repeat_sine(amplitude=50, freq_hz=10, sample_rate=256, seconds=1, channels=4)
    flat_samples = [[0.0, 0.0, 0.0, 0.0]] * 100

    provider.update(channels=FOUR_MUSE_CHANNELS, samples=good_samples, now_ms=0)
    provider.update(channels=FOUR_MUSE_CHANNELS, samples=flat_samples, now_ms=1000)
    resumed = provider.update(channels=FOUR_MUSE_CHANNELS, samples=good_samples, now_ms=2000)

    assert resumed.stable_for_ms == 0


def test_reset_clears_stability_timer() -> None:
    provider = HeuristicHeadsetFitProvider()
    samples = _repeat_sine(amplitude=50, freq_hz=10, sample_rate=256, seconds=1, channels=4)

    provider.update(channels=FOUR_MUSE_CHANNELS, samples=samples, now_ms=0)
    provider.reset()
    resumed = provider.update(channels=FOUR_MUSE_CHANNELS, samples=samples, now_ms=5000)

    assert resumed.stable_for_ms == 0


def test_to_signal_quality_metadata_round_trips_fields() -> None:
    samples = _repeat_sine(amplitude=50, freq_hz=10, sample_rate=256, seconds=1, channels=4)
    snapshot = HeuristicHeadsetFitProvider().update(channels=FOUR_MUSE_CHANNELS, samples=samples)

    metadata = to_signal_quality_metadata(snapshot)

    assert metadata.state == snapshot.state
    assert metadata.ready == snapshot.ready
    assert metadata.worn == snapshot.worn
    assert metadata.excessive_artifact == snapshot.excessive_artifact
    assert len(metadata.channels) == len(snapshot.channels)
    assert metadata.channels[0].channel.id == snapshot.channels[0].channel.id
