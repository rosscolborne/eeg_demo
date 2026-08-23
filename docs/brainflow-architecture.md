# BrainFlow EEG Pipeline Architecture

## Current Migration Shape

The browser app remains a React UI that consumes normalized EEG frames through
the existing `EegProvider` interface. BrainFlow runs in a local Python service
because BoardShim and DataFilter are native runtime APIs rather than browser Web
Bluetooth APIs.

Pipeline:

`Hardware -> BrainFlow BoardShim -> normalized EEG frame -> BrainFlow DataFilter DSP -> BrainFlow MLModel -> Focus Index -> UI`

## Device-Agnostic Pieces

- `src/domain/eeg.ts`: normalized signal/device/frame types consumed by the UI.
- `src/providers/eegProvider.ts`: frontend provider interface.
- `brainflow_service/models.py`: service-side normalized models.
- `brainflow_service/dsp.py`: preprocessing, windowing, PSD, band-power feature extraction.
- `src/metrics/attentionMetric.ts`: Focus Index metric interface and current heuristic.
- `src/training/trainingSession.ts`: session/report lifecycle.
- `src/signalQuality/headsetFitProvider.ts`: inferred quality gate from normalized frames.

## BrainFlow-Specific Pieces

- `brainflow_service/runtime.py`: BoardShim session lifecycle, board metadata,
  preset/channel lookup, BrainFlow streamer recording, and frame normalization.
- `brainflow_service/app.py`: local FastAPI bridge used by the browser.
- `src/providers/brainflowHttpProvider.ts`: frontend HTTP/SSE adapter.
- `brainflow_service/config.py`: board IDs, Muse Athena startup options, frequency
  bands, windowing, and processing parameters.
- `brainflow_service/metrics.py`: mindfulness/restfulness/focus/relax scoring
  (a Python port of the smoothing math that used to live only in
  `src/metrics/`). This is what lets a front-end other than the bundled React
  app get the same four finished scores directly from the service's HTTP/SSE
  API instead of reimplementing the scoring itself — see the README section
  "Using this service from another front-end".

## BrainFlow APIs Used

- `BoardShim(BoardIds.*, BrainFlowInputParams)`
- `prepare_session`, `start_stream`, `stop_stream`, `release_session`
- `get_current_board_data`
- `get_board_sampling_rate`
- `BoardShim.get_eeg_channels`
- `BoardShim.get_eeg_names`
- `BoardShim.get_timestamp_channel`
- `BoardShim.get_accel_channels`
- `BoardShim.get_gyro_channels`
- `BoardShim.get_optical_channels`
- `BoardShim.get_ppg_channels`
- `BoardShim.get_sampling_rate`
- `DataFilter.detrend`
- `DataFilter.perform_bandpass`
- `DataFilter.perform_bandstop`
- `DataFilter.get_psd_welch`
- `DataFilter.get_band_power`
- `DataFilter.get_nearest_power_of_two`
- `DataFilter.get_avg_band_powers`
- `MLModel`
- `BrainFlowModelParams`
- `BrainFlowMetrics.MINDFULNESS`
- `BrainFlowClassifiers.DEFAULT_CLASSIFIER`

## Recording And Replay

Recording is explicit and user controlled. Live and Synthetic streams are not
written automatically by the BrainFlow service. The browser records normalized
`SignalFrame` objects only after the user clicks `Record`, and `Download` saves
a JSON file using the app-owned format:

`eeg-demo-normalized-recording`

Replay uses the frontend `LocalReplayProvider`, not BrainFlow's
`PLAYBACK_FILE_BOARD`. The selected JSON file is parsed locally and emitted back
through the same normalized UI pipeline at approximately the original frame
timing.

Old service-generated BrainFlow streamer files are intentionally not supported
by this replay path. Several were partial or empty, and playback also requires
the correct BrainFlow master board metadata for the original source.

## Muse Athena Configuration

BrainFlow 5.22 adds `BoardIds.MUSE_S_ATHENA_BOARD`. The current live config uses:

`other_info = "preset=p1041;low_latency=true"`

Per BrainFlow's Muse Athena documentation, the default preset exposes EEG at
256 Hz. Auxiliary and ancillary presets expose motion and optical/battery data
where supported.

## Focus Index

The primary Attention Index uses BrainFlow's current `MINDFULNESS` ML metric,
scaled from `0.0-1.0` to `0-100` and smoothed in the frontend. The UI still
labels it experimental because it is not a clinical or validated focus measure.

The previous application-specific ratio remains recorded for diagnostics:

`beta power / (alpha power + theta power)`

Calibration is still recalculated every session and is not persisted. It records
the diagnostic ratio baseline, but it no longer defines the displayed Attention
Index score.

Scientific concerns to investigate later:

- BrainFlow Mindfulness is a general model, not a validated personalized attention model.
- The diagnostic beta/(alpha+theta) ratio is still scientifically weak.
- Current quality/contact is still inferred unless the device reports explicit
  metadata through BrainFlow.
- The baseline has only one condition and no persistence.
- Motion artifacts are represented in the architecture but need board-specific
  validation before they should drive user-facing claims.
