import type { SignalFrame } from "../domain/eeg";
import {
  computeBandPowers,
  defaultAttentionBands,
  type FrequencyBand,
} from "../signalProcessing/bandPower";
import type { HeadsetFitSnapshot } from "../signalQuality/headsetFitProvider";

export interface MetricProvider<TSample> {
  readonly id: string;
  readonly label: string;
  reset(): void;
  pushFrame(frame: SignalFrame, quality?: HeadsetFitSnapshot | null): TSample | null;
}

export interface AttentionMetricSample {
  metricId: "attention-index-experimental";
  label: "Attention Index — Experimental";
  atMs: number;
  displayedScore: number;
  thetaPower: number;
  alphaPower: number;
  betaPower: number;
  rawRatio: number;
  baselineRatio: number | null;
  baselineRelativeValue: number | null;
  sampleCount: number;
  channelCount: number;
  qualityState: HeadsetFitSnapshot["state"] | "unknown";
  reliable: boolean;
}

export interface CalibrationProfile {
  id: string;
  algorithmVersion: "focus-index-v1";
  createdAtMs: number;
  baselineRatio: number | null;
  acceptedWindows: number;
  rejectedWindows: number;
  rejectionReasons: Record<string, number>;
}

export interface HeuristicAttentionProviderOptions {
  bands?: {
    theta: FrequencyBand;
    alpha: FrequencyBand;
    beta: FrequencyBand;
  };
  baselineSampleCount?: number;
  smoothingAlpha?: number;
}

export class HeuristicAttentionProvider
  implements MetricProvider<AttentionMetricSample>
{
  readonly id = "attention-index-experimental";
  readonly label = "Attention Index — Experimental";

  private readonly bands;
  private readonly baselineSampleCount;
  private readonly smoothingAlpha;
  private baselineRatios: number[] = [];
  private smoothedScore: number | null = null;
  private calibrationProfile: CalibrationProfile | null = null;
  private rejectedWindows = 0;
  private rejectionReasons: Record<string, number> = {};

  constructor(options: HeuristicAttentionProviderOptions = {}) {
    this.bands = options.bands ?? defaultAttentionBands;
    this.baselineSampleCount = options.baselineSampleCount ?? 24;
    this.smoothingAlpha = options.smoothingAlpha ?? 0.22;
  }

  reset() {
    this.baselineRatios = [];
    this.smoothedScore = null;
    this.calibrationProfile = null;
    this.rejectedWindows = 0;
    this.rejectionReasons = {};
  }

  pushFrame(
    frame: SignalFrame,
    quality?: HeadsetFitSnapshot | null,
  ): AttentionMetricSample | null {
    const reliable =
      !quality || quality.ready || quality.state === "good";
    if (!reliable) {
      this.rejectWindow("poor_quality");
      return null;
    }

    const brainFlowBands = frame.features?.bandPowers?.absolute;
    const bandResult = brainFlowBands
      ? {
          channelCount: frame.channels.length,
          sampleCount: frame.samples.length,
          sampleRateHz: frame.sampleRateHz ?? 0,
          powers: brainFlowBands,
        }
      : computeBandPowers(frame, [
          this.bands.theta,
          this.bands.alpha,
          this.bands.beta,
        ]);

    if (!bandResult) {
      this.rejectWindow("insufficient_window");
      return null;
    }

    const thetaPower = bandResult.powers.theta ?? 0;
    const alphaPower = bandResult.powers.alpha ?? 0;
    const betaPower = bandResult.powers.beta ?? 0;
    const denominator = Math.max(1e-9, alphaPower + thetaPower);
    const rawRatio = betaPower / denominator;

    if (this.baselineRatios.length < this.baselineSampleCount) {
      this.baselineRatios.push(rawRatio);
    }

    const baselineRatio =
      this.baselineRatios.length > 0
        ? median(this.baselineRatios)
        : null;
    const baselineRelativeValue =
      baselineRatio && baselineRatio > 0 ? rawRatio / baselineRatio : null;
    if (this.baselineRatios.length >= this.baselineSampleCount && !this.calibrationProfile) {
      this.calibrationProfile = {
        id: crypto.randomUUID(),
        algorithmVersion: "focus-index-v1",
        createdAtMs: Date.now(),
        baselineRatio,
        acceptedWindows: this.baselineRatios.length,
        rejectedWindows: this.rejectedWindows,
        rejectionReasons: { ...this.rejectionReasons },
      };
    }
    const mappedScore = mapRelativeValueToScore(baselineRelativeValue);
    this.smoothedScore =
      this.smoothedScore === null
        ? mappedScore
        : this.smoothedScore * (1 - this.smoothingAlpha) +
          mappedScore * this.smoothingAlpha;

    return {
      metricId: "attention-index-experimental",
      label: "Attention Index — Experimental",
      atMs: frame.receivedAtMs,
      displayedScore: Math.round(clamp(this.smoothedScore, 0, 100)),
      thetaPower,
      alphaPower,
      betaPower,
      rawRatio,
      baselineRatio,
      baselineRelativeValue,
      sampleCount: bandResult.sampleCount,
      channelCount: bandResult.channelCount,
      qualityState: quality?.state ?? "unknown",
      reliable,
    };
  }

  getCalibrationProfile() {
    return this.calibrationProfile;
  }

  private rejectWindow(reason: string) {
    this.rejectedWindows += 1;
    this.rejectionReasons[reason] = (this.rejectionReasons[reason] ?? 0) + 1;
  }
}

function mapRelativeValueToScore(value: number | null) {
  if (value === null || !Number.isFinite(value)) return 50;

  return clamp(50 + Math.log2(Math.max(0.05, value)) * 22, 0, 100);
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
