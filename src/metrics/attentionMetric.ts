import type { SignalFrame } from "../domain/eeg";
import {
  computeBrainflowsNeurofeedbackScores,
  smoothScore,
} from "./neurofeedbackRatios";
import {
  computeBandPowers,
  defaultAttentionBands,
  type FrequencyBand,
} from "../signalProcessing/bandPower";
import type { HeadsetFitSnapshot } from "../signalQuality/headsetFitProvider";

const vrchatStyleEmaDecay = 0.05;
const calibratedZScoreScale = 1.5;
export interface MetricProvider<TSample> {
  readonly id: string;
  readonly label: string;
  reset(): void;
  pushFrame(frame: SignalFrame, quality?: HeadsetFitSnapshot | null): TSample | null;
}

export interface AttentionMetricSample {
  metricId: "attention-index-experimental";
  label: "BrainFlow Mindfulness";
  atMs: number;
  displayedScore: number | null;
  restfulnessScore: number | null;
  focusScore: number;
  relaxScore: number;
  thetaPower: number;
  alphaPower: number;
  betaPower: number;
  rawRatio: number;
  baselineRatio: number | null;
  rawBrainflowMindfulness: number | null;
  baselineBrainflowMindfulness: number | null;
  baselineRelativeValue: number | null;
  baselineZScore: number | null;
  rawBrainflowRestfulness: number | null;
  baselineBrainflowRestfulness: number | null;
  restfulnessBaselineRelativeValue: number | null;
  restfulnessBaselineZScore: number | null;
  scoreSource: "brainflow_mindfulness" | "unavailable";
  restfulnessScoreSource: "brainflow_restfulness" | "unavailable";
  sampleCount: number;
  channelCount: number;
  qualityState: HeadsetFitSnapshot["state"] | "unknown";
  reliable: boolean;
}

export interface CalibrationProfile {
  id: string;
  algorithmVersion: "brainflow-mindfulness-v1";
  createdAtMs: number;
  baselineRatio: number | null;
  baselineBrainflowMindfulness: number | null;
  baselineRatioSpread: number | null;
  baselineBrainflowMindfulnessSpread: number | null;
  baselineBrainflowRestfulness: number | null;
  baselineBrainflowRestfulnessSpread: number | null;
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
  readonly label = "BrainFlow Mindfulness";

  private readonly bands;
  private readonly baselineSampleCount;
  private readonly smoothingAlpha;
  private baselineRatios: number[] = [];
  private baselineBrainflowMindfulnessValues: number[] = [];
  private baselineBrainflowRestfulnessValues: number[] = [];
  private baselineFocusValues: number[] = [];
  private baselineRelaxValues: number[] = [];
  private smoothedScore: number | null = null;
  private smoothedRestfulnessScore: number | null = null;
  private smoothedFocusScore: number | null = null;
  private smoothedRelaxScore: number | null = null;
  private calibrationProfile: CalibrationProfile | null = null;
  private useBaselineRelativeDisplay = false;
  private rejectedWindows = 0;
  private rejectionReasons: Record<string, number> = {};

  constructor(options: HeuristicAttentionProviderOptions = {}) {
    this.bands = options.bands ?? defaultAttentionBands;
    this.baselineSampleCount = options.baselineSampleCount ?? 24;
    this.smoothingAlpha = options.smoothingAlpha ?? vrchatStyleEmaDecay;
  }

  reset(options: { useBaselineRelativeDisplay?: boolean } = {}) {
    this.baselineRatios = [];
    this.baselineBrainflowMindfulnessValues = [];
    this.baselineBrainflowRestfulnessValues = [];
    this.baselineFocusValues = [];
    this.baselineRelaxValues = [];
    this.smoothedScore = null;
    this.smoothedRestfulnessScore = null;
    this.smoothedFocusScore = null;
    this.smoothedRelaxScore = null;
    this.calibrationProfile = null;
    this.useBaselineRelativeDisplay = options.useBaselineRelativeDisplay ?? false;
    this.rejectedWindows = 0;
    this.rejectionReasons = {};
  }

  setBaselineRelativeDisplay(enabled: boolean) {
    this.useBaselineRelativeDisplay = enabled;
  }

  pushFrame(
    frame: SignalFrame,
    quality?: HeadsetFitSnapshot | null,
  ): AttentionMetricSample | null {
    const reliable = !quality?.excessiveArtifact;
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
    const neurofeedbackScores = computeBrainflowsNeurofeedbackScores({
      thetaPower,
      alphaPower,
      betaPower,
    });
    const denominator = Math.max(1e-9, alphaPower + thetaPower);
    const rawRatio = betaPower / denominator;
    const brainflowConcentration = readBrainflowMetric(
      frame,
      ["brainflowConcentration", "brainflow_concentration", "mindfulness"],
    );
    const brainflowRestfulness = readBrainflowMetric(
      frame,
      ["brainflowRestfulness", "brainflow_restfulness", "restfulness"],
    );

    if (this.baselineRatios.length < this.baselineSampleCount) {
      this.baselineRatios.push(rawRatio);
    }
    if (
      brainflowConcentration !== null &&
      brainflowConcentration !== undefined &&
      this.baselineBrainflowMindfulnessValues.length < this.baselineSampleCount
    ) {
      this.baselineBrainflowMindfulnessValues.push(brainflowConcentration);
    }
    if (
      brainflowRestfulness !== null &&
      brainflowRestfulness !== undefined &&
      this.baselineBrainflowRestfulnessValues.length < this.baselineSampleCount
    ) {
      this.baselineBrainflowRestfulnessValues.push(brainflowRestfulness);
    }
    if (this.baselineFocusValues.length < this.baselineSampleCount) {
      this.baselineFocusValues.push(neurofeedbackScores.focusSigned);
    }
    if (this.baselineRelaxValues.length < this.baselineSampleCount) {
      this.baselineRelaxValues.push(neurofeedbackScores.relaxSigned);
    }

    const baselineRatio =
      this.baselineRatios.length > 0
        ? median(this.baselineRatios)
        : null;
    const baselineBrainflowMindfulness =
      this.baselineBrainflowMindfulnessValues.length > 0
        ? median(this.baselineBrainflowMindfulnessValues)
        : null;
    const baselineBrainflowRestfulness =
      this.baselineBrainflowRestfulnessValues.length > 0
        ? median(this.baselineBrainflowRestfulnessValues)
        : null;
    const ratioStats = robustStats(this.baselineRatios);
    const brainflowStats = robustStats(this.baselineBrainflowMindfulnessValues);
    const brainflowRestfulnessStats = robustStats(this.baselineBrainflowRestfulnessValues);
    const focusStats = robustStats(this.baselineFocusValues);
    const relaxStats = robustStats(this.baselineRelaxValues);
    const brainflowRelativeValue =
      brainflowConcentration !== null &&
      brainflowConcentration !== undefined &&
      baselineBrainflowMindfulness &&
      baselineBrainflowMindfulness > 0
        ? brainflowConcentration / baselineBrainflowMindfulness
        : null;
    const brainflowZScore =
      brainflowConcentration !== null && brainflowConcentration !== undefined && brainflowStats
        ? (brainflowConcentration - brainflowStats.median) / brainflowStats.spread
        : null;
    const brainflowRestfulnessRelativeValue =
      brainflowRestfulness !== null &&
      brainflowRestfulness !== undefined &&
      baselineBrainflowRestfulness &&
      baselineBrainflowRestfulness > 0
        ? brainflowRestfulness / baselineBrainflowRestfulness
        : null;
    const brainflowRestfulnessZScore =
      brainflowRestfulness !== null &&
      brainflowRestfulness !== undefined &&
      brainflowRestfulnessStats
        ? (brainflowRestfulness - brainflowRestfulnessStats.median) /
          brainflowRestfulnessStats.spread
        : null;
    const focusZScore = focusStats
      ? (neurofeedbackScores.focusSigned - focusStats.median) / focusStats.spread
      : null;
    const relaxZScore = relaxStats
      ? (neurofeedbackScores.relaxSigned - relaxStats.median) / relaxStats.spread
      : null;
    const mappedFocusScore = mapZScoreToScore(focusZScore);
    const mappedRelaxScore = mapZScoreToScore(relaxZScore);
    const baselineRelativeValue = brainflowRelativeValue;
    const baselineZScore = brainflowZScore;
    const restfulnessBaselineRelativeValue = brainflowRestfulnessRelativeValue;
    const restfulnessBaselineZScore = brainflowRestfulnessZScore;
    if (this.baselineRatios.length >= this.baselineSampleCount && !this.calibrationProfile) {
      this.calibrationProfile = {
        id: crypto.randomUUID(),
        algorithmVersion: "brainflow-mindfulness-v1",
        createdAtMs: Date.now(),
        baselineRatio,
        baselineBrainflowMindfulness,
        baselineRatioSpread: ratioStats?.spread ?? null,
        baselineBrainflowMindfulnessSpread: brainflowStats?.spread ?? null,
        baselineBrainflowRestfulness,
        baselineBrainflowRestfulnessSpread: brainflowRestfulnessStats?.spread ?? null,
        acceptedWindows: this.baselineRatios.length,
        rejectedWindows: this.rejectedWindows,
        rejectionReasons: { ...this.rejectionReasons },
      };
    }
    const mappedScore = mapZScoreToScore(baselineZScore);
    const mappedRestfulnessScore = mapZScoreToScore(restfulnessBaselineZScore);
    // BrainFlow did not (yet) provide a mindfulness/restfulness reading for
    // this window -- report that honestly rather than substituting an
    // estimate derived from something else.
    const directMindfulnessScore =
      brainflowConcentration !== null && brainflowConcentration !== undefined
        ? clamp(brainflowConcentration * 100, 0, 100)
        : null;
    const directRestfulnessScore =
      brainflowRestfulness !== null && brainflowRestfulness !== undefined
        ? clamp(brainflowRestfulness * 100, 0, 100)
        : null;
    const displayScore = this.useBaselineRelativeDisplay ? mappedScore : directMindfulnessScore;
    const restfulnessDisplayScore = this.useBaselineRelativeDisplay
      ? mappedRestfulnessScore
      : directRestfulnessScore;
    if (displayScore === null) {
      this.rejectWindow("missing_metric");
    }

    this.smoothedScore =
      displayScore === null
        ? null
        : smoothScore(this.smoothedScore, displayScore, this.smoothingAlpha);
    this.smoothedRestfulnessScore =
      restfulnessDisplayScore === null
        ? null
        : smoothScore(
            this.smoothedRestfulnessScore,
            restfulnessDisplayScore,
            this.smoothingAlpha,
          );
    this.smoothedFocusScore = smoothScore(
      this.smoothedFocusScore,
      this.useBaselineRelativeDisplay
        ? mappedFocusScore ?? neurofeedbackScores.focusScore
        : neurofeedbackScores.focusScore,
      this.smoothingAlpha,
    );
    this.smoothedRelaxScore = smoothScore(
      this.smoothedRelaxScore,
      this.useBaselineRelativeDisplay
        ? mappedRelaxScore ?? neurofeedbackScores.relaxScore
        : neurofeedbackScores.relaxScore,
      this.smoothingAlpha,
    );

    return {
      metricId: "attention-index-experimental",
      label: "BrainFlow Mindfulness",
      atMs: frame.receivedAtMs,
      displayedScore:
        this.smoothedScore === null ? null : Math.round(clamp(this.smoothedScore, 0, 100)),
      restfulnessScore:
        this.smoothedRestfulnessScore === null
          ? null
          : Math.round(clamp(this.smoothedRestfulnessScore, 0, 100)),
      focusScore: Math.round(clamp(this.smoothedFocusScore, 0, 100)),
      relaxScore: Math.round(clamp(this.smoothedRelaxScore, 0, 100)),
      thetaPower,
      alphaPower,
      betaPower,
      rawRatio,
      baselineRatio,
      rawBrainflowMindfulness: brainflowConcentration ?? null,
      baselineBrainflowMindfulness,
      baselineRelativeValue,
      baselineZScore,
      rawBrainflowRestfulness: brainflowRestfulness ?? null,
      baselineBrainflowRestfulness,
      restfulnessBaselineRelativeValue,
      restfulnessBaselineZScore,
      scoreSource:
        brainflowConcentration !== null && brainflowConcentration !== undefined
          ? "brainflow_mindfulness"
          : "unavailable",
      restfulnessScoreSource:
        brainflowRestfulness !== null && brainflowRestfulness !== undefined
          ? "brainflow_restfulness"
          : "unavailable",
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

function mapZScoreToScore(value: number | null): number | null {
  if (value === null || !Number.isFinite(value)) return null;

  return 50 + Math.tanh(value / calibratedZScoreScale) * 45;
}

function readBrainflowMetric(frame: SignalFrame, keys: string[]) {
  const features = frame.features as
    | (SignalFrame["features"] & Record<string, number | null | undefined>)
    | null
    | undefined;

  for (const key of keys) {
    const value = features?.[key];
    if (value !== null && value !== undefined) return value;
  }

  return null;
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

function robustStats(values: number[]) {
  if (values.length < 4) return null;

  const center = median(values);
  const deviations = values.map((value) => Math.abs(value - center));
  const mad = median(deviations);
  const fallbackSpread = range(values) / 6;
  const spread = Math.max(mad * 1.4826, fallbackSpread);

  if (!Number.isFinite(spread) || spread <= 0) return null;

  return { median: center, spread };
}

function range(values: number[]) {
  return Math.max(...values) - Math.min(...values);
}
