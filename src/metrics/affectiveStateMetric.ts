import type { SignalFrame } from "../domain/eeg";
import {
  frontalAlphaAsymmetry,
  frontalChannels,
  groupedPowers,
  resolveBandPowers,
  temporalChannels,
} from "./bandPowerChannels";
import { baselineSampleCount, vrchatStyleEmaDecay } from "./metricConfig";
import {
  computeBrainflowsNeurofeedbackScores,
  smoothScore,
} from "./neurofeedbackRatios";
import type { FrequencyBand } from "../signalProcessing/bandPower";
import type { HeadsetFitSnapshot } from "../signalQuality/headsetFitProvider";
const neutralRadius = 0.18;

const affectiveBands = [
  { id: "theta", label: "Theta", lowHz: 4, highHz: 8 },
  { id: "alpha", label: "Alpha", lowHz: 8, highHz: 13 },
  { id: "beta", label: "Beta", lowHz: 13, highHz: 30 },
  { id: "gamma", label: "Gamma", lowHz: 30, highHz: 45 },
] satisfies FrequencyBand[];

export interface AffectiveEmotionRegion {
  label: string;
  valence: number;
  arousal: number;
}

export const affectiveEmotionRegions = [
  { label: "Tense", valence: -0.25, arousal: 0.78 },
  { label: "Angry", valence: -0.68, arousal: 0.55 },
  { label: "Frustrated", valence: -0.72, arousal: 0.25 },
  { label: "Depressed", valence: -0.74, arousal: -0.25 },
  { label: "Bored", valence: -0.58, arousal: -0.58 },
  { label: "Tired", valence: -0.25, arousal: -0.82 },
  { label: "Calm", valence: 0.25, arousal: -0.82 },
  { label: "Relaxed", valence: 0.58, arousal: -0.58 },
  { label: "Content", valence: 0.72, arousal: -0.25 },
  { label: "Happy", valence: 0.72, arousal: 0.25 },
  { label: "Delighted", valence: 0.62, arousal: 0.55 },
  { label: "Excited", valence: 0.32, arousal: 0.78 },
] satisfies AffectiveEmotionRegion[];

export interface AffectiveStateSample {
  atMs: number;
  valence: number;
  arousal: number;
  rawValence: number;
  rawArousal: number;
  calibrationActive: boolean;
  label: string;
  confidence: number;
  scoreSource: "eeg_band_power_proxy";
  thetaPower: number;
  alphaPower: number;
  betaPower: number;
  gammaPower: number;
  brainflowMindfulnessScore: number | null;
  brainflowRestfulnessScore: number | null;
  focusScore: number;
  relaxScore: number;
  reliable: boolean;
}

export interface AffectiveCalibrationState {
  status: "off" | "collecting" | "active";
  progress: number;
  required: number;
}

export class AffectiveStateProvider {
  private smoothedValence: number | null = null;
  private smoothedArousal: number | null = null;
  private smoothedBrainflowMindfulnessScore: number | null = null;
  private smoothedBrainflowRestfulnessScore: number | null = null;
  private smoothedFocusScore: number | null = null;
  private smoothedRelaxScore: number | null = null;
  private calibrationStatus: AffectiveCalibrationState["status"] = "off";
  private readonly calibrationSampleCount = baselineSampleCount;
  private calibrationValenceValues: number[] = [];
  private calibrationArousalValues: number[] = [];
  private calibrationProfile: { valence: number; arousal: number } | null = null;
  private recentValence: number[] = [];
  private recentArousal: number[] = [];

  constructor(private readonly smoothingAlpha = vrchatStyleEmaDecay) {}

  reset() {
    this.smoothedValence = null;
    this.smoothedArousal = null;
    this.smoothedBrainflowMindfulnessScore = null;
    this.smoothedBrainflowRestfulnessScore = null;
    this.smoothedFocusScore = null;
    this.smoothedRelaxScore = null;
    this.recentValence = [];
    this.recentArousal = [];
    this.resetCalibration();
  }

  startCalibration() {
    this.calibrationStatus = "collecting";
    this.calibrationValenceValues = [];
    this.calibrationArousalValues = [];
    this.calibrationProfile = null;
    this.smoothedValence = null;
    this.smoothedArousal = null;
    this.recentValence = [];
    this.recentArousal = [];
  }

  resetCalibration() {
    this.calibrationStatus = "off";
    this.calibrationValenceValues = [];
    this.calibrationArousalValues = [];
    this.calibrationProfile = null;
    this.smoothedValence = null;
    this.smoothedArousal = null;
    this.recentValence = [];
    this.recentArousal = [];
  }

  getCalibrationState(): AffectiveCalibrationState {
    return {
      status: this.calibrationStatus,
      progress:
        this.calibrationStatus === "collecting"
          ? Math.min(this.calibrationValenceValues.length, this.calibrationSampleCount)
          : this.calibrationStatus === "active"
            ? this.calibrationSampleCount
            : 0,
      required: this.calibrationSampleCount,
    };
  }

  pushFrame(
    frame: SignalFrame,
    quality?: HeadsetFitSnapshot | null,
  ): AffectiveStateSample | null {
    if (quality?.excessiveArtifact) return null;

    const resolved = resolveBandPowers(frame, affectiveBands, quality);
    if (!resolved) return null;

    const powers = resolved.powers;
    const thetaPower = finitePower(powers.theta);
    const alphaPower = finitePower(powers.alpha);
    const betaPower = finitePower(powers.beta);
    const gammaPower = finitePower(powers.gamma);
    if (thetaPower + alphaPower + betaPower + gammaPower <= 0) return null;

    const frontal = groupedPowers(
      resolved.perChannel,
      frontalChannels,
      powers,
      resolved.usableChannelKeys,
    );
    const temporal = groupedPowers(
      resolved.perChannel,
      temporalChannels,
      powers,
      resolved.usableChannelKeys,
    );
    const arousalAlpha = frontal.alpha || alphaPower;
    const arousalTheta = frontal.theta || thetaPower;
    const arousalBeta = frontal.beta || betaPower;
    const rawArousal = mapRatioToAxis(
      arousalBeta / (arousalAlpha + arousalTheta + 1e-9),
    );
    const faa = frontalAlphaAsymmetry(
      resolved.perChannel,
      resolved.usableChannelKeys,
    );
    const rawValence =
      faa === null
        ? mapRatioToAxis(alphaPower / (thetaPower + betaPower + 1e-9))
        : clamp(Math.tanh(faa / 0.8), -1, 1);
    this.acceptCalibrationSample(rawValence, rawArousal);
    const calibratedValence =
      this.calibrationProfile === null
        ? rawValence
        : clamp(rawValence - this.calibrationProfile.valence, -1, 1);
    const calibratedArousal =
      this.calibrationProfile === null
        ? rawArousal
        : clamp(rawArousal - this.calibrationProfile.arousal, -1, 1);
    const focusScores = computeBrainflowsNeurofeedbackScores({
      thetaPower: frontal.theta || thetaPower,
      alphaPower: frontal.alpha || alphaPower,
      betaPower: frontal.beta || betaPower,
    });
    const relaxScores = computeBrainflowsNeurofeedbackScores({
      thetaPower: temporal.theta || thetaPower,
      alphaPower: temporal.alpha || alphaPower,
      betaPower: temporal.beta || betaPower,
    });
    const rawBrainflowMindfulnessScore = normalizeBrainflowMindfulness(
      readBrainflowMetric(frame, [
        "brainflowConcentration",
        "brainflow_concentration",
        "mindfulness",
      ]),
    );
    const rawBrainflowRestfulnessScore = normalizeBrainflowMindfulness(
      readBrainflowMetric(frame, [
        "brainflowRestfulness",
        "brainflow_restfulness",
        "restfulness",
      ]),
    );

    this.smoothedValence =
      this.smoothedValence === null
        ? calibratedValence
        : smooth(this.smoothedValence, calibratedValence, this.smoothingAlpha);
    this.smoothedArousal =
      this.smoothedArousal === null
        ? calibratedArousal
        : smooth(this.smoothedArousal, calibratedArousal, this.smoothingAlpha);
    this.smoothedBrainflowMindfulnessScore =
      rawBrainflowMindfulnessScore === null
        ? null
        : smoothScore(
            this.smoothedBrainflowMindfulnessScore,
            rawBrainflowMindfulnessScore,
            this.smoothingAlpha,
          );
    this.smoothedBrainflowRestfulnessScore =
      rawBrainflowRestfulnessScore === null
        ? null
        : smoothScore(
            this.smoothedBrainflowRestfulnessScore,
            rawBrainflowRestfulnessScore,
            this.smoothingAlpha,
          );
    this.smoothedFocusScore = smoothScore(
      this.smoothedFocusScore,
      focusScores.focusScore,
      this.smoothingAlpha,
    );
    this.smoothedRelaxScore = smoothScore(
      this.smoothedRelaxScore,
      relaxScores.relaxScore,
      this.smoothingAlpha,
    );

    const valence = clamp(this.smoothedValence, -1, 1);
    const arousal = clamp(this.smoothedArousal, -1, 1);
    this.recentValence.push(valence);
    this.recentArousal.push(arousal);
    if (this.recentValence.length > 12) {
      this.recentValence.shift();
      this.recentArousal.shift();
    }
    const brainflowMindfulnessScore =
      rawBrainflowMindfulnessScore === null ||
      this.smoothedBrainflowMindfulnessScore === null
        ? null
        : Math.round(clamp(this.smoothedBrainflowMindfulnessScore, 0, 100));
    const brainflowRestfulnessScore =
      rawBrainflowRestfulnessScore === null ||
      this.smoothedBrainflowRestfulnessScore === null
        ? null
        : Math.round(clamp(this.smoothedBrainflowRestfulnessScore, 0, 100));

    return {
      atMs: frame.receivedAtMs,
      valence,
      arousal,
      rawValence,
      rawArousal,
      calibrationActive: this.calibrationProfile !== null,
      label: classifyAffectiveState(valence, arousal),
      confidence: estimateConfidence(this.recentValence, this.recentArousal, quality),
      scoreSource: "eeg_band_power_proxy",
      thetaPower,
      alphaPower,
      betaPower,
      gammaPower,
      brainflowMindfulnessScore,
      brainflowRestfulnessScore,
      focusScore: Math.round(clamp(this.smoothedFocusScore, 0, 100)),
      relaxScore: Math.round(clamp(this.smoothedRelaxScore, 0, 100)),
      reliable: !quality?.excessiveArtifact,
    };
  }

  private acceptCalibrationSample(rawValence: number, rawArousal: number) {
    if (this.calibrationStatus !== "collecting") return;

    this.calibrationValenceValues.push(rawValence);
    this.calibrationArousalValues.push(rawArousal);

    if (this.calibrationValenceValues.length < this.calibrationSampleCount) return;

    this.calibrationProfile = {
      valence: median(this.calibrationValenceValues),
      arousal: median(this.calibrationArousalValues),
    };
    this.calibrationStatus = "active";
    this.smoothedValence = null;
    this.smoothedArousal = null;
  }
}

function finitePower(value: number | undefined) {
  return Number.isFinite(value) ? Math.max(0, value ?? 0) : 0;
}

function mapRatioToAxis(ratio: number) {
  if (!Number.isFinite(ratio) || ratio <= 0) return 0;

  return clamp(Math.tanh(Math.log2(ratio) / 2.5), -1, 1);
}

function smooth(current: number, target: number, weight: number) {
  return current * (1 - weight) + target * weight;
}

function normalizeBrainflowMindfulness(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;

  return clamp(value * 100, 0, 100);
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

function classifyAffectiveState(valence: number, arousal: number) {
  if (Math.hypot(valence, arousal) < neutralRadius) return "Neutral";

  let nearest = affectiveEmotionRegions[0];
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (const region of affectiveEmotionRegions) {
    const distance = Math.hypot(
      valence - region.valence,
      arousal - region.arousal,
    );
    if (distance < nearestDistance) {
      nearest = region;
      nearestDistance = distance;
    }
  }

  return nearest.label;
}

function estimateConfidence(
  recentValence: number[],
  recentArousal: number[],
  quality?: HeadsetFitSnapshot | null,
) {
  const qualityFactor = quality?.ready ? 1 : quality?.state === "good" ? 0.75 : 0.45;
  const stability =
    recentValence.length < 4
      ? 0.5
      : 1 / (1 + 4 * (stdDev(recentValence) + stdDev(recentArousal)));

  return clamp(qualityFactor * stability, 0, 1);
}

function stdDev(values: number[]) {
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance =
    values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}
