import type { DeviceInfo, SignalChannel } from "../domain/eeg";
import type { AttentionMetricSample } from "../metrics/attentionMetric";
import type { HeadsetFitSnapshot } from "../signalQuality/headsetFitProvider";

export type TrainingSessionState = "idle" | "running" | "paused" | "ended";

export interface TrainingTimelineEvent {
  atMs: number;
  label: string;
}

export interface TrainingSessionSnapshot {
  state: TrainingSessionState;
  startedAtMs: number | null;
  endedAtMs: number | null;
  elapsedMs: number;
  samples: AttentionMetricSample[];
  qualitySnapshots: HeadsetFitSnapshot[];
  timeline: TrainingTimelineEvent[];
  deviceInfo: DeviceInfo | null;
}

export interface TrainingPeriod {
  label: string;
  atMs: number;
  score: number;
}

export interface TrainingSessionReport {
  durationMs: number;
  averageAttention: number | null;
  peakAttention: number | null;
  highestPeriods: TrainingPeriod[];
  lowestPeriods: TrainingPeriod[];
  attentionSeries: AttentionMetricSample[];
  bandPowerSeries: Array<{
    atMs: number;
    thetaPower: number;
    alphaPower: number;
    betaPower: number;
  }>;
  ratioSeries: Array<{
    atMs: number;
    rawRatio: number;
    baselineRelativeValue: number | null;
  }>;
  qualityTimeline: HeadsetFitSnapshot[];
  unreliablePeriods: TrainingPeriod[];
  signalQuality: {
    eegChannelCount: number;
    eegChannels: SignalChannel[];
    sampleCount: number;
  };
  timeline: TrainingTimelineEvent[];
}

export class TrainingSession {
  private state: TrainingSessionState = "idle";
  private startedAtMs: number | null = null;
  private endedAtMs: number | null = null;
  private pausedAtMs: number | null = null;
  private pausedDurationMs = 0;
  private samples: AttentionMetricSample[] = [];
  private qualitySnapshots: HeadsetFitSnapshot[] = [];
  private timeline: TrainingTimelineEvent[] = [];

  constructor(private readonly getNow = () => performance.now()) {}

  start() {
    const now = this.getNow();
    this.state = "running";
    this.startedAtMs = now;
    this.endedAtMs = null;
    this.pausedAtMs = null;
    this.pausedDurationMs = 0;
    this.samples = [];
    this.qualitySnapshots = [];
    this.timeline = [{ atMs: now, label: "Session started" }];
  }

  pause() {
    if (this.state !== "running") return;

    const now = this.getNow();
    this.state = "paused";
    this.pausedAtMs = now;
    this.timeline.push({ atMs: now, label: "Session paused" });
  }

  resume() {
    if (this.state !== "paused") return;

    const now = this.getNow();
    if (this.pausedAtMs !== null) {
      this.pausedDurationMs += now - this.pausedAtMs;
    }
    this.pausedAtMs = null;
    this.state = "running";
    this.timeline.push({ atMs: now, label: "Session resumed" });
  }

  end() {
    if (this.state === "idle" || this.state === "ended") return;

    const now = this.getNow();
    if (this.pausedAtMs !== null) {
      this.pausedDurationMs += now - this.pausedAtMs;
      this.pausedAtMs = null;
    }
    this.state = "ended";
    this.endedAtMs = now;
    this.timeline.push({ atMs: now, label: "Session ended" });
  }

  addSample(sample: AttentionMetricSample) {
    if (this.state !== "running") return;

    this.samples.push(sample);
  }

  addTimelineEvent(label: string) {
    if (this.state === "idle" || this.state === "ended") return;

    this.timeline.push({ atMs: this.getNow(), label });
  }

  addQualitySnapshot(snapshot: HeadsetFitSnapshot) {
    if (this.state !== "running") return;

    const previous = this.qualitySnapshots[this.qualitySnapshots.length - 1];
    if (previous?.updatedAtMs === snapshot.updatedAtMs) return;

    this.qualitySnapshots.push(snapshot);
  }

  snapshot(deviceInfo: DeviceInfo | null): TrainingSessionSnapshot {
    return {
      state: this.state,
      startedAtMs: this.startedAtMs,
      endedAtMs: this.endedAtMs,
      elapsedMs: this.elapsedMs(),
      samples: [...this.samples],
      qualitySnapshots: [...this.qualitySnapshots],
      timeline: [...this.timeline],
      deviceInfo,
    };
  }

  report(deviceInfo: DeviceInfo | null): TrainingSessionReport {
    const snapshot = this.snapshot(deviceInfo);
    const scores = snapshot.samples.map((sample) => sample.displayedScore);
    const eegCapability = deviceInfo?.capabilities.find(
      (capability) => capability.kind === "eeg",
    );

    return {
      durationMs: snapshot.elapsedMs,
      averageAttention: scores.length ? average(scores) : null,
      peakAttention: scores.length ? Math.max(...scores) : null,
      highestPeriods: pickPeriods(snapshot.samples, "highest"),
      lowestPeriods: pickPeriods(snapshot.samples, "lowest"),
      attentionSeries: snapshot.samples,
      bandPowerSeries: snapshot.samples.map((sample) => ({
        atMs: sample.atMs,
        thetaPower: sample.thetaPower,
        alphaPower: sample.alphaPower,
        betaPower: sample.betaPower,
      })),
      ratioSeries: snapshot.samples.map((sample) => ({
        atMs: sample.atMs,
        rawRatio: sample.rawRatio,
        baselineRelativeValue: sample.baselineRelativeValue,
      })),
      qualityTimeline: compactQualityTimeline(snapshot.qualitySnapshots),
      unreliablePeriods: pickUnreliablePeriods(snapshot.qualitySnapshots),
      signalQuality: {
        eegChannelCount: eegCapability?.channels.length ?? 0,
        eegChannels: eegCapability?.channels ?? [],
        sampleCount: snapshot.samples.reduce(
          (total, sample) => total + sample.sampleCount,
          0,
        ),
      },
      timeline: snapshot.timeline,
    };
  }

  private elapsedMs() {
    if (this.startedAtMs === null) return 0;

    const end = this.endedAtMs ?? this.getNow();
    const currentPause =
      this.pausedAtMs !== null && this.state === "paused" ? end - this.pausedAtMs : 0;

    return Math.max(0, end - this.startedAtMs - this.pausedDurationMs - currentPause);
  }
}

function pickPeriods(
  samples: AttentionMetricSample[],
  direction: "highest" | "lowest",
): TrainingPeriod[] {
  return [...samples]
    .sort((a, b) =>
      direction === "highest"
        ? b.displayedScore - a.displayedScore
        : a.displayedScore - b.displayedScore,
    )
    .slice(0, 3)
    .map((sample, index) => ({
      label: `${direction === "highest" ? "High" : "Low"} ${index + 1}`,
      atMs: sample.atMs,
      score: sample.displayedScore,
    }));
}

function average(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function compactQualityTimeline(snapshots: HeadsetFitSnapshot[]): HeadsetFitSnapshot[] {
  return snapshots.filter((snapshot, index) => {
    const previous = snapshots[index - 1];
    if (!previous) return true;

    return (
      previous.state !== snapshot.state ||
      previous.message !== snapshot.message ||
      previous.excessiveArtifact !== snapshot.excessiveArtifact ||
      previous.blockers.join("|") !== snapshot.blockers.join("|")
    );
  });
}

function pickUnreliablePeriods(snapshots: HeadsetFitSnapshot[]): TrainingPeriod[] {
  return compactQualityTimeline(snapshots)
    .filter((snapshot) => !snapshot.ready && snapshot.state !== "good")
    .slice(-6)
    .map((snapshot, index) => ({
      label: snapshot.message || `Poor signal ${index + 1}`,
      atMs: snapshot.updatedAtMs,
      score: Math.max(0, Math.round((snapshot.stableForMs / snapshot.requiredStableMs) * 100)),
    }));
}
