import { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart3,
  CirclePause,
  CirclePlay,
  ClipboardList,
  Gauge,
  LineChart,
  Play,
  Square,
  Timer,
  Waves,
} from "lucide-react";
import type { DeviceInfo, EegConnectionState, SignalFrame } from "../../domain/eeg";
import {
  HeuristicAttentionProvider,
  type AttentionMetricSample,
} from "../../metrics/attentionMetric";
import {
  TrainingSession,
  type TrainingSessionReport,
  type TrainingSessionState,
} from "../../training/trainingSession";
import type { HeadsetFitSnapshot } from "../../signalQuality/headsetFitProvider";
import { SeriesChart } from "./SeriesChart";

interface TrainingSectionProps {
  connectionState: EegConnectionState;
  deviceInfo: DeviceInfo | null;
  latestFrame: SignalFrame | null;
  fit: HeadsetFitSnapshot;
}

const defaultVideoUrl = "https://www.youtube.com/watch?v=uyb0wW0ln_g";
const baselineSamplesRequired = 24;
type TrainingPhase =
  | "idle"
  | "headset_check"
  | "baseline"
  | "calibrated"
  | "training"
  | "paused"
  | "ended";

export function TrainingSection({
  connectionState,
  deviceInfo,
  latestFrame,
  fit,
}: TrainingSectionProps) {
  const metricProviderRef = useRef(new HeuristicAttentionProvider());
  const sessionRef = useRef(new TrainingSession());
  const lastFrameSequenceRef = useRef<number | null>(null);
  const phaseBeforePauseRef = useRef<TrainingPhase>("idle");
  const [sessionState, setSessionState] = useState<TrainingSessionState>("idle");
  const [elapsedMs, setElapsedMs] = useState(0);
  const [attentionSamples, setAttentionSamples] = useState<AttentionMetricSample[]>([]);
  const [report, setReport] = useState<TrainingSessionReport | null>(null);
  const [phase, setPhase] = useState<TrainingPhase>("idle");
  const [baselineProgress, setBaselineProgress] = useState(0);
  const [hasStartedTraining, setHasStartedTraining] = useState(false);

  const eegStreaming = connectionState === "streaming" && latestFrame !== null;
  const currentScore =
    eegStreaming && attentionSamples.length > 0
      ? attentionSamples[attentionSamples.length - 1].displayedScore
      : null;
  const showVideo =
    phase === "training" || (phase === "paused" && hasStartedTraining);
  const videoEmbedUrl = useMemo(() => toYoutubeEmbedUrl(defaultVideoUrl, showVideo), [showVideo]);
  const isRunning = sessionState === "running";
  const isPaused = sessionState === "paused";
  const canPause = isRunning || isPaused;
  const canEnd = isRunning || isPaused;
  const qualityAllowsScoring = eegStreaming && !fit.excessiveArtifact;
  const primaryAction = getPrimaryAction(phase, isRunning, isPaused);

  useEffect(() => {
    if (!isRunning) return;

    const intervalId = window.setInterval(() => {
      setElapsedMs(sessionRef.current.snapshot(deviceInfo).elapsedMs);
    }, 250);

    return () => window.clearInterval(intervalId);
  }, [deviceInfo, isRunning]);

  useEffect(() => {
    if (phase !== "headset_check" || !eegStreaming) return;

    metricProviderRef.current.reset();
    sessionRef.current.addTimelineEvent(
      fit.ready ? "Stable signal" : "Headset fit check bypassed",
    );
    sessionRef.current.addTimelineEvent("Baseline calibration started");
    setBaselineProgress(0);
    setPhase("baseline");
  }, [eegStreaming, fit.ready, phase]);

  useEffect(() => {
    if (phase !== "baseline") return;
    if (!eegStreaming || fit.excessiveArtifact) {
      metricProviderRef.current.reset();
      sessionRef.current.addTimelineEvent("Baseline reset");
      setBaselineProgress(0);
      setPhase("headset_check");
    }
  }, [eegStreaming, fit.excessiveArtifact, phase]);

  useEffect(() => {
    if (!latestFrame || sessionState !== "running") return;
    if (phase !== "baseline" && phase !== "training") return;
    if (lastFrameSequenceRef.current === latestFrame.sequenceId) return;

    lastFrameSequenceRef.current = latestFrame.sequenceId;
    sessionRef.current.addQualitySnapshot(fit);
    if (!qualityAllowsScoring) return;

    const sample = metricProviderRef.current.pushFrame(latestFrame, fit);
    if (!sample) return;

    if (phase === "baseline") {
      const nextBaselineProgress = Math.min(
        baselineSamplesRequired,
        baselineProgress + 1,
      );
      setBaselineProgress(nextBaselineProgress);
      if (nextBaselineProgress >= baselineSamplesRequired) {
        sessionRef.current.addTimelineEvent("Baseline calibration complete");
        setPhase("calibrated");
      }
      return;
    }

    sessionRef.current.addSample(sample);
    setAttentionSamples((current) => [...current, sample]);
  }, [
    baselineProgress,
    fit,
    latestFrame,
    phase,
    qualityAllowsScoring,
    sessionState,
  ]);

  function startCalibration() {
    metricProviderRef.current.reset();
    sessionRef.current.start();
    sessionRef.current.addTimelineEvent("Headset check started");
    lastFrameSequenceRef.current = null;
    setAttentionSamples([]);
    setReport(null);
    setElapsedMs(0);
    setBaselineProgress(0);
    setHasStartedTraining(false);
    setSessionState("running");
    setPhase("headset_check");
  }

  function beginTraining() {
    sessionRef.current.addTimelineEvent("Training started");
    setAttentionSamples([]);
    setHasStartedTraining(true);
    setPhase("training");
  }

  function handlePrimaryAction() {
    if (phase === "calibrated") {
      beginTraining();
      return;
    }

    startCalibration();
  }

  function togglePause() {
    if (sessionState === "running") {
      sessionRef.current.pause();
      phaseBeforePauseRef.current = phase;
      setSessionState("paused");
      setPhase("paused");
      setElapsedMs(sessionRef.current.snapshot(deviceInfo).elapsedMs);
      return;
    }

    if (sessionState === "paused") {
      sessionRef.current.resume();
      setSessionState("running");
      setPhase(phaseBeforePauseRef.current === "paused" ? "training" : phaseBeforePauseRef.current);
    }
  }

  function endSession() {
    sessionRef.current.end();
    setElapsedMs(sessionRef.current.snapshot(deviceInfo).elapsedMs);
    setReport(sessionRef.current.report(deviceInfo));
    setSessionState("ended");
    setPhase("ended");
  }

  return (
    <section className="training-page" aria-label="Training">
      <header className="training-header">
        <div>
          <h1>Training</h1>
          <p>Run experimental EEG-derived training sessions from normalized signal streams.</p>
        </div>
      </header>

      <section className="training-layout">
        <article className="panel training-video-card">
          <div className="panel-header">
            <div className="panel-title">
              <div className="icon-tile">
                <Play aria-hidden="true" />
              </div>
              <div>
              <h2>Focus Training — YouTube</h2>
                <p>Watch a video while tracking a BrainFlow-derived Attention Index.</p>
              </div>
            </div>
          </div>
          <div className="training-progress-strip">
            <div className="progress-strip-copy">
              <span>Session setup</span>
              <strong>{phaseLabel(phase)}</strong>
              <small>
                {phase === "baseline" && qualityAllowsScoring
                  ? `Baseline calibration ${baselineProgress}/${baselineSamplesRequired} samples.`
                  : phase === "baseline"
                    ? "Calibration waits for EEG stream and excessive-artifact checks only."
                  : fit.blockers[0] ?? fit.message}
              </small>
            </div>
            <div className="gate-steps is-horizontal">
              {[
                "Connect device",
                "Headset check",
                "Stable signal",
                "Baseline calibration",
                "Training",
              ].map((step, index) => (
                <div
                  className={`gate-step ${
                    index <= activeGateIndex(phase, connectionState, fit.ready)
                      ? "is-active"
                      : ""
                  }`}
                  key={step}
                >
                  <small>{index + 1}</small>
                  <span>{step}</span>
                </div>
              ))}
            </div>
            {phase !== "idle" && (
              <div className="fit-progress training-fit-progress" aria-hidden="true">
                <span
                  style={{
                    width:
                      phase === "baseline"
                        ? `${(baselineProgress / baselineSamplesRequired) * 100}%`
                        : `${Math.min(100, (fit.stableForMs / fit.requiredStableMs) * 100)}%`,
                  }}
                />
              </div>
            )}
          </div>
          <div className="video-frame">
            {showVideo ? (
              <iframe
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
                src={videoEmbedUrl}
                title="Focus training YouTube video"
              />
            ) : (
              <div className="video-placeholder">
                <div className="icon-tile">
                  <Play aria-hidden="true" />
                </div>
                <strong>Video starts after calibration</strong>
                <span>Complete the headset check and baseline before training.</span>
              </div>
            )}
          </div>
        </article>

        <aside className="training-side-panel">
          <article className="panel training-session-card">
            <div className="side-panel-title">
              <div className="icon-tile">
                <Gauge aria-hidden="true" />
              </div>
              <div>
                <h2>Attention Index — Experimental</h2>
                <p>BrainFlow Mindfulness output, with diagnostic fallback if unavailable.</p>
              </div>
            </div>
            <div className="attention-score">
              <strong>{currentScore === null ? "--" : currentScore}</strong>
              <span>0-100</span>
            </div>
            <p className="metric-note">
              Experimental feedback only. This is not a validated or clinical measure
              of focus.
            </p>
          </article>

          <article className="panel session-stats-card">
            <div className="stat-line">
              <Timer aria-hidden="true" />
              <span>Elapsed</span>
              <strong>{formatDuration(elapsedMs)}</strong>
            </div>
            <div className="stat-line">
              <Waves aria-hidden="true" />
              <span>Signal quality</span>
              <strong>{fit.message}</strong>
            </div>
          </article>

          <article className="panel session-actions">
            <button
              className="primary-button"
              onClick={handlePrimaryAction}
              disabled={primaryAction.disabled}
            >
              <CirclePlay aria-hidden="true" />
              {primaryAction.label}
            </button>
            <button
              className="secondary-button"
              onClick={togglePause}
              disabled={!canPause}
            >
              <CirclePause aria-hidden="true" />
              {isPaused ? "Resume" : "Pause"}
            </button>
            <button className="secondary-button" onClick={endSession} disabled={!canEnd}>
              <Square aria-hidden="true" />
              End Session
            </button>
          </article>
        </aside>
      </section>

      <section className="panel attention-chart-card">
        <div className="panel-header">
          <div className="panel-title">
            <div className="icon-tile">
              <LineChart aria-hidden="true" />
            </div>
            <div>
              <h2>Attention Index Over Time</h2>
              <p>Smoothed BrainFlow Mindfulness output from reliable training frames.</p>
            </div>
          </div>
          <span className="panel-meta">
            {attentionSamples.length} points
          </span>
        </div>
        <SeriesChart
          emptyTitle="Start a focus session"
          emptyDescription="Attention Index samples will appear as normalized EEG frames arrive."
          height={220}
          lines={[
            {
              label: "Attention Index",
              color: "#a78bfa",
              values: attentionSamples.map((sample) => ({
                atMs: sample.atMs,
                value: sample.displayedScore,
              })),
            },
          ]}
          min={0}
          max={100}
        />
      </section>

      {report && <TrainingReport report={report} />}
    </section>
  );
}

function TrainingReport({ report }: { report: TrainingSessionReport }) {
  const sessionStartMs = report.timeline[0]?.atMs;

  return (
    <section className="training-report">
      <div className="panel-header report-heading">
        <div className="panel-title">
          <div className="icon-tile">
            <ClipboardList aria-hidden="true" />
          </div>
          <div>
            <h2>Session Report</h2>
            <p>Raw EEG measurements and derived experimental metrics are shown separately.</p>
          </div>
        </div>
      </div>

      <div className="report-summary">
        <ReportStat label="Duration" value={formatDuration(report.durationMs)} />
        <ReportStat
          label="Average Attention"
          value={formatScore(report.averageAttention)}
        />
        <ReportStat label="Peak Attention" value={formatScore(report.peakAttention)} />
        <ReportStat
          label="EEG Channels"
          value={String(report.signalQuality.eegChannelCount)}
        />
        <ReportStat
          label="Poor Signal Periods"
          value={String(report.unreliablePeriods.length)}
        />
      </div>

      <div className="report-grid">
        <article className="panel">
          <div className="panel-header compact">
            <div className="panel-title">
              <div className="icon-tile">
                <LineChart aria-hidden="true" />
              </div>
              <div>
                <h2>Derived Metric</h2>
                <p>BrainFlow-derived Attention Index over time.</p>
              </div>
            </div>
          </div>
          <SeriesChart
            emptyTitle="No attention samples"
            emptyDescription="The session ended before metric samples were produced."
            height={220}
            min={0}
            max={100}
            lines={[
              {
                label: "Attention Index",
                color: "#a78bfa",
                values: report.attentionSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.displayedScore,
                })),
              },
            ]}
          />
        </article>

        <article className="panel">
          <div className="panel-header compact">
            <div className="panel-title">
              <div className="icon-tile raw-icon">
                <BarChart3 aria-hidden="true" />
              </div>
              <div>
                <h2>Raw Band Powers</h2>
                <p>Theta, alpha, and beta power computed from EEG samples.</p>
              </div>
            </div>
          </div>
          <SeriesChart
            emptyTitle="No band power samples"
            emptyDescription="Band powers need incoming EEG frames."
            height={220}
            lines={[
              {
                label: "Theta",
                color: "#67e8f9",
                values: report.bandPowerSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.thetaPower,
                })),
              },
              {
                label: "Alpha",
                color: "#4ade80",
                values: report.bandPowerSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.alphaPower,
                })),
              },
              {
                label: "Beta",
                color: "#a78bfa",
                values: report.bandPowerSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.betaPower,
                })),
              },
            ]}
          />
        </article>

        <article className="panel">
          <div className="panel-header compact">
            <div className="panel-title">
              <div className="icon-tile">
                <Gauge aria-hidden="true" />
              </div>
              <div>
                <h2>Heuristic Ratio</h2>
                <p>Diagnostic beta divided by alpha plus theta, with baseline-relative value.</p>
              </div>
            </div>
          </div>
          <SeriesChart
            emptyTitle="No ratio samples"
            emptyDescription="The ratio appears after EEG frames are processed."
            height={220}
            lines={[
              {
                label: "Raw ratio",
                color: "#a78bfa",
                values: report.ratioSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.rawRatio,
                })),
              },
              {
                label: "Baseline-relative",
                color: "#67e8f9",
                values: report.ratioSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.baselineRelativeValue,
                })),
              },
            ]}
          />
        </article>

        <article className="panel report-detail-card">
          <div className="panel-header compact">
            <div className="panel-title">
              <div className="icon-tile raw-icon">
                <Waves aria-hidden="true" />
              </div>
              <div>
                <h2>EEG Signal</h2>
                <p>Raw measurement context reported by the provider.</p>
              </div>
            </div>
          </div>
          <dl className="detail-list">
            <div>
              <dt>Samples processed</dt>
              <dd>{report.signalQuality.sampleCount}</dd>
            </div>
            <div>
              <dt>Channels</dt>
              <dd>
                {report.signalQuality.eegChannels.map((channel) => channel.label).join(", ") ||
                  "Unavailable"}
              </dd>
            </div>
          </dl>
        </article>
      </div>

      <div className="report-compact-grid">
        <article className="panel report-detail-card">
          <div className="panel-header compact">
            <div className="panel-title">
              <div className="icon-tile raw-icon">
                <Waves aria-hidden="true" />
              </div>
              <div>
                <h2>Quality Timeline</h2>
                <p>Signal-quality state recorded during the session.</p>
              </div>
            </div>
          </div>
          <ol className="timeline-list report-scroll">
            {report.qualityTimeline.map((snapshot) => (
              <li key={`${snapshot.updatedAtMs}-${snapshot.state}`}>
                <span>
                  <time>{formatRelativeTime(snapshot.updatedAtMs, sessionStartMs)}</time>
                  {snapshot.message}
                </span>
                <strong>{snapshot.state}</strong>
              </li>
            ))}
          </ol>
        </article>

        <article className="panel report-detail-card">
          <div className="panel-header compact">
            <div className="panel-title">
              <div className="icon-tile">
                <Gauge aria-hidden="true" />
              </div>
              <div>
                <h2>Attention Periods</h2>
                <p>Highest and lowest experimental metric points.</p>
              </div>
            </div>
          </div>
          <div className="report-scroll">
            <PeriodList title="Highest" periods={report.highestPeriods} startMs={sessionStartMs} />
            <PeriodList title="Lowest" periods={report.lowestPeriods} startMs={sessionStartMs} />
            <PeriodList title="Poor Signal" periods={report.unreliablePeriods} startMs={sessionStartMs} />
          </div>
        </article>

        <article className="panel report-detail-card">
          <div className="panel-header compact">
            <div className="panel-title">
              <div className="icon-tile">
                <Timer aria-hidden="true" />
              </div>
              <div>
                <h2>Session Timeline</h2>
                <p>Training session lifecycle events.</p>
              </div>
            </div>
          </div>
          <ol className="timeline-list report-scroll">
            {report.timeline.map((event) => (
              <li key={`${event.label}-${event.atMs}`}>
                <span>
                  <time>{formatRelativeTime(event.atMs, sessionStartMs)}</time>
                  {event.label}
                </span>
              </li>
            ))}
          </ol>
        </article>
      </div>
    </section>
  );
}

function ReportStat({ label, value }: { label: string; value: string }) {
  return (
    <article className="metric-card report-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function PeriodList({
  title,
  periods,
  startMs,
}: {
  title: string;
  periods: Array<{ label: string; atMs: number; score: number }>;
  startMs?: number;
}) {
  return (
    <div className="period-list">
      <h3>{title}</h3>
      {periods.length === 0 ? (
        <p>No samples recorded.</p>
      ) : (
        periods.map((period) => (
          <div className="period-row" key={`${period.label}-${period.atMs}`}>
            <span>
              <time>{formatRelativeTime(period.atMs, startMs)}</time>
              {period.label}
            </span>
            <strong>{period.score}</strong>
          </div>
        ))
      )}
    </div>
  );
}

function toYoutubeEmbedUrl(url: string, autoplay: boolean) {
  try {
    const parsed = new URL(url);
    const params = autoplay ? "?autoplay=1&mute=1&playsinline=1" : "";
    const channelMatch = parsed.pathname.match(/\/channel\/(UC[a-zA-Z0-9_-]+)/);
    if (channelMatch?.[1]) {
      const separator = autoplay ? "&" : "?";
      return `https://www.youtube.com/embed/videoseries${params}${separator}list=UU${channelMatch[1].slice(2)}`;
    }

    const videoId =
      parsed.hostname === "youtu.be"
        ? parsed.pathname.slice(1)
        : parsed.searchParams.get("v");

    return videoId ? `https://www.youtube.com/embed/${videoId}${params}` : "";
  } catch {
    return "";
  }
}

function formatDuration(ms: number) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatScore(score: number | null) {
  return score === null ? "--" : String(Math.round(score));
}

function formatRelativeTime(atMs: number, startMs = atMs) {
  return formatDuration(atMs - startMs);
}

function phaseLabel(phase: TrainingPhase) {
  if (phase === "headset_check") return "Waiting for stable headset signal.";
  if (phase === "baseline") return "Calibrating personal baseline.";
  if (phase === "calibrated") return "Calibration complete. Start when ready.";
  if (phase === "training") return "Training session active.";
  if (phase === "paused") return "Session paused.";
  if (phase === "ended") return "Session ended.";
  return "Ready to start a focus session.";
}

function activeGateIndex(
  phase: TrainingPhase,
  connectionState: EegConnectionState,
  ready: boolean,
) {
  const connected = connectionState === "connected" || connectionState === "streaming";
  if (phase === "training" || phase === "ended") return 4;
  if (phase === "calibrated") return 3;
  if (phase === "baseline") return 3;
  if (ready) return 2;
  if (phase === "headset_check") return connected ? 1 : 0;
  return connected ? 0 : -1;
}

function getPrimaryAction(
  phase: TrainingPhase,
  isRunning: boolean,
  isPaused: boolean,
) {
  if (phase === "calibrated") {
    return { label: "Start Session", disabled: isPaused };
  }

  if (phase === "headset_check" || phase === "baseline") {
    return { label: "Calibrating...", disabled: true };
  }

  if (phase === "training") {
    return { label: "Session Active", disabled: true };
  }

  return { label: "Start Calibration", disabled: isRunning || isPaused };
}
