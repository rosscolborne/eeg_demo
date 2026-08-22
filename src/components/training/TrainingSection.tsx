import { useCallback, useEffect, useRef, useState } from "react";
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
import { InfoPopoverButton } from "../InfoPopoverButton";
import { SeriesChart } from "./SeriesChart";

declare global {
  interface Window {
    YT?: any;
    onYouTubeIframeAPIReady?: () => void;
  }
}

interface TrainingSectionProps {
  connectionState: EegConnectionState;
  deviceInfo: DeviceInfo | null;
  latestFrame: SignalFrame | null;
  fit: HeadsetFitSnapshot;
}

const defaultVideoUrl = "https://www.youtube.com/watch?v=uyb0wW0ln_g";
const baselineSamplesRequired = 24;
const defaultVideoId = getYoutubeVideoId(defaultVideoUrl);
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
  const videoStartedRef = useRef(false);
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
  const currentSample =
    eegStreaming && attentionSamples.length > 0
      ? attentionSamples[attentionSamples.length - 1]
      : null;
  const showVideo =
    phase === "training" || (phase === "paused" && hasStartedTraining);
  const isRunning = sessionState === "running";
  const isPaused = sessionState === "paused";
  const canPause = hasStartedTraining && (isRunning || isPaused);
  const canEnd = isRunning || isPaused || phase === "training";
  const qualityAllowsScoring = eegStreaming && !fit.excessiveArtifact;
  const calibrationInProgress = phase === "headset_check" || phase === "baseline";
  const canStartCalibration =
    !hasStartedTraining &&
    !calibrationInProgress &&
    sessionState !== "paused" &&
    phase !== "training";
  const canStartTraining =
    eegStreaming && !calibrationInProgress && !hasStartedTraining && !isPaused;
  const setupDetail =
    phase === "baseline" && qualityAllowsScoring
      ? `Baseline calibration ${baselineProgress}/${baselineSamplesRequired} samples.`
      : phase === "baseline"
        ? "Calibration waits for EEG stream and excessive-artifact checks only."
        : fit.blockers[0] ?? fit.message;
  const signalStatus = getTrainingSignalStatus(connectionState, fit);

  useEffect(() => {
    if (!isRunning || !hasStartedTraining) return;

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
    if (phase === "training" && !hasStartedTraining) return;
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
        setSessionState("idle");
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
    metricProviderRef.current.reset({ useBaselineRelativeDisplay: true });
    sessionRef.current = new TrainingSession();
    sessionRef.current.addTimelineEvent("Headset check started");
    lastFrameSequenceRef.current = null;
    videoStartedRef.current = false;
    setAttentionSamples([]);
    setReport(null);
    setElapsedMs(0);
    setBaselineProgress(0);
    setHasStartedTraining(false);
    setSessionState("running");
    setPhase("headset_check");
  }

  function beginTraining() {
    if (phase !== "calibrated") {
      metricProviderRef.current.reset({ useBaselineRelativeDisplay: false });
      sessionRef.current = new TrainingSession();
    } else {
      metricProviderRef.current.setBaselineRelativeDisplay(true);
    }
    videoStartedRef.current = false;
    setAttentionSamples([]);
    setElapsedMs(0);
    setHasStartedTraining(false);
    setPhase("training");
  }

  const startTrainingAfterVideoPlay = useCallback(() => {
    if (videoStartedRef.current || phase !== "training") return;

    videoStartedRef.current = true;
    sessionRef.current.start();
    sessionRef.current.addTimelineEvent("Training started");
    lastFrameSequenceRef.current = null;
    setAttentionSamples([]);
    setElapsedMs(0);
    setHasStartedTraining(true);
    setSessionState("running");
  }, [phase]);

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
    let nextReport: TrainingSessionReport | null = null;
    if (hasStartedTraining) {
      sessionRef.current.end();
      setElapsedMs(sessionRef.current.snapshot(deviceInfo).elapsedMs);
      try {
        nextReport = sessionRef.current.report(deviceInfo);
      } catch (error) {
        console.error("[Training report error]", error);
      }
    }
    videoStartedRef.current = false;
    setSessionState("ended");
    setPhase("ended");
    window.setTimeout(() => setReport(nextReport), 0);
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
                <p>Watch a video while tracking BrainFlow-derived mindfulness metrics.</p>
              </div>
            </div>
          </div>
          <div className="training-progress-strip">
            <div className="progress-strip-copy">
              <div className="training-phase-copy">
                <strong>
                  {phase === "training" && !hasStartedTraining
                    ? "Waiting for video playback."
                    : phaseLabel(phase)}
                </strong>
                {setupDetail !== signalStatus.label && <small>{setupDetail}</small>}
              </div>
              <strong className={`training-signal-status is-${signalStatus.tone}`}>
                {signalStatus.label}
              </strong>
              <div className="training-status-summary">
                <span className="training-elapsed">
                  <Timer aria-hidden="true" />
                  {formatDuration(elapsedMs)}
                </span>
              </div>
            </div>
            <div className="gate-steps is-horizontal">
              {[
                "Connect device",
                "Headset check",
                "Stable signal",
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
              <YouTubeTrainingPlayer
                onPlaying={startTrainingAfterVideoPlay}
                paused={isPaused}
                videoId={defaultVideoId}
              />
            ) : (
              <div className="video-placeholder">
                <div className="icon-tile">
                  <Play aria-hidden="true" />
                </div>
                <strong>Video starts when training begins</strong>
                <span>Start training directly, or calibrate first for baseline-relative scores.</span>
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
                <h2>Brain Metrics</h2>
                <p>BrainFlow mental-state outputs with band-ratio comparison scores.</p>
              </div>
              <InfoPopoverButton ariaLabel="Explain brain metrics" preferredSide="left">
                <p>
                  Mindfulness starts from BrainFlow's MLModel Mindfulness output,
                  then uses the session baseline and smoothing only if calibration
                  was run before training.
                </p>
                <p>
                  Restfulness uses BrainFlow's MLModel Restfulness output when
                  available. If BrainFlow does not provide that metric, the
                  value stays blank.
                </p>
                <p>
                  Focus starts from a beta/theta band-power ratio. Relax starts
                  from an alpha/theta band-power ratio. They are currently shown
                  as direct smoothed 0-100 scores unless calibration was run
                  before training.
                </p>
              </InfoPopoverButton>
            </div>
            <div className="metric-score-grid">
              <MetricScore label="Mindfulness" value={currentScore} />
              <MetricScore label="Restfulness" value={currentSample?.restfulnessScore ?? null} />
              <MetricScore label="Focus" value={currentSample?.focusScore ?? null} />
              <MetricScore label="Relax" value={currentSample?.relaxScore ?? null} />
            </div>
            <p className="metric-note">
              Experimental feedback only. This is not a validated or clinical measure
              of mental state.
            </p>
          </article>

          <article className="panel session-actions">
            <button
              className="secondary-button"
              onClick={startCalibration}
              disabled={!canStartCalibration}
            >
              <CirclePlay aria-hidden="true" />
              {calibrationInProgress
                ? `Calibrating ${baselineProgress}/${baselineSamplesRequired}`
                : phase === "calibrated"
                  ? "Re-run Calibration"
                  : "Start Calibration"}
            </button>
            <button
              className="primary-button"
              onClick={beginTraining}
              disabled={!canStartTraining}
            >
              <CirclePlay aria-hidden="true" />
              Start Training
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
              <h2>BrainFlow Metrics Over Time</h2>
              <p>Smoothed BrainFlow Mindfulness and Restfulness plus Focus and Relax ratio scores.</p>
            </div>
          </div>
          <span className="panel-meta">
            {attentionSamples.length} points
          </span>
        </div>
        <SeriesChart
          emptyTitle="Start a focus session"
          emptyDescription="BrainFlow metric samples will appear as normalized EEG frames arrive."
          height={220}
          lines={[
            {
              label: "Mindfulness",
              color: "#a78bfa",
              values: attentionSamples.map((sample) => ({
                atMs: sample.atMs,
                value: sample.displayedScore,
              })),
            },
            {
              label: "Restfulness",
              color: "#f9a8d4",
              values: attentionSamples.map((sample) => ({
                atMs: sample.atMs,
                value: sample.restfulnessScore,
              })),
            },
            {
              label: "Focus",
              color: "#67e8f9",
              values: attentionSamples.map((sample) => ({
                atMs: sample.atMs,
                value: sample.focusScore,
              })),
            },
            {
              label: "Relax",
              color: "#4ade80",
              values: attentionSamples.map((sample) => ({
                atMs: sample.atMs,
                value: sample.relaxScore,
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

function YouTubeTrainingPlayer({
  onPlaying,
  paused,
  videoId,
}: {
  onPlaying: () => void;
  paused: boolean;
  videoId: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<any>(null);
  const onPlayingRef = useRef(onPlaying);

  useEffect(() => {
    onPlayingRef.current = onPlaying;
  }, [onPlaying]);

  useEffect(() => {
    let cancelled = false;

    loadYouTubeIframeApi().then((YT) => {
      if (cancelled || !containerRef.current) return;

      playerRef.current = new YT.Player(containerRef.current, {
        videoId,
        playerVars: {
          autoplay: 1,
          controls: 1,
          modestbranding: 1,
          mute: 1,
          playsinline: 1,
          rel: 0,
        },
        events: {
          onReady: (event: any) => {
            event.target.mute();
            event.target.playVideo();
          },
          onStateChange: (event: any) => {
            if (event.data === YT.PlayerState.PLAYING) {
              onPlayingRef.current();
            }
          },
        },
      });
    });

    return () => {
      cancelled = true;
      try {
        playerRef.current?.destroy?.();
      } catch (error) {
        console.warn("Unable to destroy YouTube training player", error);
      }
      playerRef.current = null;
    };
  }, [videoId]);

  useEffect(() => {
    const player = playerRef.current;
    if (!player) return;

    if (paused) {
      player.pauseVideo?.();
    } else {
      player.playVideo?.();
    }
  }, [paused]);

  return <div className="youtube-player" ref={containerRef} />;
}

function MetricScore({
  label,
  value,
}: {
  label: string;
  value: number | null;
}) {
  return (
    <div className="metric-score-tile">
      <span>{label}</span>
      <strong>{formatScore(value)}</strong>
      <small>0-100</small>
    </div>
  );
}

function TrainingReport({ report }: { report: TrainingSessionReport }) {
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
          label="Average Mindfulness"
          value={formatScore(report.averageAttention)}
        />
        <ReportStat label="Peak Mindfulness" value={formatScore(report.peakAttention)} />
        <ReportStat
          label="EEG Channels"
          value={String(report.signalQuality.eegChannelCount)}
        />
        <ReportStat
          label="Poor Signal Periods"
          value={String(report.unreliablePeriods.length)}
        />
      </div>

      <div className="report-chart-stack">
        <article className="panel">
          <div className="panel-header compact">
            <div className="panel-title">
              <div className="icon-tile">
                <LineChart aria-hidden="true" />
              </div>
              <div>
                <h2>Derived Metric</h2>
                <p>BrainFlow mental-state outputs and ratio scores over time.</p>
              </div>
            </div>
          </div>
          <SeriesChart
            emptyTitle="No BrainFlow metric samples"
            emptyDescription="The session ended before metric samples were produced."
            height={220}
            min={0}
            max={100}
            lines={[
              {
                label: "Mindfulness",
                color: "#a78bfa",
                values: report.attentionSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.displayedScore,
                })),
              },
              {
                label: "Restfulness",
                color: "#f9a8d4",
                values: report.attentionSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.restfulnessScore,
                })),
              },
              {
                label: "Focus",
                color: "#67e8f9",
                values: report.attentionSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.focusScore,
                })),
              },
              {
                label: "Relax",
                color: "#4ade80",
                values: report.attentionSeries.map((sample) => ({
                  atMs: sample.atMs,
                  value: sample.relaxScore,
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

function getYoutubeVideoId(url: string) {
  try {
    const parsed = new URL(url);
    return parsed.hostname === "youtu.be"
      ? parsed.pathname.slice(1)
      : parsed.searchParams.get("v") ?? "";
  } catch {
    return "";
  }
}

let youtubeApiPromise: Promise<any> | null = null;

function loadYouTubeIframeApi() {
  if (window.YT?.Player) return Promise.resolve(window.YT);

  youtubeApiPromise ??= new Promise((resolve) => {
    const previousReady = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      previousReady?.();
      resolve(window.YT);
    };

    const existingScript = document.querySelector(
      'script[src="https://www.youtube.com/iframe_api"]',
    );
    if (existingScript) return;

    const script = document.createElement("script");
    script.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(script);
  });

  return youtubeApiPromise;
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
  if (phase === "training" || phase === "ended") return 3;
  if (phase === "calibrated") return 2;
  if (phase === "baseline") return 2;
  if (ready) return 2;
  if (phase === "headset_check") return connected ? 1 : 0;
  return connected ? 0 : -1;
}

function getTrainingSignalStatus(
  connectionState: EegConnectionState,
  fit: HeadsetFitSnapshot,
) {
  const connected = connectionState === "connected" || connectionState === "streaming";
  if (!connected) {
    return { label: "Connect an EEG device.", tone: "error" as const };
  }

  const anyWeakChannel = fit.channels.some((channel) => channel.state !== "good");
  if (fit.ready || (fit.state === "good" && !anyWeakChannel)) {
    return { label: fit.message || "Signal quality good", tone: "good" as const };
  }

  return {
    label: fit.message || "Check signal quality",
    tone: anyWeakChannel ? ("warning" as const) : ("error" as const),
  };
}
