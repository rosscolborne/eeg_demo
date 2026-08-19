import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BrainCircuit,
  Cable,
  ChevronDown,
  CircleAlert,
  CircleOff,
  Cpu,
  Download,
  Dumbbell,
  Gauge,
  LineChart,
  LoaderCircle,
  Radio,
  Rows3,
  Sparkles,
  Unplug,
} from "lucide-react";
import {
  HeadsetFitPanel,
  type FitCheckState,
} from "./components/quality/HeadsetFitPanel";
import { LiveEegPlot } from "./components/LiveEegPlot";
import { TrainingSection } from "./components/training/TrainingSection";
import {
  type DeviceInfo,
  type EegConnectionState,
  type EegFrameSummary,
  type SignalFrame,
  getCapability,
  summarizeEegFrame,
} from "./domain/eeg";
import type { EegProvider } from "./providers/eegProvider";
import { createAppRecording } from "./providers/localReplayProvider";
import {
  createEegProvider,
  deviceCatalog,
  getConfiguredProviderKey,
} from "./providers/providerRegistry";
import {
  HeuristicHeadsetFitProvider,
  type HeadsetFitSnapshot,
} from "./signalQuality/headsetFitProvider";

const maxHistorySamples = 640;
const fitCheckDurationMs = 3500;
type AppView = "dashboard" | "training";

function formatValue(value: number) {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(2);
}

function statusClassName(state: EegConnectionState) {
  if (state === "streaming" || state === "connected") return "connection-state is-live";
  if (state === "error") return "connection-state is-error";
  if (state === "connecting" || state === "initializing") {
    return "connection-state is-busy";
  }
  return "connection-state";
}

function StatusIcon({ state }: { state: EegConnectionState }) {
  if (state === "streaming" || state === "connected") {
    return <Radio aria-hidden="true" />;
  }

  if (state === "error") {
    return <CircleAlert aria-hidden="true" />;
  }

  if (state === "connecting" || state === "initializing") {
    return <LoaderCircle aria-hidden="true" className="spin" />;
  }

  return <CircleOff aria-hidden="true" />;
}

export default function App() {
  const providerRef = useRef<EegProvider | null>(null);
  const replayInputRef = useRef<HTMLInputElement | null>(null);
  const fitProviderRef = useRef(new HeuristicHeadsetFitProvider());
  const recordingFramesRef = useRef<SignalFrame[]>([]);
  const recordingActiveRef = useRef(false);
  const fitSnapshotRef = useRef<HeadsetFitSnapshot | null>(null);
  const fitCheckTimeoutRef = useRef<number | null>(null);
  const sampleCountRef = useRef(0);
  const frameCountRef = useRef(0);
  const [state, setState] = useState<EegConnectionState>("idle");
  const [statusDetail, setStatusDetail] = useState("");
  const [error, setError] = useState("");
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [latest, setLatest] = useState<Record<string, number>>({});
  const [sampleCount, setSampleCount] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [plotHistory, setPlotHistory] = useState<Record<string, number[]>>({});
  const [providerLabel, setProviderLabel] = useState("");
  const [selectedDeviceId, setSelectedDeviceId] = useState("brainflow-muse-athena");
  const [latestFrame, setLatestFrame] = useState<SignalFrame | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedFrameCount, setRecordedFrameCount] = useState(0);
  const [fitSnapshot, setFitSnapshot] = useState<HeadsetFitSnapshot>(() =>
    fitProviderRef.current.update({
      connectionState: "idle",
      deviceInfo: null,
      frame: null,
    }),
  );
  const [fitCheck, setFitCheck] = useState<FitCheckState>({
    status: "idle",
    startedAtMs: null,
    completedAtMs: null,
    result: null,
  });
  const [view, setView] = useState<AppView>("dashboard");

  const eegCapability = useMemo(
    () => getCapability(deviceInfo, "eeg"),
    [deviceInfo],
  );
  const selectedDevice = useMemo(() => {
    return (
      deviceCatalog.find((device) => device.id === selectedDeviceId) ??
      deviceCatalog[0]
    );
  }, [selectedDeviceId]);
  const replaySelected = selectedDevice.id === "brainflow-replay";
  const channelNames = useMemo(() => {
    return eegCapability?.channels.length
      ? eegCapability.channels.map((channel) => channel.label)
      : Object.keys(latest);
  }, [eegCapability, latest]);

  useEffect(() => {
    const providerKey = selectedDevice.providerKey ?? getConfiguredProviderKey();
    const provider = createEegProvider(providerKey, {
      onState: (nextState, detail) => {
        setState(nextState);
        setStatusDetail(detail ?? "");
        if (nextState === "disconnected" || nextState === "error") {
          recordingActiveRef.current = false;
          setRecording(false);
        }
      },
      onDeviceInfo: (info) => setDeviceInfo(info),
      onSignalFrame: (frame) => {
        if (frame.sensor !== "eeg") return;

        setLatestFrame(frame);
        if (recordingActiveRef.current) {
          recordingFramesRef.current.push(frame);
          setRecordedFrameCount(recordingFramesRef.current.length);
        }
        const summary: EegFrameSummary = summarizeEegFrame(
          frame,
          sampleCountRef.current,
          frameCountRef.current,
        );

        sampleCountRef.current = summary.totalSamples;
        frameCountRef.current = summary.frameCount;
        setLatest(summary.latestByChannel);
        setSampleCount(summary.totalSamples);
        setFrameCount(summary.frameCount);

        setPlotHistory((current) => {
          const next = { ...current };
          for (const [name, value] of Object.entries(summary.latestByChannel)) {
            next[name] = [...(next[name] ?? []), value].slice(-maxHistorySamples);
          }
          return next;
        });
      },
      onError: (providerError) => {
        setError(
          providerError.code
            ? `${providerError.message} (${providerError.code})`
            : providerError.message,
        );
      },
    });

    providerRef.current = provider;
    fitProviderRef.current.reset();
    if (fitCheckTimeoutRef.current !== null) {
      window.clearTimeout(fitCheckTimeoutRef.current);
      fitCheckTimeoutRef.current = null;
    }
    setProviderLabel(provider.descriptor.label);
    setState("idle");
    setStatusDetail("");
    setError("");
    setDeviceInfo(null);
    setLatest({});
    setSampleCount(0);
    setFrameCount(0);
    setPlotHistory({});
    setLatestFrame(null);
    setRecording(false);
    recordingActiveRef.current = false;
    setRecordedFrameCount(0);
    recordingFramesRef.current = [];
    const initialFit = fitProviderRef.current.update({
        connectionState: "idle",
        deviceInfo: null,
        frame: null,
    });
    fitSnapshotRef.current = initialFit;
    setFitSnapshot(initialFit);
    setFitCheck({
      status: "idle",
      startedAtMs: null,
      completedAtMs: null,
      result: null,
    });
    sampleCountRef.current = 0;
    frameCountRef.current = 0;

    return () => {
      if (fitCheckTimeoutRef.current !== null) {
        window.clearTimeout(fitCheckTimeoutRef.current);
        fitCheckTimeoutRef.current = null;
      }
      void provider.disconnect();
      providerRef.current = null;
    };
  }, [selectedDevice]);

  const busy =
    state === "initializing" ||
    state === "connecting" ||
    state === "disconnecting";
  const connected = state === "connected" || state === "streaming";

  useEffect(() => {
    const updateFit = () => {
      const nextFit = fitProviderRef.current.update({
          connectionState: state,
          deviceInfo,
          frame: latestFrame,
      });
      fitSnapshotRef.current = nextFit;
      setFitSnapshot(nextFit);
    };

    updateFit();
    const intervalId = window.setInterval(updateFit, 500);

    return () => window.clearInterval(intervalId);
  }, [deviceInfo, latestFrame, state]);

  async function connect() {
    if (replaySelected) {
      replayInputRef.current?.click();
      return;
    }

    setError("");
    sampleCountRef.current = 0;
    frameCountRef.current = 0;
    setLatest({});
    setSampleCount(0);
    setFrameCount(0);
    setPlotHistory({});
    setLatestFrame(null);
    fitProviderRef.current.reset();
    resetFitCheck();
    await providerRef.current?.connectAndStart();
  }

  async function connectReplay(file: File | null) {
    if (!file) return;

    setError("");
    sampleCountRef.current = 0;
    frameCountRef.current = 0;
    setLatest({});
    setSampleCount(0);
    setFrameCount(0);
    setPlotHistory({});
    setLatestFrame(null);
    fitProviderRef.current.reset();
    resetFitCheck();
    setState("connecting");
    setStatusDetail("Loading replay recording");

    try {
      await providerRef.current?.connectAndStart({ replayFile: file });
    } catch (uploadError) {
      setState("error");
      setStatusDetail("");
      setError(
        uploadError instanceof Error
          ? `Unable to load replay file: ${uploadError.message}`
          : "Unable to load replay file.",
      );
    } finally {
      if (replayInputRef.current) {
        replayInputRef.current.value = "";
      }
    }
  }

  function startRecording() {
    recordingFramesRef.current = [];
    setRecordedFrameCount(0);
    recordingActiveRef.current = true;
    setRecording(true);
  }

  function stopRecording() {
    recordingActiveRef.current = false;
    setRecording(false);
  }

  function resetFitCheck() {
    if (fitCheckTimeoutRef.current !== null) {
      window.clearTimeout(fitCheckTimeoutRef.current);
      fitCheckTimeoutRef.current = null;
    }
    setFitCheck({
      status: "idle",
      startedAtMs: null,
      completedAtMs: null,
      result: null,
    });
  }

  function runFitCheck() {
    if (fitCheckTimeoutRef.current !== null) {
      window.clearTimeout(fitCheckTimeoutRef.current);
    }

    const nowMs = performance.now();
    fitProviderRef.current.reset();
    const initialFit = fitProviderRef.current.update({
      connectionState: state,
      deviceInfo,
      frame: latestFrame,
      nowMs,
    });
    fitSnapshotRef.current = initialFit;
    setFitSnapshot(initialFit);
    setFitCheck({
      status: "running",
      startedAtMs: nowMs,
      completedAtMs: null,
      result: null,
    });

    fitCheckTimeoutRef.current = window.setTimeout(() => {
      const currentResult =
        fitSnapshotRef.current ??
        fitProviderRef.current.update({
          connectionState: state,
          deviceInfo,
          frame: latestFrame,
        });
      const result = completeFitCheckResult(currentResult);
      fitCheckTimeoutRef.current = null;
      setFitCheck({
        status: "complete",
        startedAtMs: nowMs,
        completedAtMs: performance.now(),
        result,
      });
    }, fitCheckDurationMs);
  }

  function downloadRecording() {
    const recordingPayload = createAppRecording(deviceInfo, recordingFramesRef.current);
    const blob = new Blob([JSON.stringify(recordingPayload)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    anchor.href = url;
    anchor.download = `eeg-demo-recording-${timestamp}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function disconnect() {
    recordingActiveRef.current = false;
    setRecording(false);
    await providerRef.current?.disconnect();
  }

  function showDashboard(anchorId?: string) {
    setView("dashboard");
    if (!anchorId) return;

    window.setTimeout(() => {
      document.getElementById(anchorId)?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 0);
  }

  return (
    <div className="app-frame">
      <aside className="sidebar" aria-label="Application navigation">
        <div className="brand-mark" aria-label="EEG acquisition console">
          <BrainCircuit aria-hidden="true" />
        </div>
        <nav className="nav-stack">
          <button
            className={`nav-button ${view === "dashboard" ? "is-active" : ""}`}
            onClick={() => showDashboard("connection")}
            aria-label="Dashboard"
          >
            <Cable aria-hidden="true" />
          </button>
          <button
            className="nav-button"
            onClick={() => showDashboard("fit")}
            aria-label="Headset fit"
          >
            <Activity aria-hidden="true" />
          </button>
          <button
            className="nav-button"
            onClick={() => showDashboard("chart")}
            aria-label="Live plot"
          >
            <LineChart aria-hidden="true" />
          </button>
          <button
            className={`nav-button ${view === "training" ? "is-active" : ""}`}
            onClick={() => setView("training")}
            aria-label="Training"
          >
            <Dumbbell aria-hidden="true" />
          </button>
          <button
            className="nav-button"
            onClick={() => showDashboard("channels")}
            aria-label="EEG values"
          >
            <Rows3 aria-hidden="true" />
          </button>
        </nav>
      </aside>

      <main className={`app-shell ${view === "training" ? "is-training" : ""}`}>
        {view === "dashboard" ? (
          <>
        <header className="page-header">
          <div>
            <h1>EEG Acquisition Console</h1>
            <p>Hardware-agnostic raw EEG acquisition and provider validation.</p>
          </div>
          <div className="header-badge">
            <Sparkles aria-hidden="true" />
            <span>{providerLabel || getConfiguredProviderKey()}</span>
          </div>
        </header>

        <section id="connection" className="connection-card" aria-label="Device connection controls">
          <div className={statusClassName(state)}>
            <div className="icon-tile">
              <StatusIcon state={state} />
            </div>
            <div>
              <span>Connection</span>
              <strong>{state}</strong>
              {statusDetail && <small>{statusDetail}</small>}
            </div>
          </div>
          <label className="device-selector">
            <span>Device</span>
            <div className="select-wrap">
              <Cable aria-hidden="true" />
              <select
                value={selectedDeviceId}
                onChange={(event) => setSelectedDeviceId(event.target.value)}
                disabled={busy || connected}
              >
                {deviceCatalog.map((device) => (
                  <option
                    disabled={device.disabled}
                    key={device.id}
                    value={device.id}
                  >
                    {device.disabled
                      ? `${device.label} - ${device.detail}`
                      : `${device.label} - ${device.detail}`}
                  </option>
                ))}
              </select>
              <ChevronDown aria-hidden="true" />
            </div>
          </label>
          <div className="actions">
            <input
              accept=".json"
              hidden
              onChange={(event) => void connectReplay(event.target.files?.[0] ?? null)}
              ref={replayInputRef}
              type="file"
            />
            <button className="primary-button" onClick={connect} disabled={busy || connected}>
              <Cable aria-hidden="true" />
              {replaySelected ? "Upload File" : "Connect"}
            </button>
            <button
              className="secondary-button"
              onClick={disconnect}
              disabled={!connected && !busy && state !== "error"}
            >
              <Unplug aria-hidden="true" />
              Disconnect
            </button>
            {!replaySelected && (
              <button
                className="secondary-button"
                onClick={recording ? stopRecording : startRecording}
                disabled={!connected}
              >
                <Radio aria-hidden="true" />
                {recording ? "Stop Recording" : "Record"}
              </button>
            )}
            {!replaySelected && (
              <button
                className="secondary-button"
                onClick={downloadRecording}
                disabled={recording || recordedFrameCount === 0}
              >
                <Download aria-hidden="true" />
                Download
              </button>
            )}
          </div>
        </section>

        {error && (
          <div className="error">
            <CircleAlert aria-hidden="true" />
            <span>{error}</span>
          </div>
        )}

        <section className="metrics" aria-label="EEG stream overview">
          <article className="metric-card">
            <div className="metric-icon">
              <Activity aria-hidden="true" />
            </div>
            <span>Frames</span>
            <strong>{frameCount}</strong>
            <small>{sampleCount} samples received</small>
          </article>
          <article className="metric-card">
            <div className="metric-icon">
              <Cable aria-hidden="true" />
            </div>
            <span>Device</span>
            <strong>{deviceInfo?.label ?? "Not connected"}</strong>
            <small>{deviceInfo?.model ?? "No active device"}</small>
          </article>
          <article className="metric-card">
            <div className="metric-icon">
              <Cpu aria-hidden="true" />
            </div>
            <span>Provider</span>
            <strong>{deviceInfo?.providerName ?? providerLabel}</strong>
            <small>
              {recording
                ? `Recording ${recordedFrameCount} frames`
                : recordedFrameCount > 0
                  ? `${recordedFrameCount} recorded frames ready`
                  : deviceInfo
                  ? "Adapter normalized"
                  : "Waiting for provider"}
            </small>
          </article>
          <article className="metric-card">
            <div className="metric-icon">
              <Gauge aria-hidden="true" />
            </div>
            <span>Sample Rate</span>
            <strong>
              {eegCapability?.sampleRateHz === null || !eegCapability
                ? "pending"
                : `${eegCapability.sampleRateHz} Hz`}
            </strong>
            <small>{eegCapability?.channels.length ?? channelNames.length} EEG channels</small>
          </article>
        </section>

        <HeadsetFitPanel
          check={fitCheck}
          fit={fitSnapshot}
          onRunCheck={runFitCheck}
        />

        <section id="chart" className="panel chart-card">
          <div className="panel-header">
            <div className="panel-title">
              <div className="icon-tile">
                <LineChart aria-hidden="true" />
              </div>
              <div>
                <h2>Live Plot</h2>
                <p>Recent latest sample per received frame</p>
              </div>
            </div>
            <span className="panel-meta">{channelNames.length} channels</span>
          </div>
          <div className="chart-surface">
            {channelNames.length === 0 && (
              <div className="empty-state chart-empty">
                <div className="icon-tile">
                  <LineChart aria-hidden="true" />
                </div>
                <strong>Waiting for EEG frames</strong>
                <span>Connect a provider to draw incoming channel values.</span>
              </div>
            )}
            <LiveEegPlot channelNames={channelNames} history={plotHistory} />
          </div>
        </section>

        <section id="channels" className="panel channels-card">
          <div className="panel-header">
            <div className="panel-title">
              <div className="icon-tile">
                <Rows3 aria-hidden="true" />
              </div>
              <div>
                <h2>Latest Raw EEG Values</h2>
                <p>{channelNames.join(", ") || "Waiting for channels"}</p>
              </div>
            </div>
          </div>
          <div className="channel-grid">
            {channelNames.length === 0 ? (
              <div className="empty-state channel-empty">
                <div className="icon-tile">
                  <Cable aria-hidden="true" />
                </div>
                <strong>No channels yet</strong>
                <span>Connect a provider to populate EEG channel values.</span>
              </div>
            ) : (
              channelNames.map((name) => (
                <div className="channel-row" key={name}>
                  <span className="channel-name">{name}</span>
                  <strong>{formatValue(latest[name])}</strong>
                </div>
              ))
            )}
          </div>
        </section>
          </>
        ) : (
          <TrainingSection
            connectionState={state}
            deviceInfo={deviceInfo}
            latestFrame={latestFrame}
            fit={fitSnapshot}
          />
        )}
      </main>
    </div>
  );
}

function completeFitCheckResult(result: HeadsetFitSnapshot): HeadsetFitSnapshot {
  const timerToleranceMs = 200;
  const passedByTolerance =
    result.state === "good" &&
    result.stableForMs >= result.requiredStableMs - timerToleranceMs &&
    !result.excessiveArtifact;

  if (!passedByTolerance) return result;

  return {
    ...result,
    state: "ready",
    ready: true,
    message: "Signal looks stable",
    blockers: [],
    stableForMs: result.requiredStableMs,
  };
}
