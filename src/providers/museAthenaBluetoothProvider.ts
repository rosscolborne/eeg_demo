import {
  AthenaWasmDecoder,
  BleTransport,
  checkWebBluetooth,
} from "@elata-biosciences/eeg-web-ble";
import { initEegWasm, type HeadbandFrameV1 } from "@elata-biosciences/eeg-web";
import eegWasmUrl from "@elata-biosciences/eeg-web/wasm/eeg_wasm_bg.wasm?url";
import type {
  DeviceInfo,
  EegConnectionState,
  EegProviderEvents,
  ProviderError,
  SignalFeatures,
  SignalFrame,
  SignalQualityMetadata,
  TrainingMetricSample,
} from "../domain/eeg";
import type { EegProvider, EegProviderDescriptor } from "./eegProvider";

const defaultBrainFlowServiceUrl =
  import.meta.env.VITE_BRAINFLOW_SERVICE_URL ?? "http://127.0.0.1:8000";
const analysisWindowSeconds = 2;

// The scalp electrodes, and only those, count as EEG for headset fit.
//
// BrainFlow's MUSE_S_ATHENA_BOARD reports exactly these four as its EEG
// channels (`get_eeg_channels`/`get_eeg_names`) and keeps the headband's
// AUX inputs on separate, non-EEG rows, so the fit heuristic there only
// ever grades real skin contact. The Web Bluetooth stream instead packs
// TP9/AF7/AF8/TP10 *and* AUX1-4 into one 8-channel EEG block, and an
// unconnected AUX input reads as a clean, plausible EEG trace whether or
// not the headband is on a head. Four permanently "good" channels out of
// eight is enough to satisfy HeuristicHeadsetFitProvider's
// good-channel-fraction rule, which is why the fit check passed with the
// headset sitting on the desk. Keeping only the scalp electrodes gives
// that heuristic the same four channels it grades over BrainFlow.
const scalpElectrodeIds = new Set(["tp9", "af7", "af8", "tp10"]);

/** Indices of `channelNames` that are scalp electrodes, in stream order.
 * Falls back to every channel for a layout that uses none of the Muse
 * electrode names, so an unrecognized device streams as before rather
 * than losing all of its channels. */
function scalpElectrodeIndices(channelNames: string[]): number[] {
  const indices = channelNames
    .map((name, index) => (scalpElectrodeIds.has(name.toLowerCase()) ? index : -1))
    .filter((index) => index >= 0);

  return indices.length > 0 ? indices : channelNames.map((_, index) => index);
}

export class MuseAthenaBluetoothProvider implements EegProvider {
  readonly descriptor: EegProviderDescriptor = {
    id: "muse-athena-bluetooth",
    label: "Muse Athena - Bluetooth",
    description: "Chrome Web Bluetooth via Elata EEG SDK",
  };

  private transport: BleTransport | null = null;
  private deviceInfo: DeviceInfo | null = null;
  private analysisBuffer: number[][] = [];
  private analysisInFlight = false;
  private analysisFailureReported = false;
  private emittedSequenceId = 0;
  // Backs this connection's whole server-side analysis session
  // (`analysis.py`'s `AnalysisSessionStore`, via `POST /headset-fit/sessions`)
  // -- headset fit *and* the smoothed mindfulness/restfulness/focus/relax
  // and valence/arousal scores. All of that runs in brainflow_service, not
  // here -- see `analyzeWindow`.
  private fitSessionId: string | null = null;

  constructor(
    private readonly events: EegProviderEvents,
    private readonly brainFlowServiceUrl = defaultBrainFlowServiceUrl,
  ) {}

  getDeviceInfo() {
    return this.deviceInfo;
  }

  async connectAndStart() {
    await this.disconnect("Preparing Web Bluetooth session");

    const support = checkWebBluetooth();
    if (!support.supported) {
      this.emitError(support.message, "WEB_BLUETOOTH_UNAVAILABLE");
      return;
    }

    this.events.onState("connecting", "Opening Chrome Web Bluetooth chooser");
    this.analysisBuffer = [];
    this.analysisInFlight = false;
    this.analysisFailureReported = false;
    this.emittedSequenceId = 0;
    this.fitSessionId = null;

    try {
      await Promise.all([initEegWasm(eegWasmUrl), this.startFitSession()]);

      const transport = new BleTransport({
        deviceOptions: {
          athenaDecoderFactory: () => new AthenaWasmDecoder(),
          onDisconnected: () =>
            this.events.onState("disconnected", "Bluetooth device disconnected"),
        },
        sourceName: "Muse Athena - Bluetooth",
        eegProcessing: false,
      });
      this.transport = transport;

      transport.onStatus = (status) => {
        this.events.onState(mapTransportState(status.state), status.reason);
        if (status.state === "error") {
          this.events.onError({
            message: status.reason ?? "Web Bluetooth transport error.",
            code: status.errorCode,
            recoverable: status.recoverable,
            details: status.details,
          });
        }
      };
      transport.onFrame = (frame) => {
        void this.handleFrame(frame);
      };

      await transport.connect();
      this.publishDeviceInfo(transport);
      await transport.start();
    } catch (error) {
      const providerError = formatBluetoothError(error);
      this.events.onState("error", providerError.message);
      if (!isUserCancelledBluetoothRequest(error)) {
        this.events.onError(providerError);
      }
      await this.disconnect("Bluetooth connection failed");
    }
  }

  async disconnect(reason = "Bluetooth disconnected") {
    const transport = this.transport;
    this.transport = null;
    const fitSessionId = this.fitSessionId;
    this.fitSessionId = null;

    if (transport) {
      try {
        await transport.disconnect();
      } catch (error) {
        console.warn("Unable to disconnect Web Bluetooth transport", error);
      }
    }
    if (fitSessionId) {
      try {
        await fetch(`${this.brainFlowServiceUrl}/headset-fit/sessions/${fitSessionId}`, {
          method: "DELETE",
        });
      } catch (error) {
        console.warn("Unable to stop headset-fit session", error);
      }
    }

    this.events.onState("disconnected", reason);
  }

  private async handleFrame(frame: HeadbandFrameV1) {
    const eeg = frame.eegRaw ?? frame.eeg;
    if (!this.deviceInfo) {
      this.publishDeviceInfo(this.transport, frame);
    }
    // Drop AUX1-4 before anything downstream sees them -- see
    // `scalpElectrodeIndices`.
    const electrodeIndices = scalpElectrodeIndices(eeg.channelNames);
    const electrodeNames = electrodeIndices.map((index) => eeg.channelNames[index]);
    this.analysisBuffer.push(
      ...eeg.samples.map((row) => electrodeIndices.map((index) => row[index])),
    );

    const sampleRate = eeg.sampleRateHz;
    const maxSamples = Math.max(1, Math.round(sampleRate * analysisWindowSeconds));
    if (this.analysisBuffer.length > maxSamples) {
      this.analysisBuffer = this.analysisBuffer.slice(-maxSamples);
    }
    if (this.analysisBuffer.length < maxSamples || this.analysisInFlight) {
      return;
    }

    this.analysisInFlight = true;
    const windowSamples = [...this.analysisBuffer];
    const analysis = await this.analyzeWindow(windowSamples, sampleRate, electrodeNames);
    this.analysisInFlight = false;

    const normalized: SignalFrame = {
      sensor: "eeg",
      sampleRateHz: eeg.sampleRateHz,
      channels: electrodeNames.map((name) => ({
        id: name.toLowerCase(),
        label: name,
        unit: "uV",
      })),
      samples: windowSamples,
      timestampsMs: eeg.timestampsMs?.slice(-windowSamples.length),
      receivedAtMs: frame.emittedAtMs,
      sequenceId: ++this.emittedSequenceId,
      quality: analysis?.quality ?? {
        source: "inferred",
        message: "Headset fit assessment unavailable",
      },
      features: analysis?.features ?? null,
      training: analysis?.training ?? null,
    };

    if (normalized.sequenceId <= 10 || normalized.sequenceId % 100 === 0) {
      console.log("[Normalized Web Bluetooth EEG frame]", normalized);
    }
    this.events.onSignalFrame(normalized);
  }

  /** Opens this connection's stateful analysis session in brainflow_service
   * -- see `analysis.AnalysisSessionStore`. Best-effort: if this fails
   * (service unreachable), `analyzeWindow` degrades to "no analysis"
   * rather than blocking the connection, and its own error reporting
   * already surfaces a service-down error to the user. */
  private async startFitSession() {
    try {
      const response = await fetch(`${this.brainFlowServiceUrl}/headset-fit/sessions`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as { fitSessionId?: string };
      this.fitSessionId = payload.fitSessionId ?? null;
    } catch (error) {
      this.fitSessionId = null;
      console.warn("Unable to start headset-fit session", error);
    }
  }

  /** Headset fit, and the smoothed mindfulness/restfulness/focus/relax and
   * valence/arousal scores, all happen here, entirely server-side, via
   * `POST /headset-fit/sessions/{id}/analyze-window` -- this method only
   * sends this window's samples and reads back the result. Using this
   * connection's session (rather than the stateless `/analyze-window`) is
   * what gives Web Bluetooth the same real smoothing/calibration a direct
   * BrainFlow connection's `/sessions/{id}/stream` applies -- see
   * `brainflow_service/analysis.py`. */
  private async analyzeWindow(
    samples: number[][],
    sampleRateHz: number,
    channelIds: string[],
  ): Promise<{
    features: SignalFeatures | null;
    quality: SignalQualityMetadata | null;
    training: TrainingMetricSample | null;
  } | null> {
    if (!this.fitSessionId) return null;

    try {
      const response = await fetch(
        `${this.brainFlowServiceUrl}/headset-fit/sessions/${this.fitSessionId}/analyze-window`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sampleRateHz, samples, channelIds }),
        },
      );
      if (!response.ok) {
        if (response.status === 404) {
          // The session was evicted server-side (e.g. idle timeout) --
          // get a fresh one for the next window instead of failing for
          // the rest of the connection.
          this.fitSessionId = null;
          void this.startFitSession();
        }
        throw new Error(await response.text());
      }

      const payload = (await response.json()) as {
        features?: SignalFeatures | null;
        quality?: SignalQualityMetadata | null;
        training?: TrainingMetricSample | null;
      };
      this.analysisFailureReported = false;
      return {
        features: payload.features ?? null,
        quality: payload.quality ?? null,
        training: payload.training ?? null,
      };
    } catch (error) {
      if (!this.analysisFailureReported) {
        this.analysisFailureReported = true;
        this.events.onError({
          message:
            error instanceof Error
              ? `BrainFlow analysis unavailable for Web Bluetooth stream: ${error.message}`
              : "BrainFlow analysis unavailable for Web Bluetooth stream.",
          recoverable: true,
        });
      }
      return null;
    }
  }

  /** Starts/resets this connection's valence/arousal calibration -- the
   * Bluetooth counterpart of `BrainFlowHttpProvider`'s calls to
   * `/sessions/{id}/calibration/*`, backed by the same
   * `AffectiveStateProvider` via this connection's analysis session. */
  async startAffectiveCalibration() {
    if (!this.fitSessionId) return;
    await fetch(`${this.brainFlowServiceUrl}/headset-fit/sessions/${this.fitSessionId}/calibration/start`, {
      method: "POST",
    });
  }

  async resetAffectiveCalibration() {
    if (!this.fitSessionId) return;
    await fetch(`${this.brainFlowServiceUrl}/headset-fit/sessions/${this.fitSessionId}/calibration/reset`, {
      method: "POST",
    });
  }

  private publishDeviceInfo(transport: BleTransport | null, frame?: HeadbandFrameV1) {
    const boardInfo = safeBoardInfo(transport);
    const reportedNames = boardInfo?.eeg_channel_names ?? frame?.eeg.channelNames ?? [];
    // Advertise the same channels the frames carry -- see
    // `scalpElectrodeIndices`.
    const channelNames = scalpElectrodeIndices(reportedNames).map(
      (index) => reportedNames[index],
    );
    const sampleRate = boardInfo?.sample_rate_hz ?? frame?.eeg.sampleRateHz ?? null;

    this.deviceInfo = {
      label: boardInfo?.device_name ?? "Muse Athena",
      model: boardInfo?.description ?? "Muse Athena Web Bluetooth",
      providerName: "Elata Web Bluetooth",
      capabilities: [
        {
          kind: "eeg",
          sampleRateHz: sampleRate,
          channels: channelNames.map((name) => ({
            id: name.toLowerCase(),
            label: name,
            unit: "uV",
          })),
        },
      ],
      metadata: {
        protocol: boardInfo?.protocol ?? "athena",
        source: "web-bluetooth",
        eegProcessing: "disabled",
      },
    };
    this.events.onDeviceInfo(this.deviceInfo);
  }

  private emitError(message: string, code: string) {
    const error: ProviderError = { message, code, recoverable: true };
    this.events.onState("error", message);
    this.events.onError(error);
  }
}

function safeBoardInfo(transport: BleTransport | null) {
  try {
    return transport?.getBoardInfo() as
      | {
          device_name: string;
          sample_rate_hz: number;
          eeg_channel_names: string[];
          protocol: string;
          description: string;
        }
      | undefined;
  } catch {
    return undefined;
  }
}

function mapTransportState(state: string): EegConnectionState {
  if (state === "idle") return "idle";
  if (state === "connecting") return "connecting";
  if (state === "connected") return "connected";
  if (state === "streaming") return "streaming";
  if (state === "disconnected") return "disconnected";
  if (state === "error") return "error";
  return "connected";
}

function formatBluetoothError(error: unknown): ProviderError {
  if (isUserCancelledBluetoothRequest(error)) {
    return {
      message: "Bluetooth device selection was cancelled.",
      code: "REQUEST_CANCELLED",
      recoverable: true,
    };
  }

  return {
    message:
      error instanceof Error
        ? `Web Bluetooth error: ${error.message}`
        : "Web Bluetooth error.",
    recoverable: true,
  };
}

function isUserCancelledBluetoothRequest(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("User cancelled") ||
    message.includes("User canceled") ||
    message.includes("requestDevice() chooser")
  );
}
