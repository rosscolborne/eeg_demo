import type {
  DeviceInfo,
  EegConnectionState,
  EegProviderEvents,
  ProviderError,
  SignalFrame,
} from "../domain/eeg";
import type { EegProvider, EegProviderDescriptor } from "./eegProvider";

const defaultServiceUrl =
  import.meta.env.VITE_BRAINFLOW_SERVICE_URL ?? "http://127.0.0.1:8000";

export type BrainFlowDeviceId =
  | "brainflow-muse-athena"
  | "brainflow-synthetic";

export class BrainFlowHttpProvider implements EegProvider {
  readonly descriptor: EegProviderDescriptor;
  private eventSource: EventSource | null = null;
  private sessionId: string | null = null;
  private deviceInfo: DeviceInfo | null = null;

  constructor(
    private readonly deviceId: BrainFlowDeviceId,
    label: string,
    private readonly events: EegProviderEvents,
    private readonly serviceUrl = defaultServiceUrl,
  ) {
    this.descriptor = {
      id: deviceId,
      label,
      description: "BrainFlow local service",
    };
  }

  getDeviceInfo() {
    return this.deviceInfo;
  }

  async startAffectiveCalibration() {
    if (!this.sessionId) return;
    await fetch(`${this.serviceUrl}/sessions/${this.sessionId}/calibration/start`, {
      method: "POST",
    });
  }

  async resetAffectiveCalibration() {
    if (!this.sessionId) return;
    await fetch(`${this.serviceUrl}/sessions/${this.sessionId}/calibration/reset`, {
      method: "POST",
    });
  }

  async connectAndStart() {
    await this.disconnect("Preparing a fresh BrainFlow session");
    this.events.onState("connecting", "Connecting to BrainFlow service");

    try {
      const response = await fetch(`${this.serviceUrl}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          deviceId: this.deviceId,
        }),
      });
      if (!response.ok) {
        const message = await responseErrorMessage(response);
        throw new Error(message || `BrainFlow service returned ${response.status}`);
      }

      const payload = await response.json();
      const deviceInfo = payload.deviceInfo as DeviceInfo;
      this.sessionId = payload.sessionId;
      this.deviceInfo = deviceInfo;
      this.events.onDeviceInfo(deviceInfo);
      this.events.onState("connected", "BrainFlow session prepared");
      this.openStream();
    } catch (error) {
      const providerError = formatBrainFlowError(error);
      this.events.onState("error", providerError.message);
      this.events.onError(providerError);
    }
  }

  async disconnect(reason = "Disconnecting") {
    this.eventSource?.close();
    this.eventSource = null;
    const sessionId = this.sessionId;
    this.sessionId = null;

    if (sessionId) {
      this.events.onState("disconnecting", reason);
      try {
        await fetch(`${this.serviceUrl}/sessions/${sessionId}`, { method: "DELETE" });
      } catch (error) {
        console.warn("Unable to stop BrainFlow service session", error);
      }
    }

    this.events.onState("disconnected");
  }

  private openStream() {
    if (!this.sessionId) return;

    const stream = new EventSource(`${this.serviceUrl}/sessions/${this.sessionId}/stream`);
    this.eventSource = stream;

    stream.addEventListener("state", (event) => {
      const payload = JSON.parse(event.data) as { state: EegConnectionState; detail?: string };
      this.events.onState(payload.state, payload.detail);
    });

    stream.addEventListener("signalFrame", (event) => {
      const frame = JSON.parse(event.data) as SignalFrame;
      if (frame.sequenceId <= 10 || frame.sequenceId % 100 === 0) {
        console.log("[Normalized EEG frame]", frame);
      }
      this.events.onSignalFrame(frame);
    });

    stream.addEventListener("error", (event) => {
      if ("data" in event && typeof event.data === "string" && event.data) {
        this.events.onError({ message: JSON.parse(event.data).message });
      }
    });

    stream.onerror = () => {
      this.events.onState("error", "BrainFlow stream disconnected");
    };
  }
}

async function responseErrorMessage(response: Response) {
  const text = await response.text();
  try {
    const payload = JSON.parse(text) as { detail?: unknown };
    return typeof payload.detail === "string" ? payload.detail : text;
  } catch {
    return text;
  }
}

function formatBrainFlowError(error: unknown): ProviderError {
  if (error instanceof TypeError) {
    return {
      message:
        "BrainFlow service is not reachable. Start the Python BrainFlow service on http://127.0.0.1:8000, then try Connect again.",
      recoverable: true,
    };
  }

  return {
    message:
      error instanceof Error
        ? `BrainFlow service error: ${error.message}`
        : "BrainFlow service error.",
    recoverable: true,
  };
}
