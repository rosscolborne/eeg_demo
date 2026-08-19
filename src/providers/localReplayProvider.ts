import type {
  DeviceInfo,
  EegProviderEvents,
  ProviderError,
  SignalFrame,
} from "../domain/eeg";
import type { EegProvider, EegProviderDescriptor } from "./eegProvider";

export interface AppRecording {
  format: "eeg-demo-normalized-recording";
  version: 1;
  createdAt: string;
  deviceInfo: DeviceInfo | null;
  frames: SignalFrame[];
}

export class LocalReplayProvider implements EegProvider {
  readonly descriptor: EegProviderDescriptor = {
    id: "local-replay",
    label: "Replay",
    description: "Downloaded normalized EEG recording",
  };

  private deviceInfo: DeviceInfo | null = null;
  private timeoutIds: number[] = [];
  private stopped = true;

  constructor(private readonly events: EegProviderEvents) {}

  getDeviceInfo() {
    return this.deviceInfo;
  }

  async connectAndStart(options?: { replayFile?: File | string }) {
    await this.disconnect();
    this.events.onState("initializing", "Loading replay recording");

    try {
      if (!(options?.replayFile instanceof File)) {
        throw new Error("Choose a downloaded EEG demo recording JSON file.");
      }

      const recording = parseRecording(await options.replayFile.text());
      if (recording.frames.length === 0) {
        throw new Error("The selected recording contains no EEG frames.");
      }

      this.deviceInfo = recording.deviceInfo ?? replayDeviceInfo(recording.frames[0]);
      this.events.onDeviceInfo(this.deviceInfo);
      this.events.onState("streaming", "Replaying downloaded recording");
      this.stopped = false;
      this.scheduleFrames(recording.frames);
    } catch (error) {
      const providerError = formatReplayError(error);
      this.events.onState("error", providerError.message);
      this.events.onError(providerError);
    }
  }

  async disconnect(reason = "Replay stopped") {
    this.timeoutIds.forEach((id) => window.clearTimeout(id));
    this.timeoutIds = [];
    this.stopped = true;
    this.events.onState("disconnected", reason);
  }

  private scheduleFrames(frames: SignalFrame[]) {
    const firstReceivedAt = frames[0].receivedAtMs;
    const startedAt = performance.now();

    frames.forEach((frame, index) => {
      const delay = Math.max(0, frame.receivedAtMs - firstReceivedAt);
      const timeoutId = window.setTimeout(() => {
        if (this.stopped) return;
        this.events.onSignalFrame({
          ...frame,
          receivedAtMs: startedAt + delay,
          sequenceId: index + 1,
        });
        if (index === frames.length - 1) {
          this.events.onState("disconnected", "Replay complete");
          this.stopped = true;
        }
      }, delay);
      this.timeoutIds.push(timeoutId);
    });
  }
}

export function createAppRecording(
  deviceInfo: DeviceInfo | null,
  frames: SignalFrame[],
): AppRecording {
  return {
    format: "eeg-demo-normalized-recording",
    version: 1,
    createdAt: new Date().toISOString(),
    deviceInfo,
    frames,
  };
}

function parseRecording(text: string): AppRecording {
  const parsed = JSON.parse(text) as Partial<AppRecording>;
  if (
    parsed.format !== "eeg-demo-normalized-recording" ||
    parsed.version !== 1 ||
    !Array.isArray(parsed.frames)
  ) {
    throw new Error(
      "Unsupported replay file. Use a JSON file downloaded from this app's recording button.",
    );
  }

  return parsed as AppRecording;
}

function replayDeviceInfo(frame: SignalFrame): DeviceInfo {
  return {
    label: "Replay recording",
    model: "Normalized EEG recording",
    providerName: "Local Replay",
    capabilities: [
      {
        kind: frame.sensor,
        sampleRateHz: frame.sampleRateHz,
        channels: frame.channels,
      },
    ],
  };
}

function formatReplayError(error: unknown): ProviderError {
  return {
    message:
      error instanceof Error
        ? `Replay error: ${error.message}`
        : "Replay error.",
    recoverable: true,
  };
}
