export type EegConnectionState =
  | "idle"
  | "initializing"
  | "discovering"
  | "connecting"
  | "connected"
  | "streaming"
  | "disconnecting"
  | "disconnected"
  | "error";

export type SensorKind = "eeg" | "ppg" | "accelerometer" | "gyroscope" | "battery";

export interface SignalChannel {
  id: string;
  label: string;
  unit: string;
}

export type ContactQualityState = "unknown" | "poor" | "adjusting" | "good";

export interface SignalChannelQualityMetadata {
  channelId: string;
  state: ContactQualityState;
  source: "device" | "inferred";
  score?: number;
  impedanceOhms?: number;
  message?: string;
}

export interface SignalQualityMetadata {
  source: "device" | "inferred";
  channelQualities?: SignalChannelQualityMetadata[];
  excessiveArtifact?: boolean;
  worn?: boolean;
  message?: string;
}

export interface BandPowerFeatures {
  absolute: Record<string, number>;
  relative: Record<string, number>;
  ratios: Record<string, number>;
  windowSeconds: number;
  method: "brainflow_welch_psd" | "custom_goertzel";
}

export interface SignalFeatures {
  bandPowers?: BandPowerFeatures | null;
  brainflowConcentration?: number | null;
}

export interface SensorCapability {
  kind: SensorKind;
  sampleRateHz: number | null;
  channels: SignalChannel[];
}

export interface DeviceInfo {
  label: string;
  model: string;
  providerName: string;
  firmwareVersion?: string;
  capabilities: SensorCapability[];
  metadata?: Record<string, unknown>;
}

export interface SignalFrame {
  sensor: SensorKind;
  sampleRateHz: number | null;
  channels: SignalChannel[];
  samples: number[][];
  timestampsMs?: number[];
  receivedAtMs: number;
  sequenceId: number;
  quality?: SignalQualityMetadata;
  features?: SignalFeatures | null;
}

export interface EegFrameSummary {
  latestByChannel: Record<string, number>;
  samplesInFrame: number;
  totalSamples: number;
  frameCount: number;
  sampleRateHz: number | null;
  channelNames: string[];
}

export interface ProviderError {
  message: string;
  code?: string;
  recoverable?: boolean;
  details?: Record<string, unknown>;
}

export interface EegProviderEvents {
  onState: (state: EegConnectionState, detail?: string) => void;
  onDeviceInfo: (info: DeviceInfo) => void;
  onSignalFrame: (frame: SignalFrame) => void;
  onError: (error: ProviderError) => void;
}

export function getCapability(
  device: DeviceInfo | null,
  kind: SensorKind,
): SensorCapability | null {
  return device?.capabilities.find((capability) => capability.kind === kind) ?? null;
}

export function summarizeEegFrame(
  frame: SignalFrame,
  previousTotalSamples: number,
  previousFrameCount: number,
): EegFrameSummary {
  const latestRow = frame.samples[frame.samples.length - 1] ?? [];
  const channelNames = frame.channels.map((channel) => channel.label);
  const latestByChannel = Object.fromEntries(
    frame.channels.map((channel, index) => [
      channel.label,
      latestRow[index] ?? Number.NaN,
    ]),
  );

  return {
    latestByChannel,
    samplesInFrame: frame.samples.length,
    totalSamples: previousTotalSamples + frame.samples.length,
    frameCount: previousFrameCount + 1,
    sampleRateHz: frame.sampleRateHz,
    channelNames,
  };
}
