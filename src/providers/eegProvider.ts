import type { DeviceInfo, EegProviderEvents } from "../domain/eeg";

export interface EegProviderDescriptor {
  id: string;
  label: string;
  description: string;
}

export interface EegProvider {
  readonly descriptor: EegProviderDescriptor;
  connectAndStart(options?: { replayFile?: File | string }): Promise<void>;
  disconnect(reason?: string): Promise<void>;
  getDeviceInfo(): DeviceInfo | null;
}

export type EegProviderFactory = (events: EegProviderEvents) => EegProvider;
