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
  SignalFrame,
} from "../domain/eeg";
import type { EegProvider, EegProviderDescriptor } from "./eegProvider";

export class MuseAthenaBluetoothProvider implements EegProvider {
  readonly descriptor: EegProviderDescriptor = {
    id: "muse-athena-bluetooth",
    label: "Muse Athena - Bluetooth",
    description: "Chrome Web Bluetooth via Elata EEG SDK",
  };

  private transport: BleTransport | null = null;
  private deviceInfo: DeviceInfo | null = null;

  constructor(private readonly events: EegProviderEvents) {}

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

    try {
      await initEegWasm(eegWasmUrl);

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
      transport.onFrame = (frame) => this.handleFrame(frame);

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
    if (transport) {
      try {
        await transport.disconnect();
      } catch (error) {
        console.warn("Unable to disconnect Web Bluetooth transport", error);
      }
    }

    this.events.onState("disconnected", reason);
  }

  private handleFrame(frame: HeadbandFrameV1) {
    const eeg = frame.eegRaw ?? frame.eeg;
    if (!this.deviceInfo) {
      this.publishDeviceInfo(this.transport, frame);
    }

    const normalized: SignalFrame = {
      sensor: "eeg",
      sampleRateHz: eeg.sampleRateHz,
      channels: eeg.channelNames.map((name) => ({
        id: name.toLowerCase(),
        label: name,
        unit: "uV",
      })),
      samples: eeg.samples,
      timestampsMs: eeg.timestampsMs,
      receivedAtMs: frame.emittedAtMs,
      sequenceId: frame.sequenceId,
      quality: {
        source: "inferred",
        excessiveArtifact: false,
        message: "Quality inferred from Web Bluetooth EEG stream",
      },
      features: null,
    };

    if (normalized.sequenceId <= 10 || normalized.sequenceId % 100 === 0) {
      console.log("[Normalized Web Bluetooth EEG frame]", normalized);
    }
    this.events.onSignalFrame(normalized);
  }

  private publishDeviceInfo(transport: BleTransport | null, frame?: HeadbandFrameV1) {
    const boardInfo = safeBoardInfo(transport);
    const channelNames = boardInfo?.eeg_channel_names ?? frame?.eeg.channelNames ?? [];
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
