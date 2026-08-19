# EEG Provider Architecture

The application consumes hardware through `EegProvider`, not through device SDKs,
Bluetooth objects, serial ports, or vendor-specific protocol structures.

Provider responsibilities:

- Discover and connect to the physical or simulated data source.
- Own reconnection, disconnection, and transport cleanup.
- Decode raw vendor or protocol frames.
- Emit normalized `DeviceInfo`, `SignalFrame`, lifecycle state, and provider errors.

Application responsibilities:

- Select a provider through configuration.
- Render provider/device state.
- Process and store normalized domain frames.

BrainFlow hardware now runs through the local BrainFlow service. The browser
provider is only a transport bridge to that service; it does not talk to BLE or
vendor SDKs directly.

To add hardware, add a BrainFlow device config in `brainflow_service/config.py`
and register a matching frontend option in `providerRegistry.ts`. The UI should
not import hardware-specific adapters directly.

The visible device selector is driven by `deviceCatalog`. Keep unavailable
hardware entries disabled until their adapters exist.

Use the BrainFlow Synthetic device to develop without physical hardware.
