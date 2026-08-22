import type { SignalFrame } from "../domain/eeg";
import {
  computeBandPowers,
  type FrequencyBand,
} from "../signalProcessing/bandPower";
import type { HeadsetFitSnapshot } from "../signalQuality/headsetFitProvider";

export const museEegChannels = ["tp9", "af7", "af8", "tp10"] as const;
export const frontalChannels = ["af7", "af8"] as const;
export const temporalChannels = ["tp9", "tp10"] as const;

const ratioBands = ["theta", "alpha", "beta"] as const;

export interface ResolvedBandPowers {
  powers: Record<string, number>;
  perChannel: Record<string, Record<string, number>>;
  usableChannelKeys: string[];
  channelCount: number;
  sampleCount: number;
  sampleRateHz: number;
}

export function normalizeChannelId(id: string) {
  return id.trim().toLowerCase();
}

export function resolveBandPowers(
  frame: SignalFrame,
  bands: FrequencyBand[],
  quality?: HeadsetFitSnapshot | null,
): ResolvedBandPowers | null {
  const featurePowers = frame.features?.bandPowers;
  if (frame.features != null && !featurePowers) return null;

  let perChannel = featurePowers?.perChannel ?? {};
  let powers = featurePowers?.absolute ?? {};
  let channelCount = frame.channels.length;
  let sampleCount = frame.samples.length;
  let sampleRateHz = frame.sampleRateHz ?? 0;

  if (!featurePowers?.absolute) {
    const computed = computeBandPowers(frame, bands);
    if (!computed) return null;
    perChannel = computed.perChannel;
    powers = computed.powers;
    channelCount = computed.channelCount;
    sampleCount = computed.sampleCount;
    sampleRateHz = computed.sampleRateHz;
  }

  const usable = selectChannelIds(Object.keys(perChannel), frame, quality);
  if (usable.length > 0 && Object.keys(perChannel).length > 0) {
    powers = meanPowersForChannels(perChannel, usable);
  }

  if (looksLikeHighFrequencyArtifact(powers)) return null;

  return {
    powers,
    perChannel,
    usableChannelKeys: usable,
    channelCount,
    sampleCount,
    sampleRateHz,
  };
}

export function selectChannelIds(
  availableKeys: string[],
  frame: SignalFrame,
  quality?: HeadsetFitSnapshot | null,
) {
  const available = availableKeys.length
    ? availableKeys
    : frame.channels.map((channel) => channel.id);
  const normalizedAvailable = new Map(
    available.map((key) => [normalizeChannelId(key), key]),
  );
  const museKeys = museEegChannels
    .map((id) => normalizedAvailable.get(id))
    .filter((key): key is string => Boolean(key));
  const pool = museKeys.length
    ? museKeys
    : available.filter((key) => !normalizeChannelId(key).startsWith("aux"));
  const goodIds = new Set(
    (quality?.channels ?? [])
      .filter((channel) => channel.state === "good")
      .map((channel) => normalizeChannelId(channel.channel.id)),
  );
  const goodInPool = pool.filter((key) => goodIds.has(normalizeChannelId(key)));
  return goodInPool.length > 0 ? goodInPool : pool;
}

export function meanPowersForChannels(
  perChannel: Record<string, Record<string, number>>,
  channelKeys: string[],
) {
  const powers: Record<string, number> = {};
  const bands = new Set<string>();
  for (const key of channelKeys) {
    for (const band of Object.keys(perChannel[key] ?? {})) {
      bands.add(band);
    }
  }
  for (const band of bands) {
    powers[band] = meanBand(perChannel, channelKeys, band);
  }
  return powers;
}

export function meanBand(
  perChannel: Record<string, Record<string, number>>,
  channelKeys: string[],
  band: string,
) {
  const values = channelKeys
    .map((key) => perChannel[key]?.[band])
    .filter((value): value is number => Number.isFinite(value) && value >= 0);
  if (values.length === 0) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

export function groupedPowers(
  perChannel: Record<string, Record<string, number>>,
  wantedIds: readonly string[],
  fallback: Record<string, number>,
  usableKeys?: string[],
) {
  const usable = new Set(
    (usableKeys ?? Object.keys(perChannel)).map(normalizeChannelId),
  );
  const keys = wantedIds
    .map((id) => findChannelKey(perChannel, id))
    .filter((key): key is string => key != null && usable.has(normalizeChannelId(key)));
  if (keys.length === 0) return fallback;
  return {
    theta: meanBand(perChannel, keys, "theta"),
    alpha: meanBand(perChannel, keys, "alpha"),
    beta: meanBand(perChannel, keys, "beta"),
    gamma: meanBand(perChannel, keys, "gamma"),
  };
}

export function frontalAlphaAsymmetry(
  perChannel: Record<string, Record<string, number>>,
  usableKeys: string[],
) {
  const usable = new Set(usableKeys.map(normalizeChannelId));
  const leftKey = findChannelKey(perChannel, "af7");
  const rightKey = findChannelKey(perChannel, "af8");
  if (!leftKey || !rightKey) return null;
  if (!usable.has("af7") || !usable.has("af8")) return null;

  const leftAlpha = relativeBand(perChannel[leftKey], "alpha");
  const rightAlpha = relativeBand(perChannel[rightKey], "alpha");
  if (leftAlpha === null || rightAlpha === null) return null;

  return Math.log(rightAlpha) - Math.log(leftAlpha);
}

function findChannelKey(
  perChannel: Record<string, Record<string, number>>,
  wantedId: string,
) {
  const wanted = normalizeChannelId(wantedId);
  return Object.keys(perChannel).find((key) => normalizeChannelId(key) === wanted);
}

function relativeBand(powers: Record<string, number> | undefined, band: string) {
  if (!powers) return null;
  const total = ratioBands.reduce((sum, id) => sum + Math.max(0, powers[id] ?? 0), 0);
  const value = powers[band] ?? 0;
  if (total <= 0 || value <= 0) return null;
  return value / total;
}

function looksLikeHighFrequencyArtifact(powers: Record<string, number>) {
  const theta = Math.max(0, powers.theta ?? 0);
  const alpha = Math.max(0, powers.alpha ?? 0);
  const beta = Math.max(0, powers.beta ?? 0);
  const gamma = Math.max(0, powers.gamma ?? 0);
  const total = theta + alpha + beta + gamma;
  return total > 0 && gamma / total > 0.45;
}
