export interface HeadsetFitThresholds {
  stableReadyMs: number;
  staleFrameMs: number;
  minSamplesPerFrame: number;
  minGoodChannels: number;
  minGoodChannelFraction: number;
  maxPoorChannelFraction: number;
  minRmsUv: number;
  maxRmsUv: number;
  minStdDevUv: number;
  maxGoodStdDevUv: number;
  maxGoodPeakToPeakUv: number;
  maxGoodMeanStepUv: number;
  maxGoodStepUv: number;
  maxAbsoluteUv: number;
  maxStepUv: number;
  maxClippedFraction: number;
}

export const defaultHeadsetFitThresholds: HeadsetFitThresholds = {
  stableReadyMs: 3500,
  staleFrameMs: 1500,
  minSamplesPerFrame: 16,
  minGoodChannels: 2,
  minGoodChannelFraction: 0.5,
  maxPoorChannelFraction: 0.5,
  minRmsUv: 0.35,
  maxRmsUv: 20000,
  minStdDevUv: 0.25,
  maxGoodStdDevUv: 1200,
  maxGoodPeakToPeakUv: 3600,
  maxGoodMeanStepUv: 1500,
  maxGoodStepUv: 3800,
  maxAbsoluteUv: 100000,
  maxStepUv: 6500,
  maxClippedFraction: 0.3,
};
