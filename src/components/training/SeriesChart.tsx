import { useEffect, useRef } from "react";
import { LineChart } from "lucide-react";

export interface SeriesChartLine {
  label: string;
  color: string;
  values: Array<{ atMs: number; value: number | null }>;
}

interface SeriesChartProps {
  lines: SeriesChartLine[];
  emptyTitle: string;
  emptyDescription: string;
  height?: number;
  min?: number;
  max?: number;
}

export function SeriesChart({
  lines,
  emptyTitle,
  emptyDescription,
  height = 260,
  min,
  max,
}: SeriesChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hasData = lines.some((line) =>
    line.values.some((point) => Number.isFinite(point.value)),
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * scale));
    canvas.height = Math.max(1, Math.floor(rect.height * scale));
    context.setTransform(scale, 0, 0, scale, 0, 0);

    const width = rect.width;
    const chartHeight = rect.height;
    context.clearRect(0, 0, width, chartHeight);
    context.fillStyle = "#090f20";
    context.fillRect(0, 0, width, chartHeight);
    context.strokeStyle = "rgba(255, 255, 255, 0.07)";
    context.lineWidth = 1;

    for (let y = 40; y < chartHeight; y += 40) {
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    if (!hasData) return;

    const allPoints = lines.flatMap((line) =>
      line.values.filter((point) => Number.isFinite(point.value)),
    );
    const minTime = Math.min(...allPoints.map((point) => point.atMs));
    const maxTime = Math.max(...allPoints.map((point) => point.atMs));
    const finiteValues = allPoints.map((point) => point.value ?? 0);
    const minValue = min ?? Math.min(...finiteValues);
    const maxValue = max ?? Math.max(...finiteValues);
    const valueRange = Math.max(1e-9, maxValue - minValue);
    const timeRange = Math.max(1, maxTime - minTime);

    for (const line of lines) {
      const points = line.values.filter((point) => Number.isFinite(point.value));
      if (points.length < 2) continue;

      context.strokeStyle = line.color;
      context.lineWidth = 2;
      context.beginPath();

      points.forEach((point, index) => {
        const x = ((point.atMs - minTime) / timeRange) * width;
        const y = chartHeight - (((point.value ?? 0) - minValue) / valueRange) * chartHeight;

        if (index === 0) {
          context.moveTo(x, y);
        } else {
          context.lineTo(x, y);
        }
      });

      context.stroke();
    }
  }, [hasData, height, lines, max, min]);

  return (
    <div className="series-chart" style={{ height }}>
      {!hasData && (
        <div className="empty-state chart-empty">
          <div className="icon-tile">
            <LineChart aria-hidden="true" />
          </div>
          <strong>{emptyTitle}</strong>
          <span>{emptyDescription}</span>
        </div>
      )}
      <canvas ref={canvasRef} style={{ height }} aria-label={emptyTitle} />
    </div>
  );
}
