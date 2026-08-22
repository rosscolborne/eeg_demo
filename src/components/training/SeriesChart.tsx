import { useEffect, useRef } from "react";
import { LineChart } from "lucide-react";

export interface SeriesChartLine {
  label: string;
  color: string;
  values: Array<{ atMs: number; value: number | null }>;
}

export interface SeriesChartMarker {
  atMs: number;
  label: string;
  color: string;
}

interface SeriesChartProps {
  lines: SeriesChartLine[];
  markers?: SeriesChartMarker[];
  emptyTitle: string;
  emptyDescription: string;
  height?: number;
  min?: number;
  max?: number;
}

export function SeriesChart({
  lines,
  markers = [],
  emptyTitle,
  emptyDescription,
  height = 260,
  min,
  max,
}: SeriesChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isFinitePoint = (point: { atMs: number; value: number | null }) =>
    Number.isFinite(point.atMs) && Number.isFinite(point.value);
  const hasData = lines.some((line) =>
    line.values.some(isFinitePoint),
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
      line.values.filter(isFinitePoint),
    );
    if (allPoints.length === 0) return;
    const minTime = Math.min(...allPoints.map((point) => point.atMs));
    const maxTime = Math.max(...allPoints.map((point) => point.atMs));
    const finiteValues = allPoints.map((point) => point.value ?? 0);
    const minValue = min ?? Math.min(...finiteValues);
    const maxValue = max ?? Math.max(...finiteValues);
    const valueRange = Math.max(1e-9, maxValue - minValue);
    const timeRange = Math.max(1, maxTime - minTime);

    for (const marker of markers) {
      if (marker.atMs < minTime || marker.atMs > maxTime) continue;

      const x = ((marker.atMs - minTime) / timeRange) * width;
      context.strokeStyle = marker.color;
      context.lineWidth = 1;
      context.setLineDash([4, 5]);
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, chartHeight);
      context.stroke();
      context.setLineDash([]);

      context.save();
      context.translate(Math.min(width - 10, x + 8), 14);
      context.rotate(-Math.PI / 2);
      context.fillStyle = marker.color;
      context.font = "700 11px Inter, system-ui, sans-serif";
      context.fillText(marker.label, 0, 0);
      context.restore();
    }

    for (const line of lines) {
      const points = line.values.filter(isFinitePoint);
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
  }, [hasData, height, lines, markers, max, min]);

  return (
    <div className="series-chart" style={{ height }}>
      <div className="series-chart-legend" aria-label="Chart legend">
        {lines.map((line) => (
          <span key={line.label}>
            <i style={{ background: line.color }} aria-hidden="true" />
            {line.label}
          </span>
        ))}
      </div>
      <div className="series-chart-body">
        {!hasData && (
          <div className="empty-state chart-empty">
            <div className="icon-tile">
              <LineChart aria-hidden="true" />
            </div>
            <strong>{emptyTitle}</strong>
            <span>{emptyDescription}</span>
          </div>
        )}
        <canvas ref={canvasRef} aria-label={emptyTitle} />
      </div>
    </div>
  );
}
