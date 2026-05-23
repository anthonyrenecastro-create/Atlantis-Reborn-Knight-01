type FieldHeatmapProps = {
  title: string;
  values: number[];
};

function clamp01(n: number) {
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

function toHeat(value: number) {
  const normalized = clamp01((value + 1) / 2);
  const hue = Math.round(170 - normalized * 130);
  const alpha = 0.22 + normalized * 0.72;
  return `hsla(${hue}, 90%, 62%, ${alpha.toFixed(2)})`;
}

export default function FieldHeatmap({ title, values }: FieldHeatmapProps) {
  return (
    <section className="field-heatmap-shell">
      <h3>{title}</h3>
      <div className="heatmap" aria-label={title}>
        {values.map((value, idx) => (
          <span key={idx} className="heatmap-cell" style={{ background: toHeat(value) }} />
        ))}
      </div>
    </section>
  );
}
