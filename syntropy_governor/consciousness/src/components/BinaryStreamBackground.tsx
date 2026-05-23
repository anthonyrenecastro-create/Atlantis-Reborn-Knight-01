import { useMemo } from "react";
import { useTheme } from "../context/ThemeContext";

export default function BinaryStreamBackground() {
  const { mode } = useTheme();

  const streams = useMemo(
    () =>
      Array.from({ length: 28 }).map((_, i) => ({
        id: i,
        left: `${(i / 28) * 100}%`,
        delay: `${(i * 0.12).toFixed(2)}s`,
        duration: `${(7 + (i % 8)).toFixed(1)}s`,
        text: i % 2 === 0 ? "01001011100101" : "10110100011100",
      })),
    [],
  );

  return (
    <div className={`bg-canvas ${mode}`}>
      <div className="bg-gradient" />
      <div className="bg-grid" />
      {streams.map((s) => (
        <span
          key={s.id}
          className="binary-stream"
          style={{ left: s.left, animationDelay: s.delay, animationDuration: s.duration }}
        >
          {s.text}
        </span>
      ))}
    </div>
  );
}