// A rotary knob (0..1). Drag up/down or scroll to turn; releases commit.

import { useState } from "react";
import type { PointerEvent, WheelEvent } from "react";

const SWEEP = 270.0;

function polar(cx: number, cy: number, r: number, deg: number): number[] {
  const rad = ((deg - 90.0) * Math.PI) / 180.0;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

function arc(cx: number, cy: number, r: number, start: number, end: number): string {
  const a = polar(cx, cy, r, start);
  const b = polar(cx, cy, r, end);
  const large = end - start > 180.0 ? 1 : 0;
  return `M ${a[0]} ${a[1]} A ${r} ${r} 0 ${large} 1 ${b[0]} ${b[1]}`;
}

function clamp01(v: number): number {
  return v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v;
}

export interface KnobProps {
  value: number;
  size: number;
  color: string;
  onChange: (v: number) => void;
  onCommit: () => void;
}

export function Knob({ value, size, color, onChange, onCommit }: KnobProps) {
  const [dragging, setDragging] = useState(false);

  function down(e: PointerEvent<SVGSVGElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  }

  function move(e: PointerEvent<SVGSVGElement>) {
    if (dragging) onChange(clamp01(value - e.movementY / 140.0));
  }

  function up() {
    if (dragging) {
      setDragging(false);
      onCommit();
    }
  }

  function wheel(e: WheelEvent<SVGSVGElement>) {
    onChange(clamp01(value - e.deltaY / 900.0));
  }

  const c = size / 2.0;
  const r = size / 2.0 - 4.0;
  const start = -SWEEP / 2.0;
  const end = start + SWEEP * value;
  const tip = polar(c, c, r - 6.0, end);
  return (
    <svg
      className={"knob" + (dragging ? " dragging" : "")}
      width={size}
      height={size}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onWheel={wheel}
      onMouseLeave={() => {
        if (!dragging) onCommit();
      }}
    >
      <circle cx={c} cy={c} r={r - 3.0} className="knob-body" />
      <path d={arc(c, c, r, start, start + SWEEP)} className="knob-track" />
      {value > 0.004 && <path d={arc(c, c, r, start, end)} className="knob-value" stroke={color} />}
      <line x1={c} y1={c} x2={tip[0]} y2={tip[1]} className="knob-tick" stroke={color} />
    </svg>
  );
}
