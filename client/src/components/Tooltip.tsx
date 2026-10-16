// Quick tooltips for the whole page. Any element with a `title` gets a styled tip
// shortly after the mouse rests on it (the browser's own tooltip takes a second or
// more and is easy to miss). The title moves into `data-tip` on first hover so the
// two never show together; React puts a changed title back, and the next hover
// moves it again.

import { useEffect, useState } from "react";

let _tip_timer: ReturnType<typeof setTimeout> | null = null;
// set by the mounted Tooltip each render: show(text, x, y, above) / hide()
let _tip_show: ((t: string, x: number, y: number, above: boolean) => void) | null = null;

function closest(e: Event, sel: string): Element | null {
  const t = e.target as Element | null;
  return t && t.closest ? t.closest(sel) : null;
}

function tip_over(e: Event): void {
  const el = closest(e, "[title], [data-tip]");
  if (!el) return;
  const t = el.getAttribute("title");
  if (t) {
    el.setAttribute("data-tip", t);
    el.removeAttribute("title");
  }
  const tip = el.getAttribute("data-tip");
  if (!tip) return;
  if (_tip_timer) clearTimeout(_tip_timer);
  _tip_timer = setTimeout(() => tip_place(el, tip), 280);
}

function tip_place(el: Element, tip: string): void {
  if (!el.isConnected || !_tip_show) return;
  const b = el.getBoundingClientRect();
  // near the bottom of the window (the dock): show it above
  const above = b.bottom > window.innerHeight - 140;
  _tip_show(tip, b.left + b.width / 2, above ? b.top - 8 : b.bottom + 8, above);
}

function tip_out(e: Event): void {
  const el = closest(e, "[data-tip]");
  const nxt = (e as MouseEvent).relatedTarget as Node | null;
  if (el && nxt && el.contains(nxt)) return;
  tip_hide();
}

function tip_hide(): void {
  if (_tip_timer) {
    clearTimeout(_tip_timer);
    _tip_timer = null;
  }
  if (_tip_show) _tip_show("", 0.0, 0.0, false);
}

export function Tooltip() {
  const [text, setText] = useState("");
  const [x, setX] = useState(0.0);
  const [y, setY] = useState(0.0);
  const [above, setAbove] = useState(false);

  _tip_show = (t, nx, ny, up) => {
    setText(t);
    setX(nx);
    setY(ny);
    setAbove(up);
  };

  useEffect(() => {
    document.addEventListener("mouseover", tip_over, true);
    document.addEventListener("mouseout", tip_out, true);
    document.addEventListener("mousedown", tip_hide, true);
    return () => {
      document.removeEventListener("mouseover", tip_over, true);
      document.removeEventListener("mouseout", tip_out, true);
      document.removeEventListener("mousedown", tip_hide, true);
    };
  }, []);

  if (!text) return <div className="tip-layer"></div>;
  // keep the tip on screen: clamp its centre away from the window edges
  const cx = Math.min(Math.max(x, 150.0), window.innerWidth - 150.0);
  return (
    <div className="tip-layer">
      <div className={"tip" + (above ? " above" : "")} style={{ left: String(cx) + "px", top: String(y) + "px" }}>
        {text}
      </div>
    </div>
  );
}
