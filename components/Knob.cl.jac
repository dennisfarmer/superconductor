"""A rotary knob (0..1). Drag up/down or scroll to turn; releases commit."""

glob SWEEP: float = 270.0;


def polar(cx: float, cy: float, r: float, deg: float) -> list[float] {
    rad = (deg - 90.0) * Math.PI / 180.0;
    return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}


def arc(cx: float, cy: float, r: float, start: float, end: float) -> str {
    a = polar(cx, cy, r, start);
    b = polar(cx, cy, r, end);
    large = 1 if end - start > 180.0 else 0;
    return f"M {a[0]} {a[1]} A {r} {r} 0 {large} 1 {b[0]} {b[1]}";
}


def clamp01(v: float) -> float {
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v);
}


def:pub Knob(
    value: float,
    size: int,
    color: str,
    onChange: Callable[[float], None],
    onCommit: Callable[[], None]
) -> JsxElement {
    has dragging: bool = False;

    def down(e: any) {
        e.currentTarget.setPointerCapture(e.pointerId);
        dragging = True;
    }

    def move(e: any) {
        if dragging {
            dy = e.movementY as float;
            onChange(clamp01(value - dy / 140.0));
        }
    }

    def up(e: any) {
        if dragging {
            dragging = False;
            onCommit();
        }
    }

    def wheel(e: any) {
        dy = e.deltaY as float;
        onChange(clamp01(value - dy / 900.0));
    }

    c = size / 2.0;
    r = size / 2.0 - 4.0;
    start = -SWEEP / 2.0;
    end = start + SWEEP * value;
    tip = polar(c, c, r - 6.0, end);
    return <svg
        className={"knob" + (" dragging" if dragging else "")}
        width={size}
        height={size}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onWheel={wheel}
        onMouseLeave={lambda e: MouseEvent { if not dragging { onCommit(); } }}
    >
        <circle cx={c} cy={c} r={r - 3.0} className="knob-body"/>
        <path d={arc(c, c, r, start, start + SWEEP)} className="knob-track"/>
        {if value > 0.004 {
            <path d={arc(c, c, r, start, end)} className="knob-value" stroke={color}/>
        }}
        <line x1={c} y1={c} x2={tip[0]} y2={tip[1]} className="knob-tick" stroke={color}/>
    </svg>;
}
