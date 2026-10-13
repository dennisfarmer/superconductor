"""Quick tooltips for the whole page. Any element with a `title` gets a styled tip
shortly after the mouse rests on it (the browser's own tooltip takes a second or
more and is easy to miss). The title moves into `data-tip` on first hover so the
two never show together; React puts a changed title back, and the next hover
moves it again."""

import from react { useEffect }

glob _tip_timer: any = None,
     # set by the mounted Tooltip each render: show(text, x, y, above) / hide()
     _tip_show: any = None;


def tip_over(e: any) -> None {
    global _tip_timer;
    el: any = e.target.closest("[title], [data-tip]") if e.target and e.target.closest else None;
    if not el {
        return;
    }
    t = el.getAttribute("title");
    if t {
        el.setAttribute("data-tip", t);
        el.removeAttribute("title");
    }
    tip = el.getAttribute("data-tip");
    if not tip {
        return;
    }
    if _tip_timer {
        clearTimeout(_tip_timer);
    }
    _tip_timer = setTimeout(lambda { tip_place(el, tip); }, 280);
}


def tip_place(el: any, tip: str) -> None {
    if not el.isConnected or not _tip_show {
        return;
    }
    b = el.getBoundingClientRect();
    # near the bottom of the window (the dock): show it above
    above = b.bottom > window.innerHeight - 140;
    _tip_show(tip, b.left + b.width / 2, b.top - 8 if above else b.bottom + 8, above);
}


def tip_out(e: any) -> None {
    global _tip_timer;
    el: any = e.target.closest("[data-tip]") if e.target and e.target.closest else None;
    nxt: any = e.relatedTarget;
    if el and nxt and el.contains(nxt) {
        return;
    }
    tip_hide(e);
}


def tip_hide(e: any) -> None {
    global _tip_timer;
    if _tip_timer {
        clearTimeout(_tip_timer);
        _tip_timer = None;
    }
    if _tip_show {
        _tip_show("", 0.0, 0.0, False);
    }
}


def:pub Tooltip -> JsxElement {
    global _tip_show;
    has text: str = "",
        x: float = 0.0,
        y: float = 0.0,
        above: bool = False;

    def show(t: str, nx: float, ny: float, up: bool) -> None {
        text = t;
        x = nx;
        y = ny;
        above = up;
    }
    _tip_show = show;

    useEffect(lambda {
        document.addEventListener("mouseover", tip_over, True);
        document.addEventListener("mouseout", tip_out, True);
        document.addEventListener("mousedown", tip_hide, True);
        return lambda {
            document.removeEventListener("mouseover", tip_over, True);
            document.removeEventListener("mouseout", tip_out, True);
            document.removeEventListener("mousedown", tip_hide, True);
        };
    }, []);

    if not text {
        return <div className="tip-layer"></div>;
    }
    # keep the tip on screen: clamp its centre away from the window edges
    cx = min(max(x, 150.0), window.innerWidth - 150.0);
    return <div className="tip-layer">
        <div
            className={"tip" + (" above" if above else "")}
            style={{"left": str(cx) + "px", "top": str(y) + "px"}}
        >{text}</div>
    </div>;
}
