"""The side panel beside the graph: one tab at a time (Inspect, Sound, Rules,
Typing). It collapses to a rail of tab buttons, and tabs can be hidden from
the bar. The page owns which tab is open; the body is passed in as children."""

"""[id, label, icon]"""
glob TABS: list[list[str]] = [
    ["inspect", "Inspect", "◎", "The chord you are hovering or have pinned: its path, voicing, costs and rules"],
    ["sound", "Sound", "♫", "The sound Magenta plays: a blend of text prompts"],
    ["rules", "Rules", "§", "The harmony and voice-leading rules, with the function flow chart"],
    ["typing", "Typing", "⌨", "How to type chords and Roman numerals"]
];


def:pub SidePanel(
    tab: str,
    open: bool,
    hidden: list[str],
    onTab: Callable[[str], None],
    onOpen: Callable[[bool], None],
    onHidden: Callable[[list[str]], None],
    onResize: Callable[[int], None],
    children: any
) -> JsxElement {
    has editing: bool = False;

    shown = [t for t in TABS if t[0] not in hidden or t[0] == tab];

    """Drag the panel's left border to resize it (kept between 260 px and 75% of the window)."""
    def start_drag(e: any) -> None {
        e.preventDefault();
        document.body.classList.add("resizing");
        def move(ev: any) -> None {
            onResize(int(max(260, min(window.innerWidth * 0.75, window.innerWidth - ev.clientX))));
        }
        def up(ev: any) -> None {
            window.removeEventListener("mousemove", move);
            window.removeEventListener("mouseup", up);
            document.body.classList.remove("resizing");
        }
        window.addEventListener("mousemove", move);
        window.addEventListener("mouseup", up);
    }

    def toggle_hidden(tid: str) -> None {
        if tid in hidden {
            onHidden([h for h in hidden if h != tid]);
        } else {
            onHidden(hidden + [tid]);
        }
    }

    if not open {
        return <aside className="sidepanel rail">
            <button className="tab-btn" title="Open the panel" onClick={lambda e: MouseEvent { onOpen(True); }}>«</button>
            {for t in shown {
                <button
                    key={t[0]}
                    className={"tab-btn" + (" on" if t[0] == tab else "")}
                    title={t[1] + ": " + t[3]}
                    onClick={lambda e: MouseEvent { onTab(t[0]); onOpen(True); }}
                >{t[2]}</button>
            }}
        </aside>;
    }

    return <aside className="sidepanel">
        <div className="side-resize" title="Drag to resize the panel" onMouseDown={start_drag}></div>
        <nav className="tabs">
            {for t in shown {
                <button
                    key={t[0]}
                    className={"tab" + (" on" if t[0] == tab else "")}
                    title={t[3]}
                    onClick={lambda e: MouseEvent { onTab(t[0]); }}
                >{t[1]}</button>
            }}
            <span className="tabs-sp"></span>
            <button
                className={"tab-btn" + (" on" if editing else "")}
                title="Choose which tabs are shown"
                onClick={lambda e: MouseEvent { editing = not editing; }}
            >⋮</button>
            <button className="tab-btn" title="Collapse the panel" onClick={lambda e: MouseEvent { onOpen(False); }}>»</button>
        </nav>
        {if editing {
            <div className="tab-picker">
                <span>Show tabs:</span>
                {for t in TABS {
                    <label key={t[0]}>
                        <input type="checkbox" checked={t[0] not in hidden} onChange={lambda e: ChangeEvent { toggle_hidden(t[0]); }}/>
                        {t[1]}
                    </label>
                }}
            </div>
        }}
        <div className="side-body">{children}</div>
    </aside>;
}
