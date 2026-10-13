"""The Drums page (/#drums, under the atlas's top bar): MRT2's DRUMS
input, one value per 40 ms frame. -1: the model chooses, 0: no hit, 1: a hit now.
The model picks the drum sound; this lane only says when hits happen.

Modes: Free (every frame -1, the model drums as it likes), Silent (every frame 0),
or Grid: a 4/4 step grid on the engine's beat clock. A hit cell is 1 on its first
frame and 0 for the rest of the cell (a finer subdivision gives shorter cells, so
more room for hits). Free and rest cells hold their value for the whole cell.

The grid is saved in the session (services.atlas.set_drums) and sent to the engine,
the local one or the SuperConductor sink, as settings.drum_grid."""

import from react { useEffect }
sv import from ..services.atlas { SessionView, get_state, set_drums }

glob FRAME_RATE: float = 25.0,
     PER_BEAT: list[list[any]] = [[1, "quarters"], [2, "eighths"], [4, "sixteenths"], [3, "triplets"]],
     BARS: list[int] = [1, 2, 4],
     CFG_DRUMS_HINT: str = "cfg_drums: the classifier-free guidance scale of MRT2's DRUMS input, -1 to 7 in whole steps (1 = MRT2's default; sent as the Pattern's cfg_drums). It sets how strongly the generated audio follows this lane's -1 / 0 / 1 values relative to the style prompt, which has its own scale (cfg_musiccoca, default 3). Higher: hits land where the grid puts them and 0 frames stay clear, even against a prompt that implies otherwise; very high values can pull the sound away from the prompt. 0 or below: the lane has little effect and the prompt decides. MRT2 takes this scale as a conditioning token (the notes and style scales also get extra masked forward passes), so its exact effect was learned in training.",
     # the engine's status, read by the poll timer (outside React state)
     _status_url: str = "";


def default_grid -> dict[str, any] {
    return {"mode": "silent", "bars": 1, "per_beat": 2, "cells": [-1, -1, -1, -1, -1, -1, -1, -1], "after_hit": 0, "cfg": 1.0};
}


"""The session's grid with every field present (older sessions have none)."""
def grid_of(raw: str) -> dict[str, any] {
    g = default_grid();
    if raw {
        try {
            saved: any = JSON.parse(raw);
            for k in ["mode", "bars", "per_beat", "cells", "cfg"] {
                if saved[k] or saved[k] == 0 {
                    g[k] = saved[k];
                }
            }
        } except Exception {}
    }
    n = g["bars"] * 4 * g["per_beat"];
    cells: list[int] = [];
    for i in range(n) {
        cells.append(g["cells"][i] if i < len(g["cells"]) else -1);
    }
    g["cells"] = cells;
    return g;
}


def with_field(g: dict[str, any], key: str, value: any) -> dict[str, any] {
    out = {"mode": g["mode"], "bars": g["bars"], "per_beat": g["per_beat"], "cells": g["cells"], "after_hit": g["after_hit"], "cfg": g["cfg"]};
    out[key] = value;
    return out;
}


"""New bars / subdivision: cells that still start on the same spot keep their value
(a shorter grid is cut, a longer one repeats); new in-between cells are free."""
def regrid(g: dict[str, any], bars: int, per_beat: int) -> dict[str, any] {
    old: list[int] = g["cells"];
    old_pb: int = g["per_beat"];
    cells: list[int] = [];
    for i in range(bars * 4 * per_beat) {
        if (i * old_pb) % per_beat == 0 {
            cells.append(old[int((i * old_pb) / per_beat) % len(old)]);
        } else {
            cells.append(-1);
        }
    }
    out = with_field(g, "cells", cells);
    out["bars"] = bars;
    out["per_beat"] = per_beat;
    return out;
}


"""Quick fills; `beat` is the beat number in the bar (0..3), `sub` the cell within it."""
def fill(g: dict[str, any], kind: str) -> dict[str, any] {
    pb: int = g["per_beat"];
    cells: list[int] = [];
    for i in range(len(g["cells"])) {
        sub = i % pb;
        beat = int(i / pb) % 4;
        v = -1;
        if kind == "rest" {
            v = 0;
        } elif kind == "pulse" {
            v = 1 if sub == 0 else 0;
        } elif kind == "one-three" {
            v = 1 if sub == 0 and beat % 2 == 0 else 0;
        } elif kind == "backbeat" {
            v = 1 if sub == 0 and beat % 2 == 1 else 0;
        } elif kind == "offbeats" {
            v = 1 if sub == pb - 1 - int((pb - 1) / 2) and pb > 1 else 0;
        } elif kind == "random" {
            v = Math.floor(Math.random() * 3) - 1;
        }
        cells.append(v);
    }
    out = with_field(g, "cells", cells);
    out["mode"] = "grid";
    return out;
}


"""Cell labels: 1 & 2 & ..., 1 e & a ..., 1 trip let ..."""
def cell_label(i: int, pb: int) -> str {
    sub = i % pb;
    if sub == 0 {
        return str(int(i / pb) % 4 + 1);
    }
    if pb == 2 {
        return "&";
    }
    if pb == 3 {
        return "trip" if sub == 1 else "let";
    }
    return ["", "e", "&", "a"][sub];
}


"""The values the engine sends for one bar, frame by frame (engine.Conductor._drum_token)."""
def bar_frames(g: dict[str, any], bpm: float) -> list[int] {
    per_cell = FRAME_RATE * 60.0 / max(bpm, 1.0) / g["per_beat"];
    n_frames = int(Math.ceil(FRAME_RATE * 60.0 / max(bpm, 1.0) * 4.0));
    out: list[int] = [];
    for f in range(n_frames) {
        if g["mode"] != "grid" {
            out.append(0 if g["mode"] == "silent" else -1);
            continue;
        }
        k = Math.floor(f / per_cell);
        v: int = g["cells"][k % len(g["cells"])];
        if v == 1 and not (f == 0 or Math.floor((f - 1) / per_cell) != k) {
            v = g["after_hit"];
        }
        out.append(v);
    }
    return out;
}


def value_class(v: int) -> str {
    return "hit" if v == 1 else ("rest" if v == 0 else "free");
}


def:pub DrumsPage -> JsxElement {
    has grid: dict[str, any] = default_grid(),
        loaded: bool = False,
        bpm: float = 84.0,
        now_cell: int = -1;

    async def load -> None {
        global _status_url;
        s: SessionView = await get_state();
        grid = grid_of(s.drum_grid);
        bpm = s.bpm;
        _status_url = s.engine_url + "/status";
        loaded = True;
    }

    async def poll -> None {
        if not _status_url {
            return;
        }
        try {
            resp: any = await fetch(_status_url);
            data: any = await resp.json();
            if data.settings and data.settings.bpm {
                bpm = float(data.settings.bpm);
            }
            now_cell = int(data.playing.drum_cell) if data.playing else -1;
        } except Exception {
            now_cell = -1;
        }
    }

    useEffect(lambda {
        load();
        timer = setInterval(lambda { poll(); }, 120);
        return lambda { clearInterval(timer); };
    }, []);

    async def save(g: dict[str, any]) -> None {
        grid = g;
        await set_drums(JSON.stringify(g));
    }

    def click_cell(i: int) -> None {
        # free -> hit -> rest -> free; editing the grid switches it on
        cells: list[int] = [c for c in grid["cells"]];
        v = cells[i];
        cells[i] = 1 if v == -1 else (0 if v == 1 else -1);
        g = with_field(grid, "cells", cells);
        g["mode"] = "grid";
        save(g);
    }

    if not loaded {
        return <div className="drums-page"><p className="empty">Loading the drums…</p></div>;
    }

    pb: int = grid["per_beat"];
    on_grid = grid["mode"] == "grid";
    frames = bar_frames(grid, bpm);
    frames_per_cell = FRAME_RATE * 60.0 / max(bpm, 1.0) / pb;
    bars: list[list[int]] = [];
    for b in range(grid["bars"]) {
        bars.append([b * 4 * pb + i for i in range(4 * pb)]);
    }

    return <div className="drums-page">
        <header className="drums-head">
            <div>
                <h1>Drums</h1>
                <p className="drums-sub">When hits happen. The model picks the drum sounds and plays them over the harmony from the main page.</p>
            </div>
        </header>

        <section className="drums-controls">
            <div className="drums-field">
                <span className="drums-label">Mode</span>
                <div className="seg">
                    <button className={"on" if grid["mode"] == "free" else ""} title="Every frame -1: the model decides whether and when to drum" onClick={lambda e: MouseEvent { save(with_field(grid, "mode", "free")); }}>Free</button>
                    <button className={"on" if grid["mode"] == "silent" else ""} title="Every frame 0: no drum hits" onClick={lambda e: MouseEvent { save(with_field(grid, "mode", "silent")); }}>Silent</button>
                    <button className={"on" if on_grid else ""} title="The grid below, on the beat" onClick={lambda e: MouseEvent { save(with_field(grid, "mode", "grid")); }}>Grid</button>
                </div>
            </div>
            <div className="drums-field">
                <span className="drums-label">Subdivision</span>
                <div className="seg">
                    {for opt in PER_BEAT {
                        <button key={"pb" + str(opt[0])} className={"on" if pb == opt[0] else ""} onClick={lambda e: MouseEvent { save(regrid(grid, grid["bars"], opt[0])); }}>{opt[1]}</button>
                    }}
                </div>
            </div>
            <div className="drums-field">
                <span className="drums-label">Bars</span>
                <div className="seg">
                    {for n in BARS {
                        <button key={"bars" + str(n)} className={"on" if grid["bars"] == n else ""} onClick={lambda e: MouseEvent { save(regrid(grid, n, pb)); }}>{str(n)}</button>
                    }}
                </div>
            </div>
            <label className="drums-field" title={CFG_DRUMS_HINT}>
                <span className="drums-label drums-code">cfg_drums</span>
                <input
                    type="range"
                    min="-1"
                    max="7"
                    step="1"
                    value={str(grid["cfg"])}
                    onChange={lambda e: ChangeEvent { save(with_field(grid, "cfg", float(e.target.value))); }}
                />
                <span className="drums-num">{str(grid["cfg"])}</span>
            </label>
        </section>

        <section className={"drums-grid" + ("" if on_grid else " off")}>
            {for (b, row) in enumerate(bars) {
                <div key={"bar" + str(b)} className="drums-bar">
                    <span className="drums-bar-n">{"bar " + str(b + 1)}</span>
                    <div className="drums-cells" style={{"gridTemplateColumns": "repeat(" + str(4 * pb) + ", 1fr)"}}>
                        {for i in row {
                            <button
                                key={"c" + str(i)}
                                className={"drum-cell " + value_class(grid["cells"][i]) + (" beat" if i % pb == 0 else "") + (" now" if on_grid and i == now_cell else "")}
                                title={cell_label(i, pb) + ": " + str(grid["cells"][i]) + " (click: free → hit → rest)"}
                                onClick={lambda e: MouseEvent { click_cell(i); }}
                            >
                                <span className="drum-cell-label">{cell_label(i, pb)}</span>
                                <span className="drum-cell-mark">{"●" if grid["cells"][i] == 1 else ("×" if grid["cells"][i] == 0 else "○")}</span>
                            </button>
                        }}
                    </div>
                </div>
            }}
        </section>

        <section className="drums-fills">
            <span className="drums-label">Fill</span>
            <button className="btn ghost" title="Every cell -1" onClick={lambda e: MouseEvent { save(fill(grid, "free")); }}>all free</button>
            <button className="btn ghost" title="Every cell 0" onClick={lambda e: MouseEvent { save(fill(grid, "rest")); }}>all rest</button>
            <button className="btn ghost" title="A hit on every beat" onClick={lambda e: MouseEvent { save(fill(grid, "pulse")); }}>pulse</button>
            <button className="btn ghost" title="Hits on beats 1 and 3" onClick={lambda e: MouseEvent { save(fill(grid, "one-three")); }}>1 + 3</button>
            <button className="btn ghost" title="Hits on beats 2 and 4" onClick={lambda e: MouseEvent { save(fill(grid, "backbeat")); }}>backbeat</button>
            <button className="btn ghost" disabled={pb == 1} title="Hits between the beats" onClick={lambda e: MouseEvent { save(fill(grid, "offbeats")); }}>offbeats</button>
            <button className="btn ghost" title="Each cell -1, 0 or 1 at random" onClick={lambda e: MouseEvent { save(fill(grid, "random")); }}>random</button>
        </section>

        <section className="drums-frames">
            <div className="drums-frames-head">
                <span className="drums-label">What the model gets, frame by frame (bar 1)</span>
                <span className="drums-num">{str(len(frames)) + " frames of 40 ms at " + str(int(bpm)) + " BPM · " + str(Math.round(frames_per_cell * 10.0) / 10.0) + " frames per cell"}</span>
            </div>
            <div className="drums-strip">
                {for (f, v) in enumerate(frames) {
                    <span key={"f" + str(f)} className={"drum-frame " + value_class(v)} title={"frame " + str(f) + ": " + str(v)}>{str(v)}</span>
                }}
            </div>
        </section>

        <section className="drums-legend">
            <p><span className="drum-key free">○ −1</span> no instruction: the model decides whether to hit.</p>
            <p><span className="drum-key hit">● 1</span> a hit now (on the cell's first frame).</p>
            <p><span className="drum-key rest">× 0</span> no hit for the whole cell.</p>
            <p className="hint">The grid runs on the engine's beat clock: set the tempo in the top bar. With SuperConductor, changes are heard about 0.6 s later (its buffer).</p>
        </section>
    </div>;
}
