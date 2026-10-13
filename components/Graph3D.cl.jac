"""The graph view: every progression - the current one with both beam-search
trees, and every saved one - in one 3D space.

    lead-ins ← could precede ← [step 1 … step n] → options → lookahead

The current progression sits on the front plane (z = 0) with the backward tree
growing leftwards out of its first chord and the forward tree rightwards out of
its last; each saved progression is a row on a plane further back. Chords
are 3D boxes outlined in their function's colour, their labels painted onto the
front face with the page's own font stacks and colours, joined by 3D curves.

Words: *pin* a chord on the graph (its column ignores hover; paths not through
the pins dim), *keep* it into the progression (its other options stay fanned
beside it: hover one to hear it in place, click to swap it in), *save* the whole
progression (it moves behind). Type chords into each column's dashed slot (ⓧ
removes them; a mini walker grows on from each); click a kept chord to set the
loop. Click any chord of a saved progression to play it; click again to go back.
Text faces the camera from either side: chord labels are on both faces of every
block, and floating text turns round when the camera goes behind it.

Layout runs in "layout pixels" exactly as the 2D map did; 100 of them make one
world unit. (Labels are canvas textures, not CSS-3D HTML: drei's transformed
HTML drifts off the boxes under perspective. The text box of a typing slot and
the ⓧ buttons are screen-space HTML pinned to a 3D point, which stays accurate.)
"""

sv import from ..services.atlas { SearchView, ChordView, StepView, SavedView }
import from .paths { node_map, reachable, lock_at, is_back_id }
import from react { useEffect, useRef }
import from "@react-three/fiber" { Canvas, useThree, useFrame }
import from "@react-three/drei" { OrbitControls, Html, Edges, CubicBezierLine, Line }
import from "three" { CanvasTexture, SRGBColorSpace, MOUSE, TOUCH }

glob FUNC_COLOR: dict[str, str] = {
    "T": "#5aa9ff", "TX": "#56d4c9", "PD": "#7bd88f", "D": "#ff8a65", "CAD64": "#ffb74d",
    "APP": "#c792ea", "MIX": "#ffd166", "AUG6": "#f78fb3", "NEAP": "#f4a261"
};

"""Coarse functional area, for placing typed chords beside their peers."""
glob CATEGORY: dict[str, str] = {
    "T": "tonic", "TX": "tonic", "PD": "pd", "NEAP": "pd", "D": "dom", "CAD64": "dom",
    "APP": "applied", "MIX": "colour", "AUG6": "colour"
};

glob COL_W: int = 196,
     NODE_W: int = 132,
     NODE_H: int = 34,
     ROW_H: int = 44,
     TOP: int = 42,
     LEFT: int = 18;

# 3D: world units per layout pixel, box thickness, and the gap between progression planes
glob S: float = 0.01,
     BOX_D: float = 0.12,
     Z_GAP: float = 2.6;

# box fills (the same as the 2D map's CSS)
glob FILL: str = "#18223c",
     FILL_HOT: str = "#22305a",
     FILL_SOUNDING: str = "#2a3d78";


# the page's font stacks and text colours (assets/atlas.css)
glob SERIF: str = "\"Iowan Old Style\", \"Palatino Linotype\", Palatino, Georgia, serif",
     SANS: str = "ui-sans-serif, system-ui, -apple-system, \"Segoe UI\", sans-serif",
     MONO: str = "ui-monospace, \"SF Mono\", Menlo, monospace",
     INK: str = "#e8ecf6",
     MUTED: str = "#8791ab",
     ACCENT: str = "#8fb3ff",
     WARN: str = "#ffcf66",
     TEX_SCALE: int = 4,
     _TEX: dict[str, any] = {};


"""A chord label as a texture, laid out like the 2D map's box text, cached by content."""
def chord_texture(rn: str, symbol: str, score: str, cadence: str, ink: str, w: int, h: int) -> any {
    k = rn + "|" + symbol + "|" + score + "|" + cadence + "|" + ink + "|" + str(w) + "x" + str(h);
    if k in _TEX {
        return _TEX[k];
    }
    c: any = document.createElement("canvas");
    c.width = w * TEX_SCALE;
    c.height = h * TEX_SCALE;
    g: any = c.getContext("2d");
    g.scale(TEX_SCALE, TEX_SCALE);
    small = h < 30;
    g.fillStyle = ink;
    g.font = ("13px " if small else "17px ") + SERIF;
    g.textAlign = "left";
    g.fillText(rn, 10, 13 if small else 23);
    g.textAlign = "right";
    if symbol {
        g.fillStyle = MUTED;
        g.font = "10.5px " + SANS;
        g.fillText(short(symbol), w - 8, 13);
    }
    if score {
        g.fillStyle = "#56607a";
        g.font = "9.5px " + MONO;
        g.fillText(score, w - 8, 13 if small else 29);
    }
    if cadence {
        g.fillStyle = WARN;
        g.fillRect(w - 34, 1, 33, 12);
        g.fillStyle = "#1b1b1b";
        g.font = "bold 9.5px " + SANS;
        g.textAlign = "center";
        g.fillText(cadence, w - 17.5, 10.5);
    }
    t: any = new(CanvasTexture, c);
    t.colorSpace = SRGBColorSpace;
    t.anisotropy = 8;
    _TEX[k] = t;
    return t;
}


"""A line of text as a texture; returns [texture, width in layout pixels]."""
def text_texture(text: str, font: str, color: str) -> list[any] {
    k = "t|" + text + "|" + font + "|" + color;
    if k in _TEX {
        return _TEX[k];
    }
    probe: any = document.createElement("canvas").getContext("2d");
    probe.font = font;
    w = int(probe.measureText(text).width) + 4;
    c: any = document.createElement("canvas");
    c.width = w * TEX_SCALE;
    c.height = 18 * TEX_SCALE;
    g: any = c.getContext("2d");
    g.scale(TEX_SCALE, TEX_SCALE);
    g.font = font;
    g.fillStyle = color;
    g.textBaseline = "middle";
    g.fillText(text, 2, 9);
    t: any = new(CanvasTexture, c);
    t.colorSpace = SRGBColorSpace;
    t.anisotropy = 8;
    _TEX[k] = [t, w];
    return _TEX[k];
}


"""Keeps the camera on the framing point: when it moves (a commit, a new chord, a
progression added behind), camera and orbit target shift by the same amount, so the
viewing angle the composer chose is kept."""
glob _framed_controls: any = None;


def CameraRig(x: float, y: float, z: float) -> JsxElement {
    three: any = useThree();
    useEffect(lambda {
        global _framed_controls;
        ctl: any = three.controls;
        if ctl and ctl != _framed_controls {
            # first frame for these controls: in front of the point, a little above
            _framed_controls = ctl;
            three.camera.position.set(x, y + 1.6, z + 9.5);
            ctl.target.set(x, y, z);
            ctl.update();
        } elif ctl {
            dx = x - ctl.target.x;
            dy = y - ctl.target.y;
            dz = z - ctl.target.z;
            three.camera.position.set(three.camera.position.x + dx, three.camera.position.y + dy, three.camera.position.z + dz);
            ctl.target.set(x, y, z);
            ctl.update();
        }
    }, [x, y, z, three.controls]);
    return <group/>;
}


"""Floating text that turns to face the camera: it flips round (about the vertical
axis) when the camera goes behind its plane, so it never reads mirrored."""
def Facing(position: list[float], children: any) -> JsxElement {
    ref: any = useRef(None);
    useFrame(lambda state: any, delta: any {
        g: any = ref.current;
        if g {
            g.rotation.y = 0.0 if state.camera.position.z >= position[2] else Math.PI;
        }
    });
    return <group ref={ref} position={position}>{children}</group>;
}


def wx(x: float) -> float {
    return x * S;
}


def wy(y: float) -> float {
    return 0.0 - y * S;
}


def color_of(func: str) -> str {
    if func in FUNC_COLOR {
        return FUNC_COLOR[func];
    }
    return "#8791ab";
}


def by_total(n: ChordView) -> float {
    return n.total;
}


def category(func: str) -> str {
    return CATEGORY[func] if func in CATEGORY else "other";
}


"""Insert each typed chord right after the last chord of its functional area."""
def merge_by_function(ordered: list[ChordView], extra: list[ChordView]) -> list[ChordView] {
    out: list[ChordView] = [n for n in ordered];
    for r in extra {
        at = len(out);
        for (i, n) in enumerate(out) {
            if category(n.func) == category(r.func) {
                at = i + 1;
            }
        }
        out.insert(at, r);
    }
    return out;
}


def short(symbol: str) -> str {
    return symbol if len(symbol) < 14 else symbol[:12] + "…";
}


"""Nodes of one tree grouped by layer (index 0 = the start)."""
def layers_of(start: ChordView, nodes: list[ChordView], depth: int) -> list[list[ChordView]] {
    out: list[list[ChordView]] = [];
    for L in range(depth + 1) {
        out.append([]);
    }
    for n in [start] + nodes {
        if n.layer <= depth {
            out[n.layer].append(n);
        }
    }
    return out;
}


def:pub Graph3D(
    search: SearchView,
    anim: int,
    hoverId: int,
    selectedId: int,
    soundingId: int,
    soundingStep: int,
    loopFrom: int,
    onStep: Callable[[int], None],
    showAll: bool,
    steps: list[StepView],
    altHover: list[int],
    onAltHover: Callable[[int, int], None],
    onAltLeave: Callable[[], None],
    onAltPick: Callable[[int, str], None],
    onHover: Callable[[int], None],
    onLeave: Callable[[], None],
    onSelect: Callable[[int], None],
    locks: list[int],
    typed: list[ChordView],
    onAddTyped: Callable[[bool, int, str], None],
    onRemoveTyped: Callable[[list[int]], None],
    saved: list[SavedView],
    viewing: int,
    savedSounding: int,
    onPickSaved: Callable[[int], None]
) -> JsxElement {
    has editing: str = "";   # the typing slot whose text box is open ("f2" = forward column 2)

    depth = search.depth;
    animating = anim >= 0 and anim <= depth;
    base = len(steps) - 1;
    fwd_layers = layers_of(search.start, search.nodes, depth);
    back_layers = layers_of(search.back_start, search.back_nodes, depth);
    tree_nodes: list[ChordView] = search.nodes + search.back_nodes;

    # --- columns: backward layer L at depth-L, step i at depth+i, forward layer L at depth+base+L ---
    def col_x(c: int) -> float {
        return LEFT + c * COL_W;
    }

    def x_fwd(layer: int) -> float {
        return col_x(depth + base + layer);
    }

    def x_back(layer: int) -> float {
        return col_x(depth - layer);
    }

    def px(i: int) -> float {
        return col_x(depth + i);
    }

    def x_node(n: ChordView) -> float {
        return x_back(n.layer) if n.id >= search.back_start.id else x_fwd(n.layer);
    }

    by_id = node_map(search, typed);

    def box_key(n: ChordView) -> str {
        return ("b" if is_back_id(n.id) else "f") + str(n.layer) + ":" + n.cid;
    }

    """The variant a box stands for: the first given if still reachable, else the
    cheapest reachable one, else the cheapest."""
    def pick(vs: list[ChordView], prefer_first: bool) -> ChordView {
        if prefer_first and reachable(by_id, locks, vs[0].id) {
            return vs[0];
        }
        live = [v for v in vs if reachable(by_id, locks, v.id)];
        pool = live if len(live) > 0 else vs;
        best = pool[0];
        for v in pool {
            if v.total < best.total {
                best = v;
            }
        }
        return best;
    }

    # the searched boxes, by (tree, column, chord): grown chords that match one attach to it
    kept_at: dict[str, ChordView] = {};
    for n in tree_nodes {
        k = box_key(n);
        if k not in kept_at or n.total < kept_at[k].total {
            kept_at[k] = n;
        }
    }
    attached: dict[str, list[ChordView]] = {};
    attached_keys: list[str] = [];
    # typed chords (and grown chords matching nothing searched): one box per chord,
    # linked from every parent it may follow
    group_of: dict[str, list[ChordView]] = {};
    group_keys: list[str] = [];
    for t in typed {
        k = box_key(t);
        if t.status == "grown" and k in kept_at {
            if k not in attached {
                attached[k] = [];
                attached_keys.append(k);
            }
            attached[k].append(t);
            continue;
        }
        if k not in group_of {
            group_of[k] = [];
            group_keys.append(k);
        }
        group_of[k].append(t);
    }
    rep_of: dict[str, ChordView] = {};
    for k in group_keys {
        rep_of[k] = pick(group_of[k], False);
    }

    def kept_ids(n: ChordView) -> list[int] {
        k = box_key(n);
        extra = attached[k] if k in attached and kept_at[k].id == n.id else [];
        return [n.id] + [v.id for v in extra];
    }

    def kept_rep(n: ChordView) -> ChordView {
        k = box_key(n);
        if k in attached and kept_at[k].id == n.id {
            return pick([n] + attached[k], True);
        }
        return n;
    }

    def reps_at(back: bool, L: int) -> list[ChordView] {
        return [rep_of[k] for k in group_keys if (k[0] == "b") == back and rep_of[k].layer == L];
    }

    # --- rows: each layer ordered under its parent, both trees centred on one line;
    # every searched column keeps a spare row at the bottom for the typing slot ---
    max_rows = 1;
    for L in range(depth + 1) {
        spare = 1 if L >= 1 else 0;
        max_rows = max(max_rows, len(fwd_layers[L]) + len(reps_at(False, L)) + spare);
        max_rows = max(max_rows, len(back_layers[L]) + len(reps_at(True, L)) + spare);
    }
    ypos: dict[int, float] = {};
    order_key: dict[int, float] = {};
    slot_y: dict[str, float] = {};

    def placement_key(n: ChordView) -> float {
        return order_key[n.id];
    }

    def place(tree: list[list[ChordView]], back: bool) {
        for L in range(depth + 1) {
            col = tree[L];
            for n in col {
                parent_y = ypos[n.parent] if n.parent in ypos else 0.0;
                order_key[n.id] = parent_y * 1000.0 + n.total;
            }
            ordered = sorted(col, key=by_total) if L <= 1 else sorted(col, key=placement_key);
            merged = merge_by_function(ordered, reps_at(back, L));
            rows = len(merged) + (1 if L >= 1 else 0);
            offset = (max_rows - rows) * ROW_H / 2;
            for (i, n) in enumerate(merged) {
                ypos[n.id] = TOP + offset + i * ROW_H;
            }
            slot_y[("b" if back else "f") + str(L)] = TOP + offset + len(merged) * ROW_H;
            # every variant sits in its box
            for k in group_keys {
                r = rep_of[k];
                if r.layer == L and (k[0] == "b") == back {
                    for v in group_of[k] {
                        ypos[v.id] = ypos[r.id];
                    }
                }
            }
            for k in attached_keys {
                box = kept_at[k];
                if box.layer == L and (k[0] == "b") == back and box.id in ypos {
                    for v in attached[k] {
                        ypos[v.id] = ypos[box.id];
                    }
                }
            }
        }
    }

    place(fwd_layers, False);
    place(back_layers, True);
    y0 = ypos[search.start.id];

    # --- the chains to the focused node and to every lock are drawn hot ---
    focus = hoverId if hoverId >= 0 else selectedId;
    hot: list[int] = [];
    for end in [focus] + locks {
        cur = end;
        while cur in by_id and by_id[cur].layer > 0 {
            hot.append(cur);
            cur = by_id[cur].parent;
        }
    }

    def visible(n: ChordView) -> bool {
        return (not animating) or n.layer <= anim;
    }


    def z_of(n: ChordView) -> float {
        return 0.0;
    }

    """3D curve from the right side of the box at (xl, yl) to the left side of the box
    at (xr, yr), on plane z. kind: hot | dest | typed | grown | past | dim | plain."""
    def curve(key: str, xl: float, yl: float, xr: float, yr: float, z: float, color: str, kind: str) -> JsxElement {
        x1 = wx(xl + NODE_W);
        y1 = wy(yl + NODE_H / 2);
        x2 = wx(xr);
        y2 = wy(yr + NODE_H / 2);
        mx = (x1 + x2) / 2;
        width = 2.6 if "hot" in kind else 1.4;
        opacity = 1.0 if "hot" in kind else (0.08 if "dim" in kind else (0.18 if "past" in kind else 0.55));
        dashed = "dest" in kind or "typed" in kind or "grown" in kind;
        return <CubicBezierLine
            key={key}
            start={[x1, y1, z]}
            end={[x2, y2, z]}
            midA={[mx, y1, z]}
            midB={[mx, y2, z]}
            color={color}
            lineWidth={width}
            dashed={dashed}
            dashSize={0.03 if "grown" in kind else 0.08}
            gapSize={0.06}
            transparent={True}
            opacity={opacity}
        />;
    }

    """A chord as a 3D box at layout (x, y) on plane z, its label texture on the front face."""
    def box(
        key: str, x: float, y: float, z: float, w: float, h: float, color: str, fill: str,
        opacity: float, label: any, over: any, out: any, click: any, dashed: bool = False
    ) -> JsxElement {
        hw = w * S / 2;
        hh = h * S / 2;
        fz = BOX_D / 2 + 0.004;
        return <group key={key} position={[wx(x + w / 2), wy(y + h / 2), z]} onPointerOver={over} onPointerOut={out} onClick={click}>
            <mesh>
                <boxGeometry args={[w * S, h * S, BOX_D]}/>
                <meshStandardMaterial color={fill} transparent={True} opacity={opacity}/>
                {if not dashed { <Edges color={color}/> }}
            </mesh>
            {if dashed {
                <Line
                    points={[[0.0 - hw, 0.0 - hh, fz], [hw, 0.0 - hh, fz], [hw, hh, fz], [0.0 - hw, hh, fz], [0.0 - hw, 0.0 - hh, fz]]}
                    color={color}
                    lineWidth={1.8}
                    dashed={True}
                    dashSize={0.05}
                    gapSize={0.04}
                    transparent={True}
                    opacity={opacity}
                />
            }}
            <mesh position={[0, 0, BOX_D / 2 + 0.002]}>
                <planeGeometry args={[w * S, h * S]}/>
                <meshBasicMaterial map={label} transparent={True} opacity={opacity} toneMapped={False} depthWrite={False}/>
            </mesh>
            <mesh position={[0, 0, 0.0 - BOX_D / 2 - 0.002]} rotation={[0, Math.PI, 0]}>
                <planeGeometry args={[w * S, h * S]}/>
                <meshBasicMaterial map={label} transparent={True} opacity={opacity} toneMapped={False} depthWrite={False}/>
            </mesh>
        </group>;
    }

    def chord_label(rn: str, symbol: str, score: str, cadence: str) -> any {
        return chord_texture(rn, symbol, score, cadence, INK, NODE_W, NODE_H);
    }

    """A text label centred at layout (x, y) on plane z. cls: col | name | name-on."""
    def tag(key: str, x: float, y: float, z: float, text: str, cls: str) -> JsxElement {
        font = ("600 12px " + SANS) if "on" in cls else (("12px " if "name" in cls else "11px ") + SANS);
        color = ACCENT if "on" in cls else ("#c3cadb" if "name" in cls else MUTED);
        shown = text.upper() if cls == "col" else text;
        tw = text_texture(shown, font, color);
        return <Facing key={key} position={[wx(x), wy(y), z]}>
            <mesh>
                <planeGeometry args={[tw[1] * S, 18 * S]}/>
                <meshBasicMaterial map={tw[0]} transparent={True} toneMapped={False} depthWrite={False}/>
            </mesh>
        </Facing>;
    }

    """Tree edge from a node's parent (one layer nearer the progression) to the node."""
    def tree_edge(key: str, n: ChordView, y: float, kind: str) -> JsxElement {
        py = ypos[n.parent];
        if n.id >= search.back_start.id {
            return curve(key, x_back(n.layer), y, x_back(n.layer - 1), py, 0.0, color_of(n.func), kind);
        }
        return curve(key, x_fwd(n.layer - 1), py, x_fwd(n.layer), y, 0.0, color_of(n.func), kind);
    }

    def alt_on(i: int, j: int) -> bool {
        return len(altHover) == 2 and altHover[0] == i and altHover[1] == j;
    }

    # --- the other options each kept chord was chosen from, fanned above and below it:
    # hover to hear one in place, click to swap it in ---
    others: list[list[any]] = [];
    for i in range(len(steps)) {
        others.append([o for o in steps[i].alts]);
    }

    def opt_y(j: int) -> float {
        return y0 + (int(j / 2) + 1) * ROW_H * (1 if j % 2 == 0 else -1);
    }

    height: float = (TOP + max_rows * ROW_H + 20) * 1.0;
    lo = TOP * 1.0;
    hi = height * 1.0;
    for col in others {
        if len(col) > 0 {
            lo = min(lo, opt_y(len(col) - 1 if len(col) % 2 == 0 else len(col) - 2) if len(col) > 1 else lo);
            hi = max(hi, opt_y(len(col) - 1 if len(col) % 2 == 1 else len(col) - 2) + NODE_H + 20);
        }
    }
    shift = TOP - lo;
    height = hi + shift;
    width = col_x(2 * depth + base + 1) + 20;

    def step_label(i: int) -> str {
        if base == 0 {
            return "start";
        }
        return "step " + str(i + 1) + (" · first" if i == 0 else (" · now" if i == base else ""));
    }

    """A chord box standing for variant `n` (the one hover/click act on); `ids` are
    every variant drawn in it. kind: kept | typed | grown."""
    def node_g(n: ChordView, ids: list[int], kind: str) -> JsxElement {
        x = x_node(n);
        y = ypos[n.id];
        live = reachable(by_id, locks, n.id);
        col_locked = lock_at(by_id, locks, is_back_id(n.id), n.layer) >= 0;
        hoverable = live and not col_locked;
        is_hot = any([i in hot for i in ids]);
        is_locked = any([i in locks for i in ids]);
        fill = FILL_SOUNDING if soundingId in ids else (FILL_HOT if is_hot or is_locked else FILL);
        label = chord_texture(n.rn, n.symbol, f"{n.total:.1f}", n.cadence, ACCENT if is_locked else INK, NODE_W, NODE_H);
        return <group key={"n" + str(n.id)}>
            {box(
                "b" + str(n.id), x, y, 0.0, NODE_W, NODE_H, color_of(n.func), fill, 1.0 if live else 0.25, label,
                lambda e: any { e.stopPropagation(); if hoverable { onHover(n.id); } },
                lambda e: any { if hoverable { onLeave(); } },
                lambda e: any { e.stopPropagation(); if live { onSelect(n.id); } },
                kind == "typed"
            )}
            {if is_locked {
                <mesh position={[wx(x + NODE_W / 2), wy(y + NODE_H / 2), 0.0]}>
                    <boxGeometry args={[NODE_W * S + 0.05, NODE_H * S + 0.05, BOX_D * 0.6]}/>
                    <meshBasicMaterial color="#8fb3ff" transparent={True} opacity={0.25}/>
                </mesh>
            }}
            {if kind == "typed" {
                <Html center={True} position={[wx(x + 1), wy(y + 1), BOX_D / 2 + 0.01]} zIndexRange={[200, 0]}>
                    <button className="n3-remove" title="Remove this chord" onClick={lambda e: MouseEvent { e.stopPropagation(); onRemoveTyped(ids); }}>×</button>
                </Html>
            }}
        </group>;
    }

    """A dashed slot under a column: click it to type a chord into that column."""
    def slot_g(back: bool, L: int) -> JsxElement {
        x = x_back(L) if back else x_fwd(L);
        y = slot_y[("b" if back else "f") + str(L)];
        k = ("b" if back else "f") + str(L);
        hw = NODE_W * S / 2;
        hh = NODE_H * S / 2;
        hint = text_texture("+ type a chord", "12px " + MONO, "#56607a");
        return <group key={"slot" + k} position={[wx(x + NODE_W / 2), wy(y + NODE_H / 2), 0.0]}>
            <group onClick={lambda e: any { e.stopPropagation(); editing = k; }}>
                <mesh>
                    <planeGeometry args={[NODE_W * S, NODE_H * S]}/>
                    <meshBasicMaterial transparent={True} opacity={0.0} depthWrite={False}/>
                </mesh>
                <Line
                    points={[[0.0 - hw, 0.0 - hh, 0.0], [hw, 0.0 - hh, 0.0], [hw, hh, 0.0], [0.0 - hw, hh, 0.0], [0.0 - hw, 0.0 - hh, 0.0]]}
                    color={ACCENT if editing == k else "#2f3e66"}
                    lineWidth={1.4}
                    dashed={True}
                    dashSize={0.05}
                    gapSize={0.04}
                />
                <Facing position={[0.0, 0.0, 0.001]}>
                    <mesh>
                        <planeGeometry args={[hint[1] * S, 18 * S]}/>
                        <meshBasicMaterial map={hint[0]} transparent={True} toneMapped={False} depthWrite={False}/>
                    </mesh>
                </Facing>
            </group>
            {if editing == k {
                <Html center={True} zIndexRange={[300, 0]}>
                    <input
                        className="slot-in n3-edit"
                        autoFocus={True}
                        placeholder="Dm7, V65, bVI…"
                        title="Type a chord symbol (Dm7, G7/B, Ebmaj7) or a Roman numeral (ii7, V65, V7/V, bVI); Enter adds it, Esc cancels"
                        onKeyDown={lambda e: KeyboardEvent {
                            if e.key == "Enter" and e.target.value {
                                onAddTyped(back, L, e.target.value);
                                editing = "";
                            } elif e.key == "Escape" {
                                editing = "";
                            }
                        }}
                        onBlur={lambda e: FocusEvent { editing = ""; }}
                    />
                </Html>
            }}
        </group>;
    }

    # --- saved progressions: one row each, on planes behind the current one. Only with
    # "show all progressions"; otherwise just the one being played, if any ---
    shown_saved: list[int] = [k for k in range(len(saved)) if showAll or k == viewing];

    def saved_row(k: int, slot: int, prog: SavedView) -> JsxElement {
        z = 0.0 - (slot + 1) * Z_GAP;
        on = k == viewing;
        return <group key={"saved" + str(k)}>
            {tag("sl" + str(k), px(0) - 110, y0 + NODE_H / 2, z, prog.name + " · " + prog.key_name + ("  ▶ playing" if on else "  (click to play)"), "name-on" if on else "name")}
            {for (i, st) in enumerate(prog.steps) {
                <group key={"ss" + str(k) + "-" + str(i)}>
                    {curve("se" + str(k) + "-" + str(i), px(i - 1), y0, px(i), y0, z, color_of(st.func), "hot" if on else "plain") if i > 0 else None}
                    {box(
                        "sb" + str(k) + "-" + str(i), px(i), y0, z, NODE_W, NODE_H, color_of(st.func),
                        FILL_SOUNDING if on and i == savedSounding else (FILL_HOT if on else FILL), 1.0 if on else 0.75,
                        chord_label(st.rn, st.symbol, "", st.cadence),
                        None, None,
                        lambda e: any { e.stopPropagation(); onPickSaved(k); }
                    )}
                </group>
            }}
        </group>;
    }

    # camera: like the 2D map's initial scroll, on the last chord and the options after it
    cx = wx((px(base) + x_fwd(2) + NODE_W) / 2);
    cy = wy(y0 + NODE_H / 2);
    back_z = 0.0 - len(shown_saved) * Z_GAP;

    # mouse: panning is the common move, so it gets the left button (and one finger);
    # rotating, to look at the saved progressions behind, is the right button (two fingers)
    return <div className="graph3d">
        <Canvas camera={{"position": [cx, cy + 1.6, 9.5], "fov": 45}} dpr={[1, 2]}>
            <ambientLight intensity={1.6}/>
            <directionalLight position={[4, 8, 10]} intensity={1.1}/>
            <OrbitControls
                makeDefault={True}
                enableDamping={True}
                dampingFactor={0.12}
                screenSpacePanning={True}
                mouseButtons={{"LEFT": MOUSE.PAN, "MIDDLE": MOUSE.DOLLY, "RIGHT": MOUSE.ROTATE}}
                touches={{"ONE": TOUCH.PAN, "TWO": TOUCH.DOLLY_ROTATE}}
            />
            <CameraRig x={cx} y={cy} z={back_z / 2}/>

            {for L in range(1, depth + 1) {
                <group key={"cols" + str(L)}>
                    {tag("cb" + str(L), x_back(L) + NODE_W / 2, lo - 18, 0.0, "could precede" if L == 1 else "lead-in " + str(L), "col")}
                    {tag("cf" + str(L), x_fwd(L) + NODE_W / 2, lo - 18, 0.0, "options" if L == 1 else "lookahead " + str(L), "col")}
                </group>
            }}

            {for i in range(len(steps)) {
                <group key={"opts" + str(i)}>
                    {for (j, o) in enumerate(others[i]) {
                        <group key={"o" + str(i) + "-" + str(j)}>
                            {curve(
                                "oe" + str(i) + "-" + str(j), px(i - 1), y0, px(i), opt_y(j), 0.0, color_of(o.func),
                                "hot" if alt_on(i, j) else "past"
                            ) if i > 0 else None}
                            {box(
                                "ob" + str(i) + "-" + str(j), px(i), opt_y(j), 0.0, NODE_W, NODE_H, color_of(o.func),
                                FILL_HOT if alt_on(i, j) else FILL, 1.0 if alt_on(i, j) else 0.45,
                                chord_label(o.rn, o.symbol, "", ""),
                                lambda e: any { e.stopPropagation(); onAltHover(i, j); },
                                lambda e: any { onAltLeave(); },
                                lambda e: any { e.stopPropagation(); onAltPick(i, o.cid); }
                            )}
                        </group>
                    }}
                </group>
            }}

            {for i in range(len(steps)) {
                <group key={"step" + str(i)}>
                    {tag("sl" + str(i), px(i) + NODE_W / 2, lo - 18, 0.0, step_label(i), "col")}
                    {curve("st" + str(i), px(i - 1), y0, px(i), y0, 0.0, color_of(steps[i].func), "hot") if i > 0 else None}
                    {box(
                        "sb" + str(i), px(i), y0, 0.0, NODE_W, NODE_H, color_of(steps[i].func),
                        FILL_SOUNDING if i == soundingStep else FILL, 1.0,
                        chord_label(steps[i].rn, steps[i].symbol, "", steps[i].cadence),
                        None, None,
                        lambda e: any { e.stopPropagation(); onStep(i); }
                    )}
                </group>
            }}

            {[tree_edge("e" + str(n.id), n, ypos[n.id], ("hot" if n.id in hot else "plain") + (" dest" if n.status == "dest" else "")) for n in tree_nodes if visible(n) and n.parent in ypos]}

            {[
                tree_edge("et" + str(t.id), t, ypos[t.id], ("typed" if t.status == "custom" else "grown") + (" hot" if t.id in hot else "") + ("" if reachable(by_id, locks, t.id) else " dim"))
                for t in typed if not animating and t.parent in ypos and t.id in ypos
            ]}

            {[node_g(kept_rep(n), kept_ids(n), "kept") for n in tree_nodes if visible(n)]}
            {[
                node_g(rep_of[k], [v.id for v in group_of[k]], "typed" if any([v.status == "custom" for v in group_of[k]]) else "grown")
                for k in group_keys if not animating
            ]}
            {[slot_g(False, L) for L in range(1, depth + 1) if not animating]}
            {[slot_g(True, L) for L in range(1, depth + 1) if not animating]}

            {if animating and anim < depth {
                <group>
                    {for n in fwd_layers[anim] {
                        if n.status != "dest" or n.layer == 0 {
                            <mesh key={"wf" + str(n.id)} position={[wx(x_fwd(anim) + NODE_W), wy(ypos[n.id] + NODE_H / 2), 0.08]}>
                                <sphereGeometry args={[0.05, 16, 16]}/>
                                <meshBasicMaterial color="#ffffff"/>
                            </mesh>
                        }
                    }}
                    {for n in back_layers[anim] {
                        if n.status != "dest" or n.layer == 0 {
                            <mesh key={"wb" + str(n.id)} position={[wx(x_back(anim)), wy(ypos[n.id] + NODE_H / 2), 0.08]}>
                                <sphereGeometry args={[0.05, 16, 16]}/>
                                <meshBasicMaterial color="#ffffff"/>
                            </mesh>
                        }
                    }}
                </group>
            }}

            {[saved_row(k, slot, saved[k]) for (slot, k) in enumerate(shown_saved)]}
        </Canvas>
        <div className="graph3d-hint">faint boxes beside a kept chord: hover to hear, click to swap · drag to pan · right-drag to rotate · scroll to zoom</div>
    </div>;
}
