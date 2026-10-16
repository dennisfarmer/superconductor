// The graph view: every progression - the current one with both beam-search
// trees, and every saved one - in one 3D space.
//
//     lead-ins ← could precede ← [step 1 … step n] → options → lookahead
//
// The current progression sits on the front plane (z = 0) with the backward tree
// growing leftwards out of its first chord and the forward tree rightwards out of
// its last; each saved progression is a row on a plane further back. Chords
// are 3D boxes outlined in their function's colour, their labels painted onto the
// front face with the page's own font stacks and colours, joined by 3D curves.
//
// Words: *pin* a chord on the graph (its column ignores hover; paths not through
// the pins dim), *keep* it into the progression (its other options stay fanned
// beside it: hover one to hear it in place, click to swap it in), *save* the whole
// progression (it moves behind). Type chords into each column's dashed slot (ⓧ
// removes them; a mini walker grows on from each); click a kept chord to set the
// loop. Click any chord of a saved progression to play it; click again to go back.
// Text faces the camera from either side: chord labels are on both faces of every
// block, and floating text turns round when the camera goes behind it.
//
// Layout runs in "layout pixels" exactly as the 2D map did; 100 of them make one
// world unit. (Labels are canvas textures, not CSS-3D HTML: drei's transformed
// HTML drifts off the boxes under perspective. The text box of a typing slot and
// the ⓧ buttons are screen-space HTML pinned to a 3D point, which stays accurate.)

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import type { ThreeEvent } from "@react-three/fiber";
import { CubicBezierLine, Edges, Html, Line, OrbitControls } from "@react-three/drei";
import { CanvasTexture, MOUSE, SRGBColorSpace, TOUCH } from "three";
import type { Group, Texture } from "three";
import type { AltView, ChordView, SavedView, SearchView, StepView } from "../api";
import { is_back_id, lock_at, node_map, reachable } from "./paths";
import { KIND_COLOR, VoiceLeading, worst } from "./VoiceLeading";

const FUNC_COLOR: Record<string, string> = {
  T: "#5aa9ff",
  TX: "#56d4c9",
  PD: "#7bd88f",
  D: "#ff8a65",
  CAD64: "#ffb74d",
  APP: "#c792ea",
  MIX: "#ffd166",
  AUG6: "#f78fb3",
  NEAP: "#f4a261",
};

// Coarse functional area, for placing typed chords beside their peers.
const CATEGORY: Record<string, string> = {
  T: "tonic",
  TX: "tonic",
  PD: "pd",
  NEAP: "pd",
  D: "dom",
  CAD64: "dom",
  APP: "applied",
  MIX: "colour",
  AUG6: "colour",
};

const COL_W = 196;
const NODE_W = 132;
const NODE_H = 34;
const ROW_H = 44;
const TOP = 42;
const LEFT = 18;

// 3D: world units per layout pixel, box thickness, and the gap between progression planes
const S = 0.01;
const BOX_D = 0.12;
const Z_GAP = 2.6;

// box fills (the same as the 2D map's CSS)
const FILL = "#18223c";
const FILL_HOT = "#22305a";
const FILL_SOUNDING = "#2a3d78";

// the page's font stacks and text colours (atlas.css)
const SERIF = '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif';
const SANS = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
const MONO = 'ui-monospace, "SF Mono", Menlo, monospace';
const INK = "#e8ecf6";
const MUTED = "#8791ab";
const ACCENT = "#8fb3ff";
const WARN = "#ffcf66";
const TEX_SCALE = 4;
const _TEX: Record<string, Texture> = {};
const _TEXT_TEX: Record<string, [Texture, number]> = {};

type Handler = ((e: ThreeEvent<MouseEvent>) => void) | undefined;

/** A chord label as a texture, laid out like the 2D map's box text, cached by content.
 * `alt` is the chord's other Roman numeral (VI for V/ii): drawn under the numeral,
 * so both names are on the box. */
function chord_texture(rn: string, symbol: string, cadence: string, ink: string, w: number, h: number, alt = ""): Texture {
  const k = rn + "|" + alt + "|" + symbol + "|" + cadence + "|" + ink + "|" + String(w) + "x" + String(h);
  if (k in _TEX) return _TEX[k];
  const c = document.createElement("canvas");
  c.width = w * TEX_SCALE;
  c.height = h * TEX_SCALE;
  const g = c.getContext("2d")!;
  g.scale(TEX_SCALE, TEX_SCALE);
  const small = h < 30;
  g.fillStyle = ink;
  const two = alt && !small;
  g.font = (small ? "13px " : two ? "15px " : "17px ") + SERIF;
  g.textAlign = "left";
  g.fillText(rn, 10, small ? 13 : two ? 17 : 23);
  if (two) {
    g.fillStyle = MUTED;
    g.font = "11px " + SERIF;
    g.fillText(alt, 10, 30);
    g.fillStyle = ink;
  }
  g.textAlign = "right";
  if (symbol) {
    // bottom right, clear of the cadence badge (a short box has one line: beside it)
    g.fillStyle = MUTED;
    g.font = "10.5px " + SANS;
    g.fillText(short(symbol), w - (small && cadence ? 38 : 8), small ? 13 : 29);
  }
  if (cadence) {
    g.fillStyle = WARN;
    g.fillRect(w - 34, 1, 33, 12);
    g.fillStyle = "#1b1b1b";
    g.font = "bold 9.5px " + SANS;
    g.textAlign = "center";
    g.fillText(cadence, w - 17.5, 10.5);
  }
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.anisotropy = 8;
  _TEX[k] = t;
  return t;
}

/** A line of text as a texture; returns [texture, width in layout pixels]. */
function text_texture(text: string, font: string, color: string): [Texture, number] {
  const k = "t|" + text + "|" + font + "|" + color;
  if (k in _TEXT_TEX) return _TEXT_TEX[k];
  const probe = document.createElement("canvas").getContext("2d")!;
  probe.font = font;
  const w = Math.trunc(probe.measureText(text).width) + 4;
  const c = document.createElement("canvas");
  c.width = w * TEX_SCALE;
  c.height = 18 * TEX_SCALE;
  const g = c.getContext("2d")!;
  g.scale(TEX_SCALE, TEX_SCALE);
  g.font = font;
  g.fillStyle = color;
  g.textBaseline = "middle";
  g.fillText(text, 2, 9);
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.anisotropy = 8;
  _TEXT_TEX[k] = [t, w];
  return _TEXT_TEX[k];
}

// Keeps the camera on the framing point: when it moves (a commit, a new chord, a
// progression added behind), camera and orbit target shift by the same amount, so the
// viewing angle the composer chose is kept.
let _framed_controls: any = null;

function CameraRig({ x, y, z }: { x: number; y: number; z: number }) {
  const three = useThree();
  useEffect(() => {
    const ctl: any = three.controls;
    if (ctl && ctl !== _framed_controls) {
      // first frame for these controls: in front of the point, a little above
      _framed_controls = ctl;
      three.camera.position.set(x, y + 1.6, z + 9.5);
      ctl.target.set(x, y, z);
      ctl.update();
    } else if (ctl) {
      const dx = x - ctl.target.x;
      const dy = y - ctl.target.y;
      const dz = z - ctl.target.z;
      three.camera.position.set(three.camera.position.x + dx, three.camera.position.y + dy, three.camera.position.z + dz);
      ctl.target.set(x, y, z);
      ctl.update();
    }
  }, [x, y, z, three.controls]);
  return <group />;
}

/** Floating text that turns to face the camera: it flips round (about the vertical
 * axis) when the camera goes behind its plane, so it never reads mirrored. */
function Facing({ position, children }: { position: [number, number, number]; children?: ReactNode }) {
  const ref = useRef<Group>(null);
  useFrame((state) => {
    const g = ref.current;
    if (g) g.rotation.y = state.camera.position.z >= position[2] ? 0.0 : Math.PI;
  });
  return (
    <group ref={ref} position={position}>
      {children}
    </group>
  );
}

function wx(x: number): number {
  return x * S;
}

function wy(y: number): number {
  return 0.0 - y * S;
}

function color_of(func: string): string {
  if (func in FUNC_COLOR) return FUNC_COLOR[func];
  return "#8791ab";
}

function category(func: string): string {
  return func in CATEGORY ? CATEGORY[func] : "other";
}

function sorted_by<T>(xs: T[], key: (x: T) => number): T[] {
  return [...xs].sort((a, b) => key(a) - key(b));
}

/** Insert each typed chord right after the last chord of its functional area. */
function merge_by_function(ordered: ChordView[], extra: ChordView[]): ChordView[] {
  const out = [...ordered];
  for (const r of extra) {
    let at = out.length;
    out.forEach((n, i) => {
      if (category(n.func) === category(r.func)) at = i + 1;
    });
    out.splice(at, 0, r);
  }
  return out;
}

function short(symbol: string): string {
  return symbol.length < 14 ? symbol : symbol.slice(0, 12) + "…";
}

/** Nodes of one tree grouped by layer (index 0 = the start). */
function layers_of(start: ChordView, nodes: ChordView[], depth: number): ChordView[][] {
  const out: ChordView[][] = [];
  for (let L = 0; L < depth + 1; L++) out.push([]);
  for (const n of [start, ...nodes]) if (n.layer <= depth) out[n.layer].push(n);
  return out;
}

export interface Graph3DProps {
  search: SearchView;
  anim: number;
  hoverId: number;
  selectedId: number;
  soundingId: number;
  soundingStep: number;
  loopFrom: number;
  onStep: (i: number) => void;
  showAll: boolean;
  steps: StepView[];
  altHover: number[];
  onAltHover: (i: number, j: number) => void;
  onAltLeave: () => void;
  onAltPick: (i: number, cid: string) => void;
  onHover: (id: number) => void;
  onLeave: () => void;
  onSelect: (id: number) => void;
  locks: number[];
  typed: ChordView[];
  onAddTyped: (back: boolean, layer: number, text: string) => void;
  onRemoveTyped: (ids: number[]) => void;
  onStepTyped: (i: number, text: string) => void;
  onRemoveStep: (i: number) => void;
  saved: SavedView[];
  viewing: number;
  savedSounding: number;
  onPickSaved: (k: number) => void;
}

export function Graph3D(props: Graph3DProps) {
  const { search, anim, hoverId, selectedId, soundingId, soundingStep, onStep, showAll, steps, altHover } = props;
  const { onAltHover, onAltLeave, onAltPick, onHover, onLeave, onSelect, locks, typed, onAddTyped, onRemoveTyped } = props;
  const { onStepTyped, onRemoveStep, saved, viewing, savedSounding, onPickSaved } = props;
  const [editing, setEditing] = useState(""); // the typing slot whose text box is open ("f2" = forward column 2)
  const [leadAt, setLeadAt] = useState(-1); // the step whose incoming edge shows its voice-leading diagram
  const [edgeHover, setEdgeHover] = useState(-1);

  const depth = search.depth;
  const animating = anim >= 0 && anim <= depth;
  const base = steps.length - 1;
  const fwd_layers = layers_of(search.start, search.nodes, depth);
  const back_layers = layers_of(search.back_start, search.back_nodes, depth);
  const tree_nodes = [...search.nodes, ...search.back_nodes];

  // --- columns: backward layer L at depth-L, step i at depth+i, forward layer L at depth+base+L ---
  const col_x = (c: number) => LEFT + c * COL_W;
  const x_fwd = (layer: number) => col_x(depth + base + layer);
  const x_back = (layer: number) => col_x(depth - layer);
  const px = (i: number) => col_x(depth + i);
  const x_node = (n: ChordView) => (n.id >= search.back_start.id ? x_back(n.layer) : x_fwd(n.layer));

  const by_id = node_map(search, typed);

  const box_key = (n: ChordView) => (is_back_id(n.id) ? "b" : "f") + String(n.layer) + ":" + n.cid;

  /** The variant a box stands for: the first given if still reachable, else the
   * cheapest reachable one, else the cheapest. */
  function pick(vs: ChordView[], prefer_first: boolean): ChordView {
    if (prefer_first && reachable(by_id, locks, vs[0].id)) return vs[0];
    const live = vs.filter((v) => reachable(by_id, locks, v.id));
    const pool = live.length > 0 ? live : vs;
    let best = pool[0];
    for (const v of pool) if (v.total < best.total) best = v;
    return best;
  }

  // the searched boxes, by (tree, column, chord): grown chords that match one attach to it
  const kept_at: Record<string, ChordView> = {};
  for (const n of tree_nodes) {
    const k = box_key(n);
    if (!(k in kept_at) || n.total < kept_at[k].total) kept_at[k] = n;
  }
  const attached: Record<string, ChordView[]> = {};
  const attached_keys: string[] = [];
  // typed chords (and grown chords matching nothing searched): one box per chord,
  // linked from every parent it may follow
  const group_of: Record<string, ChordView[]> = {};
  const group_keys: string[] = [];
  for (const t of typed) {
    const k = box_key(t);
    if (t.status === "grown" && k in kept_at) {
      if (!(k in attached)) {
        attached[k] = [];
        attached_keys.push(k);
      }
      attached[k].push(t);
      continue;
    }
    if (!(k in group_of)) {
      group_of[k] = [];
      group_keys.push(k);
    }
    group_of[k].push(t);
  }
  const rep_of: Record<string, ChordView> = {};
  for (const k of group_keys) rep_of[k] = pick(group_of[k], false);

  function kept_ids(n: ChordView): number[] {
    const k = box_key(n);
    const extra = k in attached && kept_at[k].id === n.id ? attached[k] : [];
    return [n.id, ...extra.map((v) => v.id)];
  }

  function kept_rep(n: ChordView): ChordView {
    const k = box_key(n);
    if (k in attached && kept_at[k].id === n.id) return pick([n, ...attached[k]], true);
    return n;
  }

  const reps_at = (back: boolean, L: number) => group_keys.filter((k) => (k[0] === "b") === back && rep_of[k].layer === L).map((k) => rep_of[k]);

  // --- rows: each layer ordered under its parent, both trees centred on one line;
  // every searched column keeps a spare row at the bottom for the typing slot ---
  let max_rows = 1;
  for (let L = 0; L < depth + 1; L++) {
    const spare = L >= 1 ? 1 : 0;
    max_rows = Math.max(max_rows, fwd_layers[L].length + reps_at(false, L).length + spare);
    max_rows = Math.max(max_rows, back_layers[L].length + reps_at(true, L).length + spare);
  }
  const ypos: Record<number, number> = {};
  const order_key: Record<number, number> = {};
  const slot_y: Record<string, number> = {};

  function place(tree: ChordView[][], back: boolean) {
    for (let L = 0; L < depth + 1; L++) {
      const col = tree[L];
      for (const n of col) {
        const parent_y = n.parent in ypos ? ypos[n.parent] : 0.0;
        order_key[n.id] = parent_y * 1000.0 + n.total;
      }
      const ordered = L <= 1 ? sorted_by(col, (n) => n.total) : sorted_by(col, (n) => order_key[n.id]);
      const merged = merge_by_function(ordered, reps_at(back, L));
      const rows = merged.length + (L >= 1 ? 1 : 0);
      const offset = ((max_rows - rows) * ROW_H) / 2;
      merged.forEach((n, i) => {
        ypos[n.id] = TOP + offset + i * ROW_H;
      });
      slot_y[(back ? "b" : "f") + String(L)] = TOP + offset + merged.length * ROW_H;
      // every variant sits in its box
      for (const k of group_keys) {
        const r = rep_of[k];
        if (r.layer === L && (k[0] === "b") === back) for (const v of group_of[k]) ypos[v.id] = ypos[r.id];
      }
      for (const k of attached_keys) {
        const box = kept_at[k];
        if (box.layer === L && (k[0] === "b") === back && box.id in ypos) for (const v of attached[k]) ypos[v.id] = ypos[box.id];
      }
    }
  }

  place(fwd_layers, false);
  place(back_layers, true);
  const y0 = ypos[search.start.id];

  // --- the chains to the focused node and to every lock are drawn hot ---
  const focus = hoverId >= 0 ? hoverId : selectedId;
  const hot: number[] = [];
  for (const end of [focus, ...locks]) {
    let cur = end;
    while (cur in by_id && by_id[cur].layer > 0) {
      hot.push(cur);
      cur = by_id[cur].parent;
    }
  }

  const visible = (n: ChordView) => !animating || n.layer <= anim;

  /** 3D curve from the right side of the box at (xl, yl) to the left side of the box
   * at (xr, yr), on plane z. kind: hot | dest | typed | grown | past | dim | plain. */
  function curve(key: string, xl: number, yl: number, xr: number, yr: number, z: number, color: string, kind: string) {
    const x1 = wx(xl + NODE_W);
    const y1 = wy(yl + NODE_H / 2);
    const x2 = wx(xr);
    const y2 = wy(yr + NODE_H / 2);
    const mx = (x1 + x2) / 2;
    const width = kind.includes("hot") ? 2.6 : 1.4;
    const opacity = kind.includes("hot") ? 1.0 : kind.includes("dim") ? 0.08 : kind.includes("past") ? 0.18 : 0.55;
    const dashed = kind.includes("dest") || kind.includes("typed") || kind.includes("grown");
    return (
      <CubicBezierLine
        key={key}
        start={[x1, y1, z]}
        end={[x2, y2, z]}
        midA={[mx, y1, z]}
        midB={[mx, y2, z]}
        color={color}
        lineWidth={width}
        dashed={dashed}
        dashSize={kind.includes("grown") ? 0.03 : 0.08}
        gapSize={0.06}
        transparent={true}
        opacity={opacity}
      />
    );
  }

  /** A chord as a 3D box at layout (x, y) on plane z, its label texture on the front face. */
  function box(
    key: string,
    x: number,
    y: number,
    z: number,
    w: number,
    h: number,
    color: string,
    fill: string,
    opacity: number,
    label: Texture,
    over: Handler,
    out: Handler,
    click: Handler,
    dashed = false
  ) {
    const hw = (w * S) / 2;
    const hh = (h * S) / 2;
    const fz = BOX_D / 2 + 0.004;
    return (
      <group key={key} position={[wx(x + w / 2), wy(y + h / 2), z]} onPointerOver={over} onPointerOut={out} onClick={click}>
        <mesh>
          <boxGeometry args={[w * S, h * S, BOX_D]} />
          <meshStandardMaterial color={fill} transparent={true} opacity={opacity} />
          {!dashed && <Edges color={color} />}
        </mesh>
        {dashed && (
          <Line
            points={[
              [0.0 - hw, 0.0 - hh, fz],
              [hw, 0.0 - hh, fz],
              [hw, hh, fz],
              [0.0 - hw, hh, fz],
              [0.0 - hw, 0.0 - hh, fz],
            ]}
            color={color}
            lineWidth={1.8}
            dashed={true}
            dashSize={0.05}
            gapSize={0.04}
            transparent={true}
            opacity={opacity}
          />
        )}
        <mesh position={[0, 0, BOX_D / 2 + 0.002]}>
          <planeGeometry args={[w * S, h * S]} />
          <meshBasicMaterial map={label} transparent={true} opacity={opacity} toneMapped={false} depthWrite={false} />
        </mesh>
        <mesh position={[0, 0, 0.0 - BOX_D / 2 - 0.002]} rotation={[0, Math.PI, 0]}>
          <planeGeometry args={[w * S, h * S]} />
          <meshBasicMaterial map={label} transparent={true} opacity={opacity} toneMapped={false} depthWrite={false} />
        </mesh>
      </group>
    );
  }

  const chord_label = (rn: string, symbol: string, cadence: string, alt = "") => chord_texture(rn, symbol, cadence, INK, NODE_W, NODE_H, alt);

  /** A text label centred at layout (x, y) on plane z. cls: col | name | name-on. */
  function tag(key: string, x: number, y: number, z: number, text: string, cls: string) {
    const font = cls.includes("on") ? "600 12px " + SANS : (cls.includes("name") ? "12px " : "11px ") + SANS;
    const color = cls.includes("on") ? ACCENT : cls.includes("name") ? "#c3cadb" : MUTED;
    const shown = cls === "col" ? text.toUpperCase() : text;
    const tw = text_texture(shown, font, color);
    return (
      <Facing key={key} position={[wx(x), wy(y), z]}>
        <mesh>
          <planeGeometry args={[tw[1] * S, 18 * S]} />
          <meshBasicMaterial map={tw[0]} transparent={true} toneMapped={false} depthWrite={false} />
        </mesh>
      </Facing>
    );
  }

  /** Tree edge from a node's parent (one layer nearer the progression) to the node. */
  function tree_edge(key: string, n: ChordView, y: number, kind: string) {
    const py = ypos[n.parent];
    if (n.id >= search.back_start.id) return curve(key, x_back(n.layer), y, x_back(n.layer - 1), py, 0.0, color_of(n.func), kind);
    return curve(key, x_fwd(n.layer - 1), py, x_fwd(n.layer), y, 0.0, color_of(n.func), kind);
  }

  const alt_on = (i: number, j: number) => altHover.length === 2 && altHover[0] === i && altHover[1] === j;

  // --- the other options each kept chord was chosen from, fanned above and below it:
  // hover to highlight one, click to swap it in ---
  const others: AltView[][] = steps.map((st) => [...st.alts]);

  const opt_y = (j: number) => y0 + (Math.trunc(j / 2) + 1) * ROW_H * (j % 2 === 0 ? 1 : -1);

  let height = (TOP + max_rows * ROW_H + 20) * 1.0;
  let lo = TOP * 1.0;
  let hi = height * 1.0;
  for (const col of others) {
    if (col.length > 0) {
      lo = Math.min(lo, col.length > 1 ? opt_y(col.length % 2 === 0 ? col.length - 1 : col.length - 2) : lo);
      hi = Math.max(hi, opt_y(col.length % 2 === 1 ? col.length - 1 : col.length - 2) + NODE_H + 20);
    }
  }
  const shift = TOP - lo;
  height = hi + shift;

  function step_label(i: number): string {
    if (base === 0) return "start";
    return "step " + String(i + 1) + (i === 0 ? " · first" : i === base ? " · now" : "");
  }

  /** A chord box standing for variant `n` (the one hover/click act on); `ids` are
   * every variant drawn in it. kind: kept | typed | grown. */
  function node_g(n: ChordView, ids: number[], kind: string) {
    const x = x_node(n);
    const y = ypos[n.id];
    const live = reachable(by_id, locks, n.id);
    const col_locked = lock_at(by_id, locks, is_back_id(n.id), n.layer) >= 0;
    const hoverable = live && !col_locked;
    const is_hot = ids.some((i) => hot.includes(i));
    const is_locked = ids.some((i) => locks.includes(i));
    const fill = ids.includes(soundingId) ? FILL_SOUNDING : is_hot || is_locked ? FILL_HOT : FILL;
    const label = chord_texture(n.rn, n.symbol, n.cadence, is_locked ? ACCENT : INK, NODE_W, NODE_H, n.alt_rn);
    return (
      <group key={"n" + String(n.id)}>
        {box(
          "b" + String(n.id),
          x,
          y,
          0.0,
          NODE_W,
          NODE_H,
          color_of(n.func),
          fill,
          live ? 1.0 : 0.25,
          label,
          (e) => {
            e.stopPropagation();
            if (hoverable) onHover(n.id);
          },
          () => {
            if (hoverable) onLeave();
          },
          (e) => {
            e.stopPropagation();
            if (live) onSelect(n.id);
          },
          kind === "typed"
        )}
        {is_locked && (
          <mesh position={[wx(x + NODE_W / 2), wy(y + NODE_H / 2), 0.0]}>
            <boxGeometry args={[NODE_W * S + 0.05, NODE_H * S + 0.05, BOX_D * 0.6]} />
            <meshBasicMaterial color="#8fb3ff" transparent={true} opacity={0.25} />
          </mesh>
        )}
        {kind === "typed" && (
          <Html center={true} position={[wx(x + 1), wy(y + 1), BOX_D / 2 + 0.01]} zIndexRange={[200, 0]}>
            <button
              className="n3-remove"
              title="Remove this chord"
              onClick={(e) => {
                e.stopPropagation();
                onRemoveTyped(ids);
              }}
            >
              ×
            </button>
          </Html>
        )}
      </group>
    );
  }

  /** A dashed slot under a column: click it to type a chord into that column. */
  function slot_g(back: boolean, L: number) {
    const k = (back ? "b" : "f") + String(L);
    return typing_slot(k, back ? x_back(L) : x_fwd(L), slot_y[k], (text) => onAddTyped(back, L, text));
  }

  /** A dashed slot at layout (x, y), keyed `k`: click it to type a chord, Enter
   * hands the text to `submit`. */
  function typing_slot(k: string, x: number, y: number, submit: (text: string) => void) {
    const hw = (NODE_W * S) / 2;
    const hh = (NODE_H * S) / 2;
    const hint = text_texture("+ type a chord", "12px " + MONO, "#56607a");
    return (
      <group key={"slot" + k} position={[wx(x + NODE_W / 2), wy(y + NODE_H / 2), 0.0]}>
        <group
          onClick={(e) => {
            e.stopPropagation();
            setEditing(k);
          }}
        >
          <mesh>
            <planeGeometry args={[NODE_W * S, NODE_H * S]} />
            <meshBasicMaterial transparent={true} opacity={0.0} depthWrite={false} />
          </mesh>
          <Line
            points={[
              [0.0 - hw, 0.0 - hh, 0.0],
              [hw, 0.0 - hh, 0.0],
              [hw, hh, 0.0],
              [0.0 - hw, hh, 0.0],
              [0.0 - hw, 0.0 - hh, 0.0],
            ]}
            color={editing === k ? ACCENT : "#2f3e66"}
            lineWidth={1.4}
            dashed={true}
            dashSize={0.05}
            gapSize={0.04}
          />
          <Facing position={[0.0, 0.0, 0.001]}>
            <mesh>
              <planeGeometry args={[hint[1] * S, 18 * S]} />
              <meshBasicMaterial map={hint[0]} transparent={true} toneMapped={false} depthWrite={false} />
            </mesh>
          </Facing>
        </group>
        {editing === k && (
          <Html center={true} zIndexRange={[300, 0]}>
            <input
              className="slot-in n3-edit"
              autoFocus={true}
              placeholder="Dm7, V65, bVI…"
              title="Type a chord symbol (Dm7, G7/B, Ebmaj7) or a Roman numeral (ii7, V65, V7/V, bVI); Enter adds it, Esc cancels"
              onKeyDown={(e) => {
                if (e.key === "Enter" && e.currentTarget.value) {
                  submit(e.currentTarget.value);
                  setEditing("");
                } else if (e.key === "Escape") {
                  setEditing("");
                }
              }}
              onBlur={() => setEditing("")}
            />
          </Html>
        )}
      </group>
    );
  }

  // --- saved progressions: one row each, on planes behind the current one. Only with
  // "show all progressions"; otherwise just the one being played, if any ---
  const shown_saved = saved.map((_, k) => k).filter((k) => showAll || k === viewing);

  function saved_row(k: number, slot: number, prog: SavedView) {
    const z = 0.0 - (slot + 1) * Z_GAP;
    const on = k === viewing;
    return (
      <group key={"saved" + String(k)}>
        {tag(
          "sl" + String(k),
          px(0) - 110,
          y0 + NODE_H / 2,
          z,
          prog.name + " · " + prog.key_name + (on ? "  ▶ playing" : "  (click to play)"),
          on ? "name-on" : "name"
        )}
        {prog.steps.map((st, i) => (
          <group key={"ss" + String(k) + "-" + String(i)}>
            {i > 0 ? curve("se" + String(k) + "-" + String(i), px(i - 1), y0, px(i), y0, z, color_of(st.func), on ? "hot" : "plain") : null}
            {box(
              "sb" + String(k) + "-" + String(i),
              px(i),
              y0,
              z,
              NODE_W,
              NODE_H,
              color_of(st.func),
              on && i === savedSounding ? FILL_SOUNDING : on ? FILL_HOT : FILL,
              on ? 1.0 : 0.75,
              chord_label(st.rn, st.symbol, st.cadence, st.alt_rn),
              undefined,
              undefined,
              (e) => {
                e.stopPropagation();
                onPickSaved(k);
              }
            )}
          </group>
        ))}
      </group>
    );
  }

  /** The edge into kept step i: click it (or its dot) to open the voice-leading
   * diagram for that change; the dot takes the colour of the worst rule on it. */
  function lead_edge(i: number) {
    const x1 = px(i - 1) + NODE_W;
    const x2 = px(i);
    const ym = wy(y0 + NODE_H / 2);
    const is_open = leadAt === i;
    const on = is_open || edgeHover === i;
    const kind = steps[i].leads.length > 0 ? worst(steps[i].leads) : "hold";
    return (
      <group key={"le" + String(i)}>
        <mesh
          position={[wx((x1 + x2) / 2), ym, 0.0]}
          onPointerOver={(e) => {
            e.stopPropagation();
            setEdgeHover(i);
            document.body.style.setProperty("cursor", "pointer");
          }}
          onPointerOut={() => {
            setEdgeHover(-1);
            document.body.style.removeProperty("cursor");
          }}
          onClick={(e) => {
            e.stopPropagation();
            setLeadAt(is_open ? -1 : i);
          }}
        >
          <planeGeometry args={[(x2 - x1) * S, 0.22]} />
          <meshBasicMaterial transparent={true} opacity={0.0} depthWrite={false} />
        </mesh>
        <mesh position={[wx((x1 + x2) / 2), ym, 0.01]}>
          <circleGeometry args={[on ? 0.07 : 0.05, 24]} />
          <meshBasicMaterial color={KIND_COLOR[kind]} toneMapped={false} />
        </mesh>
        {is_open && (
          <Html position={[wx((x1 + x2) / 2), ym + 0.12, 0.02]} zIndexRange={[400, 0]}>
            <div className="vl-anchor">
              <VoiceLeading prev={steps[i - 1]} step={steps[i]} onClose={() => setLeadAt(-1)} />
            </div>
          </Html>
        )}
      </group>
    );
  }

  // camera: like the 2D map's initial scroll, on the last chord and the options after it
  const cx = wx((px(base) + x_fwd(2) + NODE_W) / 2);
  const cy = wy(y0 + NODE_H / 2);
  const back_z = 0.0 - shown_saved.length * Z_GAP;
  const range = (a: number, b: number) => Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);

  // mouse: panning is the common move, so it gets the left button (and one finger);
  // rotating, to look at the saved progressions behind, is the right button (two fingers)
  return (
    <div className="graph3d">
      <Canvas camera={{ position: [cx, cy + 1.6, 9.5], fov: 45 }} dpr={[1, 2]}>
        <ambientLight intensity={1.6} />
        <directionalLight position={[4, 8, 10]} intensity={1.1} />
        <OrbitControls
          makeDefault={true}
          enableDamping={true}
          dampingFactor={0.12}
          screenSpacePanning={true}
          mouseButtons={{ LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE }}
          touches={{ ONE: TOUCH.PAN, TWO: TOUCH.DOLLY_ROTATE }}
        />
        <CameraRig x={cx} y={cy} z={back_z / 2} />

        {range(1, depth + 1).map((L) => (
          <group key={"cols" + String(L)}>
            {tag("cb" + String(L), x_back(L) + NODE_W / 2, lo - 18, 0.0, L === 1 ? "could precede" : "lead-in " + String(L), "col")}
            {tag("cf" + String(L), x_fwd(L) + NODE_W / 2, lo - 18, 0.0, L === 1 ? "options" : "lookahead " + String(L), "col")}
          </group>
        ))}

        {range(0, steps.length).map((i) => (
          <group key={"opts" + String(i)}>
            {others[i].map((o, j) => (
              <group key={"o" + String(i) + "-" + String(j)}>
                {i > 0
                  ? curve("oe" + String(i) + "-" + String(j), px(i - 1), y0, px(i), opt_y(j), 0.0, color_of(o.func), alt_on(i, j) ? "hot" : "past")
                  : null}
                {box(
                  "ob" + String(i) + "-" + String(j),
                  px(i),
                  opt_y(j),
                  0.0,
                  NODE_W,
                  NODE_H,
                  color_of(o.func),
                  alt_on(i, j) ? FILL_HOT : FILL,
                  alt_on(i, j) ? 1.0 : 0.45,
                  chord_label(o.rn, o.symbol, "", o.alt_rn),
                  (e) => {
                    e.stopPropagation();
                    onAltHover(i, j);
                  },
                  () => onAltLeave(),
                  (e) => {
                    e.stopPropagation();
                    onAltPick(i, o.cid);
                  }
                )}
              </group>
            ))}
          </group>
        ))}

        {range(0, steps.length).map((i) => (
          <group key={"step" + String(i)}>
            {tag("sl" + String(i), px(i) + NODE_W / 2, lo - 18, 0.0, step_label(i), "col")}
            {i > 0 ? curve("st" + String(i), px(i - 1), y0, px(i), y0, 0.0, color_of(steps[i].func), "hot") : null}
            {i > 0 ? lead_edge(i) : null}
            {box(
              "sb" + String(i),
              px(i),
              y0,
              0.0,
              NODE_W,
              NODE_H,
              color_of(steps[i].func),
              i === soundingStep ? FILL_SOUNDING : FILL,
              1.0,
              chord_label(steps[i].rn, steps[i].symbol, steps[i].cadence, steps[i].alt_rn),
              undefined,
              undefined,
              (e) => {
                e.stopPropagation();
                onStep(i);
              },
              steps[i].typed
            )}
            {steps[i].typed && (
              <Html center={true} position={[wx(px(i) + 1), wy(y0 + 1), BOX_D / 2 + 0.01]} zIndexRange={[200, 0]}>
                <button
                  className="n3-remove"
                  title="Remove this chord"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveStep(i);
                  }}
                >
                  ×
                </button>
              </Html>
            )}
          </group>
        ))}

        {tree_nodes
          .filter((n) => visible(n) && n.parent in ypos)
          .map((n) => tree_edge("e" + String(n.id), n, ypos[n.id], (hot.includes(n.id) ? "hot" : "plain") + (n.status === "dest" ? " dest" : "")))}

        {typed
          .filter((t) => !animating && t.parent in ypos && t.id in ypos)
          .map((t) =>
            tree_edge(
              "et" + String(t.id),
              t,
              ypos[t.id],
              (t.status === "custom" ? "typed" : "grown") + (hot.includes(t.id) ? " hot" : "") + (reachable(by_id, locks, t.id) ? "" : " dim")
            )
          )}

        {tree_nodes.filter((n) => visible(n)).map((n) => node_g(kept_rep(n), kept_ids(n), "kept"))}
        {group_keys
          .filter(() => !animating)
          .map((k) =>
            node_g(
              rep_of[k],
              group_of[k].map((v) => v.id),
              group_of[k].some((v) => v.status === "custom") ? "typed" : "grown"
            )
          )}
        {range(0, steps.length)
          .filter(() => !animating)
          .map((i) => typing_slot("s" + String(i), px(i), y0 + (Math.trunc((others[i].length + 1) / 2) + 1) * ROW_H, (text) => onStepTyped(i, text)))}
        {range(1, depth + 1)
          .filter(() => !animating)
          .map((L) => slot_g(false, L))}
        {range(1, depth + 1)
          .filter(() => !animating)
          .map((L) => slot_g(true, L))}

        {animating && anim < depth && (
          <group>
            {fwd_layers[anim]
              .filter((n) => n.status !== "dest" || n.layer === 0)
              .map((n) => (
                <mesh key={"wf" + String(n.id)} position={[wx(x_fwd(anim) + NODE_W), wy(ypos[n.id] + NODE_H / 2), 0.08]}>
                  <sphereGeometry args={[0.05, 16, 16]} />
                  <meshBasicMaterial color="#ffffff" />
                </mesh>
              ))}
            {back_layers[anim]
              .filter((n) => n.status !== "dest" || n.layer === 0)
              .map((n) => (
                <mesh key={"wb" + String(n.id)} position={[wx(x_back(anim)), wy(ypos[n.id] + NODE_H / 2), 0.08]}>
                  <sphereGeometry args={[0.05, 16, 16]} />
                  <meshBasicMaterial color="#ffffff" />
                </mesh>
              ))}
          </group>
        )}

        {shown_saved.map((k, slot) => saved_row(k, slot, saved[k]))}
      </Canvas>
      <div className="graph3d-hint">
        faint boxes beside a kept chord: click to swap · "+ type a chord" under a kept chord replaces it · click the dot on an edge for its
        voice leading · drag to pan · right-drag to rotate · scroll to zoom
      </div>
    </div>
  );
}
