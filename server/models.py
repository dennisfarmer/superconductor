"""Wire types: what the endpoints return to the page.

FastAPI publishes them as the OpenAPI schema the client's TypeScript types are
generated from (client/src/api-types.ts, `npm run gen-types`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MarkView(BaseModel):
    kind: str
    text: str
    cost: float


class ChordView(BaseModel):
    id: int
    parent: int
    layer: int
    first: int
    cid: str
    rn: str
    symbol: str
    func: str
    family: str
    notes: list[int]
    names: list[str]
    syntax_cost: float
    vl_cost: float
    phrase_cost: float
    step_cost: float
    total: float
    rule: str
    marks: list[MarkView]
    cadence: str
    cadence_name: str
    status: str
    alt_rn: str = ""   # the other Roman numeral for the same pitches (VI for V/ii)
    also: str = ""     # why each name applies


class LayerView(BaseModel):
    layer: int
    frontier: list[int]
    frontier_rns: list[str]
    nodes_visited: int
    edges_followed: int
    voicings_scored: int
    newly_expanded: int
    candidates: int
    kept: int
    pruned: int
    cadences: int
    ms: float


class GraphView(BaseModel):
    key_name: str
    chord_nodes: int
    progression_edges: int
    expanded_nodes: int
    vocabulary: int


class SearchView(BaseModel):
    search_id: int
    key_name: str
    start: ChordView
    nodes: list[ChordView]
    layers: list[LayerView]
    graph: GraphView
    elapsed_ms: float
    width: int
    depth: int
    options: int
    # the backward tree: what could come before the first chord (ids from BACK_BASE)
    back_start: ChordView
    back_nodes: list[ChordView]
    back_layers: list[LayerView]
    back_elapsed_ms: float


class AltView(BaseModel):
    cid: str
    rn: str
    symbol: str
    func: str
    notes: list[int]   # voiced from the chord before (or into the one after)
    alt_rn: str = ""


class LeadView(BaseModel):
    """One voice-leading rule bearing on one voice of a chord change (theory.leading)."""
    voice: int
    kind: str
    title: str
    text: str
    detail: str


class StepView(BaseModel):
    index: int
    cid: str
    rn: str
    symbol: str
    func: str
    notes: list[int]
    names: list[str]
    rule: str
    cadence: str
    degrees: list[str] = []      # scale degree of each voice, e.g. "2̂", "♯4̂"
    alt_rn: str = ""             # the other Roman numeral for the same pitches
    also: str = ""               # why each name applies
    alts: list[AltView] = []
    leads: list[LeadView] = []   # the edge into this chord from the one before
    typed: bool = False          # typed in by the composer (removable)


class SavedView(BaseModel):
    index: int
    name: str
    key_name: str
    steps: list[StepView]


class SessionView(BaseModel):
    tonic: str
    mode: str
    key_name: str
    style_texts: list[str]
    style_weights: list[float]
    texture: str
    drum_grid: str
    llm_enabled: bool
    extended: bool
    bpm: float
    beats_per_chord: int
    steps: list[StepView]
    loop_from: int
    loop_to: int
    can_undo: bool
    saved: list[SavedView]
    notice: str
    engine_url: str
    engine_online: bool
    local: bool
    major_tonics: list[str]
    minor_tonics: list[str]


class OptionResult(BaseModel):
    nodes: list[ChordView]
    notice: str
    info: str = ""   # set when another form of the typed chord was used


class PathStory(BaseModel):
    title: str = Field(description="An evocative 2-5 word name for the progression, like a chapter title.")
    mood: str = Field(description="One sentence on the emotional character of the progression.")
    explanation: str = Field(
        description="Two sentences a composer would find useful: why the progression works in terms of "
        "harmonic function, voice leading, and how it closes."
    )
