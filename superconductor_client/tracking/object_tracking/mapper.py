"""Map tracked objects to music parameters.

Three separate things:
- what an object *is*: its identity (unique id) and optional label, see identity.py
- what a parameter *does*: a `Parameter` (e.g. "nature flute" -> a style prompt)
- which object drives which parameter: `assignments`, mapping an object label
  or "#<identity id>" to one or more parameter names

An assigned object always drives its parameters. An unassigned object claims the
first free parameter that isn't reserved by an assignment. Either way it keeps
that parameter while occluded or out of frame; by default parameters are never
released (set `release_after`, in seconds unseen, to recycle them).

Each parameter has a trigger that sets its value from its object:
- "near": 1 at the reference point, falling linearly to 0 at `max_distance`
  (normalized image units)
- "held": 1 while a detected hand overlaps the object
- "visible": 1 while the object is in view
The value is then smoothed.

Parameters can be added and removed while running (`set_mappings`), e.g. from
the object library or a typed "map bulbasaur held epic strings" command.

Combos: when all objects of a `Combo` are within its `distance` of each other
(box edge to box edge, fraction of the frame width), their own parameters fade
out and the combo's parameter takes over (e.g. bulbasaur + chimchar together ->
"friendly jungle beat" instead of flute and drums). It is driven by the group
as a whole (its trigger applied to the union of their boxes). They separate at
`distance` * `release_factor` (hysteresis, so it doesn't flicker at the edge),
and the original parameters come back.
"""
import math
import time
from dataclasses import dataclass, field

PALETTE = [(80, 200, 80), (40, 140, 255), (255, 160, 60), (200, 90, 220),
           (60, 220, 220), (90, 90, 255), (220, 220, 80), (180, 180, 180)]


@dataclass
class Parameter:
    name: str
    kind: str  # "prompt" (style weight) | "temperature" | "top_k" | "cfg_musiccoca"
    prompt: str = None  # for kind="prompt"
    min: float = 0.0
    max: float = 1.0
    color: tuple = (255, 255, 255)  # BGR, for drawing
    trigger: str = "near"  # "near" | "held" | "visible"
    owner: str = None  # object label / "#id" whose mapping created this parameter

    # runtime state
    identity: int = None  # object currently driving this parameter
    value: float = 0.0
    last_seen: float = 0.0
    visible: bool = False

    @property
    def output(self):
        """value mapped into [min, max]"""
        return self.min + (self.max - self.min) * self.value


@dataclass
class Combo:
    objects: tuple  # object labels (or "#id")
    parameter: str  # name of the Parameter that replaces theirs
    distance: float = 0.05  # join when every pair is at most this far apart (box edge to edge)
    release_factor: float = 1.5  # separate beyond distance * release_factor
    trigger: str = "near"  # of the combined box

    # runtime state
    active: bool = False
    members: tuple = ()  # the objects (TrackedObject) currently combined
    box: tuple = None  # union of the members' boxes, normalized

    @property
    def key(self):
        return "combo:" + "+".join(self.objects)


def _box_gap(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return max(dx, dy)


def _overlap(a, b):
    """intersection area of two (x1, y1, x2, y2) boxes"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


@dataclass
class ParameterMapper:
    parameters: list
    assignments: dict = field(default_factory=dict)  # object label or "#id" -> parameter name(s)
    reference: tuple = (0.5, 0.5)  # normalized (x, y), in *unflipped* frame coords
    max_distance: float = 0.6
    smoothing: float = 0.3  # EMA factor per update (1 = no smoothing)
    hold_seconds: float = 1.5  # keep value while occluded, then fade out
    release_after: float = None  # free the parameter after this long unseen (None = never)
    fade_per_second: float = 1.0
    held_overlap: float = 0.15  # fraction of the object's box a hand must cover for "held"
    combos: list = field(default_factory=list)
    _last_update: float = field(default=None, repr=False)

    def __post_init__(self):
        self.assignments = {k: [v] if isinstance(v, str) else list(v) for k, v in self.assignments.items()}
        self._base = {k: list(v) for k, v in self.assignments.items()}  # from the config
        names = {p.name for p in self.parameters}
        unknown = {n for v in self.assignments.values() for n in v} - names
        if unknown:
            raise ValueError(f"assignments refer to unknown parameters {unknown}; have {names}")
        unknown = {c.parameter for c in self.combos} - names
        if unknown:
            raise ValueError(f"combos refer to unknown parameters {unknown}; have {names}")

    def get(self, name):
        return next((p for p in self.parameters if p.name == name), None)

    def parameter_for(self, identity):
        return next((p for p in self.parameters if p.identity == identity), None)

    def parameters_for(self, identity):
        return [p for p in self.parameters if p.identity == identity]

    def _assigned(self, obj):
        names = self.assignments.get(f"#{obj.identity}", []) + self.assignments.get(obj.class_name, [])
        return [p for p in self.parameters if p.name in names]

    def _reserved(self):
        return {n for v in self.assignments.values() for n in v} | {c.parameter for c in self.combos}

    def combo_for(self, identity):
        return next((c for c in self.combos if c.active and any(m.identity == identity for m in c.members)), None)

    # ---------- runtime mappings ----------

    def set_mappings(self, owner, mappings):
        """Replace the parameters `owner` (an object label or "#id") drives with
        `mappings` (library.py format). Returns the resulting parameters."""
        for param in [p for p in self.parameters if p.owner == owner]:
            self.parameters.remove(param)
        names = []
        for m in mappings:
            if m.get("parameter"):
                if self.get(m["parameter"]) is None:
                    print(f"mapping for {owner}: no parameter named {m['parameter']!r}, skipped")
                    continue
                names.append(m["parameter"])
                continue
            trigger = m.get("trigger", "near")
            kind = m.get("kind", "prompt")
            name = f"{owner} {trigger}" + ("" if kind == "prompt" else f" {kind}")
            color = tuple(m.get("color") or PALETTE[len(self.parameters) % len(PALETTE)])
            self.parameters.append(Parameter(name=name, kind=kind, prompt=m.get("prompt"),
                                             min=m.get("min", 0.0), max=m.get("max", 1.0),
                                             color=color, trigger=trigger, owner=owner))
            names.append(name)
        names = self._base.get(owner, []) + [n for n in names if n not in self._base.get(owner, [])]
        if names:
            self.assignments[owner] = names
        else:
            self.assignments.pop(owner, None)
        return [self.get(n) for n in names]

    def rename_owner(self, old, new):
        """An object got a name (e.g. "#4" saved as "bulbasaur"): keep its mappings."""
        for param in self.parameters:
            if param.owner == old:
                param.owner = new
                param.name = new + param.name[len(old):]
        for combo in self.combos:
            if old in combo.objects:
                old_key = combo.key
                combo.objects = tuple(new if o == old else o for o in combo.objects)
                for param in self.parameters:
                    if param.owner == old_key:
                        param.owner = combo.key
                    if param.identity == old_key:
                        param.identity = combo.key
        if old in self.assignments:
            names = [new + n[len(old):] if n.startswith(old + " ") else n for n in self.assignments.pop(old)]
            self.assignments.setdefault(new, [])
            self.assignments[new] += [n for n in names if n not in self.assignments[new]]

    # ---------- per frame ----------

    def _claim_free(self, obj):
        reserved = self._reserved()
        param = next((p for p in self.parameters
                      if p.identity is None and p.name not in reserved and p.owner is None), None)
        if param is not None:
            param.identity = obj.identity
            param.value = 0.0
        return param

    def _target(self, trigger, box, center, hands):
        """`box` in pixels (compared with hand boxes), `center` normalized"""
        if trigger == "held":
            area = (box[2] - box[0]) * (box[3] - box[1])
            covered = sum(_overlap(box, h.box) for h in hands)
            return 1.0 if area > 0 and covered / area >= self.held_overlap else 0.0
        if trigger == "visible":
            return 1.0
        d = math.dist(center, self.reference)
        return max(0.0, 1.0 - d / self.max_distance)

    def _update_combos(self, objects):
        """(de)activate combos; returns identities whose own parameters are replaced"""
        by_label = {}
        for obj in objects:
            if obj.identity is not None:
                by_label.setdefault(obj.class_name, obj)
                by_label.setdefault(f"#{obj.identity}", obj)
        replaced = set()
        for combo in self.combos:
            members = [by_label.get(label) for label in combo.objects]
            if any(m is None or m.box_normalized is None for m in members) or \
                    len({m.identity for m in members}) < len(members):
                combo.active, combo.members = False, ()  # one of them is gone
                continue
            limit = combo.distance * (combo.release_factor if combo.active else 1.0)
            together = all(_box_gap(a.box_normalized, b.box_normalized) <= limit
                           for i, a in enumerate(members) for b in members[i + 1:])
            if together != combo.active:
                print(f"{' + '.join(combo.objects)}: {'combined -> ' + combo.parameter if together else 'separated'}")
            combo.active = together
            combo.members = tuple(members) if together else ()
            if together:
                boxes = [m.box_normalized for m in members]
                combo.box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                             max(b[2] for b in boxes), max(b[3] for b in boxes))
                replaced.update(m.identity for m in members)
        return replaced

    def update(self, objects, now=None, hands=()):
        now = time.time() if now is None else now
        dt = 0.0 if self._last_update is None else now - self._last_update
        self._last_update = now

        replaced = self._update_combos(objects)
        seen = set()
        for obj in sorted(objects, key=lambda o: -o.confidence):
            if obj.identity is None:
                continue
            assigned = self._assigned(obj)
            if assigned:
                for param in self.parameters_for(obj.identity):
                    if param not in assigned:  # got a mapping after claiming a free parameter
                        param.identity = None
                for param in assigned:
                    if param.identity != obj.identity:
                        param.identity = obj.identity
                        param.value = 0.0
                params = assigned
            else:
                params = self.parameters_for(obj.identity)
                if not params:
                    param = self._claim_free(obj)
                    if param is None:
                        continue  # more objects than parameters
                    params = [param]
            seen.add(obj.identity)

            for param in params:
                # combined objects hand over to the combo's parameter
                target = 0.0 if obj.identity in replaced else \
                    self._target(param.trigger, obj.box, obj.center, hands)
                param.value += self.smoothing * (target - param.value)
                param.last_seen = now
                param.visible = True

        combo_params = set()
        for combo in self.combos:
            param = self.get(combo.parameter)
            combo_params.add(param.name)
            if combo.active:
                m = combo.members
                box = (min(o.box[0] for o in m), min(o.box[1] for o in m),
                       max(o.box[2] for o in m), max(o.box[3] for o in m))
                center = ((combo.box[0] + combo.box[2]) / 2, (combo.box[1] + combo.box[3]) / 2)
                target = self._target(combo.trigger, box, center, hands)
                param.identity, param.visible, param.last_seen = combo.key, True, now
            elif param.identity == combo.key:
                target = 0.0
                param.visible = False
            else:
                continue
            param.value += self.smoothing * (target - param.value)
            if not combo.active and param.value < 0.01:
                param.identity, param.value = None, 0.0

        for param in self.parameters:
            if param.identity is None or param.identity in seen or param.name in combo_params:
                continue
            param.visible = False
            missing = now - param.last_seen
            if self.release_after is not None and missing > self.release_after:
                param.identity = None
                param.value = 0.0
            elif missing > self.hold_seconds or param.trigger == "held":
                param.value = max(0.0, param.value - self.fade_per_second * dt)

    def recipe(self, base_prompt=None, base_weight=0.0):
        """Style weights for MagentaClient.update_recipe()."""
        recipe = {}
        if base_prompt and base_weight > 0:
            recipe[base_prompt] = base_weight
        for param in self.parameters:
            if param.kind == "prompt" and param.identity is not None and param.value > 0.01:
                recipe[param.prompt] = recipe.get(param.prompt, 0.0) + param.output
        return recipe

    def controls(self):
        """Sampling overrides (temperature / top_k / cfg_musiccoca) from active parameters."""
        return {param.kind: param.output for param in self.parameters
                if param.kind != "prompt" and param.identity is not None}
