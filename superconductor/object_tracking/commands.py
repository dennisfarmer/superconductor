"""Typed commands for the object library and parameter mappings.

Type them in the SuperConductor window (press / or Enter first) or in the
terminal. Objects are referred to by name ("bulbasaur") or id ("#3").

    save #3 bulbasaur              remember #3 as "bulbasaur" (adds its current views)
    save bulbasaur                 add the views seen this session to a known object
    map bulbasaur held epic orchestral strings
    map chimchar near fire taiko drum      (an existing [[parameters]] name, or a prompt)
    map #3 visible temperature 0.9 1.4
    when I'm holding bulbasaur play epic orchestral strings
    unmap bulbasaur [held]
    show bulbasaur                 what the library remembers it looks like
    forget bulbasaur               remove from the library (moved to library/.trash)
    combine bulbasaur chimchar friendly jungle beat [distance 0.05]
                                   together, they play this instead of their own mappings
    uncombine bulbasaur chimchar
    distance 0.08                  how close combined objects must be (fraction of frame width)
    list                           objects, their ids and mappings
    lock / unlock                  stop / resume creating new identities
    reset                          forget this session's unnamed objects
"""
import json
import re
import shlex

import numpy as np

from superconductor.object_tracking.library import tile
from superconductor.object_tracking.mapper import PALETTE, Combo, Parameter

CONTROL_KINDS = ("temperature", "top_k", "cfg_musiccoca")
TRIGGER_WORDS = {"near": "near", "close": "near", "distance": "near",
                 "held": "held", "hold": "held", "holding": "held", "touch": "held", "touching": "held",
                 "visible": "visible", "seen": "visible", "showing": "visible", "see": "visible"}
NATURAL = re.compile(
    r"^(?:when|while|if)\s+(?:i'?m\s+|i\s+am\s+|you\s+see\s+me\s+)?(?P<trigger>\w+)\s+(?:the\s+)?"
    r"(?P<object>#?[\w-]+)[\s,]*(?:then\s+)?(?:play|->|=>|:|=|use|add)?\s*(?P<rest>.+)$", re.IGNORECASE)


class CommandError(Exception):
    pass


class Commands:
    def __init__(self, registry, mapper, library=None):
        self.registry = registry
        self.mapper = mapper
        self.library = library
        self.mappings = {}  # owner (name or "#id") -> list of mapping dicts
        self.show_request = None  # (title, image) for the UI to display
        self._last_objects = []
        if library is not None:
            self._load_combos()
            for name, entry in library.entries.items():
                if entry.mappings:
                    self.mappings[name] = list(entry.mappings)
                    mapper.set_mappings(name, entry.mappings)

    def __call__(self, line):
        """Run one command line; returns a message for the user."""
        line = line.strip()
        if not line:
            return ""
        m = NATURAL.match(line)
        if m and m["trigger"].lower() in TRIGGER_WORDS:
            line = f"map {m['object']} {TRIGGER_WORDS[m['trigger'].lower()]} {m['rest']}"
        try:
            words = shlex.split(line)
        except ValueError:
            words = line.split()
        cmd, args = words[0].lower(), words[1:]
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            return f"unknown command {cmd!r}; try: help"
        try:
            return handler(args)
        except CommandError as e:
            return str(e)

    # ---------- helpers ----------

    def _identity(self, key):
        ident = self.registry.get(key)
        if ident is None:
            raise CommandError(f"no object {key!r} (try: list)")
        return ident

    @staticmethod
    def _owner(ident):
        return ident.user_label or f"#{ident.id}"

    @staticmethod
    def _describe(m):
        what = m.get("parameter") or m.get("prompt") or (
            f"{m.get('kind')} {m.get('min', 0)}..{m.get('max', 1)}")
        return f"{m.get('trigger', 'near')} -> {what}"

    def _apply(self, owner):
        mappings = self.mappings.get(owner, [])
        self.mapper.set_mappings(owner, mappings)
        if self.library is not None and owner in self.library.entries:
            self.library.set_mappings(owner, mappings)

    # ---------- combos (typed ones are kept in <library>/combos.json) ----------

    @property
    def _combos_path(self):
        return self.library.root / "combos.json" if self.library is not None else None

    def _load_combos(self):
        path = self._combos_path
        if path is None or not path.exists():
            return
        for c in json.loads(path.read_text()):
            try:
                self._add_combo(c["objects"], c.get("parameter"), c.get("prompt"),
                                c.get("distance"), save=False)
            except CommandError as e:
                print(f"combos.json: {e}")

    def _save_combos(self):
        path = self._combos_path
        if path is None:
            return
        typed = []
        for combo in self.mapper.combos:
            if getattr(combo, "typed", False):
                param = self.mapper.get(combo.parameter)
                entry = {"objects": list(combo.objects), "distance": combo.distance}
                if param.owner == combo.key:
                    entry["prompt"] = param.prompt
                else:
                    entry["parameter"] = combo.parameter
                typed.append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(typed, indent=1))

    def _add_combo(self, objects, parameter=None, prompt=None, distance=None, save=True):
        objects = tuple(objects)
        if len(objects) < 2:
            raise CommandError("a combo needs at least two objects")
        self._remove_combo(objects, save=False)
        combo = Combo(objects=objects, parameter="",
                      distance=distance if distance is not None else self.default_distance())
        if parameter and self.mapper.get(parameter) is not None:
            combo.parameter = parameter
        else:
            text = prompt or parameter
            if not text:
                raise CommandError("what should the combo play?")
            name = " + ".join(objects)
            self.mapper.parameters.append(Parameter(
                name=name, kind="prompt", prompt=text, owner=combo.key,
                color=PALETTE[len(self.mapper.parameters) % len(PALETTE)]))
            combo.parameter = name
        combo.typed = True
        self.mapper.combos.append(combo)
        if save:
            self._save_combos()
        return combo

    def _remove_combo(self, objects, save=True):
        objects = set(objects)
        removed = [c for c in self.mapper.combos if set(c.objects) == objects]
        for combo in removed:
            self.mapper.combos.remove(combo)
            param = self.mapper.get(combo.parameter)
            if param is not None and param.owner == combo.key:
                self.mapper.parameters.remove(param)
        if save and removed:
            self._save_combos()
        return removed

    def default_distance(self):
        return self.mapper.combos[0].distance if self.mapper.combos else 0.05

    def set_distance(self, distance, combos=None):
        for combo in combos or self.mapper.combos:
            combo.distance = max(0.0, round(distance, 3))
        self._save_combos()

    # ---------- commands ----------

    def cmd_help(self, args):
        return __doc__.split("\n\n", 2)[2].rstrip()

    def cmd_list(self, args):
        lines = []
        for ident in self.registry.identities:
            where = "in view" if any(o.identity == ident.id and not o.coasted
                                     for o in self._last_objects) else "not in view"
            lib = " [library]" if ident.library else ""
            params = [self.mapper.get(n) for n in self.mapper.assignments.get(self._owner(ident), [])]
            params += [p for p in self.mapper.parameters_for(ident.id) if p not in params]
            maps = ", ".join(f"{p.trigger} -> {p.prompt or p.name}" for p in params if p) or "no mapping"
            lines.append(f"#{ident.id} {ident.label}{lib}: {where}; {maps}")
        for c in self.mapper.combos:
            lines.append(f"combo {' + '.join(c.objects)} within {c.distance:.2f} -> {c.parameter}"
                         + (" [ACTIVE]" if c.active else ""))
        lines.append(f"{len(self.registry.pending)} unidentified track(s); new objects "
                     f"{'allowed' if self.registry.allow_new else 'locked'}")
        return "\n".join(lines)

    cmd_ls = cmd_list

    def cmd_save(self, args):
        if self.library is None:
            raise CommandError("no library configured ([tracking] library_dir)")
        if not args:
            raise CommandError("usage: save #3 bulbasaur   (or: save bulbasaur)")
        ident = self._identity(args[0])
        name = args[1] if len(args) > 1 else ident.user_label
        if not name:
            raise CommandError(f"#{ident.id} has no name yet: save #{ident.id} <name>")
        other = self.registry.get(name)
        if other is not None and other is not ident:
            # the same object was seen as two identities: keep the named one
            self.registry.merge(ident, other)
            self.mappings.setdefault(name, [])
            for m in self.mappings.pop(self._owner(ident), []):
                self._put(name, m)
            self.mapper.set_mappings(self._owner(ident), [])
            ident = other
        old_owner = self._owner(ident)
        if old_owner != name:
            self.mapper.rename_owner(old_owner, name)
            for m in self.mappings.pop(old_owner, []):
                self._put(name, m)
        ident.user_label = name
        entry = self.library.entries.get(name)
        if not ident.gallery:
            if entry is None:
                raise CommandError(f"no views of {name} yet: keep it in view for a moment and save again")
            self._apply(name)
            return f"{name}: no new views since the last save"
        mappings = self.mappings.get(name, entry.mappings if entry else [])
        self.mappings[name] = list(mappings)
        entry = self.library.save(name, np.stack(ident.gallery), ident.views, ident.hist,
                                  detected_as=dict(ident.votes.most_common(3)), mappings=mappings)
        added = len(ident.gallery)
        ident.fixed = entry.embeddings
        ident.gallery, ident.views = [], []
        ident.library = True
        self._apply(name)
        return f"saved {name} (#{ident.id}): +{added} views, {len(entry.embeddings)} total -> {entry.path}"

    def _put(self, owner, mapping):
        """add a mapping, replacing an existing one with the same trigger and kind"""
        key = (mapping.get("trigger", "near"), mapping.get("kind", "prompt"))
        maps = [m for m in self.mappings.get(owner, [])
                if (m.get("trigger", "near"), m.get("kind", "prompt")) != key]
        maps.append(mapping)
        self.mappings[owner] = maps

    def cmd_map(self, args):
        if len(args) < 2:
            raise CommandError("usage: map <object> [near|held|visible] <prompt | parameter | "
                               "temperature min max>")
        owner = self._owner(self._identity(args[0]))
        rest = args[1:]
        trigger = "near"
        if rest[0].lower() in TRIGGER_WORDS:
            trigger = TRIGGER_WORDS[rest.pop(0).lower()]
        if not rest:
            raise CommandError("what should it map to? e.g. map bulbasaur held epic strings")
        mapping = {"trigger": trigger}
        text = " ".join(rest).strip().strip("\"'")
        if rest[0].lower() in CONTROL_KINDS:
            mapping["kind"] = rest[0].lower()
            try:
                nums = [float(x) for x in rest[1:3]]
            except ValueError:
                raise CommandError(f"usage: map {owner} {trigger} {rest[0]} <min> <max>")
            if len(nums) == 2:
                mapping["min"], mapping["max"] = nums
        elif self.mapper.get(text) is not None and self.mapper.get(text).owner is None:
            mapping["parameter"] = text
        else:
            mapping["prompt"] = text
        self._put(owner, mapping)
        self._apply(owner)
        saved = "" if self.library is None or owner in self.library.entries else \
            f" (not in the library yet: save {owner} <name> to keep it)" if owner.startswith("#") else \
            f" (save {owner} to keep it)"
        return f"{owner}: {self._describe(mapping)}{saved}"

    def cmd_unmap(self, args):
        if not args:
            raise CommandError("usage: unmap <object> [near|held|visible]")
        owner = self._owner(self._identity(args[0]))
        trigger = TRIGGER_WORDS.get(args[1].lower()) if len(args) > 1 else None
        before = self.mappings.get(owner, [])
        self.mappings[owner] = [m for m in before if trigger and m.get("trigger", "near") != trigger]
        self._apply(owner)
        return f"{owner}: removed {len(before) - len(self.mappings[owner])} mapping(s)"

    def cmd_show(self, args):
        if not args:
            raise CommandError("usage: show <object>")
        ident = self.registry.get(args[0])
        name = ident.user_label if ident else args[0]
        image = None
        if self.library is not None and name in self.library.entries:
            image = self.library.contact_sheet(name)
        if ident is not None and ident.views:
            session = tile(ident.views)
            image = session if image is None else _stack(image, session)
        if image is None:
            raise CommandError(f"no saved or recent views of {args[0]!r}")
        self.show_request = (f"object: {name}", image)
        return f"showing {name}"

    def cmd_forget(self, args):
        if self.library is None or not args or args[0] not in self.library.entries:
            raise CommandError(f"usage: forget <library object>; have {sorted(self.library.entries)}"
                               if self.library else "no library configured")
        name = args[0]
        target = self.library.forget(name)
        ident = self.registry.get(name)
        if ident is not None:
            ident.library, ident.fixed = False, None
        self.mappings.pop(name, None)
        self.mapper.set_mappings(name, [])
        return f"forgot {name} (moved to {target})"

    def cmd_combine(self, args):
        distance = None
        if len(args) >= 2 and args[-2].lower() == "distance":
            try:
                distance = float(args[-1])
            except ValueError:
                raise CommandError("distance must be a number, e.g. distance 0.05")
            args = args[:-2]
        objects = []
        while args and self.registry.get(args[0]) is not None and len(objects) < 4:
            objects.append(self._owner(self.registry.get(args.pop(0))))
        if len(objects) < 2 or not args:
            raise CommandError("usage: combine <object> <object> <parameter or prompt> [distance 0.05]")
        text = " ".join(args).strip().strip("\"'")
        combo = self._add_combo(objects, parameter=text, distance=distance)
        return (f"{' + '.join(combo.objects)} within {combo.distance:.2f} -> {combo.parameter}"
                + ("" if combo.parameter == text else f" ({text})"))

    def cmd_uncombine(self, args):
        objects = [self._owner(self._identity(a)) for a in args]
        removed = self._remove_combo(objects)
        return f"removed {len(removed)} combo(s)"

    def cmd_distance(self, args):
        if not args:
            return ", ".join(f"{' + '.join(c.objects)}: {c.distance:.3f}" for c in self.mapper.combos) \
                or "no combos"
        try:
            distance = float(args[-1])
        except ValueError:
            raise CommandError("usage: distance [objects...] 0.05")
        names = {self._owner(self._identity(a)) for a in args[:-1]}
        combos = [c for c in self.mapper.combos if not names or names <= set(c.objects)]
        self.set_distance(distance, combos)
        return f"combo distance {distance:.3f}"

    def cmd_lock(self, args):
        self.registry.allow_new = False
        return "new objects locked: only known objects are tracked"

    def cmd_unlock(self, args):
        self.registry.allow_new = True
        return "new objects allowed"

    def cmd_reset(self, args):
        dropped = [i for i in self.registry.identities if not i.library]
        self.registry.reset(keep_library=True)
        for ident in dropped:
            self.mappings.pop(f"#{ident.id}", None)
            self.mapper.set_mappings(f"#{ident.id}", [])
        for param in self.mapper.parameters:
            if param.identity in {i.id for i in dropped}:
                param.identity = None
        return f"forgot {len(dropped)} unsaved object(s)"


def _stack(a, b):
    w = max(a.shape[1], b.shape[1])
    pad = lambda im: np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0)), constant_values=40)
    return np.vstack([pad(a), np.full((6, w, 3), 255, np.uint8), pad(b)])
