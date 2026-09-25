It's ready for you to run. Here's where things stand.

**Run it:**
```bash
conda activate sc_env
sc-collab                      # unsupervised mode (default)
sc-collab --mode calibration   # the old "touch the bulbasaur" flow
```
Or, without activating the environment: `make run` (webcam + `mrt2_small`), `make vision` (no music), `make test`. Extra options go in `make run ARGS="--mode calibration"`.
Each object gets its own id (`#1`, `#2`, …) the first time it's seen. Objects in the library (`library/`) are recognized by name. Each object drives its parameters; click in the window to move the crosshair.

**Keys and commands:** click an object to select it. Press `s` to save it to the library; if it has no name yet, the app asks you to type one. Press `/` (or Enter) to type a command in the window, or type the command in the terminal. `[` and `]` shrink or widen the combo distance, and `q` quits.
```
save #3 bulbasaur                        remember #3 as bulbasaur (adds its current views)
save bulbasaur                           add this session's views to a known object
map bulbasaur held epic orchestral strings
when I'm holding chimchar play fire taiko drum
map #4 visible temperature 0.9 1.4
combine bulbasaur chimchar friendly jungle beat distance 0.05
distance 0.08                            tune how close counts as together
show bulbasaur                           see what the library remembers it looks like
list / unmap / forget / lock / unlock / reset / help
```
- **Triggers:** `near` means distance to the crosshair, `held` means a hand is on the object, and `visible` means it's in view.
- **Mapping targets:** a mapping can be a prompt, a `[[parameters]]` name from `collab.toml`, or a sampling control.

**How identity works:**
- **Detection:** the detector only finds toys (`"stuffed toy", "toy"`); it never decides who is who.
- **Recognition:** a small appearance model (`yolo11n-cls`, about 3 ms per crop on the CPU) decides that. It looks at each new object 3 times in its first 8 frames and compares it with the known objects. Objects that aren't in view compete for the match jointly, so two of them can't swap.
- **Periodic checks:** each tracked object is re-checked about once a second, and immediately when it reappears. If it looks clearly more like another object, it is re-identified. This fixes swaps when the tracker revives an old track on the wrong object.
- **Unknown objects:** something that looks unlike every known object becomes a new id. Something in between waits for more evidence.
- **Coasting:** when the detector loses an object (it merged into its neighbour's box, or is half hidden), the object keeps its last box, drawn dashed with "last seen". That lasts 1 s, or indefinitely while it is next to a visible object.
- **Combos:** `[[combos]]` in `collab.toml`, or the `combine` command. When bulbasaur and chimchar are within `distance` (edge to edge, as a fraction of the frame width), "friendly jungle beat" replaces flute and drums. A yellow frame and a banner show the combination. They separate at 1.5× the distance, and the original parameters come back. While they're apart, a thin line shows the current gap next to the threshold.

**Library files:**
- `library/<name>/object.json`: the mappings (editable)
- `embeddings.npy` and `hist.npy`: what it looks like
- `views/*.jpg`: saved views
- `library/combos.json`: typed combos

`forget` moves an object to `library/.trash`. The library is seeded with one view each of bulbasaur and chimchar from a dim evening webcam frame. Press `s` on each in normal light to add better views.

**Testing without the camera:** `python tests/test_object_tracking.py` runs the unit tests. The full pipeline on a scripted scene (hide, reappear, swap, come together, overlap, separate, unknown toy, next session in dimmer light) runs with:
```bash
python scripts/simulate_tracking.py --frame photo.jpg --object bulbasaur:x1,y1,x2,y2 --object chimchar:x1,y1,x2,y2
```
It writes an annotated `var/sim/sim.mp4`.

**Watch for:** `mrt2_base` is the default. In the one run with YOLO on the CPU it managed only 0.74× real time (it needs to be above 1× to keep up), so expect audio dropouts. Run alone, it measured 51 ms per step against a 40 ms budget. The last option I know of is a 4-bit re-export of base, which needs a multi-GB checkpoint download; I haven't done it.

**Environment:** `sc_env` is built from the updated `environment.yml`, and `scripts/setup_env.sh` rebuilds it.
- **MLX:** pinned to 0.31.2, because the published MRT2 models won't load on 0.32.2.
- **mediapipe:** now 1.0.1, which is what MRT2's numpy 2 requirement allows. That release no longer includes the hand-tracking API `laptop.py` uses, so the old `sc-laptop` gesture UI won't run in this env.
- **Ultralytics (YOLO):** AGPL-3.0 licensed, which matters if this code is ever distributed.

Everything is uncommitted on `main`. Say if you'd like it on a branch.

**New files:**
- `superconductor/collab.py`
- `superconductor/collab.toml`
- `superconductor/magenta_local.py`
- `superconductor/object_tracking/` (tracker, identity, embedder, library, commands, calibration, mapper)
- `library/` (object library)
- `scripts/simulate_tracking.py`, `tests/test_object_tracking.py`
- `scripts/setup_env.sh`
