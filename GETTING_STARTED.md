It's ready for you to run. Here's where things stand.

**Run it:**
```bash
conda activate sc_env
sc-collab                      # unsupervised mode (default)
sc-collab --mode calibration   # the old "touch the bulbasaur" flow
```
Or, without activating the environment: `make server` (MRT2 `mrt2_small` server; leave it running) then `make client` in another terminal (webcam + music; `make client-remote` uses the cluster's `mrt2_base` instead), `make vision` (no music), `make test`. Extra options go in `make client ARGS="--mode calibration"`.
Each object gets a permanent id (`#1`, `#2`, …) the first time it's seen. It's cached in the library (`library/<id>/`), so it keeps its id, name and instrument when it leaves the frame and comes back, and after a restart. Click in the window to move the crosshair.

**Objects page:** `make client` opens http://localhost:8467/objects. It lists every object (those in view first) with its picture, id, name, description and instrument:
- **Name:** optional, typed by you.
- **Description:** written once per new object by a Qwen VLM on the describe server (`make describe`, see `../superconductor_describe/README.md`). "Describe again" asks for a new one.
- **Instrument and trigger:** the instrument is suggested from the description, or typed. The trigger sets when it plays: `near` (louder closer to the crosshair), `held` (a hand is on it) or `visible` (while in view). An object without an instrument is silent.
- **Combos:** tick two or more objects, type what they play together and a distance. While they're that close (edge to edge, as a fraction of the frame width), the combo replaces their own instruments; they separate at 1.5× the distance.
- **Delete** moves the object to `library/.trash`; its id isn't reused.

Edits apply immediately and are saved; there's no save step. Keys in the camera window: `[` and `]` shrink or widen the combo distance, `c` recalibrates (calibration mode), `q` quits.

**How recognition works:**
- **Detection:** the detector only finds toys (`"stuffed toy", "toy"`); it never decides who is who.
- **Recognition:** DINOv2 ViT-S/14 (about 5.5 ms per crop on the GPU) decides that. Each new track is embedded 3 times in its first 8 frames and compared with the saved views of every object not currently in view, including objects from earlier sessions. They compete for the match jointly, so two of them can't swap.
- **Periodic checks:** each tracked object is re-checked about once a second, and immediately when it reappears. If it looks more like another object than itself, it is re-identified. This fixes swaps when the tracker revives an old track on the wrong object.
- **Unknown objects:** something that looks unlike every known object becomes a new id. Something in between waits for more evidence.
- **Coasting:** when the detector loses an object (it merged into its neighbour's box, or is half hidden), the object keeps its last box, drawn dashed with "last seen". That lasts 1 s, or indefinitely while it is next to a visible object.

**Library files:** `library/<id>/object.json` (name, description, instrument, trigger), `library/combos.json`, `embeddings.npy` (DINOv2 features), `views/*.jpg` (the view for each feature). Folders from the old name-based layout (`library/bulbasaur/`) are converted on the first run.

**Checking recognition:** `python scripts/eval_reid.py record var/reid.mp4` records a clip; `python scripts/eval_reid.py eval var/reid.mp4 --objects 3` runs it with DINOv2 and with the old `yolo11n-cls` model and reports how many ids each created.

**Testing without the camera:** `python tests/test_object_tracking.py` runs the unit tests. The full pipeline on a scripted scene (hide, reappear, swap, come together, overlap, separate, unknown toy, next session in dimmer light) runs with:
```bash
python scripts/simulate_tracking.py --frame photo.jpg --object bulbasaur:x1,y1,x2,y2 --object chimchar:x1,y1,x2,y2
```
It writes an annotated `var/sim/sim.mp4`.

**Watch for:** `mrt2_base` is the default. In the one run with YOLO on the CPU it managed only 0.74× real time (it needs to be above 1× to keep up), so expect audio dropouts. Run alone, it measured 51 ms per step against a 40 ms budget.

**Environment:** `sc_env` is built from the updated `environment.yml`, and `scripts/setup_env.sh` rebuilds it.
- **MLX:** pinned to 0.31.2, because the published MRT2 models won't load on 0.32.2.
- **mediapipe:** now 1.0.1, which is what MRT2's numpy 2 requirement allows. That release no longer includes the hand-tracking API `laptop.py` uses, so the old `sc-laptop` gesture UI won't run in this env.
- **Ultralytics (YOLO):** AGPL-3.0 licensed, which matters if this code is ever distributed.

Everything is uncommitted on `main`. Say if you'd like it on a branch.

**New files:**
- `superconductor/collab.py`
- `superconductor/collab.toml`
- `superconductor/magenta_remote.py` (with `../superconductor_server/mrt2_server.py`)
- `superconductor/object_tracking/` (tracker, identity, embedder, library, calibration, mapper)
- `superconductor/web_panel.py`, `web_panel.html` (objects page), `describe_client.py`
- `library/` (object library)
- `scripts/simulate_tracking.py`, `tests/test_object_tracking.py`
- `scripts/setup_env.sh`
