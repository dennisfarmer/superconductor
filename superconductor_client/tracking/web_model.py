"""The camera page's detector model: the [tracking] YOLOE model (yoloe-11s-seg.pt), with
the [tracking] classes built in, exported to ONNX for onnxruntime-web (vision_js).

The seg model is exported as is, the same model the native client runs; the page uses
its boxes and class scores and ignores the mask outputs. The input size is fixed at
480x288 (what Ultralytics uses for a 16:9 frame at imgsz 480). The classes are written
next to the model (yoloe.json); when [tracking] classes change, the model is exported again.

    python -m superconductor_client.tracking.web_model [--config collab.toml]     (make web-model)
"""
import json
import shutil
import tempfile
from pathlib import Path

import click

INPUT = (288, 480)  # height, width


def _info(model_name, classes):
    return {"model": Path(model_name).name, "classes": list(classes), "input": list(INPUT), "head": "seg"}


def ensure_web_model(model_name, classes, web_dir):
    """`web_dir`/models/yoloe.onnx for `model_name` with `classes`, exported if missing or
    made for other classes. Returns False (the page then sends plain frames and YOLO
    runs in the client process) if the model isn't a YOLOE model or the export fails."""
    models = Path(web_dir) / "models"
    onnx, info_path = models / "yoloe.onnx", models / "yoloe.json"
    want = _info(model_name, classes)
    try:
        if onnx.exists() and json.loads(info_path.read_text()) == want:
            return True
    except (OSError, ValueError):
        pass
    if "yoloe" not in Path(model_name).name.lower() or not classes:
        print(f"browser detection needs a YOLOE model with classes ([tracking] model = {model_name}); "
              "detecting in this process")
        return False
    print(f"exporting the browser's detector ({want['model']}, classes {classes}) to {onnx} ...")
    try:
        export(model_name, classes, onnx)
    except Exception as e:  # any failure leaves compatibility mode working
        print(f"export failed ({type(e).__name__}: {e}); detecting in this process")
        return False
    info_path.write_text(json.dumps(want))
    return True


def export(model_name, classes, out):
    from ultralytics import YOLOE

    model = YOLOE(model_name)
    model.set_classes(list(classes))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        pt = Path(tmp) / "yoloe-web.pt"
        model.save(pt)
        exported = YOLOE(str(pt)).export(format="onnx", imgsz=INPUT, simplify=True)
        shutil.move(exported, out)


@click.command()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True),
              help="collab.toml to read [tracking] model / classes and [web] web_dir from")
def main(config_path):
    from superconductor_client.tracking.collab import DEFAULT_CONFIG, load_config, tracking_path

    cfg, _ = load_config(config_path or DEFAULT_CONFIG)
    t = cfg["tracking"]
    web_dir = tracking_path(cfg.get("web", {}).get("web_dir", "var/web"))
    if not ensure_web_model(t["model"], t.get("classes") or [], web_dir):
        raise SystemExit(1)
    print(f"ok: {Path(web_dir) / 'models' / 'yoloe.onnx'}")


if __name__ == "__main__":
    main()
