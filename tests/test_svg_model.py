"""SVG extraction and the icon-model backend choice. Does not download weights.

Run with `python tests/test_svg_model.py` (or pytest).
"""
import os

from superconductor_client.describe.svg_model import (
    chat_messages, default_backend, extract_svg, inference_dtype)


def test_extract_svg_keeps_one_element_and_drops_scripts():
    raw = """sure, here it is
```svg
<svg viewBox="0 0 10 10"><script>alert(1)</script>
<circle cx="1" cy="1" r="1" onclick="x()" href="javascript:alert(1)"/>
</svg>
```
and some notes after"""
    svg = extract_svg(raw)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "xmlns=" in svg
    lowered = svg.lower()
    assert "<script" not in lowered and "onclick" not in lowered and "javascript:" not in lowered
    assert "<circle" in lowered


def test_extract_svg_rejects_prose():
    assert extract_svg("no image here") is None
    assert extract_svg("") is None
    assert extract_svg(None) is None


def test_chat_messages_are_the_prompt():
    messages = chat_messages("  a small red drum  ")
    assert messages[0]["role"] == "system" and "SVG" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "a small red drum"}


def test_v100_uses_float16_and_ampere_uses_bfloat16():
    assert inference_dtype(True, 7, False) == "float16"  # V100, sm_70
    assert inference_dtype(True, 8, False) == "bfloat16"
    assert inference_dtype(False, 0, True) == "float16"
    assert inference_dtype(False, 0, False) == "float32"


def test_backend_follows_svg_backend_env():
    old = os.environ.get("SVG_BACKEND")
    try:
        os.environ["SVG_BACKEND"] = "transformers"
        assert default_backend() == "transformers"
        os.environ["SVG_BACKEND"] = "mlx"
        assert default_backend() == "mlx"
        os.environ["SVG_BACKEND"] = ""
        assert default_backend() in ("mlx", "transformers")
    finally:
        if old is None:
            os.environ.pop("SVG_BACKEND", None)
        else:
            os.environ["SVG_BACKEND"] = old


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
