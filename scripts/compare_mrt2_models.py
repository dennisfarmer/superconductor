"""Compare Magenta RT 2 exports (e.g. stock 8-bit vs 4-bit RTN vs 4-bit GPTQ).

For each variant: steady-state generation speed, a WAV per prompt (same seed
across variants, for listening), and a style-adherence score: cosine similarity
between the MusicCoCa embedding of the generated audio and of the text prompt.

Each variant runs in its own process so only one model is in memory at a time.

usage:
  python scripts/compare_mrt2_models.py \
      --variant stock8= --variant rtn4=quantized/rtn4 --variant gptq4=quantized/gptq4
  (an empty path = the stock export in ~/Documents/Magenta/magenta-rt-v2/models)
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

PROMPTS = ["disco funk", "airy wooden nature flute melody", "thundering fiery taiko drums",
           "lush orchestral strings"]


def run_variant(name, model_dir, model, out_dir, seconds, queue):
    import time

    import mlx.core as mx
    import numpy as np
    from magenta_rt import MagentaRT2StdMlxfn, paths
    from magenta_rt.config import MUSICCOCA

    if model_dir:
        paths.models_dir = lambda: Path(model_dir).resolve()
    mrt = MagentaRT2StdMlxfn(size=model)

    # speed: steady state after a warm-up call
    emb = mrt.embed_style(PROMPTS[0], use_mapper=True)
    _, state = mrt.generate(conditioning={MUSICCOCA.key: emb}, frames=25)
    t0 = time.perf_counter()
    mrt.generate(conditioning={MUSICCOCA.key: emb}, frames=100, state=state)
    ms_per_frame = (time.perf_counter() - t0) / 100 * 1000

    scores = {}
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for prompt in PROMPTS:
        mx.random.seed(0)
        text_emb = mrt.embed_style(prompt, use_mapper=True)
        wav, _ = mrt.generate(conditioning={MUSICCOCA.key: text_emb}, frames=int(seconds * 25))
        wav.write(str(out / (prompt.replace(" ", "_") + ".wav")))
        audio_emb = np.asarray(mrt.embed_style(wav), dtype=np.float32).ravel()
        t = np.asarray(mrt.embed_style(prompt), dtype=np.float32).ravel()  # text space, no mapper
        scores[prompt] = float(audio_emb @ t / (np.linalg.norm(audio_emb) * np.linalg.norm(t)))

    queue.put({"variant": name, "ms_per_frame": ms_per_frame,
               "realtime_factor": 40.0 / ms_per_frame, "style_similarity": scores})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True,
                        help="name=model_dir (empty model_dir = stock export)")
    parser.add_argument("--model", default="mrt2_base")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out-dir", default="var/compare")
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    results = []
    for spec in args.variant:
        name, _, model_dir = spec.partition("=")
        print(f"== {name} ({model_dir or 'stock'})", flush=True)
        queue = ctx.Queue()
        p = ctx.Process(target=run_variant,
                        args=(name, model_dir, args.model, args.out_dir, args.seconds, queue))
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"   failed (exit {p.exitcode})")
            continue
        results.append(queue.get())

    print(f"\n{'variant':<10} {'ms/frame':>9} {'x realtime':>11}   style similarity per prompt (higher = closer to prompt)")
    for r in results:
        sims = "  ".join(f"{v:.3f}" for v in r["style_similarity"].values())
        print(f"{r['variant']:<10} {r['ms_per_frame']:>9.1f} {r['realtime_factor']:>11.2f}   {sims}")
    print("prompts:", " | ".join(PROMPTS))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "results.json").write_text(json.dumps(results, indent=1))
    print(f"WAVs and results.json in {args.out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
