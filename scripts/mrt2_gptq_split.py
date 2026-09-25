"""GPTQ-quantize Magenta RT 2 on the cluster, export the .mlxfn on the Mac.

An .mlxfn records the device each op runs on, so an export made with MLX's CPU
backend (the only one that works on Lighthouse's V100s) runs on the CPU when
loaded on the Mac: ~45x slower than a Mac-made export. So split the work:

  quantize (cluster, CPU backend):
      loads the fp32 checkpoint, quantizes it with GPTQ (`--method gptq`,
      reusing a saved Hessians file if given, which skips calibration) or plain
      round-to-nearest (`--method rtn`), and saves every model weight to a
      .safetensors file (~1.5 GB for mrt2_base at 4-bit). No .mlxfn export.
      Loading the fp32 checkpoint peaks at ~24 GB of RAM (measured on the Mac,
      where it was killed), so this step needs the cluster.

  export (Mac, Metal):
      builds the same model with *lazy* random weights (never computed, so the
      ~10 GB fp32 model is never materialized), quantizes it (creating the same
      4-bit layer structure GPTQ uses), overwrites all weights with the
      cluster's file (strict key/shape match), and exports the .mlxfn on the GPU.

Both steps run magenta_rt.mlx.export.main with one function patched, so the
model construction is exactly the stock `mrt mlx export` path.

usage:
  # cluster (inside the Singularity container, MAGENTA_HOME set, checkpoint downloaded)
  python scripts/mrt2_gptq_split.py quantize --weights mrt2_base_gptq4.safetensors \
      --hessians exports/mrt2_base_4bit_gptq/mrt2_base_hessians.safetensors
  python scripts/mrt2_gptq_split.py quantize --method rtn --weights mrt2_base_rtn4.safetensors

  # Mac
  python scripts/mrt2_gptq_split.py export --weights mrt2_base_gptq4.safetensors \
      --output-dir quantized
"""
import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def lazy_random_init():
    """Build the random-init model without computing its weights.

    The stock path runs a dummy forward pass and then mx.eval()s every
    parameter (~10 GB fp32 for mrt2_base). Skipping that eval keeps the random
    weights as unevaluated graph nodes; they are replaced by loaded weights
    before anything needs them.
    """
    from sequence_layers.mlx import export as sl_export
    from sequence_layers.mlx.types import Sequence

    def materialize_without_eval(model, batch_size, input_spec, *, constants=None):
        x = Sequence(mx.zeros((batch_size, 1) + input_spec.shape, dtype=input_spec.dtype),
                     mx.ones((batch_size, 1), dtype=mx.bool_))
        state = model.get_initial_state(batch_size, input_spec, constants=constants)
        model.step(x, state, constants=constants)

    sl_export._materialize_deferred = materialize_without_eval


def save_weights(model, path):
    weights = dict(tree_flatten(model.parameters()))
    for v in weights.values():  # one at a time, so lazy weights never coexist in fp32
        mx.eval(v)
    mx.save_safetensors(str(path), weights)
    size = sum(v.nbytes for v in weights.values()) / 1e9
    print(f"saved {len(weights)} arrays ({size:.2f} GB) to {path}")


def quantize(args):
    from magenta_rt.mlx import export, gptq

    if args.skip_restore:
        lazy_random_init()

    if args.method == "rtn":
        original_quantize = nn.quantize

        def rtn_to_file(model, group_size=64, bits=4, **kwargs):
            original_quantize(model, group_size=group_size, bits=bits, **kwargs)
            save_weights(model, args.weights)
            sys.exit(0)  # the .mlxfn is exported on the Mac

        nn.quantize = rtn_to_file
        export.main(restore=not args.skip_restore, model_name=args.model, bits=args.bits,
                    quantize_method="default", output_name=args.output_name,
                    output_dir=args.output_dir, **args.arch)
        return

    def gptq_to_file(model, calibrate_fn, bits=4, group_size=32, block_size=128,
                     max_samples=2048, debug_identity_hessian=False, hessian_save_path=None):
        if args.hessians and Path(args.hessians).exists():
            print(f"loading saved Hessians from {args.hessians} (skipping calibration)")
            hessians = mx.load(str(args.hessians))
        else:
            hessians = gptq.capture_activations(model, calibrate_fn, max_samples,
                                                debug_identity_hessian)
            if hessian_save_path is not None:
                mx.save_safetensors(str(hessian_save_path), hessians)
                print(f"saved Hessians to {hessian_save_path}")
        gptq.gptq_adjust_weights(model, hessians, bits, group_size, block_size)
        save_weights(model, args.weights)
        sys.exit(0)  # the .mlxfn is exported on the Mac

    gptq.gptq_calibrate_and_quantize = gptq_to_file
    export.main(restore=not args.skip_restore, model_name=args.model, bits=args.bits,
                quantize_method="gptq", output_name=args.output_name,
                output_dir=args.output_dir, **args.arch)


def export_(args):
    from magenta_rt.mlx import export

    if mx.default_device() != mx.gpu:
        sys.exit("export must run on a Metal GPU (the device is recorded in the .mlxfn)")

    lazy_random_init()
    original_quantize = nn.quantize

    def quantize_then_load(model, group_size=64, bits=4, **kwargs):
        original_quantize(model, group_size=group_size, bits=bits, **kwargs)
        print(f"loading quantized weights from {args.weights}")
        model.load_weights(str(args.weights), strict=True)
        mx.eval(model.parameters())

    nn.quantize = quantize_then_load
    # restore=False: random init just to build the structure; every weight is
    # then replaced by the cluster's file
    export.main(restore=False, model_name=args.model, bits=args.bits,
                quantize_method="default", output_name=args.output_name,
                output_dir=args.output_dir, **args.arch)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["quantize", "export"])
    parser.add_argument("--weights", required=True, type=Path,
                        help="quantized weights file (written by quantize, read by export)")
    parser.add_argument("--method", choices=["gptq", "rtn"], default="gptq",
                        help="quantize: GPTQ (calibrated) or plain round-to-nearest")
    parser.add_argument("--hessians", type=Path,
                        help="quantize --method gptq: reuse a saved *_hessians.safetensors (skips calibration)")
    parser.add_argument("--model", default="mrt2_base")
    parser.add_argument("--bits", default=4, type=int)
    parser.add_argument("--output-name", default="mrt2_base",
                        help="keep a registry name (mrt2_base / mrt2_small) so MRT2 can load it")
    parser.add_argument("--output-dir", default="quantized")
    parser.add_argument("--skip-restore", action="store_true",
                        help="quantize: random weights instead of the checkpoint (for tests)")
    # architecture overrides, only for small test models; must match on both sides
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--depth-num-layers", type=int)
    # export-only speed options (weights are unaffected)
    parser.add_argument("--num-codebooks", type=int,
                        help="export: RVQ levels generated per frame (default 12; fewer = faster, lower fidelity)")
    parser.add_argument("--num-cfgs", type=int,
                        help="export: guidance batches, 0 = none, 1 = style only, 2 = style + notes (default)")
    args = parser.parse_args()
    args.arch = {k: v for k, v in (("num_layers", args.num_layers),
                                   ("depth_num_layers", args.depth_num_layers),
                                   ("num_codebooks", args.num_codebooks),
                                   ("num_cfgs", args.num_cfgs)) if v is not None}
    quantize(args) if args.step == "quantize" else export_(args)


if __name__ == "__main__":
    main()
