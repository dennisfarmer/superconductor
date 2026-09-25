# Quantizing mrt2_base to 4-bit (GPTQ) on the cluster

## Selected versions

| Component | Version | Why |
|---|---|---|
| MLX (cluster) | **`mlx[cpu]==0.32.0`** | The `aimusic_project` GPU is a **Tesla V100** (Volta, compute capability 7.0, 16 GB). `mlx[cuda12]==0.32.0` installs, but the export crashed with `cudaGraphAddKernelNode ... no kernel image is available for execution on the device`: the prebuilt mlx CUDA wheels have no GPU code for Volta. 16 GB is also below GPTQ's ~20 GB peak. The CPU backend avoids both problems, and the node's RAM is plenty. (Also, `mlx-cuda-12` was never published for 0.31.2, so `mlx[cuda]==0.31.2` can't be installed anywhere.) |
| MLX (Mac) | **`mlx==0.31.2`** (unchanged, as pinned in `sc_env`) | With the split workflow (below), only a `.safetensors` weights file crosses between machines, and the `.mlxfn` is built on the Mac. So the Mac's MLX version doesn't need to match the cluster's. (0.32.2 can't load the published MRT2 models: `[import_function] Invalid string size`.) |
| magenta-rt | **`2.0.3`** | Same version as the Mac env. |
| Python (container) | **3.12** (`python:3.12-bookworm`) | Matches the Mac env; mlx ships cp312 wheels. |
| Container runtime | **Singularity** | Available on Lighthouse (Apptainer isn't). |
| Base image | **`docker://python:3.12-bookworm`** | Debian 12, glibc 2.36. All mlx Linux wheels are `manylinux_2_35` and need glibc ≥ 2.35, but Lighthouse (RHEL 8) has glibc 2.28, so pip outside a container reports "from versions: none". A conda env doesn't help, since glibc is a system library, and conda-forge's mlx is years out of date (latest 0.9.0). |

About the backend extras: `mlx[cpu]` installs `mlx-cpu`, which needs no GPU and no `--nv`. `mlx[cuda12]` installs `mlx-cuda-12`, MLX's NVIDIA backend built against CUDA 12, bundling its CUDA libraries as pip packages; only the driver comes from the node, via `--nv`. `mlx[cuda12]` is only worth trying on a newer GPU than the V100 (Ampere/A100 or later, compute capability 8.0+) with at least ~24 GB of GPU memory. Check with `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv`.

Tested on Lighthouse (`aimusic_project` partition): driver 580.159.03 (CUDA 13.0), Tesla V100-PCIE-16GB. `mlx[cuda12]==0.32.0` installs, but crashes at the first GPU kernel. The JAX warning `An NVIDIA GPU may be present ... CUDA-enabled jaxlib is not installed` is harmless: magenta-rt only imports a small `flax` utility for loading weights, and JAX isn't used for the MLX export.

## Background

Probably yes. The CPU architecture difference shouldn't matter by itself, with conditions I haven't verified.

**Why it should work:** a `.mlxfn` file is MLX's saved compute graph with the weights inside. It isn't compiled machine code; the GPU kernels are built when the file is loaded on the Mac. So an export made on a Linux node should load on your Mac, as long as:

1. **The MLX versions match.** We already saw that published models made with an older MLX fail on 0.32.2. *(Superseded: with the split workflow below, the `.mlxfn` is built on the Mac, so versions don't need to match.)*
2. **MLX runs on the cluster at all.** On Linux it has a CUDA GPU backend (`mlx[cuda12]`) and a CPU-only backend (`mlx[cpu]`). The V100 isn't supported by the CUDA wheels, so we use the CPU backend. GPTQ calibration runs the full model about 128 times, which is slower on CPU (rough guess: minutes to under an hour, depending on core count), but it's a one-off job. On Lighthouse it needs a container (see above).
3. **Every operation the export uses is supported on the Linux backend.** The model is plain MLX code, so this is likely, but nobody may have tried the magenta-rt export there.

The cluster's big RAM fixes the GPTQ memory problem, and the checkpoint download happens there instead of on your disk.

**Memory (estimated from the code, not measured):** the peak is ~20 GB of RAM: the fp32 model (~9.8 GB) plus GPTQ's per-layer Hessians (~10 GB, all held at once). Disk use is ~21 GB: the 9.84 GB checkpoint, a ~10 GB Hessians file and the ~1.5 GB export. Request `--mem=64G`, and use scratch space for the files.

## Tiny-test result: the export must happen on the Mac

The tiny test export from the cluster (`quantized/tiny_test.mlxfn`) **loads and runs on the Mac**, on both mlx 0.31.2 and 0.32.0, and produces finite audio. But it runs at **~1,190 ms/frame**. The same tiny config exported on the Mac runs at **26 ms/frame**. An `.mlxfn` records the device each op runs on: a CPU-backend export runs on the Mac's CPU, even when the GPU is forced with `mx.set_default_device(mx.gpu)` / `mx.stream(mx.gpu)`. So a stock `mrt mlx export` on the cluster can't produce a fast Mac model.

**Fix: split the work** with `scripts/mrt2_gptq_split.py`:
- **`quantize` (cluster, CPU backend):** loads the fp32 checkpoint and runs GPTQ. If `--hessians` is given, it reuses a saved `*_hessians.safetensors` and skips calibration. It then saves **all model weights** to one `.safetensors` file (~1.5 GB for 4-bit mrt2_base) and exits, with no `.mlxfn` export.
- **`export` (Mac, Metal GPU):** builds the same model with random weights and runs the stock `nn.quantize()`. GPTQ calls the same function first, so the 4-bit layer structure is identical. It then overwrites every weight with the cluster's file (`load_weights(..., strict=True)`) and exports the `.mlxfn` on the GPU.

Both steps call the stock `magenta_rt.mlx.export.main` with one function patched, so the model is built exactly as `mrt mlx export` builds it. Tested end to end on the tiny config: the quantize step ran with MLX forced to the CPU, and the export ran on the Mac. The result ran at **15–18 ms/frame**. Reusing a saved Hessians file was tested as well.

## Steps

See `SINGULARITY.md` for more on the container setup, bind mounts, and an `sbatch` script for batch jobs.

```bash
# --- container setup (pull on a login node; compute nodes may not have internet) ---
module load singularity
export SINGULARITY_CACHEDIR=/scratch/aimusic_project_root/aimusic_project/$USER/.singularity   # keep image layers off your home quota
singularity pull py312.sif docker://python:3.12-bookworm   # Debian 12, glibc 2.36

# --- on a compute node (CPU backend: no GPU needed; more cores = faster) ---
#   salloc --account=aimusic_project --partition=aimusic_project --cpus-per-task=16 --mem=64G --time=03:00:00
singularity shell py312.sif

# inside the container (the venv only works inside it, since its Python comes from the image):
python -m venv ~/mrt-export && source ~/mrt-export/bin/activate
pip install "mlx[cpu]==0.32.0" "magenta-rt==2.0.3" flax
#   (if mlx-cuda-12 was installed earlier: pip uninstall -y mlx-cuda-12 first)
python -c "import mlx.core as mx; print(mx.__version__, mx.default_device())"   # expect: 0.32.0 Device(cpu, 0)
export MAGENTA_HOME=$PWD/magenta

mrt models init                      # MusicCoCa, used by GPTQ calibration prompts
mrt checkpoints download mrt2_base   # -> $MAGENTA_HOME/magenta-rt-v2/checkpoints/ (9.84 GB)

# copy scripts/mrt2_gptq_split.py from the repo to the cluster, then:
python mrt2_gptq_split.py quantize --weights mrt2_base_gptq4.safetensors \
    --hessians exports/mrt2_base_4bit_gptq/mrt2_base_hessians.safetensors
#   --hessians reuses the Hessians from an earlier `mrt mlx export --quantize-method gptq`
#   run (skips calibration). Without an existing file, it calibrates and saves one.
```

Then on the Mac:

```bash
# copy mrt2_base_gptq4.safetensors (~1.5 GB) back from the cluster, then:
conda activate sc_env
python scripts/mrt2_gptq_split.py export --weights mrt2_base_gptq4.safetensors --output-dir quantized
#   -> quantized/mrt2_base/mrt2_base.mlxfn (+ _state.safetensors)

# benchmark it alone (no webcam, no audio): needs < 40 ms/frame for real time
python -c "
import time; from pathlib import Path
from magenta_rt import paths, MagentaRT2StdMlxfn
from magenta_rt.config import MUSICCOCA
paths.models_dir = lambda: Path('quantized').resolve()
mrt = MagentaRT2StdMlxfn(size='mrt2_base')
emb = mrt.embed_style('disco funk', use_mapper=True)
wav, state = mrt.generate(conditioning={MUSICCOCA.key: emb}, frames=25)
t0 = time.time(); wav, state = mrt.generate(conditioning={MUSICCOCA.key: emb}, frames=100, state=state)
print(f'{(time.time() - t0) / 100 * 1000:.1f} ms/frame')"

# then with the webcam + tracking:
sc-collab --model mrt2_base --model-dir quantized
```

Notes, checked against the magenta-rt 2.0.3 code:
- Set `MAGENTA_HOME` once and don't pass `--download-path`. The export reads checkpoints from `$MAGENTA_HOME/magenta-rt-v2/checkpoints/`, which is also where the download goes by default.
- GPTQ calibration embeds text prompts with MusicCoCa, so run `mrt models init` first.
- The Mac export writes `<output-dir>/<output-name>/<output-name>.mlxfn`. The output name defaults to `mrt2_base`, a name MRT2's loader accepts. `sc-collab --model-dir quantized` (or `model_dir = "quantized"` in `collab.toml`) loads it instead of the stock `mrt2_base`.
- The Mac export builds the model with random weights before overwriting them. The random fp32 weights are converted to bf16 and then quantized, so the peak is roughly the size of the fp32 model (~10 GB, estimated). That should fit in 18 GB of RAM; close other heavy apps first.
- The `.mlxfn` the cluster's stock GPTQ export produces is CPU-bound and not useful on the Mac. Only the `mrt2_base_hessians.safetensors` from that run is worth keeping, for `--hessians`.
