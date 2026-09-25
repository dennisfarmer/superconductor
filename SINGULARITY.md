# Singularity containers on Lighthouse

Singularity runs a container image as your own user on a cluster node, with no root or Docker daemon needed. The container brings its own Linux userland: its own glibc, Python and system libraries. Your files, the node's CPUs and (with `--nv`) its NVIDIA GPUs stay available inside.

**Why we need it:** Lighthouse runs RHEL 8 with **glibc 2.28**. Many modern Python wheels, including every mlx Linux wheel (`manylinux_2_35`), need glibc ≥ 2.35. Outside a container, pip reports `Could not find a version that satisfies the requirement ... (from versions: none)`. Conda can't fix it: glibc is a system library, not a package in the env. A container based on a newer distro fixes it. `python:3.12-bookworm` is Debian 12, with glibc 2.36.

Lighthouse provides **Singularity** (`module load singularity`); Apptainer, its renamed successor, isn't installed. The commands are the same apart from the name.

## One-time setup

```bash
module load singularity

# Image layers are big; keep the cache (and images) off your home quota.
export SINGULARITY_CACHEDIR=/scratch/aimusic_project_root/aimusic_project/$USER/.singularity
mkdir -p $SINGULARITY_CACHEDIR

# Pull on a LOGIN node: compute nodes may not have internet access.
# Converts a Docker Hub image into a single read-only .sif file.
cd /scratch/aimusic_project_root/aimusic_project/$USER
singularity pull py312.sif docker://python:3.12-bookworm
```

Add the `module load` and `export SINGULARITY_CACHEDIR=...` lines to `~/.bashrc` if you use this often.

## Interactive use

```bash
salloc --account=aimusic_project --partition=aimusic_project --gpus=1 --mem=64G --cpus-per-task=4 --time=02:00:00
module load singularity
singularity shell --nv py312.sif        # opens a shell inside the container
Singularity> nvidia-smi                 # GPU visible thanks to --nv
Singularity> ldd --version | head -1    # glibc 2.36 inside, vs 2.28 outside
```

- **`--nv`** passes the node's NVIDIA driver and GPU devices into the container. Without it, GPU code falls back to CPU or fails. It isn't needed for CPU-only work, such as the mlx export (see the GPU note below).
- `exit` leaves the container.

### GPU note: the aimusic_project partition has a Tesla V100

`nvidia-smi` on the partition shows a **Tesla V100-PCIE-16GB** (Volta, compute capability 7.0), driver 580.159.03 (CUDA 13.0). Prebuilt GPU wheels only include code for certain GPU generations. Those that skip Volta fail with `no kernel image is available for execution on the device`, even though `--nv` and the driver work. mlx's CUDA wheels are one example, so the MRT2 export uses the CPU backend. Check a package's supported GPU generations before assuming a GPU speedup here:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

## Python environments inside the container

The image is read-only, so install packages into a venv under your home or scratch directory:

```bash
Singularity> python -m venv ~/mrt-export
Singularity> source ~/mrt-export/bin/activate
Singularity> pip install "mlx[cpu]==0.32.0" "magenta-rt==2.0.3" flax
```

- **Only use the venv inside the container.** Its `python` points at the container's interpreter, which doesn't exist on the host.
- **Keep one venv per image.** If you switch images, create a fresh venv.
- If packages from `~/.local` (host `pip install --user`) leak in and conflict, set `export PYTHONNOUSERSITE=1`.

## Files and bind mounts

By default Singularity mounts your **home directory**, **the current directory** and `/tmp` into the container, so they appear at the same paths. Other paths, such as `/scratch`, may or may not be mounted depending on the site config. If a scratch path is missing inside the container, bind it explicitly:

```bash
singularity shell --nv --bind /scratch py312.sif
```

Anything written to a bound path persists after the container exits. Anything written elsewhere inside the container is lost, or fails because the image is read-only.

## Batch jobs (non-interactive)

`singularity exec` runs one command in the container and exits, which makes it the right fit for `sbatch`. Example for the 4-bit GPTQ quantize step (see `QUANTIZE.md`; copy `scripts/mrt2_gptq_split.py` into `$WORK` first):

```bash
#!/bin/bash
#SBATCH --job-name=mrt2-gptq
#SBATCH --account=aimusic_project
#SBATCH --partition=aimusic_project
#SBATCH --cpus-per-task=16      # CPU backend: the V100 isn't supported by mlx's CUDA wheels
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=mrt2-gptq-%j.log

module load singularity
WORK=/scratch/aimusic_project_root/aimusic_project/$USER
cd $WORK

singularity exec --bind /scratch py312.sif bash -c '
  set -e
  source ~/mrt-export/bin/activate
  export MAGENTA_HOME=$PWD/magenta
  mrt models init
  mrt checkpoints download mrt2_base
  # GPTQ weights only; the .mlxfn must be exported on the Mac (see QUANTIZE.md)
  python mrt2_gptq_split.py quantize --weights mrt2_base_gptq4.safetensors \
      --hessians exports/mrt2_base_4bit_gptq/mrt2_base_hessians.safetensors
'
```

Submit it with `sbatch mrt2-gptq.sbatch`, and follow progress with `tail -f mrt2-gptq-<jobid>.log`.

## Quick reference

| Command | What it does |
|---|---|
| `singularity pull NAME.sif docker://IMAGE:TAG` | Download a Docker image as a `.sif` file |
| `singularity shell [--nv] IMG.sif` | Interactive shell inside the container |
| `singularity exec [--nv] IMG.sif CMD ...` | Run one command inside the container |
| `singularity run IMG.sif` | Run the image's default command |
| `--nv` | Expose NVIDIA GPUs and driver |
| `--bind /host/path[:/container/path]` | Mount extra host directories |
| `singularity cache clean` | Free space used by `SINGULARITY_CACHEDIR` |

## Other uses for this project

The same approach fits any cluster tool that needs a newer Linux than RHEL 8, such as the MagentaRT server once it's moved to mrt2_base. Pick an image with the right userland, e.g. `docker://nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04` when you need CUDA system libraries rather than pip-bundled ones. Then keep the Python env in a venv next to it.
