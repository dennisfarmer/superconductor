# Changelog

Releases are cut from the `client` branch: **Actions** → **release** → **Run workflow**. That tags the current `client` and `server` commits and publishes

- `ghcr.io/dennisfarmer/superconductor_client:client`
- `ghcr.io/dennisfarmer/superconductor_server:server`

The release list keeps every version's commits. GHCR keeps those two images and the two dev images (`:client-dev`, `:server-dev`). Write notes for the next release under Unreleased, on this branch and on `client`. After it ships, move those bullets under a version heading.

## [Unreleased]

- Playback (`/stream/playback`) and input (`/stream/input`) are separate websockets. One player on playback; the client steers generation on input.
- The release image is `server_container/Dockerfile`. macOS setup uses `environment_macbook.yml`; lh2300 uses `requirements_lighthouse.txt`. Checkpoints stay on scratch (`MAGENTA_HOME`).
- `make sif` pulls the latest release image. `make sif-dev` pulls the dev image pushed with `make image-push`.
- `make server` runs setup when the environment is missing or its inputs changed. `make server-singularity` pulls the release image when `superconductor_server.sif` is not in the checkout.

## Release workflow

How images and releases work. The workflow file is `.github/workflows/release.yml` on the `client` branch (the default branch). This branch is the server half of that release: the workflow checks it out for the server image and for this changelog. GitHub shows **Run workflow** on `client`.

**Actions** → **release** → **Run workflow** checks out the current `client`, `server`, and `atlas` commits. Pushing a branch does not build an image. Each `CHANGELOG.md` needs a `## [Unreleased]` section with at least one bullet; those bullets are the release notes. After it ships, move the bullets under a version heading on both branches.

The release is `vX.Y.Z` (next patch, or minor/major if chosen on the button; the first is `v0.1.0`). The git tag points at the client commit. The notes and the `release.json` asset record the client, server, and atlas commits. The release list keeps every version. A later release moves the two release image tags; the dev tags stay until the next laptop push.

| Image | Tag | Published by | Pull |
|---|---|---|---|
| Client release | `ghcr.io/dennisfarmer/superconductor_client:client` | the release button | `make sif` in the client checkout, and `make client-singularity` when `superconductor_client.sif` is absent |
| Server release | `ghcr.io/dennisfarmer/superconductor_server:server` | the same button | `make sif`, and `make server-singularity` when `superconductor_server.sif` is absent |
| Client dev | `ghcr.io/dennisfarmer/superconductor_client:client-dev` | `make image-push` in the client checkout | `make sif-dev` there, and `make client-singularity-dev` when `superconductor_client-dev.sif` is absent |
| Server dev | `ghcr.io/dennisfarmer/superconductor_server:server-dev` | `make image-push` here | `make sif-dev`, and `make server-singularity-dev` when `superconductor_server-dev.sif` is absent |

`make image` and `make image-push` publish the dev tag. `make sif` and `make sif-dev` always pull (`singularity pull --force`), which is how a `.sif` already on disk picks up a newer tag. The run targets pull only when their `.sif` is absent, and they do not run `make setup`.

`make server` runs `make setup` when the environment is missing or the setup inputs changed: `environment_macbook.yml` on macOS, or `requirements_lighthouse.txt` on lh2300. The stamp is `.cache/setup-stamp`. When it matches and the environment and checkpoints are on disk, the check is a few file tests. `make setup` itself still runs the installer. `make client` in the client checkout does the same for that environment. Singularity runs leave the checkpoints on scratch (`MAGENTA_HOME`); the image does not contain them.
