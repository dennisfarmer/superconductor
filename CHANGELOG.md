# Changelog

Releases are cut by hand: **Actions** → **release** → **Run workflow**. That tags the current `client` and `server` commits and publishes

- `ghcr.io/dennisfarmer/superconductor:client`
- `ghcr.io/dennisfarmer/superconductor_server:server`

The release list keeps every version's commits. GHCR keeps those two images and the two dev images (`:client-dev`, `:server-dev`). Write notes for the next release under Unreleased, on this branch and on `server`. After it ships, move those bullets under a version heading.

## [Unreleased]

## [0.1.0] - 2026-10-24

- Playback and input are two websockets. The client container connects to the server on `/stream/input`.
- Where the music server is lives in `superconductor_client/addresses.toml`: this computer, the SSH tunnel, the cluster, and Docker. `make client` asks which, or take `--where`. Playback and TouchDesigner use the same names.
- Describe stays off unless `make describe` or `DESCRIBE=1`. On the container gateway the conducting page is `/conduct/`.
- Atlas can stop sending harmony or drums, reset that lane once, and leave it for another program. That Atlas commit is the one baked into the client image.
- `make sif` pulls the latest release image. `make sif-dev` pulls the dev image pushed from a laptop with `make image-push`.
- `make client` runs setup when the environment is missing or its inputs changed. `make client-singularity` pulls the release image when `superconductor_client.sif` is not in the checkout.

## Release workflow

How images and releases work. The workflow file is `.github/workflows/release.yml` on this branch (`client`, the default branch). GitHub shows **Run workflow** for that file.

**Actions** → **release** → **Run workflow** checks out the current `client`, `server`, and `atlas` commits. Pushing a branch does not build an image. Each `CHANGELOG.md` needs a `## [Unreleased]` section with at least one bullet; those bullets are the release notes. After it ships, move the bullets under a version heading on both branches.

The release is `vX.Y.Z` (next patch, or minor/major if chosen on the button; the first is `v0.1.0`). The git tag points at the client commit. The notes and the `release.json` asset record the client, server, and atlas commits. The release list keeps every version. A later release moves the two release image tags; the dev tags stay until the next laptop push.

| Image | Tag | Published by | Pull |
|---|---|---|---|
| Client release | `ghcr.io/dennisfarmer/superconductor:client` | the release button | `make sif`, and `make client-singularity` when `superconductor_client.sif` is absent |
| Server release | `ghcr.io/dennisfarmer/superconductor_server:server` | the same button | `make sif` in the server checkout, and `make server-singularity` when `superconductor_server.sif` is absent |
| Client dev | `ghcr.io/dennisfarmer/superconductor:client-dev` | `make image-push` here | `make sif-dev`, and `make client-singularity-dev` when `superconductor_client-dev.sif` is absent |
| Server dev | `ghcr.io/dennisfarmer/superconductor_server:server-dev` | `make image-push` in the server checkout | `make sif-dev` there, and `make server-singularity-dev` when `superconductor_server-dev.sif` is absent |

`make image` and `make image-push` publish the dev tag. `make sif` and `make sif-dev` always pull (`singularity pull --force`), which is how a `.sif` already on disk picks up a newer tag. The run targets pull only when their `.sif` is absent, and they do not run `make setup`.

`make client` runs `make setup` when the environment is missing or the setup inputs changed: `environment.yml` and the macOS package pins, or `client_container/requirements.txt` on lh2300, plus `pyproject.toml`. The stamp is `.cache/setup-stamp`. When it matches and the environment is on disk, the check is a few file tests. `make setup` itself still runs the installer. `make server` in the server checkout does the same for that environment.
