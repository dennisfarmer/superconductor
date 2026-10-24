#!/usr/bin/env bash
# Source this on lh2300. Running it as a program cannot leave `module load`
# or the exports in your shell.
#
# The copy make sources is:
#   /scratch/aimusic_project_root/aimusic_project/shared_data/.singularity/singularity-setup.sh
#
# One directory:
#   /scratch/aimusic_project_root/aimusic_project/shared_data/.singularity/
#     cache/               SINGULARITY_CACHEDIR
#     docker-config.json   SINGULARITY_AUTHFILE
#     ghcr.token           one line, a GitHub PAT with read:packages
#
# The GHCR username is dennisfarmer. That is the account that owns the images.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script so the module and the exports stay in your shell:" >&2
  echo "  source $0" >&2
  exit 1
fi

# make runs this from a non-interactive shell, which has not defined `module` yet.
if ! type module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
fi
if ! type module >/dev/null 2>&1; then
  echo "module is not available in this shell." >&2
  return 1
fi
module load singularity

base="/scratch/aimusic_project_root/aimusic_project/shared_data/.singularity"
# GitHub personal access token (classic) with read:packages, chmod 600.
token_file="$base/ghcr.token"

mkdir -p "$base/cache"

export SINGULARITY_CACHEDIR="$base/cache"
export SINGULARITY_AUTHFILE="$base/docker-config.json"

if [[ ! -r "$token_file" ]]; then
  echo "Cannot read $token_file" >&2
  return 1
fi

token=$(tr -d '[:space:]' < "$token_file")
if [[ -z "$token" ]]; then
  echo "$token_file is empty." >&2
  return 1
fi

printf '%s\n' "$token" | singularity registry login \
  --username dennisfarmer \
  --password-stdin \
  --authfile "$SINGULARITY_AUTHFILE" \
  docker://ghcr.io

echo "Cache: $SINGULARITY_CACHEDIR"
echo "Auth:  $SINGULARITY_AUTHFILE"
echo "Logged in to ghcr.io as dennisfarmer."
