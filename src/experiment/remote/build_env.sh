#!/usr/bin/env bash
set -euo pipefail

# experiment/remote/build_env.sh
# Build or check shared Python environment
# Usage: bash build_env.sh <root> <name> <lockhash> <snapshot_dir> [extra ...]

ROOT="${1:?ROOT required}"
NAME="${2:?NAME (cpu|gpu) required}"
LOCKHASH="${3:?LOCKHASH required}"
SNAPSHOT="${4:?snapshot_dir required}"
shift 4
# Guarded rather than EXTRAS=("$@"): macOS bash 3.2 (the local-mode tests) treats an
# empty array under `set -u` as an unbound variable.
EXTRAS=()
if [ $# -gt 0 ]; then EXTRAS=("$@"); fi

ENVDIR="$ROOT/envs/$NAME"
VENV="$ENVDIR/.venv"
STAMP="$ENVDIR/lock.sha256"

export PATH="$HOME/.local/bin:$PATH"

# Take flock if available to avoid concurrent syncs on the same environment
LOCK_FILE="$ROOT/envs/.$NAME.lock"
if command -v flock &>/dev/null; then
  exec 3>"$LOCK_FILE"
  flock 3
fi

# Check if environment is up to date
if [[ -f "$STAMP" ]] && [[ -x "$VENV/bin/python" ]]; then
  if [[ "$(cat "$STAMP")" == "$LOCKHASH" ]]; then
    echo ">> env $NAME is up to date" >&2
    echo "VENV=$VENV"
    exit 0
  fi
fi

# (Re)build environment
if ! command -v uv &>/dev/null; then
  echo "FATAL: uv not found" >&2
  exit 1
fi

# Build uv sync arguments with extras
mkdir -p "$ENVDIR"
declare -a SYNC_ARGS=(--frozen --no-install-workspace)
for extra in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do
  SYNC_ARGS+=(--extra "$extra")
done

# Run uv sync from snapshot directory
# Note: --no-install-workspace because the venv is shared by every commit, so
# installing the project - or the harness, a workspace member - would pin it to
# whichever snapshot built it first. Jobs instead run $VENV/bin/python with the
# snapshot's own source dirs on PYTHONPATH.
( cd "$SNAPSHOT" && UV_PROJECT_ENVIRONMENT="$VENV" uv sync "${SYNC_ARGS[@]}" >&2 )

# The atari extra additionally needs the ALE PR #707 wheel force-installed over the
# PyPI build; GSP_VENV points that script at this shared env instead of $PROJECT/.venv.
has_extra() {
  local want="$1" have
  for have in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do
    [[ "$have" == "$want" ]] && return 0
  done
  return 1
}

if has_extra atari; then
  if has_extra cuda; then
    GSP_VENV="$VENV" bash "$SNAPSHOT/scripts/install_ale_wheel.sh" --cuda >&2
  else
    GSP_VENV="$VENV" bash "$SNAPSHOT/scripts/install_ale_wheel.sh" >&2
  fi
fi

# Write stamp only after everything succeeded
echo "$LOCKHASH" > "$STAMP"

# Output
echo "VENV=$VENV"
