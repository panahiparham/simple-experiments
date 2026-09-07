#!/usr/bin/env bash
set -euo pipefail

# experiment/remote/build_env.sh
# Build or check shared Python environment
# Usage: bash build_env.sh <root> <name> <lockhash> <snapshot_dir> <post_sync> \
#          [extra ...]
# post_sync is a snapshot-relative script to run after the sync, or empty for none.

ROOT="${1:?ROOT required}"
NAME="${2:?NAME (cpu|gpu) required}"
LOCKHASH="${3:?LOCKHASH required}"
SNAPSHOT="${4:?snapshot_dir required}"
POST_SYNC="${5?post_sync required (empty for none)}"
shift 5
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

# A project can name a script of its own to run after the sync - installing a build
# uv cannot express, say. It runs from the snapshot, is told the venv it must target
# and the extras that were synced, and decides for itself what to do with them.
if [ -n "$POST_SYNC" ]; then
  HOOK="$SNAPSHOT/$POST_SYNC"
  [ -f "$HOOK" ] || { echo "FATAL: post_sync script not in the snapshot: $POST_SYNC" >&2
                      exit 1; }
  SYNCED_EXTRAS=""
  if [ ${#EXTRAS[@]} -gt 0 ]; then SYNCED_EXTRAS="${EXTRAS[*]}"; fi
  echo ">> post_sync: $POST_SYNC" >&2
  ( cd "$SNAPSHOT" \
    && EXPERIMENT_VENV="$VENV" EXPERIMENT_EXTRAS="$SYNCED_EXTRAS" bash "$HOOK" >&2 )
fi

# Write stamp only after everything succeeded
echo "$LOCKHASH" > "$STAMP"

# Output
echo "VENV=$VENV"
