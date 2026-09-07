#!/usr/bin/env bash
set -euo pipefail

# experiment/remote/prepare.sh
# Prepare a run directory: extract commit, setup results symlink, write manifest
# Usage: bash prepare.sh <root> <runid> <sha> <label> <exp_reldir> <project>

ROOT="${1:?ROOT required}"
RUNID="${2:?RUNID required}"
SHA="${3:?SHA required}"
LABEL="${4:?LABEL required}"
EXP_RELDIR="${5:?exp_reldir required}"
PROJECT="${6:?PROJECT required}"

BARE="$ROOT/$PROJECT.git"
RUNDIR="$ROOT/runs/$RUNID"

# Hard-fail if bare repo is missing
if [[ ! -d "$BARE" ]]; then
  echo "FATAL: bare repo not found at $BARE. Run setup_cluster.py to initialize." >&2
  exit 1
fi

# Hard-fail if RUNDIR already exists
if [[ -d "$RUNDIR" ]]; then
  echo "FATAL: RUNDIR already exists at $RUNDIR" >&2
  exit 1
fi

# Verify commit is present in bare repo
if ! git --git-dir="$BARE" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  echo "FATAL: commit $SHA not found in bare repo. Did the push land?" >&2
  exit 1
fi

# Create run directory and extract commit
mkdir -p "$RUNDIR/logs"
git --git-dir="$BARE" archive --format=tar "$SHA" | tar -x -C "$RUNDIR" >&2

# Create shared results directory and replace extracted one with symlink
# Every commit's runs accumulate in one place, so the laptop syncs an experiment
# with a single rsync and the harness's run_id dedup resumes across commits.
mkdir -p "$ROOT/results/$LABEL"
RESULTS_LINK_DIR="$RUNDIR/$EXP_RELDIR/results"
if [[ -e "$RESULTS_LINK_DIR" ]] || [[ -L "$RESULTS_LINK_DIR" ]]; then
  rm -rf "$RESULTS_LINK_DIR"
fi
mkdir -p "$(dirname "$RESULTS_LINK_DIR")"
ln -s "$ROOT/results/$LABEL" "$RESULTS_LINK_DIR" >&2

# Write manifest
CREATED=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$RUNDIR/MANIFEST" <<EOF
runid=$RUNID
sha=$SHA
label=$LABEL
created=$CREATED
EOF

# Output
echo "RUNDIR=$RUNDIR"
