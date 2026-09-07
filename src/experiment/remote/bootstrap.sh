#!/usr/bin/env bash
set -euo pipefail

# experiment/remote/bootstrap.sh
# Initialize cluster environment: directories, bare repo, uv, Python
# Usage: bash bootstrap.sh <root> <project>

ROOT="${1:?ROOT (cluster root dir) required}"
PROJECT="${2:?PROJECT (bare repo name) required}"

# Create directory structure
mkdir -p "$ROOT/runs" "$ROOT/envs" "$ROOT/results" >&2

# Initialize bare repo if absent
BARE="$ROOT/$PROJECT.git"
if [[ ! -d "$BARE" ]]; then
  git init --quiet --bare "$BARE" >&2
fi

# Ensure uv is available
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >&2
fi
if ! command -v uv &>/dev/null; then
  echo "FATAL: uv not found after installation attempt" >&2
  exit 1
fi

UV="$(command -v uv)"

# Install Python 3.13 (failure is warning, not fatal; uv sync can still resolve it)
uv python install 3.13 >&2 || true

# Detect Slurm account candidates: basenames of dirs under ~/projects/, excluding def-sponsor00
declare -a ACCOUNTS=()
for dir in "$HOME"/projects/*/; do
  [[ -d "$dir" ]] || continue
  acct="$(basename "$dir")"
  [[ "$acct" != "def-sponsor00" ]] || continue
  ACCOUNTS+=("$acct")
done
ACCOUNTS_STR="${ACCOUNTS[*]+${ACCOUNTS[*]}}"

# Output
echo "ROOT=$ROOT"
echo "BARE=$BARE"
echo "UV=$UV"
echo "ACCOUNTS=$ACCOUNTS_STR"
