#!/usr/bin/env bash
set -euo pipefail

# experiment/remote/delete_runs.sh
# Delete runs from every database under an experiment's results dir, mirrors too
# Usage: bash delete_runs.sh <results_dir> <component> <run_id>...

python3 - "$@" <<'PY'
import sqlite3
import sys
from pathlib import Path

results, component, *run_ids = sys.argv[1:]
placeholders = ", ".join("?" for _ in run_ids)
where = f"FROM runs WHERE run_id IN ({placeholders})"
paths = [*Path(results).rglob(f"{component}.db")]
paths += Path(results).rglob(f"*.parts/{component}/part-*.db")
removed = set()
for path in sorted(paths):
    conn = sqlite3.connect(str(path), timeout=60.0)
    try:
        found = conn.execute(f"SELECT run_id {where}", run_ids)
        removed.update(run for (run,) in found)
        conn.execute(f"DELETE {where}", run_ids)
        conn.commit()
    finally:
        conn.close()
print(f"REMOVED={len(removed)}")
PY
