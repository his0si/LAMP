#!/usr/bin/env bash
# Install experiment-only extras into the existing LAMP_acpl env.
# Re-run safely: pip handles already-installed packages idempotently.
set -euo pipefail

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "LAMP_acpl" ]; then
  echo "[install_extras] LAMP_acpl env not active. Run: conda activate LAMP_acpl" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")"/.. && pwd)"
python -m pip install -r "${HERE}/requirements-extra.txt"

echo
echo "[install_extras] Done. Timeloop + Accelergy require a separate install."
echo "  See experiments/README.md §3 (Hardware simulator install)."
