#!/usr/bin/env bash
# Install Timeloop + Accelergy into the LAMP_acpl conda environment.
#
# Works without sudo by using conda-forge for build deps and patching the
# gcc-13 incompatibility (missing <cstdint> includes) in Timeloop v3.0.3.
#
# Expected wall time: 5-10 minutes (one-time).
# Re-running is safe: existing symlinks are overwritten in place.

set -euo pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "LAMP_acpl" ]; then
  echo "[install_timeloop] Run 'conda activate LAMP_acpl' first." >&2
  exit 1
fi

ROOT=/home/heeseo/LAMP/acpl
THIRD=$ROOT/third_party
mkdir -p "$THIRD"

echo "[install_timeloop] 1/5 build deps via conda-forge"
conda install -y -c conda-forge scons libconfig yaml-cpp ncurses cmake \
  gcc_linux-64 gxx_linux-64 git >/dev/null

echo "[install_timeloop] 2/5 Accelergy (pure Python)"
if [ ! -d "$THIRD/accelergy" ]; then
  git clone --depth 1 https://github.com/Accelergy-Project/accelergy.git "$THIRD/accelergy"
fi
( cd "$THIRD/accelergy" && pip install -q . )

echo "[install_timeloop] 3/5 Timeloop v3.0.3 (NVlabs)"
if [ ! -d "$THIRD/timeloop_nv" ]; then
  git clone https://github.com/NVlabs/timeloop.git "$THIRD/timeloop_nv"
fi
cd "$THIRD/timeloop_nv"
git fetch --tags --quiet
git -c advice.detachedHead=false checkout v3.0.3 >/dev/null 2>&1
ln -sf ../pat-public/src/pat src/pat

echo "[install_timeloop] 4/5 patch gcc-13 <cstdint> in headers"
FILES=$(grep -rln "std::uint[0-9]\+_t\|std::int[0-9]\+_t" include/ src/ || true)
for f in $FILES; do
  python -c "
import sys, re
p = '$f'
txt = open(p).read()
if '<cstdint>' in txt: sys.exit(0)
m = re.search(r'^#define [A-Z_]+\$', txt, re.MULTILINE) or re.search(r'^#include[^\n]*\$', txt, re.MULTILINE)
pos = (m.end() + 1) if m else 0
open(p,'w').write(txt[:pos] + '\n#include <cstdint>\n' + txt[pos:])
"
done

echo "[install_timeloop] 5/5 scons build"
export CPATH=$CONDA_PREFIX/include:${CPATH:-}
export LIBRARY_PATH=$CONDA_PREFIX/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
scons -j"$(nproc)" >/dev/null

# Expose binaries + libs into the conda env
for f in bin/*;  do ln -sf "$(realpath "$f")" "$CONDA_PREFIX/bin/$(basename "$f")"; done
for f in lib/*;  do ln -sf "$(realpath "$f")" "$CONDA_PREFIX/lib/$(basename "$f")"; done

# Persist runtime LD_LIBRARY_PATH so the binaries find libconfig++/yaml-cpp.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/timeloop_ld.sh" <<'A'
export _LAMP_PREV_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
A
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/timeloop_ld.sh" <<'D'
export LD_LIBRARY_PATH="$_LAMP_PREV_LD_LIBRARY_PATH"
unset _LAMP_PREV_LD_LIBRARY_PATH
D

echo
echo "[install_timeloop] Done."
echo "Re-source the env to pick up LD_LIBRARY_PATH:"
echo "  conda deactivate && conda activate LAMP_acpl"
echo
echo "Sanity:"
echo "  which timeloop-mapper accelergy"
echo "  timeloop-mapper /home/heeseo/LAMP/acpl/third_party/timeloop_nv/configs/mapper/sample.yaml"
