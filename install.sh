#!/usr/bin/env bash
# Install CPU sglang + this plugin into a fresh venv.
#
# Idempotent: reuses an existing venv at $VENV_DIR and an existing sglang
# checkout at $SGLANG_REPO. Re-runs the build steps in either case (they
# no-op if up-to-date).
#
# Environment overrides:
#     VENV_DIR     -- venv path                  (default: ./.venv)
#     SGLANG_REPO  -- where to clone sglang into (default: ./sglang)
#     SGLANG_TAG   -- which sglang tag to build  (default: v0.5.11)
#
# Requires: uv, git, a C++ toolchain (for sgl-kernel).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
SGLANG_REPO="${SGLANG_REPO:-${REPO_ROOT}/sglang}"
SGLANG_TAG="${SGLANG_TAG:-v0.5.11}"

log() { echo "[install.sh] $*"; }
die() { echo "[install.sh] error: $*" >&2; exit 1; }

command -v uv  >/dev/null 2>&1 || die "uv not on PATH; see https://docs.astral.sh/uv/"
command -v git >/dev/null 2>&1 || die "git not on PATH"

# Swap pyproject_<variant>.toml in for pyproject.toml around a command and
# restore the original even if the command fails. $1=dir, then the rest is
# the shell command to run inside it.
swap_pyproject_and_run() {
    local dir="$1"; shift
    pushd "${dir}" >/dev/null
    cp pyproject.toml pyproject.toml.installsh.bak
    cp pyproject_cpu.toml pyproject.toml
    local rc=0
    ( "$@" ) || rc=$?
    mv -f pyproject.toml.installsh.bak pyproject.toml
    popd >/dev/null
    return "${rc}"
}

# --- 1. sglang checkout ---
if [[ ! -d "${SGLANG_REPO}" ]]; then
    log "cloning sglang ${SGLANG_TAG} into ${SGLANG_REPO}"
    git clone --depth 1 --branch "${SGLANG_TAG}" \
        https://github.com/sgl-project/sglang.git "${SGLANG_REPO}"
else
    log "reusing existing sglang checkout at ${SGLANG_REPO}"
fi

[[ -f "${SGLANG_REPO}/python/pyproject_cpu.toml" ]] || \
    die "${SGLANG_REPO} does not look like a sglang checkout (no python/pyproject_cpu.toml)"

# --- 2. venv ---
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "creating Python 3.12 venv at ${VENV_DIR}"
    uv venv --python 3.12 "${VENV_DIR}"
else
    log "reusing existing venv at ${VENV_DIR}"
fi

PY="${VENV_DIR}/bin/python"
export UV_LINK_MODE=copy

# --- 3. CPU torch + build deps ---
log "installing CPU torch + kernel build deps"
uv pip install --python "${PY}" \
    --index https://download.pytorch.org/whl/cpu \
    --index https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    'torch==2.9.0' scikit-build-core 'setuptools>=64' wheel ninja cmake

# --- 4. sglang-cpu ---
log "building sglang-cpu"
swap_pyproject_and_run "${SGLANG_REPO}/python" \
    uv pip install --python "${PY}" --no-build-isolation \
        --index https://download.pytorch.org/whl/cpu \
        --index https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        .

# --- 5. sglang-kernel-cpu (compiles C++, ~40 s) ---
log "building sglang-kernel-cpu (compiles C++; may take ~40 s)"
swap_pyproject_and_run "${SGLANG_REPO}/sgl-kernel" \
    uv pip install --python "${PY}" --no-build-isolation .

# --- 6. plugin editable ---
log "installing sglang-tenstorrent editable from ${REPO_ROOT}"
uv pip install --python "${PY}" -e "${REPO_ROOT}" --no-deps

# --- 7. smoke check ---
log "smoke check"
SGLANG_TENSTORRENT_MOCK=1 "${PY}" -c \
    "import sglang_tenstorrent; print('activate ->', sglang_tenstorrent.activate())"

log "done. activate the venv with: source ${VENV_DIR}/bin/activate"
