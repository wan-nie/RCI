#!/usr/bin/env bash
# =============================================================================
# Install local QM packages managed as Git submodules:
# SOLAX + clic_clib + clic
#
# The parent repository pins the exact commit for every submodule.
# This script initializes those submodules, then builds and installs them.
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Color output helpers
# --------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Retry helper
# Usage: retry <max_attempts> <delay_seconds> <command> [args...]
# --------------------------------------------------------------------------
retry() {
    local max_attempts=$1
    local delay=$2
    shift 2

    local attempt=1

    until "$@"; do
        if (( attempt >= max_attempts )); then
            error "Command failed after ${max_attempts} attempts: $*"
        fi

        warn "Attempt ${attempt}/${max_attempts} failed. Retrying in ${delay}s..."
        sleep "${delay}"
        ((attempt++))
    done
}

# --------------------------------------------------------------------------
# Locate the repository root, independent of the current working directory
# --------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="${ROOT_DIR}/packages"

# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------
info "Checking prerequisites..."

command -v git >/dev/null 2>&1 || \
    error "git not found. Please install git first."

command -v python >/dev/null 2>&1 || \
    error "python not found. Please activate or install Python first."

python -m pip --version >/dev/null 2>&1 || \
    error "pip is not available for the current Python environment."

command -v gcc >/dev/null 2>&1 || \
    error "gcc not found. Please install GCC first."

command -v g++ >/dev/null 2>&1 || \
    error "g++ not found. Please install G++ first."

# --------------------------------------------------------------------------
# Initialize Git submodules
#
# This checks out the exact commits recorded by the current RCI commit.
# It does NOT update packages to the latest upstream main branch.
# --------------------------------------------------------------------------
cd "${ROOT_DIR}"

[[ -f ".gitmodules" ]] || error \
    ".gitmodules not found. Please clone this repository with Git rather than downloading a source ZIP."

info "Downloading pinned Git submodules..."
retry 5 10 git submodule update --init --recursive

# --------------------------------------------------------------------------
# Verify required submodules
# --------------------------------------------------------------------------
for pkg in SOLAX clic_clib clic; do
    [[ -d "${PACKAGES_DIR}/${pkg}" ]] || \
        error "Required submodule is missing: packages/${pkg}"
done

# --------------------------------------------------------------------------
# SOLAX
# --------------------------------------------------------------------------
info "Installing SOLAX..."

# Compatibility fallback for old SOLAX commits without packaging metadata.
# This generated file stays inside the SOLAX submodule and is not committed
# to the RCI parent repository.
if [[ ! -f "${PACKAGES_DIR}/SOLAX/pyproject.toml" && \
      ! -f "${PACKAGES_DIR}/SOLAX/setup.py" ]]; then

    warn "SOLAX has no pyproject.toml or setup.py; generating a local setup.py."

    cat > "${PACKAGES_DIR}/SOLAX/setup.py" <<'EOF'
from setuptools import find_packages, setup

setup(
    name="solax",
    packages=find_packages(),
)
EOF
fi

(
    cd "${PACKAGES_DIR}/SOLAX"
    pip install --no-deps -e .
)

# --------------------------------------------------------------------------
# clic_clib
# --------------------------------------------------------------------------
info "Cleaning stale macOS artifacts in clic_clib..."

find "${PACKAGES_DIR}/clic_clib" -name "*darwin*.so" -delete
find "${PACKAGES_DIR}/clic_clib" -name "*.dylib" -delete

info "Building and installing clic_clib (OpenMP + O3)..."

(
    cd "${PACKAGES_DIR}/clic_clib"

    CC=gcc \
    CXX=g++ \
    CFLAGS="-O3 -fopenmp -march=native" \
    CXXFLAGS="-O3 -fopenmp -march=native" \
    LDFLAGS="-fopenmp" \
    pip install --no-deps -e .
)

# --------------------------------------------------------------------------
# clic
# --------------------------------------------------------------------------
info "Building and installing clic (OpenMP + O3)..."

(
    cd "${PACKAGES_DIR}/clic"

    CC=gcc \
    CXX=g++ \
    CFLAGS="-O3 -fopenmp -march=native" \
    CXXFLAGS="-O3 -fopenmp -march=native" \
    LDFLAGS="-fopenmp" \
    pip install --no-deps -e .
)

# --------------------------------------------------------------------------
# Verify local package imports
# --------------------------------------------------------------------------
info "Verifying local package imports..."

python - <<'PY'
import solax
import clic_clib
import clic

print(f"solax:     {solax.__file__}")
print(f"clic_clib: {clic_clib.__file__}")
print(f"clic:      {clic.__file__}")
PY

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo ""
info "============================================================"
info "All local QM packages installed successfully."
info "============================================================"
