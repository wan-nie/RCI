#!/usr/bin/env bash
# =============================================================================
# Install local packages: SOLAX + CLIC_CLIB + CLIC
# Re-run safe + auto-retry on network errors
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Color output helpers
# --------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --------------------------------------------------------------------------
# Retry helper
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
        (( attempt++ ))
    done
}

# --------------------------------------------------------------------------
# pip install wrapper
# --------------------------------------------------------------------------
pip_install() {
    info "pip install $*"
    retry 5 10 pip install "$@"
}

# --------------------------------------------------------------------------
# git clone wrapper
# --------------------------------------------------------------------------
git_clone() {
    local repo=$1
    info "Cloning ${repo}..."
    retry 5 10 git clone "${repo}"
}

# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------
info "Checking prerequisites..."
command -v git &>/dev/null || error "git not found. Please install git first."
command -v gcc &>/dev/null || error "gcc not found. Please install GCC first."

# --------------------------------------------------------------------------
# Prepare packages directory
# --------------------------------------------------------------------------
PACKAGES_DIR="$(pwd)/packages"
mkdir -p "${PACKAGES_DIR}"
cd "${PACKAGES_DIR}"
info "Working directory: ${PACKAGES_DIR}"

# --------------------------------------------------------------------------
# SOLAX
# --------------------------------------------------------------------------
if [[ -d "${PACKAGES_DIR}/SOLAX" ]]; then
    warn "Directory 'SOLAX' already exists — skipping."
else
    git_clone https://github.com/pavlobilous/SOLAX
    cd SOLAX

    if [[ ! -f setup.py ]]; then
        info "Generating setup.py for SOLAX..."
        cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="solax",
    packages=find_packages()
)
EOF
    fi

    pip_install -e .
    cd "${PACKAGES_DIR}"
fi

# --------------------------------------------------------------------------
# clic_clib
# --------------------------------------------------------------------------
if [[ -d "${PACKAGES_DIR}/clic_clib" ]]; then
    warn "Directory 'clic_clib' already exists — skipping."
else
    git_clone https://github.com/bslhrzg/clic_clib.git
    cd clic_clib

    info "Cleaning stale macOS build artifacts..."
    find . -name "*darwin*.so" -delete
    find . -name "*.dylib"     -delete

    info "Building and installing clic_clib (OpenMP + O3)..."
    CC=gcc CXX=g++ \
    CFLAGS="-O3 -fopenmp -march=native" \
    CXXFLAGS="-O3 -fopenmp -march=native" \
    LDFLAGS="-fopenmp" \
    retry 5 10 pip install -e .
    cd "${PACKAGES_DIR}"
fi

# --------------------------------------------------------------------------
# clic
# --------------------------------------------------------------------------
if [[ -d "${PACKAGES_DIR}/clic" ]]; then
    warn "Directory 'clic' already exists — skipping."
else
    git_clone https://github.com/bslhrzg/clic.git
    cd clic

    info "Building and installing clic (OpenMP + O3)..."
    CC=gcc CXX=g++ \
    CFLAGS="-O3 -fopenmp -march=native" \
    CXXFLAGS="-O3 -fopenmp -march=native" \
    LDFLAGS="-fopenmp" \
    retry 5 10 pip install -e .
    cd "${PACKAGES_DIR}"
fi

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo ""
info "============================================================"
info " All local packages installed successfully!"
info "============================================================"
