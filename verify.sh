#!/usr/bin/env bash
# verify.sh — Single verification entrypoint for the vrag governance harness.
# Automatically uses .venv (or uv if available) with fallback to user Python environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

USER_SITE="$(python3 -m site --user-site 2>/dev/null || true)"
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="${ROOT_DIR}/src:${USER_SITE}:${PYTHONPATH:-}"
export PATH="${ROOT_DIR}/.venv/bin:${HOME}/Library/Python/3.9/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

if ! "$PYTHON_BIN" -c "import fitz" >/dev/null 2>&1; then
    echo "Notice: PyMuPDF ('fitz') not detected in ${PYTHON_BIN}. Run: ${PYTHON_BIN} -m pip install pymupdf" >&2
fi

if ! "$PYTHON_BIN" -c "from sentence_transformers import SentenceTransformer, CrossEncoder" >/dev/null 2>&1; then
    echo "Notice: sentence-transformers dependency conflict detected in ${PYTHON_BIN}. Run: ${PYTHON_BIN} -m pip install 'tokenizers>=0.22.0,<=0.23.0' 'huggingface-hub>=0.34.0,<1.0'" >&2
fi

echo "============================================================"
echo " [vrag] Stage 1/4: Ruff Lint Check"
echo "============================================================"
ruff check src/ tests/

echo "============================================================"
echo " [vrag] Stage 2/4: Ruff Format Check"
echo "============================================================"
ruff format --check src/ tests/

echo "============================================================"
echo " [vrag] Stage 3/4: Domain Law Invariant Suite (LAWS.md)"
echo "============================================================"
"$PYTHON_BIN" -m pytest tests/laws/ -v -m "law"

echo "============================================================"
echo " [vrag] Stage 4/4: Unit Test Suite"
echo "============================================================"
"$PYTHON_BIN" -m pytest tests/unit/ -v

echo "============================================================"
echo " [vrag] ALL VERIFICATION CHECKS PASSED"
echo "============================================================"
