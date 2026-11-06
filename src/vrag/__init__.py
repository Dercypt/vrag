"""Verification-Driven RAG (vrag) comparative benchmarking package."""

from __future__ import annotations

import os

# Prevent macOS Intel OpenMP thread-pool collisions across PyMuPDF, SciPy, Torch & FAISS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
