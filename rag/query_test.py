#!/usr/bin/env python3
"""Quick test of RAG retrieval quality. Run after build_index.py."""
from pathlib import Path

def main() -> None:
    # TODO: Load ChromaDB client + collection, embedder. Query e.g. "What does John 3:16 say?"
    # Print top-5 results (reference + text). Verify John 3:16 is in top results.
    raise NotImplementedError("Implement query test per guide Section 12. Optional sanity check.")
