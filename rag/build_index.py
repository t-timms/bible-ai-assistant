#!/usr/bin/env python3
"""
Build ChromaDB vector index from Bible JSON.
Uses nomic-embed-text-v1.5 with search_document prefix. See guide Section 12.
"""
from pathlib import Path

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw"
    # TODO: Load Bible JSON (e.g. bible_kjv.json), create PersistentClient, get_or_create_collection.
    # TODO: SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True).
    # TODO: For each verse: document = f"search_document: {text}", ids = ref, metadata = book, chapter, verse, reference.
    # TODO: Encode in batches (e.g. 1000), collection.add(...). Persist to rag/chroma_db.
    raise NotImplementedError(
        "RAG index builder skeleton. Implement per guide Section 12. Requires data/raw/ Bible JSON."
    )


if __name__ == "__main__":
    main()
