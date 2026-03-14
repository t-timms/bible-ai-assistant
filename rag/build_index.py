#!/usr/bin/env python3
"""
Build ChromaDB vector index from Bible JSON.
Uses nomic-embed-text-v1.5 with search_document prefix. See docs/WALKTHROUGH.md Step 12.
"""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

# ChromaDB: use persistent client; we add raw embeddings (no built-in embedder)
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("Install chromadb: pip install chromadb>=0.4.0")

BATCH_SIZE = 500
COLLECTION_NAME = "bible_verses"
DOCUMENT_PREFIX = "search_document: "


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw"
    db_path = project_root / "rag" / "chroma_db"

    # Prefer bible_web.json (from Step 8), fallback to bible.json
    for name in ("bible_web.json", "bible.json"):
        bible_file = raw_path / name
        if bible_file.exists():
            break
    else:
        raise FileNotFoundError(
            f"No Bible JSON found in {raw_path}. Run Step 8 (dataset build) first and ensure data/raw/bible_web.json exists."
        )

    print(f"Loading verses from {bible_file}...")
    with open(bible_file, "r", encoding="utf-8") as f:
        verses = json.load(f)

    if not isinstance(verses, list):
        # Nested format: { book: { chapter: { verse: text } } } -> flatten
        flat = []
        for book, chapters in verses.items():
            for chapter, verse_dict in chapters.items():
                for verse, text in verse_dict.items():
                    flat.append({
                        "book": book,
                        "chapter": int(chapter) if isinstance(chapter, str) else chapter,
                        "verse": int(verse) if isinstance(verse, str) else verse,
                        "text": text,
                    })
        verses = flat

    print(f"Loaded {len(verses)} verses. Building embeddings with nomic-embed-text-v1.5...")

    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path), settings=Settings(anonymized_telemetry=False))

    # Delete existing collection so we can rebuild from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Bible verses for RAG retrieval"},
    )

    ids = []
    metadatas = []
    documents = []

    for v in verses:
        ref = f"{v['book']} {v['chapter']}:{v['verse']}"
        ids.append(ref)
        metadatas.append({
            "book": v["book"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "reference": ref,
        })
        documents.append(DOCUMENT_PREFIX + v["text"])

    for i in range(0, len(documents), BATCH_SIZE):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]
        embeddings = model.encode(batch_docs, batch_size=min(BATCH_SIZE, 256), show_progress_bar=False)
        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_docs,
            metadatas=batch_meta,
        )
        print(f"  Indexed {min(i + BATCH_SIZE, len(documents))}/{len(documents)} verses.")

    print(f"Done. ChromaDB index saved to {db_path}")


if __name__ == "__main__":
    main()
