#!/usr/bin/env python3
"""
Build ChromaDB vector index + BM25 sparse index + passage collection from Bible JSON.

Creates three artifacts in rag/chroma_db/:
  1. bible_verses collection  -- individual verse embeddings (nomic-embed-text-v1.5)
  2. bible_passages collection -- 5-verse passage windows for parent-child retrieval
  3. bm25_index.pkl           -- pickled BM25Okapi index for hybrid search
"""
import contextlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError("Install chromadb: pip install chromadb>=0.4.0") from e

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError("Install rank_bm25: pip install rank_bm25>=0.2.2") from e

BATCH_SIZE = 500
VERSES_COLLECTION = "bible_verses"
PASSAGES_COLLECTION = "bible_passages"
DOCUMENT_PREFIX = "search_document: "
PASSAGE_WINDOW = 5
PASSAGE_STRIDE = 3


def _load_verses(raw_path: Path) -> list[dict]:
    for name in ("bible_web.json", "bible.json"):
        bible_file = raw_path / name
        if bible_file.exists():
            break
    else:
        raise FileNotFoundError(
            f"No Bible JSON found in {raw_path}. "
            "Ensure data/raw/bible_web.json exists."
        )

    print(f"Loading verses from {bible_file}...")
    with open(bible_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    flat = []
    for book, chapters in data.items():
        for chapter, verse_dict in chapters.items():
            for verse, text in verse_dict.items():
                flat.append({
                    "book": book,
                    "chapter": int(chapter) if isinstance(chapter, str) else chapter,
                    "verse": int(verse) if isinstance(verse, str) else verse,
                    "text": text,
                })
    return flat


def _build_verse_index(
    verses: list[dict], model: SentenceTransformer, client: chromadb.ClientAPI
) -> tuple[list[str], list[str]]:
    """Build individual verse collection. Returns (ids, documents) for BM25."""
    with contextlib.suppress(Exception):
        client.delete_collection(VERSES_COLLECTION)

    collection = client.create_collection(
        name=VERSES_COLLECTION,
        metadata={"description": "Individual Bible verses for RAG retrieval"},
    )

    ids, metadatas, documents = [], [], []
    for v in verses:
        ref = f"{v['book']} {v['chapter']}:{v['verse']}"
        ids.append(ref)
        metadatas.append({
            "book": v["book"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "reference": ref,
        })
        documents.append(DOCUMENT_PREFIX + ref + ": " + v["text"])

    for i in range(0, len(documents), BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]
        embeddings = model.encode(
            batch_docs, batch_size=min(BATCH_SIZE, 256), show_progress_bar=False
        )
        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_docs,
            metadatas=batch_meta,
        )
        print(f"  [verses] Indexed {min(i + BATCH_SIZE, len(documents))}/{len(documents)}")

    return ids, documents


def _build_passage_index(
    verses: list[dict], model: SentenceTransformer, client: chromadb.ClientAPI
) -> None:
    """Build parent passage collection with overlapping 5-verse windows."""
    import gc
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except ImportError:
        pass

    with contextlib.suppress(Exception):
        client.delete_collection(PASSAGES_COLLECTION)

    collection = client.create_collection(
        name=PASSAGES_COLLECTION,
        metadata={"description": "5-verse passage windows for parent-child retrieval"},
    )

    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for v in verses:
        key = f"{v['book']}|{v['chapter']}"
        by_chapter[key].append(v)

    ids, metadatas, documents = [], [], []
    for _key, chapter_verses in by_chapter.items():
        chapter_verses.sort(key=lambda x: x["verse"])
        for i in range(0, len(chapter_verses), PASSAGE_STRIDE):
            window = chapter_verses[i:i + PASSAGE_WINDOW]
            if not window:
                continue
            book = window[0]["book"]
            chapter = window[0]["chapter"]
            first_v = window[0]["verse"]
            last_v = window[-1]["verse"]

            passage_id = f"{book} {chapter}:{first_v}-{last_v}"
            child_ids = [f"{v['book']} {v['chapter']}:{v['verse']}" for v in window]
            passage_text = " ".join(
                f"{v['book']} {v['chapter']}:{v['verse']}: {v['text']}" for v in window
            )

            ids.append(passage_id)
            metadatas.append({
                "book": book,
                "chapter": chapter,
                "first_verse": first_v,
                "last_verse": last_v,
                "child_ids": json.dumps(child_ids),
                "reference": passage_id,
            })
            documents.append(DOCUMENT_PREFIX + passage_text)

    passage_batch = 100
    for i in range(0, len(documents), passage_batch):
        batch_ids = ids[i:i + passage_batch]
        batch_docs = documents[i:i + passage_batch]
        batch_meta = metadatas[i:i + passage_batch]
        embeddings = model.encode(
            batch_docs, batch_size=32, show_progress_bar=False
        )
        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_docs,
            metadatas=batch_meta,
        )
        print(f"  [passages] Indexed {min(i + passage_batch, len(documents))}/{len(documents)}")

    print(f"  Created {len(ids)} passage windows")


def _build_bm25_index(ids: list[str], documents: list[str], db_path: Path) -> None:
    """Build and pickle a BM25Okapi index for sparse retrieval."""
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    bm25_path = db_path / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "documents": documents}, f)
    print(f"  BM25 index saved to {bm25_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw"
    db_path = project_root / "rag" / "chroma_db"
    db_path.mkdir(parents=True, exist_ok=True)

    verses = _load_verses(raw_path)
    print(f"Loaded {len(verses)} verses.")

    print("Loading embedding model (nomic-embed-text-v1.5)...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    client = chromadb.PersistentClient(
        path=str(db_path), settings=Settings(anonymized_telemetry=False)
    )

    print("\n--- Building verse index ---")
    ids, documents = _build_verse_index(verses, model, client)

    print("\n--- Building passage index ---")
    _build_passage_index(verses, model, client)

    print("\n--- Building BM25 index ---")
    _build_bm25_index(ids, documents, db_path)

    print(f"\nDone. All indexes saved to {db_path}")


if __name__ == "__main__":
    main()
