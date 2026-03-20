#!/usr/bin/env python3
"""Quick test of RAG retrieval quality. Run after build_index.py."""
from pathlib import Path


def main() -> None:
    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("Install chromadb and sentence-transformers. pip install chromadb sentence-transformers")
        raise SystemExit(1) from e

    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "rag" / "chroma_db"
    if not db_path.exists():
        print(f"ChromaDB not found at {db_path}. Run: python rag/build_index.py")
        raise SystemExit(1)

    client = chromadb.PersistentClient(path=str(db_path), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("bible_verses")
    # trust_remote_code required by nomic-embed-text-v1.5 for custom pooling
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    query = "What does John 3:16 say?"
    embedding = model.encode(["search_query: " + query], show_progress_bar=False)
    results = collection.query(
        query_embeddings=embedding.tolist(),
        n_results=5,
        include=["documents", "metadatas"],
    )

    print(f"Query: {query}\n")
    print("Top 5 results:")
    if not results["metadatas"] or not results["metadatas"][0]:
        print("  (no results returned)")
        return
    for meta, doc in zip(results["metadatas"][0], results["documents"][0], strict=True):
        ref = meta.get("reference", "?")
        text = doc.replace("search_document: ", "", 1) if doc.startswith("search_document:") else doc
        print(f"  {ref}: {text[:80]}...")
    # Sanity: John 3:16 should often be in top 5 for this query
    refs = [m.get("reference", "") for m in results["metadatas"][0]]
    if any("John 3:16" in r for r in refs):
        print("\n✓ John 3:16 found in top results.")
    else:
        print("\n(John 3:16 not in top 5 for this query; retrieval is still working.)")


if __name__ == "__main__":
    main()
