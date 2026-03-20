#!/usr/bin/env python3
"""
Convert World English Bible from TehShrike format to our flat format.

The TehShrike repo (https://github.com/TehShrike/world-english-bible) has one JSON
file per book in the json/ folder. Each file is an array of objects with types like
"paragraph text" and "line text" containing chapterNumber, verseNumber, and value.

This script:
  1. Reads all json/*.json files from the TehShrike repo folder you provide
  2. Extracts verse text (paragraph text / line text), grouped by book, chapter, verse
  3. Writes data/raw/bible_web.json as a flat array of { book, chapter, verse, text }

Usage:
  python training/convert_web_tehshrike.py path/to/world-english-bible
  (e.g. after cloning: python training/convert_web_tehshrike.py ../world-english-bible)

Then run: python training/dataset_builder.py --max-examples 50000
"""
import json
import re
from pathlib import Path


def filename_to_book_name(filename: str) -> str:
    """Turn e.g. '1chronicles' into '1 Chronicles', 'genesis' into 'Genesis'."""
    stem = filename.replace(".json", "").strip()
    # Insert space before capital letters: 1Chronicles -> 1 Chronicles
    spaced = re.sub(r"(\d)([A-Za-z])", r"\1 \2", stem)
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", spaced)
    return spaced.title() if spaced else stem


def extract_verses_from_book_array(arr: list, book: str) -> list[dict]:
    """From one book's array of typed objects, return list of { book, chapter, verse, text }."""
    # Group text by (chapter, verse); some verses span multiple "paragraph text" or "line text" objects
    chunks: dict[tuple[int, int], list[str]] = {}
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        if kind not in ("paragraph text", "line text"):
            continue
        ch = obj.get("chapterNumber")
        v = obj.get("verseNumber")
        val = obj.get("value")
        if ch is None or v is None or val is None:
            continue
        key = (int(ch), int(v))
        chunks.setdefault(key, []).append(str(val).strip())
    verses = []
    for (ch, v), texts in sorted(chunks.items()):
        verses.append({
            "book": book,
            "chapter": ch,
            "verse": v,
            "text": " ".join(texts).strip(),
        })
    return verses


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Convert TehShrike WEB JSON to flat bible_web.json")
    parser.add_argument("tehshrike_repo", type=Path, help="Path to world-english-bible repo (contains json/ folder)")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: data/raw/bible_web.json)")
    args = parser.parse_args()

    json_dir = args.tehshrike_repo / "json"
    if not json_dir.is_dir():
        raise FileNotFoundError(f"json folder not found: {json_dir}. Point to the world-english-bible repo root.")

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (raw_dir / "bible_web.json")

    all_verses = []
    for jpath in sorted(json_dir.glob("*.json")):
        book_name = filename_to_book_name(jpath.name)
        try:
            arr = json.loads(jpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  WARNING: Skipping {jpath.name} (malformed JSON: {e})")
            continue
        verses = extract_verses_from_book_array(arr, book_name)
        all_verses.extend(verses)
        print(f"  {jpath.name}: {len(verses)} verses")

    out_path.write_text(json.dumps(all_verses, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(all_verses)} verses to {out_path}")


if __name__ == "__main__":
    main()
