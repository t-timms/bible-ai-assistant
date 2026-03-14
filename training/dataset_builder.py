#!/usr/bin/env python3
"""
Build Bible Q&A dataset for fine-tuning.
Reads from data/raw/, outputs data/processed/train.json in Qwen3 chat format.
Supports JSON array of verses with keys: book (or book_name), chapter, verse, text (or content).
Usage:
  python training/dataset_builder.py
  python training/dataset_builder.py --max-examples 50000 --input data/raw/bible_kjv.json
"""
from pathlib import Path
import json
import argparse


def load_system_prompt(project_root: Path) -> str:
    path = project_root / "prompts" / "system_prompt.txt"
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def find_raw_bible(raw_dir: Path) -> Path:
    for name in ("bible.json", "bible_web.json", "bible_kjv.json", "en_bbe.json"):
        p = raw_dir / name
        if p.exists():
            return p
    for f in raw_dir.glob("*.json"):
        return f
    raise FileNotFoundError(
        f"No Bible JSON found in {raw_dir}. "
        "Add a JSON array of verses (each with book, chapter, verse, text) e.g. bible.json"
    )


def _flatten_nested_bible(nested: dict) -> list[dict]:
    """Convert nested structure { book: { chapter: { verse: text } } } to list of verse records."""
    verses = []
    for book, chapters in nested.items():
        if not isinstance(chapters, dict):
            continue
        for ch_str, versed in chapters.items():
            if not isinstance(versed, dict):
                continue
            try:
                ch_num = int(ch_str)
            except (ValueError, TypeError):
                continue
            for v_str, text in versed.items():
                try:
                    v_num = int(v_str)
                except (ValueError, TypeError):
                    continue
                if not text or not str(text).strip():
                    continue
                verses.append({
                    "book": str(book),
                    "chapter": ch_num,
                    "verse": v_num,
                    "text": str(text).strip(),
                })
    return verses


def verse_to_record(item: dict) -> dict | None:
    """Normalize a verse object. Returns dict with book, chapter, verse, text or None if invalid."""
    book = item.get("book") or item.get("book_name") or item.get("b")
    chapter = item.get("chapter") or item.get("c")
    verse = item.get("verse") or item.get("v")
    text = item.get("text") or item.get("content") or item.get("verse_text") or item.get("t")
    if book is None or chapter is None or verse is None or text is None:
        return None
    return {"book": str(book), "chapter": int(chapter), "verse": int(verse), "text": str(text).strip()}


def build_verse_lookup_examples(verses: list[dict], system_prompt: str, max_examples: int | None) -> list[dict]:
    """One Q&A per verse: 'What does X Y:Z say?' -> verse text + brief meaning."""
    examples = []
    for v in verses:
        if max_examples is not None and len(examples) >= max_examples:
            break
        ref = f"{v['book']} {v['chapter']}:{v['verse']}"
        user = f"What does {ref} say?"
        # Assistant: quote verse and one sentence of meaning (model will learn to cite and explain)
        assistant = f"{ref} says: \"{v['text']}\" This verse is part of Scripture; its meaning can be understood in context of the surrounding passage and the whole Bible."
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })
    return examples


def add_theology_examples(system_prompt: str, count: int = 500) -> list[dict]:
    """Static theology Q&A templates to add variety. Expand as needed."""
    templates = [
        ("What does the Bible teach about forgiveness?", "Scripture teaches that forgiveness is central to the Christian life. God forgives us through Christ (Ephesians 1:7), and we are called to forgive others (Matthew 6:14-15, Colossians 3:13). Jesus modeled this on the cross (Luke 23:34)."),
        ("What does the Bible say about love?", "The Bible teaches that God is love (1 John 4:8) and that we are to love God and love our neighbor (Matthew 22:37-39). Love is patient, kind, and keeps no record of wrongs (1 Corinthians 13:4-7)."),
        ("What does the Bible say about faith?", "The Bible defines faith as confidence in what we hope for and assurance about what we do not see (Hebrews 11:1). We are saved by grace through faith (Ephesians 2:8-9)."),
        ("What does John 3:16 mean?", "John 3:16 (NIV) says: 'For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.' This verse summarizes the gospel: God's love, the gift of Jesus, and the promise of eternal life for those who believe."),
        ("Who was the Apostle Paul?", "Paul was an apostle who wrote much of the New Testament. He was originally a persecutor of the church (Acts 8:3) and was converted on the road to Damascus (Acts 9). He preached the gospel to the Gentiles and established churches."),
    ]
    out = []
    while len(out) < count:
        for q, a in templates:
            if len(out) >= count:
                break
            out.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            })
    return out[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bible Q&A training dataset.")
    parser.add_argument("--input", type=Path, default=None, help="Path to raw Bible JSON (default: auto-detect in data/raw/)")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: data/processed/train.json)")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap total examples (default: no cap)")
    parser.add_argument("--add-theology", type=int, default=500, help="Number of theology template examples to add (default: 500)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(project_root)
    input_path = args.input or find_raw_bible(raw_dir)
    output_path = args.output or (out_dir / "train.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = json.loads(input_path.read_text(encoding="utf-8"))

    # Support (1) list of verse objects or (2) nested dict: book -> chapter -> verse -> text (e.g. WEB from TehShrike)
    if isinstance(raw, dict) and raw and not isinstance(next(iter(raw.values())), (str, int, float)):
        verses = _flatten_nested_bible(raw)
    elif isinstance(raw, list):
        verses = []
        for item in raw:
            rec = verse_to_record(item)
            if rec and rec["text"]:
                verses.append(rec)
    else:
        verses = []
        rec = verse_to_record(raw)
        if rec and rec["text"]:
            verses.append(rec)

    if not verses:
        raise ValueError("No valid verses found. Each JSON item needs book, chapter, verse, and text (or content).")

    print(f"Loaded {len(verses)} verses from {input_path.name}")

    examples = build_verse_lookup_examples(verses, system_prompt, args.max_examples)
    theology = add_theology_examples(system_prompt, min(args.add_theology, 500))
    examples.extend(theology)

    output_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
