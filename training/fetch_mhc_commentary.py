#!/usr/bin/env python3
"""Build data/raw_v2/mhc_commentary.json from Matthew Henry's Commentary.

Source: codeberg.org/revisedcommonversion/matthew-henry-commentary
License: CC0 / public domain dedication (no rights reserved).

The upstream repo stores one markdown file per chapter, each split into
``### Verses N-N`` / ``### Verse N`` sections. This script clones it (shallow,
to a temp dir), parses those sections into flat records, and writes a single
JSON blob consumed by ``training/build_dataset_v2.py``'s ``gen_grounded_exegesis``.

Usage:
    python training/fetch_mhc_commentary.py            # clone + build
    python training/fetch_mhc_commentary.py --src DIR  # reuse an existing clone
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess  # nosec B404 - fixed git clone of a known public repo
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = "https://codeberg.org/revisedcommonversion/matthew-henry-commentary"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw_v2" / "mhc_commentary.json"

# dir-name -> canonical book string (must normalize, under build_dataset_v2's
# book_key(), to the same thing the KJV corpus keys do: lowercase, digit ordinals)
_DIR_TO_BOOK = {
    "song-of-solomon": "Song of Solomon",
}


def _book_from_dir(name: str) -> str:
    if name in _DIR_TO_BOOK:
        return _DIR_TO_BOOK[name]
    parts = name.split("-")
    if parts[0] in {"1", "2", "3"}:
        return parts[0] + " " + " ".join(p.capitalize() for p in parts[1:])
    return " ".join(p.capitalize() for p in parts)


_HEAD = re.compile(r"^###\s+Verses?\s+(\d+)(?:\s*[-–]\s*(\d+))?\s*$", re.MULTILINE)
_CH_FROM_NAME = re.compile(r"(?:Chapter|Psalm)\s+0*(\d+)", re.IGNORECASE)


def _clean(text: str) -> str:
    text = text.replace("\\'", "'").replace('\\"', '"')
    text = re.sub(r"`([^`]*)`", r"\1", text)  # drop the `[1.]` enumerator backticks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)  # unwrap hard line breaks
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_chapter(md: str, book: str, chapter: int) -> list[dict]:
    out: list[dict] = []
    heads = list(_HEAD.finditer(md))
    for i, m in enumerate(heads):
        v_start = int(m.group(1))
        v_end = int(m.group(2)) if m.group(2) else v_start
        body = md[m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(md)]
        body = _clean(body)
        if len(body) < 200:
            continue
        out.append(
            {
                "book": book,
                "chapter": chapter,
                "verse_start": v_start,
                "verse_end": v_end,
                "text": body,
            }
        )
    return out


def build(src: Path) -> dict:
    records: list[dict] = []
    for book_dir in sorted(p for p in src.iterdir() if p.is_dir() and not p.name.startswith(".")):
        book = _book_from_dir(book_dir.name)
        for md_file in sorted(book_dir.glob("MHC*.md")):
            cm = _CH_FROM_NAME.search(md_file.name)
            if not cm:
                continue
            chapter = int(cm.group(1))
            records.extend(
                parse_chapter(md_file.read_text(encoding="utf-8", errors="replace"), book, chapter)
            )
    records.sort(key=lambda r: (r["book"], r["chapter"], r["verse_start"]))
    digest = hashlib.sha256(
        "\n".join(
            f"{r['book']} {r['chapter']}:{r['verse_start']} {r['text']}" for r in records
        ).encode("utf-8")
    ).hexdigest()
    return {
        "meta": {
            "source": "Matthew Henry's Commentary on the Whole Bible",
            "license": "CC0 / public domain",
            "via": REPO,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "records": len(records),
            "sha256": digest,
        },
        "records": records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--src", type=Path, help="existing clone dir (skips git clone)")
    ap.add_argument("--out", type=Path, default=OUT)
    ns = ap.parse_args()

    if ns.src:
        blob = build(ns.src)
    else:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "mhc"
            print(f"cloning {REPO} ...")
            subprocess.run(  # nosec B603 B607 - fixed args, known public repo
                ["git", "clone", "--depth", "1", "-q", REPO, str(dest)], check=True
            )
            blob = build(dest)

    if blob["meta"]["records"] < 3000:
        sys.exit(f"only {blob['meta']['records']} records parsed — aborting (expected ~4000+)")

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {blob['meta']['records']} records -> {ns.out}")
    print(f"sha256 {blob['meta']['sha256']}")


if __name__ == "__main__":
    main()
