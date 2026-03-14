# Data

Bible datasets for fine-tuning and RAG. This project uses the **World English Bible (WEB)** as the primary translation (clear, modern English; public domain).

## Layout

| Path | Purpose |
|------|---------|
| `raw/` | Original Bible source files (JSON/CSV). Not committed (see .gitignore). |
| `processed/` | Cleaned, chat-format training data. Not committed. |
| `sample.json` | 10–20 example Q&A pairs for documentation. Committed. |

## Dataset Format

Training data must follow the Qwen3 chat format. Each example is a JSON object with a `messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "[Your system prompt]"},
    {"role": "user", "content": "What does John 3:16 say and mean?"},
    {"role": "assistant", "content": "John 3:16 reads: 'For God so loved the world...' (NIV). This verse captures..."}
  ]
}
```

See the main guide (Section 8) for data sources (e.g. scrollmapper/bible_databases, World English Bible, CCEL) and types of Q&A pairs to include.

## Targets

- **First run:** 30,000–50,000 examples (quality over quantity).
- **Types:** Direct verse lookup, cross-references, theology, character studies, constitution-testing, uncertainty examples, thinking-mode (complex reasoning).

## Raw JSON format

The dataset builder expects a JSON **array** of verse objects. Each object can use any of these field names:

- **Book:** `book`, `book_name`, or `b`
- **Chapter:** `chapter` or `c`
- **Verse:** `verse` or `v`
- **Text:** `text`, `content`, `verse_text`, or `t`

Example: `[{"book": "Genesis", "chapter": 1, "verse": 1, "text": "In the beginning..."}, ...]`

Place the file in `data/raw/` as `bible.json` or `bible_kjv.json`. Then run `python training/dataset_builder.py` (optionally `--max-examples 50000`).

## Build Script

Use `training/dataset_builder.py` to generate `processed/train.json` from raw sources and templates. Commit checkpoint: **v0.2.0** after dataset is ready.
