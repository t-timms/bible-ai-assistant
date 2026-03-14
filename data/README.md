# Data

Bible datasets for fine-tuning and RAG.

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

## Build Script

Use `training/dataset_builder.py` to generate `processed/train.json` from raw sources and templates. Commit checkpoint: **v0.2.0** after dataset is ready.
