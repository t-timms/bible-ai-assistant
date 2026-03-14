# Step 8: Get WEB and Build the Dataset (Learn as You Go)

This guide walks you through getting the World English Bible into the project and building the training dataset, with a short explanation for each step so you see why we do it.

---

## What we’re doing (big picture)

1. **Get the WEB text** — We need the Bible in a form the computer can read (JSON). The TehShrike repo has WEB as one JSON file per book, in a format that’s good for display but not yet what our dataset builder expects.
2. **Convert to one flat file** — We’ll run a small script that reads all those book files and writes a single `bible_web.json` where each item is one verse: `{ book, chapter, verse, text }`. That’s the format our dataset builder knows how to use.
3. **Build the training dataset** — The dataset builder reads that flat file and the system prompt, and creates Q&A pairs (“What does Psalm 27:1 say?” → “Psalm 27:1 says: …”) plus some theology examples. It writes `data/processed/train.json`, which we’ll use later for fine-tuning.

---

## Step A: Clone the World English Bible repo

**What:** We’re copying the TehShrike “world-english-bible” repository to your machine so we can read its JSON files.

**Why:** The repo is the standard way to get the WEB in JSON. We don’t change it; we only read from it.

**Do this:** Open a terminal (Anaconda Prompt or any terminal). Go to a folder *next to* (or above) your project—for example your Desktop or a folder where you keep repos. You don’t have to put it inside `bible-ai-assistant`.

```bash
# Go to a folder where you want to put the WEB repo (e.g. Desktop or same folder as bible-ai-assistant's parent)
cd c:\Users\ttimm\Desktop\John

# Clone the repo (this creates a folder called world-english-bible)
git clone https://github.com/TehShrike/world-english-bible.git
```

**After it finishes:** You should see a new folder `world-english-bible` with a `json` folder inside it. Each file in `json` is one book (e.g. `genesis.json`, `psalms.json`). We’ll point our converter at this folder next.

---

## Step B: Convert WEB to one flat JSON file

**What:** We run a script that reads every `json/*.json` file in the repo, pulls out the verse text (from “paragraph text” and “line text” objects), groups by book, chapter, and verse, and writes a single file: `data/raw/bible_web.json`, which is a JSON array of `{ "book", "chapter", "verse", "text" }`.

**Why:** Our dataset builder expects either a flat list of verses or a nested object. TehShrike’s format is one file per book with mixed “paragraph text” / “line text” objects, so this conversion step gives us one file in the shape the builder understands.

**Do this:** From your **bible-ai-assistant** project root, with the **bible-ai-assistant** conda env activated:

```bash
cd c:\Users\ttimm\Desktop\John\bible-ai-assistant
conda activate bible-ai-assistant

# Point the script at the repo you just cloned (path to the repo root, not the json folder)
python training/convert_web_tehshrike.py c:\Users\ttimm\Desktop\John\world-english-bible
```

**What you’ll see:** The script prints how many verses it found in each book file, then says it wrote `data/raw/bible_web.json` with the total count (about 31k verses for the full Bible). No errors means the conversion worked.

**Check (optional):** Open `data/raw/bible_web.json` in an editor. You should see entries like:

```json
{ "book": "Genesis", "chapter": 1, "verse": 1, "text": "In the beginning God created the heavens and the earth." }
```

That’s exactly what the dataset builder needs.

---

## Step C: Build the training dataset (Q&A pairs)

**What:** The dataset builder reads `data/raw/bible_web.json` and your system prompt, and creates training examples. For each verse it makes one Q&A: user asks “What does [reference] say?”, assistant answers with the verse and a short line that it’s Scripture. It also adds 500 theology examples (forgiveness, love, faith, etc.). It writes everything to `data/processed/train.json` in the Qwen3 chat format (system / user / assistant messages).

**Why:** Fine-tuning needs many examples in that exact chat format. The model will learn to answer verse questions from Scripture and to stay on topic.

**Do this:** Still in the project root with the env activated:

```bash
python training/dataset_builder.py --max-examples 50000
```

**What you’ll see:** It prints how many verses it loaded and how many examples it wrote. The total will be (number of verse examples, capped at 50k) + 500 theology examples. The file `data/processed/train.json` will be large (many MB). Don’t commit it (it’s in `.gitignore`).

**Check (optional):** Open the start of `data/processed/train.json`. You should see objects with a `messages` array containing `system`, `user`, and `assistant` messages, matching the format in `data/sample.json`.

---

## Step D: Optional — Tag the checkpoint

When you’re happy with the dataset, you can mark this in git:

```bash
git add training/convert_web_tehshrike.py training/dataset_builder.py docs/STEP8_WEB_DATASET.md
git commit -m "Add WEB converter and Step 8 dataset guide"
git tag -a v0.2.0 -m "Dataset ready: WEB, 50k Q&A examples"
git push origin main
git push origin v0.2.0
```

(Only commit code and docs; `data/raw/bible_web.json` and `data/processed/train.json` stay local and are ignored by git.)

---

## Summary

| Step | What | Why |
|------|------|-----|
| A | Clone TehShrike/world-english-bible | Get WEB in JSON (one file per book). |
| B | Run convert_web_tehshrike.py | Turn per-book JSON into one flat `bible_web.json` for the builder. |
| C | Run dataset_builder.py --max-examples 50000 | Build Q&A training data in Qwen3 format. |
| D | Tag v0.2.0 (optional) | Mark “dataset ready” in history. |

After Step C you’re ready for Phase 3: fine-tuning (guide Section 9).
