# Step-by-Step Walkthrough

This document walks you through the Bible AI Assistant project with explanations so you learn as you go. Follow in order.

**See also:** [PROJECT_JOURNEY.md](PROJECT_JOURNEY.md) — the story behind key decisions (overfitting fix, simplified prompt, post-processing) for interviews and content.

---

## Step 1: Connect Your Code to GitHub ✅

**What we did:**

1. **`git init`** — Creates a `.git` folder in your project. Git will now track every file here and remember changes over time.

2. **`git remote add origin https://github.com/omnipotence-eth/bible-ai-assistant.git`** — Tells Git where your “backup” lives. `origin` is the usual name for the main remote (your GitHub repo). Nothing has been sent to GitHub yet; we only saved the URL.

**Next:** We add all files, commit, and push. A **commit** is a snapshot of the project at a point in time. **Push** sends your commits to GitHub so they’re stored in the cloud and you can work from other machines or share the repo.

---

## Step 2: First Commit and Push

(Commands are run in the next section. Here’s what they mean.)

- **`git add .`** — Stages every file in the project (respecting `.gitignore`, so models and `.env` are not added).
- **`git commit -m "..."`** — Creates a snapshot with the given message. Use present tense and a version for milestones.
- **`git branch -M main`** — Renames the current branch to `main` (GitHub’s default).
- **`git push -u origin main`** — Sends your commits to GitHub. `-u` sets `main` to track `origin/main`, so future `git push` is enough.
- **`git tag -a v0.1.0 -m "..."`** — Creates an annotated tag (a named bookmark for “version 0.1.0”).
- **`git push origin v0.1.0`** — Pushes that tag to GitHub so you can see it under Releases/Tags.

After this, your scaffold is on GitHub and you have a clear starting point (v0.1.0).

---

## Step 3: Phase 1 — Environment and Base Model

We’ll do this next. It includes:

1. **Installing the toolchain** — Git (done), Miniconda, CUDA 12.8+, Node.js, Docker, Ollama, etc.
2. **Creating a conda environment** — An isolated Python 3.11 environment named **`bible-ai-assistant`** (same as the repo) so this project’s packages don’t conflict with others.
3. **Installing PyTorch nightly** — Your RTX 5070 Ti (Blackwell) needs CUDA 12.8+; stable PyTorch doesn’t support it yet, so we use the nightly build.
4. **Installing project dependencies** — `requirements.txt` (training, RAG, API, voice).
5. **Logging into Hugging Face and W&B** — So you can download models and log training runs.
6. **Downloading Qwen3 4B** — The base model you’ll fine-tune.

We’ll go through each of these one at a time so you understand why each step matters.

---

## Step 4: Create the Conda Environment (`bible-ai-assistant`)

**Why we use conda:** Conda gives you an isolated environment—Python 3.11 and all packages (PyTorch, Unsloth, etc.) live here and won’t clash with other projects or your system Python. Naming it `bible-ai-assistant` matches the repo so you can remember it.

**Do this in Anaconda Prompt** (search for it in the Start menu—use this instead of regular PowerShell or Command Prompt so conda is available):

```bash
# Go to your project folder (replace with your path)
cd c:\Users\YOUR_USERNAME\Desktop\John\bible-ai-assistant

# Create the environment with Python 3.11 (-y skips “are you sure?” prompts)
conda create -n bible-ai-assistant python=3.11 -y

# Activate it (you’ll need to do this every time you open a new terminal for this project)
conda activate bible-ai-assistant
```

**Check it worked:** Run `python --version`. You should see **Python 3.11.x**. Run `conda info --envs` and you should see `bible-ai-assistant` with an asterisk next to it when it’s active.

**Tip:** Every new Anaconda Prompt window starts with the base env. Always run `conda activate bible-ai-assistant` before working in this project.

---

## Step 5: Install PyTorch Nightly (CUDA 12.8) and Dependencies

**Why PyTorch nightly:** Your RTX 5070 Ti uses NVIDIA’s Blackwell architecture (compute capability sm_120). Stable PyTorch doesn’t support it yet—you need a **nightly** build compiled for **CUDA 12.8+**. (If you haven’t installed CUDA Toolkit 12.8+, do that first: https://developer.nvidia.com/cuda-downloads .)

**Do this in Anaconda Prompt** with `bible-ai-assistant` activated:

```bash
# 1. PyTorch nightly with CUDA 12.8 (required for RTX 5070 Ti)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 2. Project dependencies (training, RAG, API, voice)
pip install -r requirements.txt
```

**Verify PyTorch and GPU:** Run:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
python -c "import torch; print('Arch list:', torch.cuda.get_arch_list())"
```

You should see `CUDA available: True`, your GPU name, and **`sm_120`** in the arch list. If `sm_120` is missing, reinstall the nightly; do not use stable PyTorch for this GPU.

**Next:** Log into Hugging Face and Weights & Biases (Step 6), then download the base model (Step 7).

---

## Step 6: Log into Hugging Face and Weights & Biases

**Why:** Hugging Face hosts the base model (Qwen3 4B) and will store your fine-tuned model. W&B (Weights & Biases) records training runs (loss curves, hyperparameters) so you can compare experiments. Both need one-time login from your terminal.

---

### 6a. Create and use a Hugging Face token

1. **Open:** https://huggingface.co/settings/tokens (log in or sign up if needed).
2. **Choose token type:**
   - **Write** — Simplest. Full read + write (download base model, create repos, upload your fine-tuned model, update for benchmarking). Use this unless you want extra restrictions.
   - **Fine-grained** — More secure: limit what the token can do (e.g. only write to **Model repos**, or only to a specific repo like `YOUR_USERNAME/bible-qwen3-4b`). If the token is ever leaked, only those resources are affected. Good for open source when you want one token just for this project.
3. **Create token:** Click **“Create new token”**.
   - **Name:** e.g. `bible-ai-assistant`.
   - **Type:** **Write** (easiest) or **Fine-grained** (see below).
   - If **Fine-grained:** under permissions, grant at least **Read** (for downloading base model) and **Write** or **Create/update repos** for **Models** (so you can upload and update your fine-tuned model). Optionally restrict to a specific repo once you’ve created it.
   - Click **“Generate token”**.
3. **Copy the token:** Click the copy icon next to the token. It looks like `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (starts with `hf_`). You won’t see it again after you leave the page, so copy it now.
4. **In Anaconda Prompt** (with `bible-ai-assistant` activated and project folder as current directory), run:
   ```bash
   huggingface-cli login
   ```
5. When it says **“Enter your token”** or **“Paste your token”**:
   - **Right‑click** in the terminal window to paste (Windows pastes with right‑click in most terminals).
   - Or type **Ctrl+Shift+V** (some terminals use this instead of Ctrl+V).
   - Do **not** type the token by hand—one wrong character and it will fail. Paste only.
   - Press **Enter** after pasting.
6. You should see **“Login successful”** or **“Token is valid”**. If you see an error, create a new token (no spaces before/after when pasting) and try again.

---

### 6b. Create and use a Weights & Biases API key

1. **Open:** https://wandb.ai/authorize (log in or sign up at https://wandb.ai if needed).
2. **Copy your API key:** On the authorize page you’ll see your key (a long string). Click to copy it.
3. **In Anaconda Prompt**, run:
   ```bash
   wandb login
   ```
4. When it says **“Paste your API key”** or **“Enter your API key”**:
   - **Right‑click** to paste (or **Ctrl+Shift+V** if that’s what your terminal uses).
   - Press **Enter**.
5. You should see **“Successfully logged in”** or similar. W&B will save the key so you usually won’t need to paste it again on this machine.

---

### Quick reference (both in one go)

In Anaconda Prompt, from the project folder with `bible-ai-assistant` active:

```bash
huggingface-cli login
# When prompted: paste HF token (hf_...), Enter

wandb login
# When prompted: paste W&B API key, Enter
```

**Next:** Download the base model (Step 7).

---

## Step 7: Download Qwen3 4B Base Model

**Why:** Qwen3-4B-Instruct-2507 is the model you’ll fine-tune on Bible Q&A. It’s about 8GB; downloading to `models/base_model/` keeps everything in one place and respects `.gitignore` (models are not committed).

**Do this in Anaconda Prompt** with `bible-ai-assistant` activated, from the **project root** (e.g. `c:\Users\YOUR_USERNAME\Desktop\John\bible-ai-assistant`):

```bash
# Download to models/base_model (creates folder; ~10–20 min, ~8GB). Exclude .msgpack to save space.
# Preferred (new CLI):
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/base_model --exclude "*.msgpack"
# Legacy (still works, but deprecated):
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/base_model --exclude "*.msgpack"
```

**Note:** If you see a warning that `huggingface-cli download` is deprecated, use `hf download` next time. If you see “Xet Storage” / “hf_xet” messages, the download still uses regular HTTP; installing `pip install hf_xet` is optional for faster future downloads.

**Check:** When it finishes, `models/base_model/` should contain config, tokenizer, and model weights (e.g. `.safetensors`). Do not commit this folder (it’s in `.gitignore`).

**Phase 1 complete.** Optionally tag the repo as **v0.1.1** to mark “base model downloaded” in history (`git tag -a v0.1.1 -m "Base model downloaded"` then `git push origin v0.1.1`). Next: **Phase 2** — building the Bible dataset (Step 8).

---

## Step 8: Phase 2 — Build the Bible Dataset (WEB)

**Why:** Fine-tuning quality depends on dataset quality. We use **~1,600 diverse examples** (quality over quantity) across verse lookups, RAG-grounded examples, thematic Q&A, general assistant, meta-questions, multi-turn, and refusals. A short system prompt (~15 lines) is embedded in each example—long prompts cause instruction leakage in 4B models.

**What we’re doing (big picture):**

1. **Get the WEB text** — We need the Bible in a form the computer can read (JSON). The TehShrike repo has WEB as one JSON file per book, in a format that’s good for display but not yet what our dataset builder expects.
2. **Convert to one flat file** — We’ll run a small script that reads all those book files and writes a single `bible_web.json` where each item is one verse: `{ book, chapter, verse, text }`. That’s the format our dataset builder knows how to use.
3. **Build the training dataset** — The dataset builder reads that flat file and the system prompt, and creates Q&A pairs (“What does Psalm 27:1 say?” → “Psalm 27:1 says: …”) plus some theology examples. It writes `data/processed/train.json`, which we’ll use later for fine-tuning.

---

### Step 8a: Clone the World English Bible repo

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

### Step 8b: Convert WEB to one flat JSON file

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

### Step 8c: Build the training dataset (Q&A pairs)

**What:** The dataset builder reads `data/raw/bible_web.json` and your system prompt, and creates training examples. For each verse it makes one Q&A: user asks “What does [reference] say?”, assistant answers with the verse and a short line that it’s Scripture. It also adds 500 theology examples (forgiveness, love, faith, etc.). It writes everything to `data/processed/train.json` in the Qwen3 chat format (system / user / assistant messages).

**Why:** Fine-tuning needs many examples in that exact chat format. The model will learn to answer verse questions from Scripture and to stay on topic.

**Do this:** Still in the project root with the env activated:

```bash
python training/dataset_builder.py
```

**What you’ll see:** It prints how many verses it loaded and how many examples it wrote. The total will be ~1,600 diverse examples (verse lookups, RAG-grounded, thematic, general assistant, meta-questions, multi-turn, refusals). The file `data/processed/train.json` will be a few MB. Don’t commit it (it’s in `.gitignore`).

**Check (optional):** Open the start of `data/processed/train.json`. You should see objects with a `messages` array containing `system`, `user`, and `assistant` messages, matching the format in `data/sample.json`.

---

### Step 8d: Optional — Tag the checkpoint

When you’re happy with the dataset, you can mark this in git:

```bash
git add training/convert_web_tehshrike.py training/dataset_builder.py
git commit -m "Add WEB converter and dataset builder"
git tag -a v0.2.0 -m "Dataset ready: WEB, ~1.6k diverse Q&A examples"
git push origin main
git push origin v0.2.0
```

(Only commit code and docs; `data/raw/bible_web.json` and `data/processed/train.json` stay local and are ignored by git.)

**Phase 2 checkpoint:** When `train.json` is ready, tag **v0.2.0** (“Dataset builder complete, ~1.6k diverse Bible Q&A”). Then continue to Phase 3: fine-tuning (Step 9).

---

## Step 9: Phase 3 — Fine-tune Your Model (qwen3-4b-bible-John)

**What you’re about to do (big picture):**

1. **Check your environment** – Conda, GPU, and data are ready.
2. **Run the training script** – The script loads the base model, adds small “adapter” weights (LoRA), and trains them on your Bible Q&A data.
3. **Save the adapter** – Only the new weights are saved (as **qwen3-4b-bible-John**), not the full 4B model, so it’s small and fast to use later.

**Why “qwen3-4b-bible-John”?**  
- **qwen3-4b** = base model (Qwen3, 4 billion parameters).  
- **bible** = trained on Bible (WEB) Q&A.  
- **John** = your chosen name so this run and the saved folder are clearly yours.

---

### Step 9a: Open a terminal and activate the environment

1. Open your terminal (PowerShell or Command Prompt).
2. Go to the project folder and activate conda:

```powershell
cd c:\Users\ttimm\Desktop\John\bible-ai-assistant
conda activate bible-ai-assistant
```

You should see `(bible-ai-assistant)` at the start of your prompt. All commands assume you’re in this folder so paths like `data/processed/train.json` and `models/qwen3-4b-bible-John` work correctly.

---

### Step 9b: (Optional) Check your GPU and training data

**GPU:**

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

You want `CUDA: True` and your GPU name (e.g. RTX 5070 Ti). Training uses the GPU; if CUDA is False, it would run on CPU and be extremely slow.

**Training data:**

```powershell
python -c "from pathlib import Path; p = Path('data/processed/train.json'); print('Exists:', p.exists()); print('Size (MB):', p.stat().st_size / (1024*1024) if p.exists() else 0)"
```

You should see `Exists: True` and a size of roughly 5–20 MB for ~1,600 examples. If this file is missing, run the dataset builder first (Step 8 above).

---

### Step 9c: Understand what the script will do

When you run the trainer, it will:

| Step | What happens |
|------|-------------------------------|
| 1 | Load **Qwen3-4B-Instruct** (from Hugging Face or your local `models/base_model`). |
| 2 | Load **data/processed/train.json** (Bible Q&A in system/user/assistant format). |
| 3 | Turn each Q&A into one text string using the model’s “chat template” (so the model sees it as a normal conversation). |
| 4 | Add **LoRA** adapters (small extra weights) to the model and train only those for **2 epochs** (anti-overfitting: LR 1e-4, LoRA r=8, dropout 0.15, 10% eval split). |
| 5 | Log progress to **Weights & Biases** (project: `bible-ai`, run name: **qwen3-4b-bible-John**). |
| 6 | Save checkpoints to **checkpoints/** every 500 steps. |
| 7 | At the end, save the final LoRA adapter and tokenizer to **models/qwen3-4b-bible-John**. |

**Why LoRA?** Training the full 4B parameters would need a lot of VRAM and time. LoRA trains only a small set of extra weights (adapters). You get most of the benefit with less memory and a smaller file to store and load.

---

### Step 9d: Run training

From the project root, run:

```powershell
python training/train_unsloth.py --run-name qwen3-4b-bible-John
```

If you use the script’s default run name, you can also just run:

```powershell
python training/train_unsloth.py
```

(The default is already set to **qwen3-4b-bible-John**, so the W&B run and the folder **models/qwen3-4b-bible-John** will use that name.)

**Using a local base model (if you downloaded it earlier):**

```powershell
python training/train_unsloth.py --run-name qwen3-4b-bible-John --model-path models/base_model
```

**What you’ll see:** Loading messages for the model and dataset; training steps with loss values; logs every 50 steps and checkpoint saves every 500 steps. On an RTX 5070 Ti, expect roughly **2–6 hours** depending on data size and settings.

**If you get an error:**  
- **“Training data not found”** → Run the dataset builder (Step 8) first.  
- **“CUDA out of memory”** → In **training/config.yaml** try lowering `per_device_train_batch_size` (e.g. to 2) and/or `max_seq_length` (e.g. 1024).  
- **Login/API errors** → Make sure you’re logged in: `huggingface-cli login` and `wandb login`.

---

### Step 9e: Watch progress (optional)

- **In the terminal:** Loss should generally go down over time.  
- **In Weights & Biases:** Go to [wandb.ai](https://wandb.ai), open project **bible-ai**, and find the run **qwen3-4b-bible-John**. You’ll see loss curves and system stats.

If loss doesn’t decrease or you see NaNs, something may be wrong (e.g. learning rate too high, bad data). W&B helps you compare runs later.

---

### Step 9f: When training finishes

When the script exits normally, you’ll see something like:

```
Saved LoRA adapter to c:\Users\ttimm\Desktop\John\bible-ai-assistant\models\qwen3-4b-bible-John
```

**What’s in that folder:** **adapter_config.json** and **adapter_model.safetensors** (the LoRA weights), plus **tokenizer** files so you can load the same tokenizer when you use the model.

**Quick reference:**

| Goal | Command |
|------|--------|
| Train with default name (qwen3-4b-bible-John) | `python training/train_unsloth.py` |
| Train with custom name | `python training/train_unsloth.py --run-name YourName` |
| Use local base model | `python training/train_unsloth.py --model-path models/base_model` |
| Where adapter is saved | `models/qwen3-4b-bible-John` (or whatever you passed to `--run-name`) |
| Where config lives | `training/config.yaml` |

---

## Step 10: Phase 3 — Merge adapters and evaluate

After training, merge the LoRA adapter into the base model, then run evaluation.

**Merge adapters:**

```powershell
# Default (uses models/qwen3-4b-bible-John)
python training/merge_adapters.py

# For v2 or v3 runs, specify the LoRA path:
python training/merge_adapters.py --lora-path models/qwen3-4b-bible-John-v3
```

This produces a full model (e.g. `models/qwen3-4b-bible-John-v3-merged`) that you can convert to GGUF. The output folder name matches `{lora_path}-merged`.

**Evaluate:**

```powershell
python training/evaluate.py
```

Pass = verse retrieval + constitution compliance (5/5). Ensure zero fabricated verses and constitution pass before moving to Phase 4.

**Checkpoint:** **v0.3.0** — Fine-tuning complete. Next: Step 11 (quantize to GGUF and Ollama).

---

## Step 11: Phase 4 — Quantize to GGUF and Run in Ollama

Follow this after **Phase 3** (training, merge, evaluation). Goal: convert your merged model to GGUF, load it in Ollama, and run a local smoke test.

**What you’re doing:**

1. **Convert** the merged Hugging Face model (e.g. `models/qwen3-4b-bible-John-v3-merged`) to GGUF (float16) with `--outtype f16`.
2. **Quantize** to Q4_K_M for smaller size and faster inference (Ollama / llama.cpp).
3. **Create an Ollama model** using a Modelfile (system prompt + GGUF path).
4. **Smoke test** with `ollama run bible-assistant "What does John 3:16 say?"`

**Checkpoint:** **v0.4.0** — Model quantized to GGUF, tested in Ollama locally.

---

### Prerequisites

- **Ollama** installed: https://ollama.com/download/windows  
- **llama.cpp** (for conversion and quantize). Two options:
  - **Option A:** Clone and build locally (recommended for full control).
  - **Option B:** Use a Python GGUF converter that can read Hugging Face models (e.g. from `llama.cpp` repo).

---

### Option A: Using llama.cpp (clone and build)

#### 1. Clone and prepare llama.cpp

```powershell
cd c:\Users\ttimm\Desktop\John
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

Install Python deps for the converter:

```powershell
pip install -r requirements.txt
```

(Use your `bible-ai-assistant` conda env so `torch` etc. are available.)

#### 2. Convert merged model to GGUF (f16)

From the **project root** (`bible-ai-assistant`), point the converter at your merged model. For v3 (recommended):

```powershell
cd c:\Users\ttimm\Desktop\John\bible-ai-assistant
python ..\llama.cpp\convert_hf_to_gguf.py models\qwen3-4b-bible-John-v3-merged --outfile models\qwen3-4b-bible-John-v3-f16.gguf --outtype f16
```

For v2: use `models\qwen3-4b-bible-John-v2-merged` and `qwen3-4b-bible-John-v2-f16.gguf` instead.

If the script name or path differs (e.g. `convert-hf-to-gguf.py`), adjust. Some llama.cpp versions use different names; check the repo’s root for `convert*hf*gguf*.py`.

#### 3. Build llama.cpp and quantize

On Windows, build with CMake (see https://github.com/ggerganov/llama.cpp#build). You need **CMake** and **Visual Studio Build Tools** (C++) installed on the machine.

```powershell
cd c:\Users\ttimm\Desktop\John\llama.cpp
mkdir build
cd build
cmake .. -DGGUF_CUDA=ON
cmake --build . --config Release
```

The quantize tool is built as **llama-quantize** (not `quantize`). Build it explicitly if needed:

```powershell
cmake --build . --config Release --target llama-quantize
```

Then quantize to Q4_K_M (from `llama.cpp/build`):

```powershell
.\bin\Release\llama-quantize.exe ..\..\bible-ai-assistant\models\qwen3-4b-bible-John-v3-f16.gguf ..\..\bible-ai-assistant\models\qwen3-4b-bible-John-v3-q4_k_m.gguf Q4_K_M
```

(If the executable is elsewhere, run `dir bin\Release` to find it. Paths assume you’re in `llama.cpp/build`.)

#### 4. Create Ollama Modelfile

From **project root**:

```powershell
cd c:\Users\ttimm\Desktop\John\bible-ai-assistant
```

Create the Modelfile from your system prompt:

```powershell
python deployment/pc/generate_modelfile.py
```

This writes `deployment/pc/Modelfile` using `prompts/system_prompt.txt` and an **absolute path** to the GGUF so Ollama finds the local file (a relative path can be treated as a model name to pull). The generated Modelfile includes `num_predict 256`, `repeat_penalty 1.65`, and `repeat_last_n 128` to limit response length and discourage repetition. By default, `generate_modelfile.py` points to `qwen3-4b-bible-John-v3-q4_k_m.gguf`; edit its `GGUF_PATH` if you use a different run (e.g. v2 or f16).

**After editing `prompts/system_prompt.txt`** (e.g. tone guidelines, meta-questions), regenerate the Modelfile and recreate the Ollama model so changes take effect:
```powershell
python deployment/pc/generate_modelfile.py
ollama create bible-assistant -f deployment/pc/Modelfile
```

Or create `deployment/pc/Modelfile` by hand (copy from `Modelfile.template` and edit). Use an **absolute path** in `FROM` on Windows (e.g. `C:/Users/.../models/qwen3-4b-bible-John-v3-q4_k_m.gguf`) so `ollama create` does not try to pull from the registry. Include:

```
FROM C:/path/to/your/bible-ai-assistant/models/qwen3-4b-bible-John-v3-q4_k_m.gguf
SYSTEM """..."""   # full contents of prompts/system_prompt.txt (simplified ~15 lines)
PARAMETER temperature 0.2
PARAMETER num_ctx 2048
PARAMETER num_predict 256
PARAMETER repeat_penalty 1.65
PARAMETER repeat_last_n 128
```

**Important:** Use the entire contents of `prompts/system_prompt.txt` in the SYSTEM block. The system prompt is intentionally short for 4B models to reduce instruction leakage. `num_predict` caps output length; `repeat_penalty` and `repeat_last_n` discourage looping.

Create the Ollama model (from project root). If `ollama` is not in your PATH, use the full path to `ollama.exe` (e.g. `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` on Windows):

```powershell
cd c:\Users\ttimm\Desktop\John\bible-ai-assistant
ollama create bible-assistant -f deployment/pc/Modelfile
```

#### 5. Smoke test

Start Ollama (or rely on the app), then:

```powershell
ollama run bible-assistant "What does John 3:16 say?"
```

You should get a verse-accurate answer. The session may wait for more input; type **/bye** to exit. If the model repeats indefinitely, ensure the Modelfile includes `PARAMETER num_predict 256` and recreate the model.

**One-shot test (single response, no chat loop):** From PowerShell:  
`Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -ContentType "application/json" -Body '{"model":"bible-assistant","prompt":"What does Psalm 23:1 say?","stream":false,"options":{"num_predict":256}}'`  
From Command Prompt (if Ollama is not in PATH): use the full path to `ollama.exe` (e.g. `"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"`).

Then continue to **Phase 5: RAG** (Step 12).

---

### Option B: Alternative converters

- **Unsloth:** Some Unsloth workflows can export to GGUF; check their docs for “save GGUF” or “export llama.cpp”.
- **Other tools:** Any pipeline that turns a Hugging Face CausalLM into a GGUF file (then quantize with llama.cpp’s `quantize`) is fine. The Modelfile and Ollama steps stay the same.

---

### Troubleshooting (Step 11)

- **“pull model manifest: file does not exist”** — Use an **absolute path** in the Modelfile’s `FROM` line (e.g. generate with `python deployment/pc/generate_modelfile.py`, which writes an absolute path).
- **“ollama” not recognized** — Use the full path to `ollama.exe` (e.g. `"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"` on Windows) or open a new terminal after installing Ollama.
- **Model repeats forever** — Add `PARAMETER num_predict 256` to the Modelfile, regenerate, and run `ollama create bible-assistant -f deployment/pc/Modelfile` again.
- **“Failed to load model”** — Ensure `FROM` points to the correct `.gguf` path and you run `ollama create` from a directory that makes that path valid.
- **Slow or OOM** — Q4_K_M is usually fine on 16GB; if needed, use a smaller `num_ctx` or a smaller quantization.
- **Wrong behavior** — Ensure the Modelfile’s `SYSTEM` block is exactly your `prompts/system_prompt.txt` (no truncation).
- **Qwen3 tokenizer error during convert** — If the converter fails on tokenizer load, use a llama.cpp converter that supports Qwen3 (e.g. with a hub tokenizer fallback); see project notes.
- **"llama-quantize" not recognized** — The exe may be at `build\bin\Release\llama-quantize.exe` (when using `cmake -B build`) or `build\Release\llama-quantize.exe` depending on your CMake/Visual Studio setup. Check both locations.

---

## Step 12: Phase 5 — RAG (Retrieval-Augmented Generation)

After your model runs in Ollama (Step 11), add a **RAG layer** so answers are grounded in retrieved verses. The RAG server sits between the client (e.g. OpenClaw or a UI) and Ollama: it finds relevant Scripture for each user question, injects it into the prompt, then forwards to Ollama.

**What you’re doing:**

1. **Build a vector index** — Turn `data/raw/bible_web.json` into a ChromaDB index using **nomic-embed-text-v1.5** (with `search_document:` prefix). One-time step.
2. **Run the RAG server** — FastAPI app that accepts OpenAI-style `/v1/chat/completions`, retrieves top-k verses for the last user message, augments the prompt, and forwards to Ollama. Handles **meta-questions** (e.g. “What can you do?”) without verse retrieval—the model answers directly. Strips OpenClaw metadata and suffix instructions (e.g. “Answer in quotes, then add explanation”) so capability questions get natural responses.
3. **Test** — Call the RAG server (or use `query_test.py` to sanity-check retrieval). Checkpoint: **v0.5.0**.

**Checkpoint:** **v0.5.0** — RAG layer complete; ChromaDB indexed, server augments prompts and forwards to Ollama.

---

### Prerequisites

- **Step 8 done** — `data/raw/bible_web.json` (or `bible.json`) must exist.
- **Ollama** running with `bible-assistant` (Step 11).
- **Dependencies** — `chromadb`, `sentence-transformers`, `fastapi`, `uvicorn`, `httpx` (in `requirements.txt`).

---

### Step 12a: Build the ChromaDB index

From the **project root** with `bible-ai-assistant` conda env activated:

```powershell
cd c:\Users\ttimm\Desktop\John\bible-ai-assistant
conda activate bible-ai-assistant

python rag/build_index.py
```

**What you’ll see:** The script loads verses from `data/raw/bible_web.json`, encodes them in batches with **nomic-embed-text-v1.5** (prefix `search_document:`), and writes the index to **rag/chroma_db/**. First run downloads the embedding model; then indexing takes a few minutes. Do not commit `rag/chroma_db/` (add to `.gitignore` if needed).

---

### Step 12b: (Optional) Sanity-check retrieval

```powershell
python rag/query_test.py
```

This queries for “What does John 3:16 say?” and prints the top 5 retrieved verses. You should see John 3:16 in or near the top results.

---

### Step 12c: Start the RAG server

From the project root:

```powershell
uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081
```

The server listens on port **8081** and forwards chat requests to Ollama (default `http://localhost:11434`). Ensure Ollama is running and has the `bible-assistant` model.

**Environment (optional):**

- **OLLAMA_URL** — Default `http://localhost:11434`. Set if Ollama is on another host.
- **RAG_TOP_K** — Number of verses to inject (default `5`).

---

### Step 12d: Test the RAG endpoint

With the server running, from another terminal:

```powershell
curl -X POST http://localhost:8081/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"bible-assistant\", \"messages\": [{\"role\": \"user\", \"content\": \"What does Psalm 23:1 say?\"}]}"
```

You should get a JSON response with the model’s answer, grounded in the retrieved verses the server injected.

**Using from OpenClaw / UI:** Point the client at `http://localhost:8081/v1` instead of `http://localhost:11434/v1` so that all chat goes through RAG.

---

### Troubleshooting (Step 12)

- **“ChromaDB index not found”** — Run `python rag/build_index.py` first.
- **“No Bible JSON found”** — Ensure Step 8 is done and `data/raw/bible_web.json` exists.
- **Slow first request** — The RAG server loads the embedder and ChromaDB on first use; later requests are faster.
- **Ollama connection refused** — Start Ollama and ensure `bible-assistant` is available (`ollama list`).
- **Responses start with "Answer:" or sound robotic** — Regenerate the Modelfile from the updated system prompt and recreate the model: `python deployment/pc/generate_modelfile.py` then `ollama create bible-assistant -f deployment/pc/Modelfile`. The system prompt includes tone guidelines to avoid stiff phrasing.
- **“What can you do?” returns verses instead of capabilities** — The RAG server detects meta-questions and skips retrieval. If this still happens, regenerate the Modelfile (`python deployment/pc/generate_modelfile.py`) and recreate the model (`ollama create bible-assistant -f deployment/pc/Modelfile`) so the updated system prompt (META-QUESTIONS section) is applied.
- **Responses start with “Answer:” or sound robotic** — Regenerate the Modelfile and recreate the model (see Modelfile note in Step 11). The system prompt forbids “Answer:” and instructs natural, conversational tone.

---

## Step 13: Phase 6 — OpenClaw + Telegram

After RAG is running (Step 12), add the **OpenClaw** agent and **Telegram** so you can chat with the Bible assistant from your phone or any client. OpenClaw talks to your RAG server (which forwards to Ollama); Telegram is one interface.

**What you’re doing:**

1. **Install OpenClaw** — Node.js agent that routes messages to your LLM endpoint.
2. **Point OpenClaw at the RAG server** — Use `http://localhost:8081/v1` so every request is augmented with retrieved verses.
3. **Create a Telegram bot** — Get a token from @BotFather and add it to `.env`.
4. **Run the full stack** — Ollama + RAG server + OpenClaw; connect Telegram to the bot.

**Checkpoint:** **v0.6.0** — Full dev stack: Ollama + RAG + OpenClaw + Telegram.

---

### Prerequisites

- **Steps 11–12 done** — Ollama with `bible-assistant`, RAG server on port 8081, ChromaDB index built.
- **Node.js** (LTS) installed — OpenClaw is an npm package.
- **Telegram account** — For @BotFather and testing.

---

### Step 13a: Install OpenClaw

From any terminal (PowerShell or Command Prompt):

```powershell
npm install -g openclaw
```

Verify: `openclaw --version` (or `npx openclaw --help`).

---

### Step 13b: Configure OpenClaw to use the RAG server

Run the onboarding wizard so OpenClaw uses your local RAG endpoint (not Ollama directly):

```powershell
openclaw onboard
```

When prompted for the **LLM / API endpoint**, use:

- **Base URL:** `http://localhost:8081/v1`
- **Model name:** `bible-assistant` (must match the model name your RAG server expects).

This way all chat goes through the RAG server, which retrieves verses and forwards to Ollama.

**SOUL.md (constitution):** OpenClaw may ask for or use a SOUL.md file that describes the agent’s behavior. You can point it at your project’s **`prompts/system_prompt.txt`** or a short SOUL.md that references the Bible assistant’s role (e.g. “You are a Bible AI Assistant…”). The RAG server and Ollama Modelfile already inject the full system prompt; SOUL.md can be a brief high-level description for the agent framework. The RAG server automatically strips OpenClaw metadata (e.g. “Sender (untrusted metadata)”) from user messages before processing, so capability questions like “What can you do?” work correctly.

---

### Step 13c: Create a Telegram bot and add token

1. In Telegram, open a chat with **@BotFather**.
2. Send `/newbot` and follow the prompts (name and username for the bot).
3. Copy the **token** BotFather returns (e.g. `7123456789:AAH...`).
4. In your project, ensure `.env` exists (copy from `.env.example` if needed) and add or set:
   ```bash
   TELEGRAM_BOT_TOKEN=your_actual_token_here
   ```
   Never commit `.env`; it is in `.gitignore`.

---

### Step 13d: Run the full stack and connect Telegram

1. **Terminal 1 — Ollama:** Ensure Ollama is running and `bible-assistant` is loaded (e.g. start the Ollama app or `ollama serve`).
2. **Terminal 2 — RAG server:**
   ```powershell
   cd c:\Users\ttimm\Desktop\John\bible-ai-assistant
   conda activate bible-ai-assistant
   uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081
   ```
3. **Terminal 3 — OpenClaw:** Start the OpenClaw gateway (use `openclaw gateway`, not `openclaw start`):
   ```powershell
   openclaw gateway
   ```
   Ensure OpenClaw is configured to read `TELEGRAM_BOT_TOKEN` from `.env` (or set it in the OpenClaw config). The gateway connects to the RAG server at `http://localhost:8081/v1`.
4. **Telegram:** Open your bot in Telegram and send a message (e.g. “What does John 3:16 say?”). You should get a response from the Bible model via RAG + Ollama.

**Checkpoint:** **v0.6.0** — Full dev stack with Telegram.

---

### Troubleshooting (Step 13)

- **OpenClaw can’t reach the LLM** — Ensure the RAG server is running on 8081 and that the endpoint in OpenClaw is `http://localhost:8081/v1` (and that you use `bible-assistant` as the model name).
- **Telegram bot doesn’t respond** — Check that `TELEGRAM_BOT_TOKEN` is set in `.env` and that OpenClaw is running and connected to Telegram (see OpenClaw docs for gateway/connector setup).
- **Wrong or empty answers** — Confirm RAG is working: run `python rag/query_test.py` and test the RAG server with curl (Step 12d). Then ensure OpenClaw is pointing at the RAG server, not directly at Ollama.
- **"What can you do?" returns verses instead of capabilities** — The RAG server detects meta-questions and skips verse retrieval; if this still fails, ensure you are on the latest RAG server and that OpenClaw metadata is being stripped (check logs).

---

## Step 14 onward: Phase 7+ (Voice, Deployment)

- **Phase 7:** Voice + Gradio — Faster-Whisper (STT), Kokoro TTS, voice tab in UI.
- **Phase 8:** Jetson + VPS — Edge deployment, Tailscale, production.

See **docs/DEVELOPMENT_WORKFLOW.md** for the full phase list and version tags. For a quick index of all docs, see **docs/README.md**.

---

*Glory to God. We’ll build this in a way that honors the Word and serves others.*
