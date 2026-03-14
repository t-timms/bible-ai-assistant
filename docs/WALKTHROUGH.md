# Step-by-Step Walkthrough

This document walks you through the Bible AI Assistant project with explanations so you learn as you go. Follow in order.

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

## Step 8: Phase 2 — Build the Bible Dataset

**Why:** Fine-tuning quality depends on dataset quality. We need 30k–50k Q&A examples in Qwen3 chat format (system + user + assistant) so the model learns to answer from Scripture and follow the constitution.

**8a. Get raw Bible text (WEB)**

This project uses the **World English Bible (WEB)**. Download WEB in JSON:

- **[TehShrike/world-english-bible](https://github.com/TehShrike/world-english-bible)** — clone or download the repo; copy the WEB JSON file into `data/raw/` and name it `bible_web.json` or `bible.json`. If the file is nested (book → chapter → verse → text), the dataset builder will flatten it.
- **Or** [scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases) — get WEB from the Formats folder; if it’s CSV or another format, convert to a JSON array of `{ "book", "chapter", "verse", "text" }` and save as `data/raw/bible.json`.

The dataset builder expects either a JSON **array** of verse objects or a **nested** object (book → chapter → verse → text); it will normalize to the format it needs.

**8b. Run the dataset builder**

From the project root with `bible-ai-assistant` activated:

```bash
python training/dataset_builder.py
```

By default this reads `data/raw/bible.json`, loads the system prompt from `prompts/system_prompt.txt`, generates verse-lookup and optional theology/constitution examples, and writes `data/processed/train.json`. Use `--max-examples` to cap the number (e.g. `--max-examples 50000`).

**8c. Check output**

- Open `data/processed/train.json` (or inspect the first few lines). Each entry should have a `messages` array with `system`, `user`, and `assistant` roles matching `data/sample.json`.
- Target for first run: **30,000–50,000** examples. More variety (theology, character studies, constitution tests) can be added later.

**Phase 2 checkpoint:** When `train.json` is ready, tag **v0.2.0** (“Dataset builder complete, 50k Bible Q&A”). Then continue to Phase 3: fine-tuning (guide Section 9).

---

*Glory to God. We’ll build this in a way that honors the Word and serves others.*
