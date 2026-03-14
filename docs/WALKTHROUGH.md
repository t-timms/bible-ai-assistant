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
2. **Creating a conda environment** — An isolated Python 3.11 environment so this project’s packages don’t conflict with others.
3. **Installing PyTorch nightly** — Your RTX 5070 Ti (Blackwell) needs CUDA 12.8+; stable PyTorch doesn’t support it yet, so we use the nightly build.
4. **Installing project dependencies** — `requirements.txt` (training, RAG, API, voice).
5. **Logging into Hugging Face and W&B** — So you can download models and log training runs.
6. **Downloading Qwen3 4B** — The base model you’ll fine-tune.

We’ll go through each of these one at a time so you understand why each step matters.

---

*Glory to God. We’ll build this in a way that honors the Word and serves others.*
