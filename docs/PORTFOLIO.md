# Bible AI Assistant — Portfolio & Interview Guide

Professional language for resume, portfolio sites, and technical interviews. All claims are factually accurate as of the current codebase (March 2026).

---

## One-Liner (Elevator Pitch)

**"A Bible Q&A system that fine-tunes Qwen3.5-4B with bf16 LoRA, uses hybrid RAG (dense + BM25 + reranking), and serves it locally via Ollama with a voice-enabled chat UI."**

---

## Resume Bullets

- **"Fine-tuned a 4B LLM for Bible Q&A using bf16 LoRA (Unsloth, PEFT, TRL) on ~1,800 diverse examples, achieving verse-accurate responses with constitutional guardrails."**
- **"Built a hybrid RAG pipeline (ChromaDB + nomic-embed-text-v1.5 + BM25 + Reciprocal Rank Fusion + cross-encoder reranking) serving 31,000+ verses with sub-second retrieval."**
- **"Deployed a fine-tuned model to Ollama via GGUF quantization, with FastAPI middleware for OpenAI-compatible RAG augmentation and optional ORPO preference alignment."**
- **"Implemented evaluation pipelines (keyword scoring + LLM-as-judge) and iterated on data diversity to resolve overfitting and instruction leakage."**

---

## Portfolio Summary (3–4 Sentences)

**Bible AI Assistant** is a domain-specific Q&A system that combines LLM fine-tuning with retrieval-augmented generation. The base model (Qwen3.5-4B) is adapted using bf16 LoRA on ~1,800 curated examples across verse lookups, theology, topical questions, cross-references, and refusal scenarios, with optional ORPO for preference alignment. A hybrid RAG pipeline (dense + sparse retrieval, Reciprocal Rank Fusion, and cross-encoder reranking) retrieves relevant verses from 31,000+ indexed passages, augments prompts with context, and forwards requests to a locally hosted model. The system exposes an OpenAI-compatible API, a Gradio chat interface with optional voice (Faster-Whisper STT + Kokoro TTS), and supports deployment on PC or edge hardware (e.g., Jetson Orin).

---

## Interview Talking Points

| Topic | What to Say |
|-------|-------------|
| **Architecture** | "Fine-tuned 4B LLM plus RAG. Retrieval is hybrid: dense search with ChromaDB and nomic-embed-text-v1.5, BM25 sparse search, Reciprocal Rank Fusion to merge rankings, and a cross-encoder reranker. A FastAPI middleware augments user prompts with retrieved context and forwards to Ollama." |
| **Why bf16 LoRA (not QLoRA)** | "Qwen3.5 doesn't recommend 4-bit quantization with Unsloth; it produces poor output. We use bf16 LoRA instead—full precision base, low-rank adapters—so we get quality while fitting in 16GB VRAM. On Blackwell GPUs (RTX 5070 Ti), bf16 is required; fp16 causes NaNs." |
| **Why RAG** | "RAG grounds answers in retrieved verses, reduces hallucination, and supports citation. Without it, a 4B model tends to fabricate references. With hybrid retrieval (dense + BM25 + reranking), we get high recall and precision for both verse lookups and thematic questions." |
| **Technical Challenges** | "Overfitting from 31,000 repetitive examples, Blackwell GPU compatibility (bf16-only), tokenizer differences between Qwen3 and Qwen3.5—solved with diverse data (~1,800 examples across 7 categories), shortened system prompt, and text-only tokenizer for training." |
| **Outcome** | "Accurate verse lookups and topical answers with citations; constitutional guardrails for safe, theologically consistent behavior; optional voice mode for hands-free use." |

---

## Skills to Highlight

| Skill | Where It Shows |
|-------|----------------|
| **LLM fine-tuning** | bf16 LoRA, PEFT, Unsloth, TRL, optional ORPO |
| **RAG** | ChromaDB, nomic-embed-text-v1.5, hybrid retrieval, Reciprocal Rank Fusion, bge-reranker-v2-m3 |
| **MLOps** | W&B, evaluation scripts, GGUF quantization, Ollama |
| **Backend** | FastAPI, OpenAI-compatible API design |
| **Deployment** | GGUF, Ollama, optional Docker (Kokoro TTS), Jetson docs |
