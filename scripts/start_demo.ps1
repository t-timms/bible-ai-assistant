# Bible AI Assistant — print demo launch steps (Ollama + RAG + Gradio).
# Does not start services; run each command in its own terminal.

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Write-Host ""
Write-Host "=== Bible AI Assistant — local demo ===" -ForegroundColor DarkYellow
Write-Host ""
Write-Host "Repo: $root"
Write-Host ""
Write-Host "1) Ollama" -ForegroundColor Cyan
Write-Host "   Ensure Ollama is running. Model example: bible-assistant-orpo"
Write-Host ""
Write-Host "2) RAG (port 8081)" -ForegroundColor Cyan
Write-Host "   cd `"$root`""
Write-Host "   conda activate <your-env>"
Write-Host "   uvicorn rag.rag_server:app --host 0.0.0.0 --port 8081"
Write-Host ""
Write-Host "3) Gradio UI (port 7860)" -ForegroundColor Cyan
Write-Host "   cd `"$root`""
Write-Host "   conda activate <your-env>"
Write-Host "   python ui/app.py"
Write-Host "   Open: http://127.0.0.1:7860"
Write-Host ""
Write-Host "Optional: Kokoro TTS on 8880 — see deployment/pc/voice_setup.md"
Write-Host "Docs: docs/DEMO_LAUNCH.md"
Write-Host ""
