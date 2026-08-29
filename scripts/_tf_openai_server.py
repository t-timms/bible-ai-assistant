"""Minimal OpenAI-compatible /v1/chat/completions server over a local HF model.

stdlib http.server only (no fastapi/uvicorn) so it runs in the bible-orpo env.
Used for benchmarking when neither GGUF/Ollama nor vLLM can serve the model
(Qwen3.5-4B hybrid: no llama.cpp support yet; vLLM UVA path broken under WSL2).
Greedy decoding, non-thinking, ChatML stops.

  MODEL_PATH=models/qwen3.5-4b-bible-v2-merged PORT=8001 SERVED_NAME=bible-v2-4b \
    python scripts/_tf_openai_server.py
"""

from __future__ import annotations

import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "models/qwen3.5-4b-bible-v2-merged")
PORT = int(os.environ.get("PORT", "8001"))
SERVED_NAME = os.environ.get("SERVED_NAME", "bible-eval")

print(f"[tf-server] loading {MODEL_PATH} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
)
model.eval()
_EOT = tok.convert_tokens_to_ids("<|im_end|>")
_STOP = [t for t in (_EOT, tok.eos_token_id) if t is not None]
print(f"[tf-server] ready on :{PORT} (stop ids {_STOP})", flush=True)


def _content_str(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return "" if c is None else str(c)


def _generate(messages: list[dict], max_tokens: int) -> tuple[str, int, int]:
    msgs = [
        {"role": m.get("role", "user"), "content": _content_str(m.get("content"))} for m in messages
    ]
    try:
        text = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if not text.rstrip().endswith("</think>"):
            text += "<think>\n\n</think>\n\n"
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    n_prompt = ids.shape[1]
    max_new = max(1, min(int(max_tokens or 512), 1024))
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=_STOP,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = out[0][n_prompt:]
    return tok.decode(gen, skip_special_tokens=True).strip(), n_prompt, int(gen.shape[0])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send(200, {"status": "ok", "model": SERVED_NAME})
        elif self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": SERVED_NAME, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            completion, n_p, n_c = _generate(req.get("messages", []), req.get("max_tokens", 512))
        except Exception as e:  # noqa: BLE001 - surface any failure to the client
            self._send(500, {"error": {"message": repr(e)}})
            return
        self._send(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": SERVED_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": n_p,
                    "completion_tokens": n_c,
                    "total_tokens": n_p + n_c,
                },
            },
        )


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
