"""Strip model chain-of-thought from user-facing text (no FastAPI/Chroma imports)."""

from __future__ import annotations

import re

# Qwen / Ollama: XML-style <think> ... </think> (see docs/benchmark_runs JSON)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_TAGS = re.compile(
    re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE) + r"\s*",
    re.DOTALL,
)
_THINK_OPEN_ONLY = re.compile(re.escape(_THINK_OPEN) + r".*", re.DOTALL)
# Line is only an open or close tag (handles weird spacing / partial streams)
_ORPHAN_THINK_LINE = re.compile(
    rf"(?m)^\s*(?:{re.escape(_THINK_OPEN)}|{re.escape(_THINK_CLOSE)})\s*$",
)
# BOM / NBSP / ZW* before tags — otherwise startswith("<think>") never fires in real Ollama output
_LEADING_TAG_CHUNK = re.compile(
    r"^[\ufeff\u00a0\u200b-\u200d\u3000\u2028\u2029\s]*"
    r"(?:\x3C/?\s*think\s*\x3E)"
    r"[ \t\n\r]*",
    re.IGNORECASE,
)
_LEAD_WS = re.compile(r"^[\ufeff\u00a0\u200b-\u200d\u3000\u2028\u2029\s]+")
# Whole line is any <...think...> variant (some GGUF builds add tokens inside the tag)
_ORPHAN_FLEX_THINK_LINE = re.compile(
    r"(?m)^[\s\u200b-\u200d\ufeff]*\x3C[^\n\r]*?think[^\n\r]*?\x3E\s*$",
    re.IGNORECASE,
)


def _strip_leading_think_xml_flex(text: str) -> str:
    """Remove leading `<...think...>` chunks (handles nonstandard Qwen/Ollama tag bodies)."""
    t = text
    while True:
        u = _LEAD_WS.sub("", t)
        if not u.startswith("<"):
            break
        end = u.find(">")
        if end == -1:
            break
        tag = u[: end + 1]
        if "think" not in tag.lower():
            break
        t = u[end + 1 :].lstrip(" \t\n\r")
    return t


def _strip_leading_think_markers(text: str) -> str:
    """Peel stray think open/close tags from the start (handles invisible leading whitespace)."""
    t = _strip_leading_think_xml_flex(text)
    while True:
        m = _LEADING_TAG_CHUNK.match(t)
        if not m:
            break
        t = t[m.end() :]
    t = _strip_leading_think_xml_flex(t)
    return t


def _strip_thinking_process_paragraphs(text: str) -> str:
    """Remove leading 'Thinking Process:' + numbered **...** analysis before the real answer."""
    if not text:
        return text
    t = text.strip()
    if not t.lower().startswith("thinking process:"):
        return text
    pos = 0
    while True:
        m = re.search(r"\n\s*\n\s*", t[pos:])
        if not m:
            break
        start_content = pos + m.end()
        rest = t[start_content:].lstrip()
        if not rest:
            break
        # Strong signals that the user-facing answer begins here
        if rest[0] in '"\'"\u201c\u2018':
            return rest
        if re.match(
            r"([A-Z][a-z]{1,22}\s+\d+:\d+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d+:\d+)",
            rest,
        ):
            return rest
        if re.match(
            r"(I'll|I can |I cannot |I'm a |Scripture |The Bible |According to |In [A-Z])",
            rest,
            re.IGNORECASE,
        ):
            return rest
        pos = start_content
    return t


def _strip_verbose_thinking_process(text: str) -> str:
    """Remove long Qwen plans: Thinking Process + **Retrieve Verse:** scaffolding."""
    if not text:
        return text
    head = text[:12000].lower()
    if "thinking process:" not in head and "**analyze the request**" not in head:
        return text
    # Model often ends planning with **Retrieve Verse:** then the quote
    m = re.search(r"(?is)\*\*\s*Retrieve\s+Verse:\*\*\s*", text)
    if m:
        return text[m.end() :].strip()
    # Fallback: first substantial quoted verse opener after planning noise
    for needle in (
        '"For God',
        "\u201cFor God",
        '"The Lord',
        "\u201cThe Lord",
        '"In the beginning',
        '"All things',
        '"Trust ',
        '"Love',
        '"Now faith',
        '"If you forgive',
    ):
        i = text.find(needle)
        if i > 0:
            return text[i:].strip()
    # Line-start quote deep in the buffer (verse line)
    m = re.search(r"(?m)^\s*[\"'\u201c\u2018][A-Za-z]", text[200:])
    if m:
        return text[200 + m.start() :].strip()
    return text


def strip_model_thinking(text: str | None) -> str:
    """Remove think-tag blocks and plain 'Thinking Process:' preambles."""
    if text is None:
        return text  # type: ignore[return-value]
    if not text:
        return text
    # Unify newlines so DOTALL + Windows CRLF behave like saved JSON
    text = "\n".join(text.splitlines())
    # Paired Qwen/Ollama blocks first: flex XML stripper treats leading `</think>` like
    # `<think>` and would remove only the opener, leaving body + `</think>` unmatched.
    cleaned = _THINK_TAGS.sub("", text)
    cleaned = _strip_leading_think_markers(cleaned)
    cleaned = _THINK_TAGS.sub("", cleaned)
    cleaned = _strip_leading_think_markers(cleaned)
    cleaned = _THINK_OPEN_ONLY.sub("", cleaned)
    cleaned = _ORPHAN_THINK_LINE.sub("", cleaned)
    cleaned = _ORPHAN_FLEX_THINK_LINE.sub("", cleaned)
    cleaned = cleaned.strip()
    cleaned = _strip_leading_think_markers(cleaned)
    cleaned = _strip_thinking_process_paragraphs(cleaned)
    cleaned = _strip_verbose_thinking_process(cleaned)
    cleaned = _strip_leading_think_xml_flex(cleaned.strip())
    # Paired-tag removal can leave a BOM glued to the answer (e.g. \ufeff after empty </think>…</think>).
    cleaned = re.sub(r"^\ufeff+", "", cleaned.strip())
    return cleaned.strip()
