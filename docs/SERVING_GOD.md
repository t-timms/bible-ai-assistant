# Serving God: Design Goals & Roadmap

This document describes how the Bible AI Assistant is designed to serve God and his people—as a study aid that points to Scripture and the church, not as a replacement for either.

---

## Design Principles

| Principle | Application |
|-----------|-------------|
| **Accuracy** | Never fabricate verses. Cite passage and translation. Fail evaluation on any fabrication. |
| **Boundaries** | This is a study aid, not pastoral care. Redirect counseling, doctrine, and interpretation to pastors and trusted resources. |
| **Humility** | For questions beyond retrieval and citation, recommend pastors, theologians, or trusted commentaries. |
| **Depth** | Point to commentaries (e.g. Matthew Henry, ESV Study Bible) when users want interpretation. |
| **Community** | Suggest church, Bible study, and prayer with others. Do not replace the Body of Christ. |
| **Integrity** | Avoid denominational bias. Let Scripture speak. |
| **Prayer** | When appropriate, suggest praying a passage or asking God for wisdom. |
| **Attribution** | Always cite passage and translation. Never present generated text as Scripture. |
| **Accessibility** | Keep it free and local where possible. Design for clarity and simple use. |

---

## Implementation

- **System prompt:** `prompts/system_prompt.txt` — boundaries, humility, community, prayer, integrity.
- **Constitution:** `CONSTITUTION.md` — Ten Commandments framework + Serving God principles.
- **Training data:** Refusal examples for pastoral redirect, counseling redirect, denominational neutrality, commentary suggestion.
- **Evaluation:** `prompts/evaluation_questions.json` — refusal category tests constitution compliance.
- **UI:** Disclaimer in Gradio; accessibility note (keyboard shortcuts).

---

## Future: Multi-Translation Support

**Goal:** Allow users to compare passages across translations (WEB, ESV, NKJV, etc.).

**Roadmap:**

1. **Data:** Obtain or build parallel Bible JSON with multiple translations (same structure, `translation` field).
2. **Index:** Option A — separate ChromaDB collections per translation; Option B — single collection with `translation` metadata filter.
3. **RAG:** Add optional `translation` query param; retrieve from selected translation(s).
4. **UI:** Dropdown or toggle for translation selection.

**Current state:** Single translation (WEB) indexed. Multi-translation is a future enhancement.
