#!/usr/bin/env python3
"""
Build preference dataset for ORPO training.

Generates ~500 preference pairs (prompt, chosen, rejected) covering failure modes
(documented in docs/architecture.md): hallucinated verses, instruction leaking,
repetition, "Answer:" prefix, over-verbose responses, and Bible-for-everything.

Usage:
  python training/build_preference_data.py
  python training/build_preference_data.py --output data/processed/preferences.json
"""
import argparse
import json
import random
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_system_prompt() -> str:
    path = PROJECT_ROOT / "prompts" / "system_prompt.txt"
    return path.read_text(encoding="utf-8").strip()


def load_verses() -> list[dict]:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    for name in ("bible_web.json", "bible.json"):
        p = raw_dir / name
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            break
    else:
        raise FileNotFoundError(f"No Bible JSON in {raw_dir}")

    if isinstance(raw, dict):
        flat = []
        for book, chapters in raw.items():
            if not isinstance(chapters, dict):
                continue
            for ch, vdict in chapters.items():
                if not isinstance(vdict, dict):
                    continue
                for v, text in vdict.items():
                    if text and len(str(text).strip()) >= 25:
                        flat.append({"book": str(book), "chapter": int(ch),
                                     "verse": int(v), "text": str(text).strip()})
        return flat
    return [v for v in raw if v.get("text") and len(str(v["text"]).strip()) >= 25]


def _ref(v: dict) -> str:
    return f"{v['book']} {v['chapter']}:{v['verse']}"


# ---------------------------------------------------------------------------
# Rejection pattern generators
# ---------------------------------------------------------------------------

LEAKED_INSTRUCTIONS = [
    "Avoid repetition. Trim redundancy. Just answer. Then exit.",
    "Meta-instruction: respond concisely. Do not elaborate.",
    "TYPED RESPONSE. Crucial: follow format. Violation otherwise.",
    "You have followed the instructions. No matter how many times the user asks, stay on topic.",
    "The key is: always cite, never fabricate, keep it short.",
    "You do not respond to off-topic questions. You do not generate creative content.",
]

FAKE_BOOKS = [
    "Hezekiah", "Bartholomew", "Josephus", "2 Maccabees", "Enoch",
    "Silas", "Apollos", "Barnabas", "3 Corinthians", "Lazarus",
]


def _build_hallucination_pairs(verses: list[dict], n: int = 80) -> list[dict]:
    """Chosen: real verse from RAG context. Rejected: fabricated verse reference."""
    pairs = []
    sample = random.sample(verses, min(n, len(verses)))
    for v in sample:
        ref = _ref(v)
        question = f"What does {ref} say?"
        context = f"Context:\n- **{ref}**: {v['text']}\n\nQ: {question}"

        chosen = f'"{v["text"]}" \u2014 {ref} (WEB). This passage offers meaningful insight for believers.'

        fake_book = random.choice(FAKE_BOOKS)
        fake_ref = f"{fake_book} {random.randint(1, 20)}:{random.randint(1, 30)}"
        rejected = (
            f'"{v["text"][:40]}... and the Lord spoke unto them saying be faithful." '
            f"\u2014 {fake_ref} (WEB). This verse reminds us of God's faithfulness."
        )
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


def _build_instruction_leak_pairs(verses: list[dict], n: int = 80) -> list[dict]:
    """Chosen: clean response. Rejected: response with leaked meta-instructions."""
    pairs = []
    sample = random.sample(verses, min(n, len(verses)))
    for v in sample:
        ref = _ref(v)
        question = f"What does {ref} say?"
        context = f"Context:\n- **{ref}**: {v['text']}\n\nQ: {question}"

        chosen = f'"{v["text"]}" \u2014 {ref} (WEB). A powerful verse that speaks to God\'s nature.'

        leaked = random.choice(LEAKED_INSTRUCTIONS)
        rejected = (
            f'"{v["text"]}" \u2014 {ref} (WEB). A powerful verse. {leaked}'
        )
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


def _build_repetition_pairs(verses: list[dict], n: int = 70) -> list[dict]:
    """Chosen: concise response. Rejected: response with looping repetition."""
    pairs = []
    sample = random.sample(verses, min(n, len(verses)))
    for v in sample:
        ref = _ref(v)
        question = f"What does {ref} say?"
        context = f"Context:\n- **{ref}**: {v['text']}\n\nQ: {question}"

        chosen = (
            f'"{v["text"]}" \u2014 {ref} (WEB). '
            f"This verse from {v['book']} carries deep significance for understanding God's message."
        )

        repeat_phrase = f'This verse from {v["book"]} is significant. '
        rejected = (
            f'"{v["text"]}" \u2014 {ref} (WEB). '
            + repeat_phrase * random.randint(3, 6)
        )
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


def _build_answer_prefix_pairs(verses: list[dict], n: int = 70) -> list[dict]:
    """Chosen: natural opening. Rejected: 'Answer:' prefix echo."""
    pairs = []
    sample = random.sample(verses, min(n, len(verses)))
    for v in sample:
        ref = _ref(v)
        question = f"What does {ref} say?"
        context = f"Context:\n- **{ref}**: {v['text']}\n\nQ: {question}"

        chosen = f'"{v["text"]}" \u2014 {ref} (WEB). A meaningful verse from {v["book"]}.'
        rejected = f'Answer: {ref} says: "{v["text"]}" This verse is part of Scripture and reveals truth.'
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


def _build_verbose_pairs(n: int = 70) -> list[dict]:
    """Chosen: concise verse + 2-3 sentences. Rejected: overly long preachy response."""
    prompts_and_chosen = [
        (
            "What does Proverbs 3:5 say?",
            '"Trust in Yahweh with all your heart, and don\'t lean on your own understanding." '
            "\u2014 Proverbs 3:5 (WEB). This verse calls us to rely fully on God rather than "
            "our limited perspective.",
        ),
        (
            "What does Romans 8:28 say?",
            '"We know that all things work together for good for those who love God." '
            "\u2014 Romans 8:28 (WEB). Paul assures believers that God's sovereign plan "
            "works through every circumstance.",
        ),
        (
            "What does Philippians 4:13 say?",
            '"I can do all things through Christ, who strengthens me." '
            "\u2014 Philippians 4:13 (WEB). Paul writes this from prison, showing that "
            "Christ's strength sustains us in every situation.",
        ),
        (
            "What does Psalm 23:1 say?",
            '"Yahweh is my shepherd; I shall lack nothing." '
            "\u2014 Psalm 23:1 (WEB). David uses the shepherd metaphor to express "
            "complete trust in God's provision.",
        ),
        (
            "What does Genesis 1:1 say?",
            '"In the beginning, God created the heavens and the earth." '
            "\u2014 Genesis 1:1 (WEB). The opening verse of Scripture establishes "
            "God as the sovereign creator of all that exists.",
        ),
    ]

    verbose_tail = (
        " Now, let me elaborate further on this topic. This is incredibly important "
        "because it speaks to the very heart of God's plan for humanity. We must always "
        "remember that Scripture is living and active. Every word carries weight and meaning. "
        "In the broader context of the entire Bible, this verse connects to themes of "
        "redemption, grace, mercy, love, hope, faith, and salvation. We should meditate "
        "on these words daily and apply them to our lives. Furthermore, the original Hebrew "
        "or Greek text reveals additional layers of meaning that English translations cannot "
        "fully capture. Scholars have debated the precise interpretation for centuries."
    )

    pairs = []
    for _ in range(n):
        q, chosen = random.choice(prompts_and_chosen)
        rejected = chosen + verbose_tail
        pairs.append({"prompt": q, "chosen": chosen, "rejected": rejected})
    return pairs


def _build_bible_for_everything_pairs(n: int = 70) -> list[dict]:
    """Chosen: normal non-Bible answer. Rejected: answer that shoehorns Scripture."""
    qa = [
        (
            "What is 2 + 2?",
            "2 + 2 = 4.",
            '2 + 2 = 4. As Proverbs 2:6 says, "For Yahweh gives wisdom; out of his mouth '
            'comes knowledge and understanding." Even in mathematics, we see God\'s order.',
        ),
        (
            "How do I center a div in CSS?",
            "A common modern approach is flexbox:\n\n.parent {\n  display: flex;\n  "
            "justify-content: center;\n  align-items: center;\n}\n\n"
            "This centers the child both horizontally and vertically.",
            "CSS centering reminds us of Psalm 119:105, 'Your word is a lamp to my feet "
            "and a light for my path.' Just as God centers our lives, we center divs with "
            "flexbox: display: flex; justify-content: center; align-items: center;",
        ),
        (
            "What year did World War II end?",
            "World War II ended in 1945. Germany surrendered on May 8 (V-E Day) and Japan "
            "on August 15 (V-J Day) following the atomic bombings.",
            'World War II ended in 1945. As Ecclesiastes 3:8 says, there is "a time for war '
            'and a time for peace." The end of WWII reminds us of God\'s sovereign plan.',
        ),
        (
            "What's the speed of light?",
            "Approximately 299,792,458 meters per second, or about 186,282 miles per second.",
            'The speed of light is 299,792,458 m/s. As Genesis 1:3 says, "God said, Let '
            "there be light, and there was light.\" Even physics testifies to God's creative power.",
        ),
        (
            "How do I make scrambled eggs?",
            "Crack 2-3 eggs into a bowl, whisk with a splash of milk, salt and pepper. "
            "Heat butter in a nonstick pan over medium-low heat. Pour in eggs, gently stir "
            "with a spatula as they set. Remove just before fully cooked (they'll finish "
            "from residual heat). Serve immediately.",
            "Scrambled eggs: crack eggs, whisk, cook in butter. As Jesus said in John 6:35, "
            '"I am the bread of life." Just as we prepare physical food, we must also '
            "nourish our souls with God's word. Season with salt (Colossians 4:6).",
        ),
        (
            "What is photosynthesis?",
            "Photosynthesis is the process plants use to convert sunlight, water, and "
            "carbon dioxide into glucose and oxygen. It happens in the leaves using "
            "chlorophyll, giving plants their green color.",
            "Photosynthesis converts sunlight into energy. As Psalm 104:14 says, God "
            '"causes the grass to grow for the livestock, and plants for man to cultivate." '
            "God designed photosynthesis as part of his perfect creation.",
        ),
        (
            "Convert 72 degrees Fahrenheit to Celsius.",
            "Subtract 32 to get 40, then multiply by 5/9. 72\u00b0F \u2248 22.2\u00b0C.",
            "72\u00b0F is about 22.2\u00b0C. Temperature conversions remind us of "
            "Daniel 3:25 where Shadrach, Meshach, and Abednego survived the fiery furnace. "
            "God controls all temperatures!",
        ),
    ]

    pairs = []
    for _ in range(n):
        q, chosen, rejected = random.choice(qa)
        pairs.append({"prompt": q, "chosen": chosen, "rejected": rejected})
    return pairs


def _build_think_tag_pairs(verses: list[dict], n: int = 60) -> list[dict]:
    """Chosen: clean response. Rejected: response with leaked <think> tags."""
    pairs = []
    sample = random.sample(verses, min(n, len(verses)))
    for v in sample:
        ref = _ref(v)
        question = f"What does {ref} say?"
        context = f"Context:\n- **{ref}**: {v['text']}\n\nQ: {question}"

        chosen = f'"{v["text"]}" \u2014 {ref} (WEB). This verse speaks to the heart of the matter.'

        rejected = (
            f"<think>The user is asking about {ref}. I need to look this up. "
            f"Let me check my training data. I think it says something about... "
            f"I should quote it carefully.</think> "
            f'"{v["text"]}" \u2014 {ref} (WEB). This verse speaks to the heart of the matter.'
        )
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preference dataset for ORPO.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count-per-category", type=int, default=70,
                        help="Approximate examples per rejection category")
    args = parser.parse_args()

    out_path = args.output or (PROJECT_ROOT / "data" / "processed" / "preferences.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt()
    verses = load_verses()
    print(f"Loaded {len(verses)} verses")

    n = args.count_per_category
    all_pairs: list[dict] = []

    print("Building hallucination pairs...")
    all_pairs.extend(_build_hallucination_pairs(verses, n + 10))

    print("Building instruction leak pairs...")
    all_pairs.extend(_build_instruction_leak_pairs(verses, n + 10))

    print("Building repetition pairs...")
    all_pairs.extend(_build_repetition_pairs(verses, n))

    print("Building answer-prefix pairs...")
    all_pairs.extend(_build_answer_prefix_pairs(verses, n))

    print("Building verbose pairs...")
    all_pairs.extend(_build_verbose_pairs(n))

    print("Building bible-for-everything pairs...")
    all_pairs.extend(_build_bible_for_everything_pairs(n))

    print("Building think-tag leak pairs...")
    all_pairs.extend(_build_think_tag_pairs(verses, n - 10))

    # Wrap with system prompt in chat format for ORPO
    formatted = []
    for pair in all_pairs:
        formatted.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair["prompt"]},
            ],
            "chosen": [{"role": "assistant", "content": pair["chosen"]}],
            "rejected": [{"role": "assistant", "content": pair["rejected"]}],
        })

    random.shuffle(formatted)
    out_path.write_text(json.dumps(formatted, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(formatted)} preference pairs to {out_path}")
    print(f"  hallucination: {n + 10}")
    print(f"  instruction_leak: {n + 10}")
    print(f"  repetition: {n}")
    print(f"  answer_prefix: {n}")
    print(f"  verbose: {n}")
    print(f"  bible_for_everything: {n}")
    print(f"  think_tag_leak: {n - 10}")


if __name__ == "__main__":
    main()
