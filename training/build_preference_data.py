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

    if not raw:
        raise ValueError(f"Bible JSON file is empty: {p}")

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
                        flat.append(
                            {
                                "book": str(book),
                                "chapter": int(ch),
                                "verse": int(v),
                                "text": str(text).strip(),
                            }
                        )
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
    "Hezekiah",
    "Bartholomew",
    "Josephus",
    "2 Maccabees",
    "Enoch",
    "Silas",
    "Apollos",
    "Barnabas",
    "3 Corinthians",
    "Lazarus",
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
        rejected = f'"{v["text"]}" \u2014 {ref} (WEB). A powerful verse. {leaked}'
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

        repeat_phrase = f"This verse from {v['book']} is significant. "
        rejected = f'"{v["text"]}" \u2014 {ref} (WEB). ' + repeat_phrase * random.randint(3, 6)
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
        rejected = (
            f'Answer: {ref} says: "{v["text"]}" This verse is part of Scripture and reveals truth.'
        )
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


def _build_verbose_pairs(verses: list[dict], n: int = 70) -> list[dict]:
    """Chosen: concise verse + 2 sentences. Rejected: same + preachy wall-of-text tail.

    Uses randomly sampled verses — like all other generators — so the model learns a
    general "be concise" principle rather than memorising 5 fixed prompts repeated 14×.
    """
    _verbose_tail = (
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
    sample = random.sample(verses, min(n, len(verses)))
    for v in sample:
        ref = _ref(v)
        question = f"What does {ref} say?"
        context = f"Context:\n- **{ref}**: {v['text']}\n\nQ: {question}"
        chosen = (
            f'"{v["text"]}" \u2014 {ref} (WEB). '
            f"This passage from {v['book']} offers meaningful guidance for the faithful."
        )
        rejected = chosen + _verbose_tail
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


def _build_bible_for_everything_pairs(n: int = 70) -> list[dict]:
    """Chosen: normal factual answer. Rejected: same answer with Scripture shoehorned in.

    30 diverse QA topics so each appears ~2× at default n=70 — avoiding the memorisation
    trap of the previous 7-topic list (each appeared ~10×).
    """
    qa = [
        # Math / Numbers
        (
            "What is 2 + 2?",
            "2 + 2 = 4.",
            '2 + 2 = 4. As Proverbs 2:6 says, "For Yahweh gives wisdom; out of his mouth '
            "comes knowledge and understanding.\" Even arithmetic reveals God's order.",
        ),
        (
            "How many sides does a hexagon have?",
            "A hexagon has six sides.",
            "A hexagon has six sides. This reminds us of the six days of creation in "
            "Genesis 1 — God's perfect design is found even in geometry.",
        ),
        (
            "What is the Pythagorean theorem?",
            "In a right triangle, a\u00b2 + b\u00b2 = c\u00b2, where c is the hypotenuse.",
            "The Pythagorean theorem states a\u00b2 + b\u00b2 = c\u00b2. As Proverbs 11:1 says, "
            '"A false balance is an abomination to Yahweh, but accurate weights are his delight." '
            "Mathematical truth is part of God's order.",
        ),
        (
            "What is the area of a circle with radius r?",
            "The area of a circle is \u03c0r\u00b2.",
            "The area of a circle is \u03c0r\u00b2. As Isaiah 40:22 says, God "
            '"sits above the circle of the earth." Even \u03c0 points to his infinite precision.',
        ),
        (
            "Convert 72 degrees Fahrenheit to Celsius.",
            "Subtract 32 to get 40, then multiply by 5/9. 72\u00b0F \u2248 22.2\u00b0C.",
            "72\u00b0F is about 22.2\u00b0C. This reminds us of Daniel 3:25 where "
            "Shadrach, Meshach, and Abednego survived the fiery furnace by God's protection.",
        ),
        (
            "Convert 100 degrees Celsius to Fahrenheit.",
            "100\u00b0C = 212\u00b0F. The formula is (C \u00d7 9/5) + 32.",
            "100\u00b0C = 212\u00b0F. The boiling point of water speaks to Revelation 21:8, "
            '"the lake that burns with fire and sulfur." Even temperature scales proclaim God\'s truth.',
        ),
        # Science
        (
            "What's the speed of light?",
            "Approximately 299,792,458 meters per second.",
            'The speed of light is 299,792,458 m/s. As Genesis 1:3 says, "God said, Let '
            "there be light, and there was light.\" Even physics testifies to God's creative power.",
        ),
        (
            "What is photosynthesis?",
            "Photosynthesis is the process plants use to convert sunlight, water, and "
            "carbon dioxide into glucose and oxygen using chlorophyll in their leaves.",
            "Photosynthesis converts sunlight into energy. As Psalm 104:14 says, God "
            '"causes the grass to grow for the livestock, and plants for man to cultivate." '
            "God designed photosynthesis as part of his perfect creation.",
        ),
        (
            "What is DNA?",
            "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions "
            "for all known living organisms.",
            'DNA carries our genetic code. As Psalm 139:14 says, we are "fearfully and '
            "wonderfully made.\" Even molecular biology reveals God's handiwork.",
        ),
        (
            "What is the chemical formula for water?",
            "The chemical formula for water is H\u2082O — two hydrogen atoms bonded to one oxygen atom.",
            "Water is H\u2082O. As John 4:14 says, Jesus offers living water "
            '"that will become in him a spring of water welling up to eternal life." '
            "Even chemistry points to spiritual truth.",
        ),
        (
            "What is the boiling point of water at sea level?",
            "Water boils at 100\u00b0C (212\u00b0F) at standard atmospheric pressure.",
            "Water boils at 100\u00b0C. As Revelation 3:15-16 says, God prefers us "
            '"hot or cold — not lukewarm." Even the states of water teach spiritual lessons.',
        ),
        (
            "What is the approximate distance from Earth to the Moon?",
            "The average distance from Earth to the Moon is about 384,400 km (238,855 miles).",
            "The Moon is about 384,400 km away. As Genesis 1:16 says, God "
            '"made the two great lights — the greater light to rule the day and the lesser '
            'light to rule the night." Astronomy proclaims his glory.',
        ),
        (
            "What is the speed of sound in air?",
            "The speed of sound in air at room temperature is approximately 343 meters per second.",
            "Sound travels at ~343 m/s. As Romans 10:17 says, "
            '"So faith comes from hearing, and hearing through the word of Christ." '
            "Even acoustics has spiritual meaning.",
        ),
        (
            "How does a microwave heat food?",
            "A microwave emits radiation that causes water molecules in food to vibrate rapidly, "
            "generating heat from molecular friction.",
            "Microwaves excite water molecules to heat food. As Hebrews 4:12 says, "
            '"For the word of God is living and active, sharper than any two-edged sword." '
            "God's word penetrates deeper than any microwave.",
        ),
        # History / Geography
        (
            "What year did World War II end?",
            "World War II ended in 1945. Germany surrendered on May 8 (V-E Day); Japan on "
            "September 2 (V-J Day).",
            'World War II ended in 1945. As Ecclesiastes 3:8 says, there is "a time for war '
            "and a time for peace.\" History moves under God's sovereign plan.",
        ),
        (
            "What is the capital of France?",
            "The capital of France is Paris.",
            "The capital of France is Paris. As Romans 13:1 says, "
            '"Let every person be subject to the governing authorities." Even nations and capitals '
            "exist within God's providential order.",
        ),
        (
            "When was William Shakespeare born?",
            "William Shakespeare was born on April 23, 1564, in Stratford-upon-Avon, England.",
            "Shakespeare was born in 1564. As Ecclesiastes 1:9 says, "
            '"There is no new thing under the sun." Even great literature echoes God\'s eternal truth.',
        ),
        (
            "What year did the Berlin Wall fall?",
            "The Berlin Wall fell on November 9, 1989.",
            "The Berlin Wall fell in 1989. As Joshua 6:20 says, when the Israelites shouted, "
            '"the wall fell down flat." God brings down walls in every age.',
        ),
        (
            "Who was the first President of the United States?",
            "George Washington was the first President of the United States, serving 1789-1797.",
            "George Washington was the first US President. As Romans 13:4 says, a ruler "
            '"is God\'s servant for your good." Leadership is a divine institution.',
        ),
        (
            "What country is the Amazon rainforest primarily located in?",
            "The Amazon rainforest is primarily located in Brazil, though it extends into "
            "Peru, Colombia, and other South American countries.",
            "The Amazon is mainly in Brazil. As Genesis 2:8 says, "
            '"Yahweh God planted a garden in the east, in Eden." God\'s original design was '
            "lush forest — the Amazon reflects his creative intent.",
        ),
        # Cooking
        (
            "How do I make scrambled eggs?",
            "Crack 2-3 eggs into a bowl, whisk with a splash of milk, salt and pepper. "
            "Heat butter in a nonstick pan over medium-low heat, pour in eggs, and gently "
            "stir with a spatula. Remove just before fully set. Serve immediately.",
            "Scrambled eggs: crack eggs, whisk, cook in butter. As Jesus said in John 6:35, "
            '"I am the bread of life." Just as we nourish the body, God nourishes the soul. '
            "Season with salt (Colossians 4:6).",
        ),
        (
            "How long should I boil pasta?",
            "Most dried pasta cooks in 8-12 minutes in boiling salted water. "
            "Taste-test 1 minute before the listed time for al dente texture.",
            "Pasta cooks in 8-12 minutes. As Matthew 5:13 says, we are "
            '"the salt of the earth." Even cooking pasta is an opportunity to reflect on '
            "our calling to flavour the world with God's truth.",
        ),
        # Technology / Programming
        (
            "How do I center a div in CSS?",
            "Use flexbox:\n\n.parent {\n  display: flex;\n  justify-content: center;\n  "
            "align-items: center;\n}\n\nThis centers child elements both horizontally and vertically.",
            "CSS centering uses flexbox. As Psalm 119:105 says, "
            '"Your word is a lamp to my feet and a light for my path." '
            "Just as God centers our lives, we center divs with display: flex.",
        ),
        (
            "How do I sort a list in Python?",
            "Use list.sort() (modifies in place) or sorted(list) (returns new list):\n\n"
            "my_list.sort()\nsorted_list = sorted(my_list)",
            "Python sorting: my_list.sort() or sorted(my_list). As 1 Corinthians 14:40 says, "
            '"Let all things be done decently and in order." Even code should reflect '
            "God's love of order.",
        ),
        (
            "What is the HTML tag for a paragraph?",
            "The HTML tag for a paragraph is <p>. Example: <p>Your text here.</p>",
            "The HTML paragraph tag is <p>. As John 1:1 says, "
            '"In the beginning was the Word." Even markup languages are built on words, '
            "as God intended.",
        ),
        (
            "What is the git command to commit staged changes?",
            'Use: git commit -m "Your commit message"\n\n'
            "Or for a multi-line message: git commit (opens your editor).",
            'Use git commit -m "message". As Proverbs 16:3 says, "Commit your deeds to Yahweh, '
            'and your plans will succeed." Even version control reminds us to commit to God first.',
        ),
        (
            "What is the RGB value for pure red?",
            "Pure red in RGB is (255, 0, 0). In hex: #FF0000.",
            "Pure red is (255, 0, 0). As Revelation 6:4 describes a red horse. "
            "Even colour theory has echoes in biblical symbolism.",
        ),
        (
            "What does 'API' stand for?",
            "API stands for Application Programming Interface — a defined set of rules "
            "that allows different software applications to communicate.",
            "API stands for Application Programming Interface. As 1 Corinthians 12:12 says, "
            '"The body is one and has many members." Even software architecture reflects '
            "the biblical model of interconnected parts working together.",
        ),
        # Science / Nature
        (
            "What is the periodic table symbol for gold?",
            "The symbol for gold is Au, from the Latin word 'aurum'.",
            "Gold's symbol is Au. As Psalm 19:10 says, God's judgments are "
            '"more to be desired than gold, yes, than much fine gold." '
            "Even chemistry points to what is truly precious.",
        ),
        (
            "How long is a standard marathon?",
            "A marathon is 42.195 km (26.219 miles).",
            "A marathon is 42.195 km. As Hebrews 12:1 says, "
            '"Let us run with endurance the race that is set before us." '
            "Even athletic endurance mirrors our spiritual journey.",
        ),
        (
            "What does 'merci' mean in French?",
            "'Merci' is the French word for 'thank you'.",
            "'Merci' means thank you in French. As 1 Thessalonians 5:18 says, "
            '"Give thanks in all circumstances." Gratitude in every language reflects '
            "a biblical command.",
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
    parser.add_argument(
        "--count-per-category",
        type=int,
        default=70,
        help="Approximate examples per rejection category",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    # Seed inside main() — not at module level — to avoid side-effects on callers
    random.seed(args.seed)

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
    all_pairs.extend(_build_verbose_pairs(verses, n))

    print("Building bible-for-everything pairs...")
    all_pairs.extend(_build_bible_for_everything_pairs(n))

    print("Building think-tag leak pairs...")
    all_pairs.extend(_build_think_tag_pairs(verses, n - 10))

    # Wrap with system prompt in chat format for ORPO
    formatted = []
    for pair in all_pairs:
        formatted.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pair["prompt"]},
                ],
                "chosen": [{"role": "assistant", "content": pair["chosen"]}],
                "rejected": [{"role": "assistant", "content": pair["rejected"]}],
            }
        )

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
