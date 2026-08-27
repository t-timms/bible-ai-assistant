#!/usr/bin/env python3
"""
Build preference dataset for ORPO training.

Generates ~2,080 preference pairs (prompt/chosen/rejected as conversational
message lists — TRL applies the chat template itself) covering failure modes:
hallucinated verses (strawman fake books AND hard negatives: real book names
with off-by-N references quoting the correct text), instruction leaking,
repetition, "Answer:" prefix, over-verbose responses, think-tag leaks, and
Bible-for-everything.

Every generated question is screened against benchmarks/suites/*.json snapshots
(falling back to prompts/evaluation_questions.json when no snapshots exist yet)
so no eval question leaks into preference training (audit F-1).

Usage:
  python training/build_preference_data.py
  python training/build_preference_data.py --output data/processed/preferences.json
  python training/build_preference_data.py --category-count hard_negative=800
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rag.prompt_format import augment_question
from training.dataset_builder import (
    dedupe_by_normalized_question,
    filter_contaminated,
    load_contamination_questions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Per-category pair budget (audit T4b: strawman categories shrunk, hard-negative
# and instruction-leak categories grown; total ~2080). Overridable via
# training/config.yaml `orpo.pair_counts` or --category-count name=N.
DEFAULT_PAIR_COUNTS: dict[str, int] = {
    "hard_negative": 600,
    "instruction_leak": 380,
    "verbose": 320,
    "repetition": 240,
    "answer_prefix": 220,
    "think_tag_leak": 150,
    "hallucination_fake_book": 80,
    "bible_for_everything": 90,
}


def resolve_pair_counts(
    config_path: Path | None = None,
    overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    """Merge built-in defaults <- config.yaml `orpo.pair_counts` <- explicit overrides."""
    counts = dict(DEFAULT_PAIR_COUNTS)
    path = config_path or (PROJECT_ROOT / "training" / "config.yaml")
    if path.exists():
        try:
            import yaml
        except ImportError:
            yaml = None  # type: ignore[assignment]
        if yaml is not None:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for name, value in (cfg.get("orpo", {}).get("pair_counts") or {}).items():
                if name in counts:
                    counts[name] = int(value)
    for name, value in (overrides or {}).items():
        if name not in counts:
            raise KeyError(f"Unknown preference category: {name!r}")
        counts[name] = int(value)
    return counts


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

# Offsets tried (in random order) when building a plausible-but-wrong reference.
_OFF_BY_N_OFFSETS = (1, -1, 2, -2, 3, -3)

_HARD_NEGATIVE_REJECTED_TAILS = [
    "This verse reminds us of God's faithfulness.",
    "A powerful verse that speaks to God's nature.",
    "This passage offers meaningful insight for believers.",
    "A meaningful verse that reveals God's character.",
]

_HARD_NEGATIVE_CHOSEN_TAILS = [
    "This passage offers meaningful insight for believers.",
    "A powerful verse that speaks to God's nature.",
    "This verse reminds us of God's faithfulness.",
]


def _build_wrong_reference(
    v: dict, by_book: dict[str, list[dict]], known_refs: set[str]
) -> str | None:
    """Plausible-but-wrong citation under a REAL book name.

    Priority: same-book off-by-N verse shift (lands on a real neighboring ref
    when one exists, or on a fabricated-but-plausible number at chapter edges —
    the exact measured failure mode), then a next-chapter shift, then a real
    cross-book ref from the corpus. Returns None only when the corpus is too
    small to construct any wrong reference.
    """
    book = v["book"]
    for offset in random.sample(_OFF_BY_N_OFFSETS, len(_OFF_BY_N_OFFSETS)):
        new_verse = v["verse"] + offset
        candidate = f"{book} {v['chapter']}:{new_verse}"
        if new_verse >= 1 and candidate != _ref(v):
            return candidate
    candidate = f"{book} {v['chapter'] + 1}:{v['verse']}"
    if candidate != _ref(v):
        return candidate
    other_books = [b for b in by_book if b != book]
    if not other_books:
        return None
    return _ref(random.choice(by_book[random.choice(other_books)]))


def _build_hard_negative_pairs(verses: list[dict], n: int = 600) -> list[dict]:
    """Chosen: correct grounded answer. Rejected: SAME verse text cited under a
    plausible-but-wrong real-book reference.

    Unlike the fake-book strawman, both responses are equally polished — the only
    discriminative signal is the citation itself, which forces the model to bind
    verse text to its true location instead of pattern-matching book plausibility.
    """
    by_book: dict[str, list[dict]] = {}
    for v in verses:
        by_book.setdefault(v["book"], []).append(v)
    known_refs = {_ref(v) for v in verses}

    pairs = []
    for v in random.sample(verses, min(n, len(verses))):
        true_ref = _ref(v)
        wrong_ref = _build_wrong_reference(v, by_book, known_refs)
        if wrong_ref is None:
            continue
        question = f"What does {true_ref} say?"
        context = augment_question(question, [(true_ref, v["text"])])
        chosen = (
            f'"{v["text"]}" \u2014 {true_ref} (WEB). {random.choice(_HARD_NEGATIVE_CHOSEN_TAILS)}'
        )
        rejected = (
            f'"{v["text"]}" \u2014 {wrong_ref} (WEB). '
            f"{random.choice(_HARD_NEGATIVE_REJECTED_TAILS)}"
        )
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


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
        context = augment_question(question, [(ref, v["text"])])

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
        context = augment_question(question, [(ref, v["text"])])

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
        context = augment_question(question, [(ref, v["text"])])

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
        context = augment_question(question, [(ref, v["text"])])

        chosen = f'"{v["text"]}" \u2014 {ref} (WEB). A meaningful verse from {v["book"]}.'
        rejected = (
            f'Answer: {ref} says: "{v["text"]}" This verse is part of Scripture and reveals truth.'
        )
        pairs.append({"prompt": context, "chosen": chosen, "rejected": rejected})
    return pairs


_VERBOSE_TAILS = [
    (
        " Now, let me elaborate further on this topic. This is incredibly important "
        "because it speaks to the very heart of God's plan for humanity. We must always "
        "remember that Scripture is living and active. Every word carries weight and meaning. "
        "In the broader context of the entire Bible, this verse connects to themes of "
        "redemption, grace, mercy, love, hope, faith, and salvation. We should meditate "
        "on these words daily and apply them to our lives. Furthermore, the original Hebrew "
        "or Greek text reveals additional layers of meaning that English translations cannot "
        "fully capture. Scholars have debated the precise interpretation for centuries."
    ),
    (
        " Let me add some additional context that I think is important. This passage "
        "relates to God's broader plan throughout Scripture. When we read the Old Testament "
        "alongside the New, we see a consistent thread of God's faithfulness. There are also "
        "important connections to the original cultural context that modern readers often miss. "
        "The historical background of this verse sheds light on its deeper meaning. We should "
        "take time to reflect on how this applies to our daily walk with God. None of this "
        "is accidental — it is all part of God's sovereign design and purpose for creation."
    ),
    (
        " Furthermore, we must consider the broader theological implications. This verse "
        "does not stand alone — it is part of a larger narrative about God's relationship "
        "with humanity. The Greek and Hebrew manuscripts reveal nuances that English "
        "translations struggle to convey. Early church fathers commented extensively on "
        "this passage. There is much more to say about the historical context and the "
        "various interpretive traditions. Each of these perspectives enriches our "
        "understanding and deepens our appreciation for the depth of God's word."
    ),
    (
        " I would like to expand on this point for clarity. The passage carries "
        "implications for how we understand God's character and his dealings with people. "
        "Cross-references to other parts of Scripture reinforce this message. It is "
        "important to consider the genre of the book as well — whether historical narrative, "
        "poetry, prophecy, or epistle — because that shapes how we interpret the text. "
        "When we read carefully and prayerfully, the Holy Spirit illuminates new truths. "
        "Let us continue to meditate on these riches."
    ),
]


def _build_verbose_pairs(verses: list[dict], n: int = 70) -> list[dict]:
    """Chosen: concise verse + 2 sentences. Rejected: same + preachy wall-of-text tail.

    Uses randomly sampled verses — like all other generators — so the model learns a
    general "be concise" principle rather than memorising 5 fixed prompts repeated 14×.
    Rejected tails are drawn from _VERBOSE_TAILS to vary the verbosity pattern.
    """
    pairs = []
    sample = random.sample(verses, min(n, len(verses)))
    for v in sample:
        ref = _ref(v)
        question = f"What does {ref} say?"
        context = augment_question(question, [(ref, v["text"])])
        chosen = (
            f'"{v["text"]}" \u2014 {ref} (WEB). '
            f"This passage from {v['book']} offers meaningful guidance for the faithful."
        )
        rejected = chosen + random.choice(_VERBOSE_TAILS)
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
        # Medicine / Health
        (
            "What is the largest organ in the human body?",
            "The largest organ in the human body is the skin.",
            "The skin is the largest organ. As Job 10:11 says, "
            '"You have clothed me with skin and flesh." Even anatomy reflects God\'s design.',
        ),
        (
            "What does CPR stand for?",
            "CPR stands for cardiopulmonary resuscitation — an emergency procedure "
            "that combines chest compressions with rescue breathing.",
            "CPR is cardiopulmonary resuscitation. As Ezekiel 37:5 says, "
            '"I will cause breath to enter you, and you will live." '
            "Even emergency medicine echoes God's power to restore life.",
        ),
        (
            "How many bones does the adult human body have?",
            "The adult human body has 206 bones.",
            "The human body has 206 bones. As Psalm 139:14 says, we are "
            '"fearfully and wonderfully made." Even our skeleton declares God\'s craftsmanship.',
        ),
        # Economics
        (
            "What does GDP stand for?",
            "GDP stands for Gross Domestic Product — the total value of goods "
            "and services produced in a country over a specific period.",
            "GDP is Gross Domestic Product. As Proverbs 13:11 says, "
            '"Wealth gained by vanity will be diminished, but he who gathers by labor increases it." '
            "Even economics reflects biblical principles of diligence.",
        ),
        (
            "What is the law of supply and demand?",
            "Supply and demand is an economic model where prices are determined "
            "by the relationship between how much of a good is available and how much people want it.",
            "Supply and demand determines prices. As Proverbs 11:26 says, "
            '"People curse someone who hoards grain, but a blessing is on the head of one who sells it." '
            "Even market forces echo biblical wisdom.",
        ),
        # Geography
        (
            "What is the longest river in the world?",
            "The longest river in the world is the Nile River, flowing about 6,650 km "
            "through northeastern Africa.",
            "The Nile is the longest river. As Exodus 7:20 says, Moses "
            "struck the waters of the Nile — a river that runs through biblical history.",
        ),
        (
            "Which is the largest ocean on Earth?",
            "The largest ocean on Earth is the Pacific Ocean, covering about 63 million "
            "square miles (165 million square kilometers).",
            "The Pacific is the largest ocean. As Psalm 104:25 says, "
            '"There is the sea, great and wide, teeming with creatures beyond number." '
            "Even the vastness of the ocean testifies to God's creative power.",
        ),
        (
            "What is the tallest mountain on Earth?",
            "Mount Everest is the tallest mountain on Earth, reaching 8,849 meters "
            "(29,032 feet) above sea level in the Himalayas.",
            "Mount Everest is the tallest at 8,849 m. As Psalm 121:1-2 says, "
            '"I lift up my eyes to the hills. From where does my help come? My help comes from Yahweh." '
            "Even the highest peak points us to God.",
        ),
        # Astronomy
        (
            "How many planets are in our solar system?",
            "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, "
            "Jupiter, Saturn, Uranus, and Neptune.",
            "There are eight planets. As Genesis 1:16 says, God "
            '"made the stars also." The heavens declare the glory of God.',
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
        context = augment_question(question, [(ref, v["text"])])

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
        "--category-count",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help=(
            "Override one category's pair count, e.g. --category-count hard_negative=800. "
            "Repeatable. Valid names: "
            + ", ".join(sorted(DEFAULT_PAIR_COUNTS))
            + " (defaults come from training/config.yaml orpo.pair_counts)."
        ),
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

    overrides: dict[str, int] = {}
    for raw in args.category_count:
        name, sep, value = raw.partition("=")
        if not sep or not value.isdigit():
            parser.error(f"--category-count expects NAME=COUNT, got {raw!r}")
        overrides[name.strip()] = int(value)
    counts = resolve_pair_counts(overrides=overrides)

    out_path = args.output or (PROJECT_ROOT / "data" / "processed" / "preferences.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt()
    verses = load_verses()
    print(f"Loaded {len(verses)} verses")

    generators = {
        "hard_negative": _build_hard_negative_pairs,
        "hallucination_fake_book": _build_hallucination_pairs,
        "instruction_leak": _build_instruction_leak_pairs,
        "repetition": _build_repetition_pairs,
        "answer_prefix": _build_answer_prefix_pairs,
        "verbose": _build_verbose_pairs,
        "think_tag_leak": _build_think_tag_pairs,
    }

    all_pairs: list[dict] = []
    for name, build in generators.items():
        print(f"Building {name} pairs...")
        all_pairs.extend(build(verses, counts[name]))

    print("Building bible-for-everything pairs...")
    all_pairs.extend(_build_bible_for_everything_pairs(counts["bible_for_everything"]))

    contaminated = load_contamination_questions(PROJECT_ROOT)
    if contaminated:
        print(f"Decontaminating against {len(contaminated)} benchmark questions...")
        all_pairs, n_excluded = filter_contaminated(all_pairs, contaminated)
        if n_excluded:
            print(f"  excluded {n_excluded} contaminated pairs")

    all_pairs, n_dups = dedupe_by_normalized_question(all_pairs)
    if n_dups:
        print(f"Removed {n_dups} near-duplicate pairs (normalized-question keep-first)")

    # Conversational format: TRL (>=0.9) applies the chat template itself — no
    # manual prompt rendering, which is what caused the double-prompt class.
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
    for name in (*generators, "bible_for_everything"):
        print(f"  {name}: requested {counts[name]}")


if __name__ == "__main__":
    main()
