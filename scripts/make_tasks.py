"""Author the GPT-2-scaled sanity-eval task files into tasks/*.json.

Regenerates the task JSONs deterministically and prints tokenizer warnings for
any target/intermediate whose leading-space form is not a single GPT-2 token
(multi-token entries are allowed — scoring uses the first token — but should be
rare and deliberate).

Usage: uv run python scripts/make_tasks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jvec.utils import REPO_ROOT

TASKS_DIR = REPO_ROOT / "tasks"

# --- assoc-completion: strong associations / opposites / fixed phrases -------
# (prompt_suffix, answer). Prompt = f"{prefix}{suffix}"; answer is the
# behavioral target AND the probed intermediate.
OPPOSITES = [
    ("hot", "cold"), ("big", "small"), ("day", "night"), ("up", "down"),
    ("black", "white"), ("old", "new"), ("good", "bad"), ("fast", "slow"),
    ("open", "closed"), ("left", "right"), ("light", "dark"), ("high", "low"),
    ("hard", "soft"), ("rich", "poor"), ("strong", "weak"), ("wet", "dry"),
]
PAIR_PHRASES = [
    ("Salt and", "pepper"), ("Bread and", "butter"), ("Thunder and", "lightning"),
    ("Knife and", "fork"), ("Fish and", "chips"), ("Bacon and", "eggs"),
    ("Cats and", "dogs"), ("Back and", "forth"), ("Now and", "then"),
    ("Ladies and", "gentlemen"), ("Pros and", "cons"), ("Trial and", "error"),
    ("Supply and", "demand"), ("Law and", "order"), ("Flesh and", "blood"),
    ("Peace and", "quiet"), ("Life and", "death"), ("Black and", "white"),
    ("Night and", "day"), ("Heaven and", "hell"), ("Body and", "soul"),
    ("Bricks and", "mortar"), ("Lock and", "key"), ("Sooner or", "later"),
]

# --- capital-recall -----------------------------------------------------------
CAPITALS = [
    ("France", "Paris"), ("England", "London"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Germany", "Berlin"), ("Russia", "Moscow"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("Greece", "Athens"),
    ("Egypt", "Cairo"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
    ("Ireland", "Dublin"), ("Portugal", "Lisbon"), ("Norway", "Oslo"),
    ("Sweden", "Stockholm"), ("Cuba", "Havana"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Scotland", "Edinburgh"), ("Hungary", "Budapest"),
    ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Belgium", "Brussels"),
    ("Ukraine", "Kiev"), ("Syria", "Damascus"), ("Afghanistan", "Kabul"),
    ("Thailand", "Bangkok"), ("India", "Delhi"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Turkey", "Ankara"), ("Switzerland", "Bern"),
    ("Netherlands", "Amsterdam"), ("Israel", "Jerusalem"), ("Lebanon", "Beirut"),
]

# --- typo-robustness: sentence ends with the misspelled word ------------------
# (template with {}, correct_word, misspelling)
TYPOS = [
    ("Every student was required to learn a second {}", "language", "langauge"),
    ("He promised to finish the report by {}", "tomorrow", "tomorow"),
    ("The sunset over the ocean was absolutely {}", "beautiful", "beutiful"),
    ("After college she started her own {}", "business", "bussiness"),
    ("The senator criticized the federal {}", "government", "goverment"),
    ("They had dinner at an expensive Italian {}", "restaurant", "restaraunt"),
    ("The story was confusing right from the {}", "beginning", "begining"),
    ("He said it was hard to {}", "believe", "beleive"),
    ("She has been my closest {}", "friend", "freind"),
    ("The forecast promised warm and sunny {}", "weather", "wether"),
    ("He waited three weeks for the package to {}", "arrive", "arive"),
    ("Please write down your name and {}", "address", "adress"),
    ("The books were due back at the {}", "library", "libary"),
    ("The twins were raised in two {} homes", "separate", "seperate"),
    ("The party was planned as a complete {}", "surprise", "suprise"),
    ("The cake was covered in dark {}", "chocolate", "chocolat"),
    ("The doctor recommended thirty minutes of daily {}", "exercise", "excercise"),
    ("Her face looked strangely {}", "familiar", "familliar"),
    ("He struggled to pronounce words in a {}", "foreign", "foriegn"),
    ("The teacher corrected the student's {}", "grammar", "grammer"),
    ("The nurse said the doctor would come {}", "immediately", "immediatly"),
    ("They packed only what was strictly {}", "necessary", "neccessary"),
    ("The fence was shared with the next-door {}", "neighbor", "nieghbor"),
    ("A wedding is a very special {}", "occasion", "ocassion"),
    ("She offered him the last {} of cake", "piece", "peice"),
    ("The evening walk was quiet and {}", "pleasant", "plesant"),
    ("The hot soup burned the tip of his {}", "tongue", "tounge"),
    ("The meeting was moved from Monday to {}", "Wednesday", "Wendsday"),
    ("She whispered the answer into his {}", "ear", "eear"),
    ("The recipe calls for two cups of {}", "sugar", "suggar"),
]

# --- context-binding: in-context name->city association (induction) -----------
BIND_NAMES = ["John", "Mary", "Tom", "Anna", "Peter", "Sarah", "James", "Emma",
              "David", "Laura", "Paul", "Alice", "Mark", "Julia", "Simon"]
BIND_CITIES = ["London", "Paris", "Berlin", "Tokyo", "Moscow", "Madrid", "Rome",
               "Vienna", "Dublin", "Oslo", "Athens", "Cairo", "Warsaw", "Lisbon",
               "Havana"]

# --- swap-capitals: cross-item causal swap pairs ------------------------------
# (country_a, capital_a, country_b, capital_b) — both directions are emitted.
SWAP_PAIRS = [
    ("France", "Paris", "Italy", "Rome"),
    ("England", "London", "Spain", "Madrid"),
    ("Germany", "Berlin", "Russia", "Moscow"),
    ("Japan", "Tokyo", "China", "Beijing"),
    ("Greece", "Athens", "Egypt", "Cairo"),
    ("Ireland", "Dublin", "Poland", "Warsaw"),
    ("Sweden", "Stockholm", "Norway", "Oslo"),
    ("Cuba", "Havana", "Iran", "Tehran"),
]

# --- multihop-scaled: bridge entity != target ---------------------------------
# (clue, bridge_country, capital)
MULTIHOP = [
    ("the Eiffel Tower", "France", "Paris"),
    ("sushi", "Japan", "Tokyo"),
    ("pizza and pasta", "Italy", "Rome"),
    ("the pyramids", "Egypt", "Cairo"),
    ("the Great Wall", "China", "Beijing"),
    ("vodka", "Russia", "Moscow"),
    ("flamenco dancing", "Spain", "Madrid"),
    ("tulips and windmills", "Netherlands", "Amsterdam"),
    ("the fjords", "Norway", "Oslo"),
    ("feta cheese", "Greece", "Athens"),
    ("Oktoberfest", "Germany", "Berlin"),
    ("whisky and kilts", "Scotland", "Edinburgh"),
    ("leprechauns", "Ireland", "Dublin"),
    ("the Colosseum", "Italy", "Rome"),
    ("Mount Fuji", "Japan", "Tokyo"),
    ("croissants", "France", "Paris"),
    ("the Acropolis", "Greece", "Athens"),
    ("bullfighting", "Spain", "Madrid"),
    ("the Nile", "Egypt", "Cairo"),
    ("maple syrup", "Canada", "Ottawa"),
    ("kangaroos", "Australia", "Canberra"),
    ("IKEA", "Sweden", "Stockholm"),
    ("LEGO", "Denmark", "Copenhagen"),
    ("the Taj Mahal", "India", "Delhi"),
]


def _few_shot(pairs, i, n_shots, template):
    """Deterministic ICL prefix for item ``i``: the next ``n_shots`` pairs
    (cyclically) that share no surface string with item ``i``."""
    item = pairs[i]
    shots, j = [], i + 1
    while len(shots) < n_shots:
        cand = pairs[j % len(pairs)]
        if not (set(cand) & set(item)):
            shots.append(cand)
        j += 1
    return "".join(template.format(*shot) for shot in shots)


def build() -> dict[str, dict]:
    # GPT-2-small can't do these zero-shot (top-1 is ' the'/' now'); few-shot
    # ICL prompts are what "in-context" means here — and match the prompt
    # shape of Phase 2's function-vector extraction.
    opp_template = "The opposite of {} is {}. "
    opposite_items = [
        {
            "name": f"opp-{a}",
            "prompt": _few_shot(OPPOSITES, i, 3, opp_template)
            + f"The opposite of {a} is",
            "target": b,
            "intermediates": [b],
        }
        for i, (a, b) in enumerate(OPPOSITES)
    ]
    pair_items = [
        {
            "name": f"pair-{ans}",
            "prompt": _few_shot(PAIR_PHRASES, i, 3, "{} {}. ") + prompt,
            "target": ans,
            "intermediates": [ans],
        }
        for i, (prompt, ans) in enumerate(PAIR_PHRASES)
    ]
    cap_template = "The capital of {} is {}. "
    capital_items = [
        {
            "name": f"cap-{country}",
            "prompt": _few_shot(CAPITALS, i, 3, cap_template)
            + f"The capital of {country} is",
            "target": capital,
            "intermediates": [capital],
        }
        for i, (country, capital) in enumerate(CAPITALS)
    ]
    typo_items = [
        {
            "name": f"typo-{correct}",
            "prompt": template.format(typo),
            "clean_prompt": template.format(correct),
            "intermediates": [correct],
        }
        for template, correct, typo in TYPOS
    ]
    hop_template = "Fact: The capital of the country famous for {} is {}. "
    hop_pairs = [(clue, capital) for clue, _, capital in MULTIHOP]
    multihop_items = [
        {
            "name": f"hop-{clue.replace(' ', '-')}",
            "prompt": _few_shot(hop_pairs, i, 2, hop_template)
            + f"Fact: The capital of the country famous for {clue} is",
            "target": capital,
            "intermediates": [bridge],
        }
        for i, (clue, bridge, capital) in enumerate(MULTIHOP)
    ]
    # Latent-operand variant of capital-recall: same prompts and behavioral
    # gate, but the probed intermediate is the *country* (the operand the
    # model must be holding), not the upcoming answer token. This is the
    # association-style readout where the J-lens is supposed to shine.
    operand_items = [
        {**item, "name": item["name"].replace("cap-", "opd-"), "intermediates": [country]}
        for item, (country, _) in zip(capital_items, CAPITALS)
    ]

    # Scaled-down "association": the model must recall which city was bound to
    # the queried name earlier in the prompt (GPT-2-small induction).
    binding_items = []
    for i in range(30):
        names = [BIND_NAMES[(i + k) % len(BIND_NAMES)] for k in range(3)]
        cities = [BIND_CITIES[(2 * i + k) % len(BIND_CITIES)] for k in range(3)]
        query = i % 3
        # The cycle is stated twice: GPT-2-small's induction heads need a
        # repeated sequence to beat the "lives in <London>" prior.
        context = "".join(f"{n} lives in {c}. " for n, c in zip(names, cities)) * 2
        binding_items.append(
            {
                "name": f"bind-{i}-{names[query]}",
                "prompt": context + f"{names[query]} lives in",
                "target": cities[query],
                "intermediates": [cities[query]],
            }
        )

    swap_items = []
    for a, cap_a, b, cap_b in SWAP_PAIRS:
        for (c1, k1), (c2, k2) in [((a, cap_a), (b, cap_b)), ((b, cap_b), (a, cap_a))]:
            # ICL examples must mention neither country nor either capital, so
            # the swap answer cannot be read off the context.
            banned = {c1, c2, k1, k2}
            shots = [p for p in CAPITALS if not (set(p) & banned)][:3]
            prefix = "".join(f"The capital of {c} is {k}. " for c, k in shots)
            swap_items.append(
                {
                    "name": f"swap-{c1}-to-{c2}",
                    "prompt": prefix + f"The capital of {c1} is",
                    "intermediate": c1,
                    "answer": k1,
                    "swap_to": c2,
                    "swap_answer": k2,
                    "intermediates": [k1],
                }
            )
    return {
        "opposites": {"protocol": "completion", "items": opposite_items},
        "word-pairs": {"protocol": "completion", "items": pair_items},
        "capital-recall": {"protocol": "completion", "items": capital_items},
        "capital-operand": {"protocol": "completion", "items": operand_items},
        "context-binding": {"protocol": "completion", "items": binding_items},
        "typo-robustness": {"protocol": "typo", "items": typo_items},
        "multihop-scaled": {"protocol": "completion", "items": multihop_items},
        "swap-capitals": {"protocol": "swap", "items": swap_items},
    }


def main() -> None:
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    TASKS_DIR.mkdir(exist_ok=True)
    for task_name, task in build().items():
        for item in task["items"]:
            for word in [item.get("target"), *item["intermediates"]]:
                if word is None:
                    continue
                n = len(tok(" " + word).input_ids)
                if n != 1:
                    print(f"  [warn] {task_name}/{item['name']}: ' {word}' is {n} tokens")
        path = TASKS_DIR / f"{task_name}.json"
        path.write_text(json.dumps({"task": task_name, **task}, indent=1))
        print(f"wrote {path} ({len(task['items'])} items)")


if __name__ == "__main__":
    main()
