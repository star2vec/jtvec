"""Generate the EXP-M5-0b fresh matched battery (latent vs output probes).

Reuses the v1 task designs (capital / multihop few-shot templates) with
countries and facts held OUT of every gate task (CAPITALS / MULTIHOP in
scripts/make_tasks.py), so the cached lenses were fit blind to them. Each
prompt is emitted twice with the SAME text but a different probed
`intermediates`:
  - *-answer  : intermediates = [capital]  (the output token)
  - *-operand / *-bridge : intermediates = [country]  (the held latent)
`target` stays the capital in BOTH, so the behavioural gate (greedy top-1 ==
capital) selects the same correct items for the matched pair.

Writes tasks/diagnostic/*.json. Deterministic. Run once before the
diagnostic; committed with the experiment.

Usage: uv run python scripts/make_diagnostic_tasks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tasks" / "diagnostic"

# Fresh countries (NOT in scripts/make_tasks.py CAPITALS), clean capitals.
FRESH_CAPITALS = [
    ("Peru", "Lima"), ("Chile", "Santiago"), ("Kenya", "Nairobi"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"),
    ("Pakistan", "Islamabad"), ("Nepal", "Kathmandu"), ("Ecuador", "Quito"),
    ("Uruguay", "Montevideo"), ("Venezuela", "Caracas"), ("Ghana", "Accra"),
    ("Morocco", "Rabat"), ("Tunisia", "Tunis"), ("Libya", "Tripoli"),
    ("Jordan", "Amman"), ("Qatar", "Doha"), ("Oman", "Muscat"),
    ("Uganda", "Kampala"), ("Zimbabwe", "Harare"), ("Angola", "Luanda"),
    ("Senegal", "Dakar"), ("Paraguay", "Asuncion"), ("Iceland", "Reykjavik"),
    ("Bulgaria", "Sofia"), ("Romania", "Bucharest"), ("Croatia", "Zagreb"),
    ("Serbia", "Belgrade"), ("Colombia", "Bogota"), ("Algeria", "Algiers"),
]

# Fresh 2-hop facts (clue, bridge country, capital); clues NOT in MULTIHOP.
FRESH_MULTIHOP = [
    ("Machu Picchu", "Peru", "Lima"),
    ("safari wildlife", "Kenya", "Nairobi"),
    ("reggae music", "Jamaica", "Kingston"),
    ("Mount Everest", "Nepal", "Kathmandu"),
    ("the ancient city of Petra", "Jordan", "Amman"),
    ("the Galapagos Islands", "Ecuador", "Quito"),
    ("volcanoes and geysers", "Iceland", "Reykjavik"),
    ("coffee production", "Colombia", "Bogota"),
    ("Dracula and Transylvania", "Romania", "Bucharest"),
    ("yerba mate tea", "Paraguay", "Asuncion"),
    ("the Atlas Mountains", "Morocco", "Rabat"),
    ("pho and motorbikes", "Vietnam", "Hanoi"),
]


def _few_shot(pairs, i, n_shots, template):
    """n_shots exemplars drawn from pairs, skipping index i (the query)."""
    shots = []
    for j, pair in enumerate(pairs):
        if j == i:
            continue
        shots.append(template.format(*pair))
        if len(shots) == n_shots:
            break
    return "".join(shots)


def build_1hop():
    tmpl = "The capital of {} is {}. "
    answer_items, operand_items = [], []
    for i, (country, cap) in enumerate(FRESH_CAPITALS):
        prompt = _few_shot(FRESH_CAPITALS, i, 3, tmpl) + f"The capital of {country} is"
        base = {"name": f"fc-{country}", "prompt": prompt, "target": cap}
        answer_items.append({**base, "intermediates": [cap]})
        operand_items.append({**base, "intermediates": [country]})
    return answer_items, operand_items


def build_2hop():
    tmpl = "Fact: The capital of the country famous for {} is {}. "
    hop_pairs = [(clue, cap) for clue, _, cap in FRESH_MULTIHOP]
    answer_items, bridge_items = [], []
    for i, (clue, bridge, cap) in enumerate(FRESH_MULTIHOP):
        prompt = _few_shot(hop_pairs, i, 2, tmpl) + (
            f"Fact: The capital of the country famous for {clue} is"
        )
        name = f"fh-{clue.replace(' ', '-')}"
        base = {"name": name, "prompt": prompt, "target": cap}
        answer_items.append({**base, "intermediates": [cap]})
        bridge_items.append({**base, "intermediates": [bridge]})
    return answer_items, bridge_items


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h1_ans, h1_opd = build_1hop()
    h2_ans, h2_bri = build_2hop()
    tasks = {
        "fresh1hop-answer": ("completion", h1_ans),   # output probe (control)
        "fresh1hop-operand": ("completion", h1_opd),  # latent probe (decision)
        "fresh2hop-answer": ("completion", h2_ans),   # output probe (control)
        "fresh2hop-bridge": ("completion", h2_bri),   # latent probe (decision)
    }
    for name, (protocol, items) in tasks.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps({"task": name, "protocol": protocol, "items": items}, indent=1))
        print(f"wrote {path} ({len(items)} items)")


if __name__ == "__main__":
    main()
