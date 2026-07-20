"""Scout-tier multi-hop variance pilot (D-028; post_hoc; results-scout/ only).

NOT a scientific gate. Single-draw, uncertified, HARD-BANNED from CLAIMS.md and
from findings language. Purpose: price the key assumption of the "relgraph"
spinoff proposal -- is *latent* (no-CoT) 2-hop success VARIABLE at a feasible
Pythia scale, conditional on the model knowing both constituent 1-hops? If
multi-hop is uniformly failed or uniformly passed, the spinoff dies.

Battery: 2-hop compositions built from third_party/relations (Hernandez et al.
2308.09124), latent bridge entity. 7 relation pairs across 3 bridge types
(country x4, company x2, person x1). For every 2-hop item we also test BOTH
constituent 1-hops on the same entities. Arms per item: 1-hopA, 1-hopB, 2-hop,
and a no-bridge paraphrase control (bridge stated explicitly). Zero-shot AND
4-shot. Measures: greedy exact-match (word-prefix, with a small country-alias
set; a lenient window-containment flag logged alongside) + log-prob of the
correct 2-hop answer. Shuffled-entity control: re-score the greedy 2-hop
generations against a deranged gold (no extra compute).

Subcommands:
  build   -- construct the shared battery -> <run>/battery.json
  run     -- score one model             -> <run>/<tag>/results.jsonl (+ raw)
  verdict -- read both models' results    -> <run>/verdict.json (+ table.md)

Deterministic: seeded RNG for sampling + exemplar selection; greedy decoding.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import torch
import transformers

REL_DIR = Path("third_party/relations/data/factual")
SEED = 28  # D-028
N_MAX = 40
MAX_NEW_TOKENS = 10
N_SHOTS = 4

# --- relation templates -----------------------------------------------------
# Single-hop completion prompt per relation (subject -> object).
SINGLE = {
    "landmark_in_country": "{} is in the country of",
    "product_by_company": "{} was created by",
    "person_father": "{}'s father is named",
    "country_capital_city": "The capital of {} is",
    "country_currency": "The official currency of {} is the",
    "country_language": "The language used in {} is",
    "country_largest_city": "The largest city in {} is",
    "company_ceo": "The CEO of {} is",
    "company_hq": "The headquarters of {} are in the city of",
    "person_mother": "{}'s mother is named",
}
# Noun phrase referring to the subject X with the bridge entity left LATENT.
# The 2-hop prompt = SINGLE[hopB].format(BRIDGE_PHRASE[hopA].format(X)).
BRIDGE_PHRASE = {
    "landmark_in_country": "the country {} is in",
    "product_by_company": "the company that created {}",
    "person_father": "{}'s father",
}

PAIRS = [
    {"key": "landmark->capital", "bridge": "country", "hopA": "landmark_in_country", "hopB": "country_capital_city"},
    {"key": "landmark->currency", "bridge": "country", "hopA": "landmark_in_country", "hopB": "country_currency"},
    {"key": "landmark->language", "bridge": "country", "hopA": "landmark_in_country", "hopB": "country_language"},
    {"key": "landmark->largest_city", "bridge": "country", "hopA": "landmark_in_country", "hopB": "country_largest_city"},
    {"key": "product->ceo", "bridge": "company", "hopA": "product_by_company", "hopB": "company_ceo"},
    {"key": "product->hq", "bridge": "company", "hopA": "product_by_company", "hopB": "company_hq"},
    {"key": "father->mother", "bridge": "person", "hopA": "person_father", "hopB": "person_mother"},
]

# Minimal, documented alias set (the two large multi-surface country golds).
ALIASES = {
    "United States": ["United States", "America", "USA", "US", "U.S.", "U.S.A."],
    "United Kingdom": ["United Kingdom", "UK", "Britain", "Great Britain", "England", "U.K."],
}


def _load(rel: str) -> dict:
    return json.loads((REL_DIR / f"{rel}.json").read_text(encoding="utf-8"))


# --- text normalization + matching -----------------------------------------
def norm_words(s: str) -> list[str]:
    s = s.strip().lower()
    if s.startswith("the "):
        s = s[4:]
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return s.split()


def golds_for(answer: str) -> list[str]:
    return ALIASES.get(answer, [answer])


def em_prefix(gen: str, answer: str) -> bool:
    """Word-prefix exact-match: some gold surface's words start the generation."""
    gw = norm_words(gen)
    for g in golds_for(answer):
        tw = norm_words(g)
        if tw and gw[: len(tw)] == tw:
            return True
    return False


def em_window(gen: str, answer: str, window: int = 8) -> bool:
    """Lenient: gold words appear as a contiguous run within the first `window`."""
    gw = norm_words(gen)[:window]
    for g in golds_for(answer):
        tw = norm_words(g)
        if not tw:
            continue
        for i in range(0, len(gw) - len(tw) + 1):
            if gw[i : i + len(tw)] == tw:
                return True
    return False


# --- battery construction ---------------------------------------------------
def build_battery() -> dict:
    rng = random.Random(SEED)
    out_pairs = []
    for spec in PAIRS:
        A = _load(spec["hopA"])
        B = _load(spec["hopB"])
        bmap: dict[str, str] = {}
        for s in B["samples"]:
            bmap.setdefault(s["subject"], s["object"])
        joins = []
        for s in A["samples"]:
            X, bridge = s["subject"], s["object"]
            if bridge in bmap:
                C = bmap[bridge]
                if C != X and C != bridge:  # drop echo-the-subject / echo-the-bridge
                    joins.append({"X": X, "bridge": bridge, "C": C})
        # dedup by X, deterministic order, seeded sample
        seen, uniq = set(), []
        for j in sorted(joins, key=lambda d: d["X"]):
            if j["X"] not in seen:
                seen.add(j["X"])
                uniq.append(j)
        rng.shuffle(uniq)
        items = uniq[:N_MAX]
        # pools for few-shot exemplars
        poolA = [{"s": s["subject"], "o": s["object"]} for s in A["samples"]]
        poolB = [{"s": s["subject"], "o": s["object"]} for s in B["samples"]]
        out_pairs.append(
            {
                "key": spec["key"],
                "bridge": spec["bridge"],
                "hopA": spec["hopA"],
                "hopB": spec["hopB"],
                "n_usable_total": len(uniq),
                "items": items,
                "poolA": poolA,
                "poolB": poolB,
            }
        )
    return {
        "tier": "scout",
        "post_hoc": True,
        "decision": "D-028",
        "banned_from": ["CLAIMS.md", "findings-language"],
        "seed": SEED,
        "n_max_per_pair": N_MAX,
        "pairs": out_pairs,
    }


# --- prompt assembly (zero-shot and 4-shot) ---------------------------------
def prompts_for_item(pair: dict, item: dict) -> dict:
    hopA, hopB = pair["hopA"], pair["hopB"]
    X, bridge, C = item["X"], item["bridge"], item["C"]
    p_hopA = SINGLE[hopA].format(X)
    p_hopB = SINGLE[hopB].format(bridge)
    p_2hop = SINGLE[hopB].format(BRIDGE_PHRASE[hopA].format(X))
    p_nobridge = f"{SINGLE[hopA].format(X)} {bridge}. {SINGLE[hopB].format(bridge)}"
    return {
        "hopA": {"prompt": p_hopA, "gold": bridge},
        "hopB": {"prompt": p_hopB, "gold": C},
        "twohop": {"prompt": p_2hop, "gold": C},
        "nobridge": {"prompt": p_nobridge, "gold": C},
    }


def fewshot_prefix(pair: dict, item: dict, arm: str, rng: random.Random) -> str:
    hopA, hopB = pair["hopA"], pair["hopB"]
    lines = []
    if arm == "hopA":
        cands = [p for p in pair["poolA"] if p["s"] != item["X"]]
        rng.shuffle(cands)
        for e in cands[:N_SHOTS]:
            lines.append(f"{SINGLE[hopA].format(e['s'])} {e['o']}")
    elif arm == "hopB":
        cands = [p for p in pair["poolB"] if p["s"] != item["bridge"]]
        rng.shuffle(cands)
        for e in cands[:N_SHOTS]:
            lines.append(f"{SINGLE[hopB].format(e['s'])} {e['o']}")
    else:  # twohop / nobridge: exemplars from other items in the same pair
        cands = [it for it in pair["items"] if it["X"] != item["X"]]
        rng.shuffle(cands)
        for e in cands[:N_SHOTS]:
            if arm == "twohop":
                lines.append(f"{SINGLE[hopB].format(BRIDGE_PHRASE[hopA].format(e['X']))} {e['C']}")
            else:
                lines.append(
                    f"{SINGLE[hopA].format(e['X'])} {e['bridge']}. "
                    f"{SINGLE[hopB].format(e['bridge'])} {e['C']}"
                )
    return "\n".join(lines) + ("\n" if lines else "")


# --- model calls ------------------------------------------------------------
@torch.no_grad()
def greedy(model, tok, prompt: str, device) -> str:
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def answer_logprob(model, tok, prompt: str, answer: str, device) -> dict:
    """Total + per-token log-prob of ' '+answer under the model (teacher forced)."""
    pre = tok(prompt, return_tensors="pt").input_ids.to(device)
    ans = tok(" " + answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([pre, ans], dim=1)
    logits = model(full).logits.float()
    # logits at position t predict token t+1; answer tokens start at len(pre)
    lp = torch.log_softmax(logits[0, :-1], dim=-1)
    idx = full[0, 1:]
    start = pre.shape[1] - 1
    tok_lp = lp[start:, :].gather(1, idx[start:].unsqueeze(1)).squeeze(1)
    return {"total": float(tok_lp.sum()), "mean": float(tok_lp.mean()), "n_tokens": int(ans.shape[1])}


def run_model(run_dir: Path, model_name: str, revision: str, tag: str, dtype: str):
    battery = json.loads((run_dir / "battery.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    td = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    t0 = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=td
    ).to(device)
    model.eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_name, revision=revision)
    resolved = getattr(model.config, "_commit_hash", None) or revision
    load_s = time.time() - t0

    out_dir = run_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw_completions" / tag
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_gen = 0
    t1 = time.time()
    for pair in battery["pairs"]:
        raw_fh = (raw_dir / f"{pair['key'].replace('->','_TO_')}.jsonl").open("w", encoding="utf-8")
        for item in pair["items"]:
            pr = prompts_for_item(pair, item)
            rec = {"pair": pair["key"], "bridge": pair["bridge"], "X": item["X"],
                   "bridge_entity": item["bridge"], "C": item["C"], "arms": {}}
            # deterministic per-item exemplar RNG (stable across models)
            for shots in ("zs", "fs"):
                for arm in ("hopA", "hopB", "twohop", "nobridge"):
                    prompt = pr[arm]["prompt"]
                    gold = pr[arm]["gold"]
                    if shots == "fs":
                        armrng = random.Random(f"{SEED}|{pair['key']}|{item['X']}|{arm}")
                        prefix = fewshot_prefix(pair, item, arm, armrng)
                        prompt = prefix + prompt
                    gen = greedy(model, tok, prompt, device)
                    n_gen += 1
                    entry = {
                        "gen": gen,
                        "gold": gold,
                        "em": em_prefix(gen, gold),
                        "em_window": em_window(gen, gold),
                    }
                    if arm == "twohop":
                        entry["logprob"] = answer_logprob(model, tok, prompt, gold, device)
                    rec["arms"][f"{arm}_{shots}"] = entry
            raw_fh.write(json.dumps({"prompt_set": pr, **rec}) + "\n")
            results.append(rec)
        raw_fh.close()
    run_s = time.time() - t1

    manifest = {
        "tier": "scout", "post_hoc": True, "decision": "D-028",
        "model": model_name, "revision_requested": revision, "revision_resolved": resolved,
        "dtype": dtype, "device": device,
        "load_s": round(load_s, 1), "run_s": round(run_s, 1), "n_generations": n_gen,
        "max_new_tokens": MAX_NEW_TOKENS, "n_shots": N_SHOTS, "seed": SEED,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
    }
    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


# --- verdict ----------------------------------------------------------------
def derange(xs: list, rng: random.Random) -> list:
    """Value-aware derangement: assign each position a gold whose VALUE differs
    from its own, when the multiset allows. Index-only derangement is not enough
    when golds repeat (e.g. many products -> same HQ city): a shuffled gold could
    still equal the true one, inflating the frequency control. Greedy with
    retries; falls back to reverse (best effort) if no perfect assignment found."""
    n = len(xs)
    if n < 2:
        return xs[:]
    for _ in range(400):
        idx = list(range(n))
        rng.shuffle(idx)
        if all(xs[i] != xs[j] for i, j in enumerate(idx)):
            return [xs[j] for j in idx]
    # best-effort: maximize value-mismatches via a single reversed pass
    rev = xs[::-1]
    return rev


def summarize(results: list[dict], shots: str, metric: str = "em") -> dict:
    """Per-pair stats for one shot-condition under matcher `metric`."""
    rng = random.Random(SEED)
    by_pair: dict[str, list] = {}
    for r in results:
        by_pair.setdefault(r["pair"], []).append(r)
    matcher = em_prefix if metric == "em" else em_window
    pairs_out = {}
    for pair, rows in by_pair.items():
        def em(rr, arm):
            return rr["arms"][f"{arm}_{shots}"][metric]
        hopA = [r for r in rows if em(r, "hopA")]
        hopB = [r for r in rows if em(r, "hopB")]
        both = [r for r in rows if em(r, "hopA") and em(r, "hopB")]
        two_all = sum(em(r, "twohop") for r in rows)
        two_both = sum(em(r, "twohop") for r in both)
        nb_both = sum(em(r, "nobridge") for r in both)
        # shuffled-entity control on 2-hop: rescore greedy gens vs deranged gold
        gens = [r["arms"][f"twohop_{shots}"]["gen"] for r in rows]
        golds = [r["C"] for r in rows]
        dgolds = derange(golds, rng)
        shuf = sum(matcher(g, dg) for g, dg in zip(gens, dgolds))
        lp = [r["arms"][f"twohop_{shots}"]["logprob"]["mean"] for r in rows if "logprob" in r["arms"][f"twohop_{shots}"]]
        lp_both = [r["arms"][f"twohop_{shots}"]["logprob"]["mean"] for r in both]
        # frequency control: real 2-hop EM must clear 2x the shuffled-gold EM
        # (else apparent success is base-rate on low-diversity golds, e.g. a pair
        # whose gold is one dominant value). n_gold_distinct records the diversity.
        control_pass = bool(two_all > 0 and two_all >= 2 * shuf)
        pairs_out[pair] = {
            "n": len(rows),
            "hopA_rate": round(len(hopA) / len(rows), 3),
            "hopB_rate": round(len(hopB) / len(rows), 3),
            "n_both": len(both),
            "twohop_rate_all": round(two_all / len(rows), 3),
            "cond_twohop_rate": round(two_both / len(both), 3) if both else None,
            "nobridge_rate_both": round(nb_both / len(both), 3) if both else None,
            "twohop_em_count": int(two_all),
            "shuffled_twohop_count": int(shuf),
            "n_gold_distinct": len(set(golds)),
            "control_pass": control_pass,
            "logprob_mean_all": round(sum(lp) / len(lp), 3) if lp else None,
            "logprob_mean_both": round(sum(lp_both) / len(lp_both), 3) if lp_both else None,
        }
    return pairs_out


def verdict(run_dir: Path, tags: list[str]):
    all_scales = {}
    for tag in tags:
        p = run_dir / tag / "results.jsonl"
        if not p.exists():
            continue
        results = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        scale = {}
        for metric in ("em", "em_window"):
            for shots in ("zs", "fs"):
                per_pair = summarize(results, shots, metric)
                # a pair is admissible only with n_both>=5 AND a passing frequency
                # control (real 2-hop EM >= 2x shuffled-gold EM).
                def admissible(v):
                    return v["cond_twohop_rate"] is not None and v["n_both"] >= 5 and v["control_pass"]
                # VARIANCE-EXISTS if >=3 admissible pairs land cond_twohop in [0.2,0.8]
                band = [
                    k for k, v in per_pair.items()
                    if admissible(v) and 0.2 <= v["cond_twohop_rate"] <= 0.8
                ]
                label = "VARIANCE-EXISTS" if len(band) >= 3 else None
                if label is None:
                    rates = [v["cond_twohop_rate"] for v in per_pair.values() if admissible(v)]
                    if rates and all(r <= 0.2 for r in rates):
                        label = "FLAT-FAIL"
                    elif rates and all(r >= 0.8 for r in rates):
                        label = "FLAT-PASS"
                    else:
                        label = "MIXED-INCONCLUSIVE"
                scale[f"{metric}:{shots}"] = {"label": label, "band_pairs": band, "per_pair": per_pair}
        all_scales[tag] = scale
    (run_dir / "verdict.json").write_text(json.dumps(all_scales, indent=2), encoding="utf-8")
    _write_table(run_dir, all_scales)
    print(json.dumps({t: {s: all_scales[t][s]["label"] for s in all_scales[t]} for t in all_scales}, indent=2))


def _write_table(run_dir: Path, all_scales: dict):
    lines = ["# Scout multi-hop variance -- verdict table (D-028, scout tier, post_hoc)\n"]
    lines.append("Labels are scout observations, not findings. Banned from CLAIMS.md. "
                 "EM = greedy word-prefix exact-match; `em_window` = lenient (gold in "
                 "first-8-word window). `shuf` = real-2hop-EM v shuffled-gold control.\n")
    for tag, scale in all_scales.items():
        for cond in scale:  # e.g. "em:zs", "em_window:fs"
            metric, shots = cond.split(":")
            v = scale[cond]
            lab_shot = "zero-shot" if shots == "zs" else "4-shot"
            lab_metric = "strict-prefix" if metric == "em" else "lenient-window"
            lines.append(f"\n## {tag}  [{lab_shot}, {lab_metric}]  -> **{v['label']}**"
                         + (f"  (band: {', '.join(v['band_pairs'])})" if v["band_pairs"] else "") + "\n")
            lines.append("| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for k, p in v["per_pair"].items():
                ctrl = "ok" if p["control_pass"] else "FAIL"
                lines.append(
                    f"| {k} | {p['n']} | {p['hopA_rate']} | {p['hopB_rate']} | {p['n_both']} | "
                    f"{p['twohop_rate_all']} | {p['cond_twohop_rate']} | {p['nobridge_rate_both']} | "
                    f"{p['twohop_em_count']}v{p['shuffled_twohop_count']} | {ctrl} | {p['n_gold_distinct']} | {p['logprob_mean_both']} |"
                )
    (run_dir / "verdict_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--dtype", default="fp16")
    v = sub.add_parser("verdict")
    v.add_argument("--run", required=True)
    v.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    run_dir = Path(a.run)
    if a.cmd == "build":
        run_dir.mkdir(parents=True, exist_ok=True)
        bat = build_battery()
        (run_dir / "battery.json").write_text(json.dumps(bat, indent=2), encoding="utf-8")
        n = sum(len(p["items"]) for p in bat["pairs"])
        print(f"battery: {len(bat['pairs'])} pairs, {n} items")
        for p in bat["pairs"]:
            print(f"  {p['key']:24} bridge={p['bridge']:8} items={len(p['items']):3} (usable={p['n_usable_total']})")
    elif a.cmd == "run":
        run_model(run_dir, a.model, a.revision, a.tag, a.dtype)
    elif a.cmd == "verdict":
        verdict(run_dir, a.tags)


if __name__ == "__main__":
    main()
