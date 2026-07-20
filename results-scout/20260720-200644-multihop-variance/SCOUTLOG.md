# SCOUTLOG — multi-hop variance pilot (D-028)

**Tier:** scout. **post_hoc.** Single-run, uncertified. HARD-BANNED from
CLAIMS.md and from findings language (D-024/D-028 precedent). Labels below
(VARIANCE-EXISTS / FLAT-FAIL / FLAT-PASS / MIXED) are scout observations, not
findings. This document stays at the preview/scan register.

Machine: win32 RTX 2000 Ada laptop (8.59 GB VRAM), torch 2.13.0+cu130.
Parent commit at run time: recorded in each `<tag>/manifest.json`.

## Purpose
Price the key assumption of the (not-yet-in-repo) "relgraph" spinoff: is
*latent* (no-CoT) 2-hop success VARIABLE — some chains pass, some fail — at a
Pythia scale we can afford, conditional on the model knowing both constituent
1-hops? If multi-hop is uniformly failed or uniformly passed, relgraph dies and
the taxonomy proceeds alone. Not building relgraph; measuring its precondition.

## Substrates
- pythia-1.4b @ fedc38a16eea3bd36a96b906d78d11d2ce18ed79 (D-023 pin), fp16 inference.
- pythia-2.8b @ 2a259cdd96a4beb1cdf467512e3904197345f6a9 (main resolved 2026-07-20
  via HfApi), fp16 inference. fp16 chosen (scout tier permits) to fit 8 GB and
  for speed; dtype recorded in manifest.

## Battery (build: `scripts/scout_multihop.py build`)
From third_party/relations (Hernandez et al. 2308.09124), 2-hop compositions
with a LATENT bridge entity; 7 relation pairs across 3 bridge types, 265 items:

| pair | bridge | items | usable joins |
|---|---|---|---|
| landmark->capital | country | 40 | 457 |
| landmark->currency | country | 40 | 393 |
| landmark->language | country | 40 | 457 |
| landmark->largest_city | country | 40 | 507 |
| product->ceo | company | 40 | 398 |
| product->hq | company | 25 | 25 |
| father->mother | person | 40 | 50 |

Joins are on the shared bridge entity (A.object == B.subject). Items where the
final answer C equals the subject X or the bridge B are dropped (no
echo-the-input shortcut). Seed 28, deterministic sampling + exemplar selection.

Arms per item (each zero-shot AND 4-shot): 1-hopA, 1-hopB, 2-hop (latent
bridge), no-bridge paraphrase (bridge stated explicitly, two-sentence — the
format control). Measures: greedy exact-match (word-prefix, small country alias
set {US,UK}; lenient window-containment logged alongside as `em_window`) and
log-prob of the correct 2-hop answer. Shuffled-entity control: the greedy 2-hop
generations re-scored against a deranged gold (no extra compute) — guards
against exact-match succeeding by base-rate/frequency alone.

## Verdict statistic
Per pair, among items where BOTH 1-hops pass: the conditional 2-hop success
rate `cond_twohop_rate` (n_both). Per scale x shot condition:
- VARIANCE-EXISTS if >= 3 pairs land cond_twohop in [0.2, 0.8] (n_both >= 5);
- FLAT-FAIL / FLAT-PASS if all such pairs sit <= 0.2 / >= 0.8;
- else MIXED-INCONCLUSIVE.

## Compute estimate (recorded BEFORE the run; >10-min rule)
1.4b fp16 smoke on this GPU: load 7.2 s; greedy 10-tok ~0.21 s/gen (zs),
~0.18 s/gen (4-shot, 55-tok prompt); peak VRAM 2.85 GB.
- Per model: 265 items x 8 greedy gens = 2120 gens x ~0.2 s ~= 7 min, plus
  530 teacher-forced logprob forwards (~0.5 min). ~= 8 min for 1.4b.
- 2.8b fp16 ~2x per-token + larger weights: est. ~15-20 min; peak VRAM est.
  ~6-6.5 GB (5.6 GB weights + activations) < 8.59 GB.
- Both scales: ~25-30 min scientific compute. Under the 12 h LAW; single
  laptop; no A100 flag. Downloads (1.4b ~2.8 GB done; 2.8b ~5.6 GB) are setup.

## Run log
- pythia-1.4b @ fedc38a, fp16: load 7.2 s, run 350.8 s (~5.8 min), 2120
  greedy gens + 530 logprob forwards, peak VRAM 2.88 GB. results in
  `pythia-1.4b/`, raw in `raw_completions/pythia-1.4b/`.
- pythia-2.8b @ 2a259cd, fp16: run 611.8 s (~10.2 min), 2120 gens, peak VRAM
  5.64 GB (GPU 5.6/8.0 GB in use mid-run; no OOM). results in `pythia-2.8b/`.
- Total scientific compute ~16 min (both scales), vs ~25-30 min estimate;
  under the 12 h LAW. Verdict is post_hoc from stored results.jsonl.

## Scan result (scout observations — NOT findings; banned from CLAIMS.md)
Full grid in `verdict_table.md`; machine-readable in `verdict.json`. Labels
per scale x shot x matcher. A pair is *admissible* for the variance test only
with n_both >= 5 AND a passing frequency control (real 2-hop EM >= 2x the
shuffled-gold EM).

- **Zero-shot, both scales: MIXED-INCONCLUSIVE.** Obscure-landmark hop-A
  mostly fails zero-shot, so the both-pass populations are thin; where a pair
  is admissible the latent 2-hop sits at/near floor (cond-2hop 0.0 on
  capital/largest_city). Too thin to call FLAT-FAIL.
- **4-shot, both scales: VARIANCE-EXISTS** (strict AND lenient matcher). The
  four country-bridge landmark pairs are all admissible and land in
  [0.2, 0.8]:
  - cond-2hop 1.4b: capital 0.375, currency 0.391, language 0.679,
    largest_city 0.500 (n_both 23-28).
  - cond-2hop 2.8b: capital 0.280, currency 0.429, language 0.800,
    largest_city 0.455 (n_both 22-30).
  Neither uniformly failed nor uniformly passed; the spread is stable across
  the two scales.
- Ordering is consistent across scales: **most compositional** =
  landmark->language (0.68 / 0.80), then largest_city, currency; **least** =
  landmark->capital (0.375 / 0.28). Intuitive (language = low-entropy 2nd hop;
  capital = most specific).
- **No-bridge control is high** (0.65-1.0, including zero-shot) on the same
  both-pass items -> the 2-hop drop is not a prompt-format artifact; the cost
  is specifically the LATENT bridge.
- **Frequency control did its job.** product->hq is a base-rate trap: its
  golds are 92% one value ("Kyoto", 23/25 Nintendo products), shuffled EM ==
  real EM (23v23) -> excluded. product->ceo real ~ shuffled -> excluded.
  father->mother has no both-pass items at either scale (obscure grandparents).
- Scale 1.4b -> 2.8b: modest, non-uniform movement (language 0.68->0.80;
  capital 0.375->0.28); the variance is present at both scales, not a
  small-model artifact. n_both grows at 2.8b (stronger 1-hops).

Caveats (scout-grade): single draw; strict EM is a conservative lower bound on
multi-word golds (lenient `em_window` logged alongside, same labels); country
aliases limited to {US, UK}; landmark hop-A obscurity is the main brake on the
zero-shot both-pass populations. All admissible signal is carried by the
country bridge; company/person bridges did not yield admissible pairs here.
