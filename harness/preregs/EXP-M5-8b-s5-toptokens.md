# Preregistration — EXP-M5-8b: S5 top-k unembed-token diagnostic

- experiment-id: EXP-M5-8b
- status: **ORDERED** by Ecaterina 2026-07-23 (ruling on the EXP-M5-8 S5
  HETEROGENEOUS 1/4 result). Post-hoc diagnostic on the SAME S5 steering vectors;
  NOT a re-spec — re-cutting the readout words is refused under the
  anti-harvesting clause. This asks what the vectors actually POINT AT (their top
  unembed tokens), not whether they hit the words I picked, so it cannot be gamed.
- claim: none (diagnostic). Feeds the interpretation of EXP-M5-8's S5 verdict.
- models: EleutherAI/pythia-410m@9879c9b.
- author + date: Claude (build), 2026-07-23, per Ecaterina's ruling.

## Hypothesis

If EXP-M5-8's S5 HETEROGENEOUS (1/4) is a READOUT-WORD ARTIFACT, then all four
steering vectors' top unembed tokens are attribute-content-aligned (they point at
the right semantic field) even though only sentiment hit my chosen label words.
If the heterogeneity is REAL, only the sentiment vector's top tokens are
attribute-aligned; the other three point elsewhere. HYPOTHESIS tier.

## Decision rule (branches fixed by Ecaterina BEFORE the dump)

- **All four output-aligned in their top tokens** → the 1/4 was a readout-word
  artifact; S5 output-alignment is TYPE-GENERAL.
- **Only sentiment aligned** → HETEROGENEOUS is real; S5 is NOT a type-general
  output-aligned profile, reported as such.
- **Mixed** → reported per-vector, no forced species call.

The authoritative alignment call is Ecaterina's read of the RAW top tokens — not
an automated bar. A convenience flag (do the authored pos_w appear in the top-20?)
is reported but is explicitly NOT decisive.

## What counts as failure / honesty

Diagnostic — no pass/fail gate. The top tokens are dumped verbatim for human
judgement; the authored readout words are NEVER used to decide alignment (using
them would be the re-cutting that is refused). All four named vectors reported;
none curated.

## Estimator plan

Re-extract the four S5 steering vectors EXACTLY as EXP-M5-8 (mean-difference over
pos/neg answer states, 12-of-16 seed-varied subsample, 3 draws), and average the
per-layer raw direction over the 3 draws (the representative direction; per-draw
consistency of the top-1 token also reported). For each vector, at representative
band layers, dump the top-k tokens of unembed(mean_dir_l) [LOGIT lens, output
space] and unembed(J_l · mean_dir_l) [J-lens] over a cached lens draw.

## Sample plan

4 attributes × 3 extraction draws (averaged) × representative band layers
{low, mid, high}; top-15 tokens per (vector, lens, layer). Lens draw 0 as the
representative J-lens draw (top-1 token stability across the 3 lens draws noted).
Raw token lists retained.

## Resource estimate (Mac tier, 410M, MPS fp32)

Re-extraction ≈ 4 attrs × 12×2 × 3 draws ≈ 290 forwards + unembed matmuls +
one cached lens load. Projected ~2–3 min wall, peak ~2 GB. Under the 12 h LAW.

## Ratification

Ordered by Ecaterina 2026-07-23 as the non-confounded diagnostic to adjudicate
the S5 HETEROGENEOUS result; branches fixed above before the dump. Re-specifying
S5 with new readout words is refused under the anti-harvesting clause.
