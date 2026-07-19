# TAXONOMY_DESIGN.md — Workspace residency as a property of representation type

Status: design document (pre-prereg). Nothing here is a finding. Each Hn
becomes a HYPOTHESIS entry in CONSTRAINTS.md; each experiment gets its own
prereg with ratified constants before any run. Predictions below are
registered guesses — the point of the matrix is to be fillable-in-advance
and falsifiable.

## Construct

A direction (or operator) is *workspace-resident* to the degree it scores on
five axes, each measured by an already-gated or to-be-gated instrument:

- A1 decodability — identity/content rank through the M1-gated J-lens,
  vs logit-lens and norm-matched-random baselines, median over >= 3 lens
  draws x >= 3 species draws (E1 machinery; lens-draw marginalization is
  mandatory per the E1 lesson).
- A2 causal potency — ablation removes the species' function vs matched
  sham (M3 fv-direction-ablation pattern, per-species function metric).
- A3 basis-mediation — lens-coordinate intervention vs raw-direction
  intervention gap (E3 machinery; the novel axis).
- A4 report coupling — ablation moves the gated continuous report_score of
  the species' content (E2 machinery; P3-class prior-corrected score;
  power caveat from E2 carried into every prereg).
- A5 flexibility — one edit propagates to >= 3 distinct downstream queries
  (Anthropic France->China pattern; capability-gated at small scale).

Entry requirement per species: an M2-style stability certificate
(convergence under re-extraction, >= 3 draws, sham arm, positive+negative
instrument controls) BEFORE any axis is measured.

## Species roster

- S1 concept/entity directions. Extraction: mean-difference / answer-state
  directions on certified tasks (capital-recall answers; entity identity).
  Largely existing apparatus (M1/M3).
- S2 function vectors (fv_todd, M2-certified). DONE — E1/E2/E3 results
  stand as this species' axis values.
- S3 relational operators (LRE; Hernandez et al. ICLR 2024). Affine maps
  s -> o estimated by Jacobian at few-shot context. Measured twice: the
  OPERATOR (as machinery) and its ARGUMENTS/OUTPUTS (as content).
  Adaptation note: A1/A3 need operator-specific definitions (decode W's
  action on a probe set; apply-in-lens-coords vs apply-raw).
- S4 binding vectors (Feng & Steinhardt-style binding IDs). Capability-
  gated: requires the substrate to pass binding-task baselines.
- S5 (optional, contrast) behavioral steering direction (sentiment-class
  mean-difference; refusal unavailable on base Pythia).

## Prediction matrix (registered guesses; H = high, L = low, M = mixed)

| species | A1 decode | A2 potency | A3 basis | A4 report | A5 flex |
|---|---|---|---|---|---|
| S1 concept        | H | H | H | H | H |
| S2 FV             | L (E1: confirmed) | H (E2: confirmed) | L (E3: confirmed) | L/inverted (E2: confirmed) | L |
| S3 LRE operator   | L | H | L | L | M |
| S3 LRE output     | H | H | H | H | H |
| S4 binding        | L | H | L | L | L |
| S5 steering       | M | H | M | M | M |

## Hypotheses

- H1 (dichotomy, confirmatory): S1 and S2 occupy opposite poles of the
  matrix at preregistered bars. Anchor result; both poles already have
  supporting certified evidence (M1 +0.60 lens-coord swaps; E1/E3
  negatives).
- H2 (axis coupling): A1 and A3 agree per species. A dissociation
  (decodable-but-basis-agnostic or dark-but-basis-mediated) in any species
  is a headline refinement: verbalizable != workspace-functional.
- H3 (report specificity): A4 tracks residency; dark-species ablation
  spares or RAISES report readouts (the E2 fv-report rise becomes a
  predicted signature, not an anomaly).
- H4 (relational split): the LRE operator is dark while its outputs are
  resident; workspace ablation impairs relation NAMING but not relation
  APPLICATION.
- H5 (dark boundary): binding vectors are potent and dark on all other
  axes.

## Falsification / outcome semantics

- All species one profile -> residency construct fails; report as
  deflation: the J-lens indexes output-proximity, not representation type.
- Matrix as predicted -> the taxonomy claim, with H2/H4 cells as depth.
- Partial -> per-cell surprises are individually reportable; every cell is
  pre-registered so no outcome is unpublishable.

## Scope

Primary substrate: Pythia-1.4B (LRE/binding capability risk at 410M —
baseline-gate before committing; escalate to 2.8B if 1.4B fails gates).
410M kept as small anchor where capable. One replication scale minimum.
Emergence-over-training per species = Section 6 / sequel, reusing the
validated revision-keyed sweep pipeline.

## Non-goals (this paper)

No consciousness claims. No multi-token lens extension. No new lens
variants. The emergence sweep runs only after the adult-model matrix is
filled.
