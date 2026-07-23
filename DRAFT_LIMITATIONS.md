# Limitations (paper-bound running ledger; folds into DRAFT.md §Limitations)

Acknowledged open questions the write-up must carry, not drop. Language-linted
(DRAFT*.md). Each entry cites its LABNOTES decision.

## L1 — FV re-extraction can produce degenerate endpoints; draw-stability is conditional on healthy extractions

The S2 function-vector draw-stability figures (EXP-M5-8 S2 min_pairwise_cosine
0.983–0.997; M2 same-pipeline 0.959–0.997, reproduced cross-machine to the digit)
are measured over HEALTHY extractions — draws whose function vectors steer at full
strength. On identical weights, a v1 checkpoint-sweep ENDPOINT re-extraction once
produced a DEGENERATE function vector (capitalize induction collapsed +38.8% →
+1.8%), giving cross-draw cosine 0.43–0.61 to the healthy Phase-2 vector — the v1
VERIFIED entry, correctly scoped to that sweep endpoint rather than demoted.

So Todd-style FV re-extraction OCCASIONALLY produces degenerate endpoints, and the
CAUSE was never bounded: the controlled fresh-draw experiment that would
characterise the failure rate (v1's `15_fv_stability.py`) was written as the
blocking response but never ran (the project pivoted to v2). Consequence for this
paper: every draw-stability figure is conditional on healthy extractions, and the
estimator's degeneracy tail — rare intrinsic tail vs sweep-harness artifact —
remains an ACKNOWLEDGED OPEN QUESTION, not a resolved one. (D-035.)
