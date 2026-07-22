# EXP-M5-2 — post-hoc raw replay (labeled post-hoc forever)

Read-only diagnostic run AFTER the gate, to substantiate the uniform control
failure (surprise -> replay rule). NOT part of the gate; no threshold/spec change.

## Question
Why does the shuffled-relation negative control FAIL on the factual relations
(shuffled operator ~ as faithful as the real one: country_capital 0.75 vs 0.67,
food 0.733 vs 0.733)?

## Method
For held-out country_capital_city subjects, compare per subject: logit-lens@L12
(mt.lm_head applied to the raw subject state h at L12, NO operator), the REAL
operator top-1, the SHUFFLED-label operator top-1, and the true object.

## Result (h_layer=12, k_estimate=6)
```
subj             logitlens@L12    real_op        shuf_op        true
China            ' is'            ' Beijing'     ' Beijing'     Beijing
United States    ' is'            ' Washington'  ' Washington'  Washington D.C.
India            ' is'            ' New'         ' Delhi'       New Delhi
Colombia         ' is'            ' Bog'         ' Bog'         Bogota
Peru             ' is'            ' L'           ' L'           Lima
Germany          ' is'            '\n'           ' L'           Berlin
Turkey           ' is'            ' Istanbul'    ' Istanbul'    Ankara (both -> largest city)
Saudi Arabia     ' is'            ' S'           ' M'           Riyadh
```

## Reading (scoped; instrument diagnostic, not a model finding)
- logit-lens@L12 predicts " is" for every subject -> the raw subject state h at
  L12 does NOT pre-encode the object. (This corrects an on-the-fly hypothesis
  that h was answer-encoding; it is not.)
- The linear operator W does the country->capital mapping (real op recovers
  Beijing/Washington/Bog/L...). The SHUFFLED-label operator recovers the SAME
  true objects.
- Mechanism: JacobianIclMeanEstimator derives (W, b) from the model's local
  computation dz/dh at the exemplar subjects. That Jacobian reflects the MODEL's
  true relation computation and is insensitive to permuting the ICL exemplars'
  object LABELS (the surface text changes; the model still computes the true
  capital, so dz/dh still maps to true capitals). Averaging over shuffled-label
  exemplars therefore still yields a country->true-capital operator.
- Consequence: the ratified negative control (permute s->o labels among ICL
  exemplars) does NOT create a null for a Jacobian-based estimator, so
  "shuffled ~ real" is expected and does not, by itself, establish the operator
  as invalid. A control that actually nulls the relational signal would have to
  break it in the Jacobian (e.g. estimate on non-relational / random-token
  prompts, or apply an UNRELATED relation's operator to these subjects) — a
  spec question for Ecaterina, NOT adopted here.

Separately, the linguistic/commonsense relations fail the POSITIVE control
(real held-out faithfulness 0.07-0.43 < 0.60) at h_layer=12 — a distinct issue
(operator weak on these relations at this layer); layer is NOT re-picked here.
