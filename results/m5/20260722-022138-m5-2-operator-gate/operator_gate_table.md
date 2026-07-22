# EXP-M5-2 S3 operator-gate — 0/8 converged; S3-set-certified=False

Functional gate (prereg EXP-M5-2). Certificate + sign-off are Ecaterina's.
Substrate EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79, h_layer=12, k_estimate=6, n_draws=3. Bars: {'agreement': 0.9, 'output_cosine': 0.95, 'faithfulness_pos': 0.6, 'shuffled_neg': 0.1, 'n_relations_certify': 6}

| relation | n_probe | top1-agree | out-cos | faith(pos) | shuffled(neg) | ctrl | CONVERGED | W-cos(desc) |
|---|---|---|---|---|---|---|---|---|
| factual/country_capital_city | 12 | 0.75 | 0.9519 | 0.75 | 0.6667 | FAIL | no | 0.921 |
| factual/food_from_country | 15 | 0.8 | 0.9528 | 0.7333 | 0.7333 | FAIL | no | 0.9016 |
| factual/product_by_company | 30 | 0.1667 | 0.92 | 0.4333 | 0.3667 | FAIL | no | 0.7574 |
| linguistic/adj_antonym | 30 | 0.0 | 0.9204 | 0.0667 | 0.0 | FAIL | no | 0.7869 |
| linguistic/verb_past_tense | 30 | 0.0 | 0.908 | 0.0667 | 0.0 | FAIL | no | 0.824 |
| linguistic/word_first_letter | 30 | 0.2 | 0.9301 | 0.4 | 0.1667 | FAIL | no | 0.8235 |
| commonsense/object_superclass | 30 | 0.2 | 0.9144 | 0.2333 | 0.1667 | FAIL | no | 0.8519 |
| commonsense/work_location | 19 | 0.3684 | 0.9205 | 0.2632 | 0.2632 | FAIL | no | 0.8606 |
