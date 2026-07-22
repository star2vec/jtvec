# EXP-M5-2b S3 marginalized/cosine criterion — 2/8 certified -> **H-S3-HARD**

Final budgeted amendment cycle. Positive criterion = pre-registered re-analysis of results/m5/20260722-022138-m5-2-operator-gate (post-hoc); D-034 negative estimated fresh. Certificate + sign-off are Ecaterina's.
Substrate EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79, h_layer=12. Bars: {'output_cosine': 0.95, 'marginalized_faith': 0.6, 'negative': 0.1, 'n_relations_certify': 6}

| relation | n_probe | out-cos (>=0.95) | marg-faith (>=0.60) | per-draw faith | raw-agree | donor neg (<=0.10) | ctrl | CERTIFIED |
|---|---|---|---|---|---|---|---|---|
| factual/country_capital_city | 12 | 0.9519 | 0.75 | [0.75, 0.75, 0.6667] | 0.75 | 0.0 (adj_antonym) | ok | YES |
| factual/food_from_country | 15 | 0.9528 | 0.7333 | [0.6667, 0.7333, 0.7333] | 0.8 | 0.0 (adj_antonym) | ok | YES |
| factual/product_by_company | 30 | 0.92 | 0.4 | [0.3, 0.4333, 0.4667] | 0.1667 | 0.0 (adj_antonym) | FAIL | no |
| linguistic/adj_antonym | 30 | 0.9204 | 0.1 | [0.0667, 0.0667, 0.1] | 0.0 | 0.0333 (country_capital_city) | FAIL | no |
| linguistic/verb_past_tense | 30 | 0.908 | 0.1 | [0.0333, 0.0667, 0.0667] | 0.0 | 0.1667 (country_capital_city) | FAIL | no |
| linguistic/word_first_letter | 30 | 0.9301 | 0.4 | [0.3, 0.4, 0.4667] | 0.2 | 0.2 (country_capital_city) | FAIL | no |
| commonsense/object_superclass | 30 | 0.9144 | 0.4667 | [0.8, 0.2333, 0.2333] | 0.2 | 0.0 (country_capital_city) | FAIL | no |
| commonsense/work_location | 19 | 0.9205 | 0.3158 | [0.3158, 0.2632, 0.2105] | 0.3684 | 0.0526 (country_capital_city) | FAIL | no |

close-but-under = FAIL (H-S3-HARD); final budgeted cycle
