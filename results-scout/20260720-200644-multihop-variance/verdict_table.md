# Scout multi-hop variance -- verdict table (D-028, scout tier, post_hoc)

Labels are scout observations, not findings. Banned from CLAIMS.md. EM = greedy word-prefix exact-match; `em_window` = lenient (gold in first-8-word window). `shuf` = real-2hop-EM v shuffled-gold control.


## pythia-1.4b  [zero-shot, strict-prefix]  -> **MIXED-INCONCLUSIVE**  (band: landmark->currency)

| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| landmark->capital | 40 | 0.25 | 0.9 | 8 | 0.0 | 0.0 | 1.0 | 0v0 | FAIL | 12 | -5.328 |
| landmark->currency | 40 | 0.475 | 0.425 | 8 | 0.125 | 0.375 | 1.0 | 5v0 | ok | 13 | -1.921 |
| landmark->language | 40 | 0.3 | 0.025 | 0 | 0.275 | None | None | 11v0 | ok | 10 | None |
| landmark->largest_city | 40 | 0.3 | 0.5 | 7 | 0.0 | 0.0 | 0.857 | 0v0 | FAIL | 10 | -7.329 |
| product->ceo | 40 | 0.325 | 0.0 | 0 | 0.0 | None | None | 0v0 | FAIL | 11 | None |
| product->hq | 25 | 0.4 | 0.92 | 10 | 0.0 | 0.0 | 1.0 | 0v0 | FAIL | 2 | -4.032 |
| father->mother | 40 | 0.175 | 0.025 | 1 | 0.025 | 0.0 | 0.0 | 1v0 | ok | 38 | -3.623 |

## pythia-1.4b  [4-shot, strict-prefix]  -> **VARIANCE-EXISTS**  (band: landmark->capital, landmark->currency, landmark->language, landmark->largest_city)

| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| landmark->capital | 40 | 0.65 | 0.925 | 24 | 0.25 | 0.375 | 0.75 | 10v1 | ok | 12 | -2.347 |
| landmark->currency | 40 | 0.7 | 0.75 | 23 | 0.325 | 0.391 | 0.957 | 13v3 | ok | 13 | -1.348 |
| landmark->language | 40 | 0.75 | 0.925 | 28 | 0.675 | 0.679 | 1.0 | 27v2 | ok | 10 | -0.734 |
| landmark->largest_city | 40 | 0.7 | 0.9 | 26 | 0.35 | 0.5 | 0.731 | 14v0 | ok | 10 | -2.741 |
| product->ceo | 40 | 0.675 | 0.425 | 13 | 0.2 | 0.154 | 0.769 | 8v5 | FAIL | 11 | -1.585 |
| product->hq | 25 | 0.88 | 0.92 | 21 | 0.92 | 0.952 | 1.0 | 23v23 | FAIL | 2 | -0.449 |
| father->mother | 40 | 0.25 | 0.1 | 0 | 0.025 | None | None | 1v1 | FAIL | 38 | None |

## pythia-1.4b  [zero-shot, lenient-window]  -> **MIXED-INCONCLUSIVE**  (band: landmark->currency, landmark->largest_city)

| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| landmark->capital | 40 | 0.275 | 0.9 | 9 | 0.0 | 0.0 | 1.0 | 0v0 | FAIL | 12 | -6.036 |
| landmark->currency | 40 | 0.475 | 0.85 | 15 | 0.6 | 0.8 | 1.0 | 24v1 | ok | 13 | -2.487 |
| landmark->language | 40 | 0.35 | 0.05 | 1 | 0.35 | 0.0 | 1.0 | 14v0 | ok | 10 | -2.291 |
| landmark->largest_city | 40 | 0.35 | 0.5 | 8 | 0.1 | 0.25 | 0.75 | 4v0 | ok | 10 | -7.775 |
| product->ceo | 40 | 0.375 | 0.0 | 0 | 0.0 | None | None | 0v0 | FAIL | 11 | None |
| product->hq | 25 | 0.44 | 0.92 | 11 | 0.0 | 0.0 | 1.0 | 0v0 | FAIL | 2 | -3.979 |
| father->mother | 40 | 0.2 | 0.025 | 1 | 0.025 | 0.0 | 0.0 | 1v0 | ok | 38 | -3.623 |

## pythia-1.4b  [4-shot, lenient-window]  -> **VARIANCE-EXISTS**  (band: landmark->capital, landmark->currency, landmark->language, landmark->largest_city)

| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| landmark->capital | 40 | 0.65 | 0.925 | 24 | 0.25 | 0.375 | 0.75 | 10v1 | ok | 12 | -2.347 |
| landmark->currency | 40 | 0.7 | 1.0 | 28 | 0.425 | 0.536 | 0.964 | 17v3 | ok | 13 | -1.427 |
| landmark->language | 40 | 0.75 | 0.95 | 28 | 0.675 | 0.679 | 1.0 | 27v2 | ok | 10 | -0.734 |
| landmark->largest_city | 40 | 0.7 | 0.9 | 26 | 0.35 | 0.5 | 0.731 | 14v0 | ok | 10 | -2.741 |
| product->ceo | 40 | 0.675 | 0.425 | 13 | 0.2 | 0.154 | 0.769 | 8v5 | FAIL | 11 | -1.585 |
| product->hq | 25 | 0.88 | 0.92 | 21 | 0.92 | 0.952 | 1.0 | 23v23 | FAIL | 2 | -0.449 |
| father->mother | 40 | 0.25 | 0.1 | 0 | 0.025 | None | None | 1v1 | FAIL | 38 | None |

## pythia-2.8b  [zero-shot, strict-prefix]  -> **MIXED-INCONCLUSIVE**  (band: landmark->currency)

| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| landmark->capital | 40 | 0.2 | 0.925 | 7 | 0.0 | 0.0 | 1.0 | 0v0 | FAIL | 12 | -4.291 |
| landmark->currency | 40 | 0.55 | 0.3 | 8 | 0.2 | 0.375 | 0.5 | 8v1 | ok | 13 | -1.576 |
| landmark->language | 40 | 0.35 | 0.0 | 0 | 0.15 | None | None | 6v0 | ok | 10 | None |
| landmark->largest_city | 40 | 0.375 | 0.7 | 11 | 0.025 | 0.0 | 0.818 | 1v0 | ok | 10 | -4.927 |
| product->ceo | 40 | 0.45 | 0.0 | 0 | 0.0 | None | None | 0v0 | FAIL | 11 | None |
| product->hq | 25 | 0.4 | 0.92 | 10 | 0.12 | 0.3 | 1.0 | 3v3 | FAIL | 2 | -2.591 |
| father->mother | 40 | 0.175 | 0.0 | 0 | 0.0 | None | None | 0v0 | FAIL | 38 | None |

## pythia-2.8b  [4-shot, strict-prefix]  -> **VARIANCE-EXISTS**  (band: landmark->capital, landmark->currency, landmark->language, landmark->largest_city)

| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| landmark->capital | 40 | 0.675 | 0.925 | 25 | 0.2 | 0.28 | 0.96 | 8v2 | ok | 12 | -2.479 |
| landmark->currency | 40 | 0.775 | 0.875 | 28 | 0.35 | 0.429 | 0.964 | 14v2 | ok | 13 | -1.022 |
| landmark->language | 40 | 0.75 | 1.0 | 30 | 0.725 | 0.8 | 0.933 | 29v1 | ok | 10 | -0.597 |
| landmark->largest_city | 40 | 0.6 | 0.85 | 22 | 0.275 | 0.455 | 0.818 | 11v0 | ok | 10 | -2.522 |
| product->ceo | 40 | 0.675 | 0.575 | 20 | 0.175 | 0.3 | 0.65 | 7v6 | FAIL | 11 | -1.287 |
| product->hq | 25 | 0.92 | 1.0 | 23 | 0.92 | 0.957 | 1.0 | 23v23 | FAIL | 2 | -0.466 |
| father->mother | 40 | 0.275 | 0.1 | 0 | 0.0 | None | None | 0v1 | FAIL | 38 | None |

## pythia-2.8b  [zero-shot, lenient-window]  -> **MIXED-INCONCLUSIVE**  (band: landmark->language)

| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| landmark->capital | 40 | 0.375 | 0.925 | 14 | 0.0 | 0.0 | 1.0 | 0v0 | FAIL | 12 | -5.285 |
| landmark->currency | 40 | 0.575 | 0.85 | 18 | 0.65 | 0.944 | 1.0 | 26v2 | ok | 13 | -2.179 |
| landmark->language | 40 | 0.425 | 0.825 | 14 | 0.4 | 0.5 | 1.0 | 16v1 | ok | 10 | -2.841 |
| landmark->largest_city | 40 | 0.425 | 0.7 | 12 | 0.025 | 0.0 | 0.833 | 1v0 | ok | 10 | -4.965 |
| product->ceo | 40 | 0.45 | 0.35 | 7 | 0.0 | 0.0 | 0.286 | 0v0 | FAIL | 11 | -2.335 |
| product->hq | 25 | 0.4 | 0.92 | 10 | 0.12 | 0.3 | 1.0 | 3v3 | FAIL | 2 | -2.591 |
| father->mother | 40 | 0.175 | 0.0 | 0 | 0.0 | None | None | 0v0 | FAIL | 38 | None |

## pythia-2.8b  [4-shot, lenient-window]  -> **VARIANCE-EXISTS**  (band: landmark->capital, landmark->currency, landmark->language, landmark->largest_city)

| pair | n | 1hopA | 1hopB | n_both | 2hop(all) | **cond-2hop** | nobridge(both) | shuf(real v shuf) | ctrl | #golds | logprob(both) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| landmark->capital | 40 | 0.675 | 0.925 | 25 | 0.2 | 0.28 | 0.96 | 8v2 | ok | 12 | -2.479 |
| landmark->currency | 40 | 0.775 | 1.0 | 31 | 0.375 | 0.452 | 0.968 | 15v2 | ok | 13 | -1.126 |
| landmark->language | 40 | 0.75 | 1.0 | 30 | 0.725 | 0.8 | 0.933 | 29v1 | ok | 10 | -0.597 |
| landmark->largest_city | 40 | 0.6 | 0.85 | 22 | 0.275 | 0.455 | 0.818 | 11v0 | ok | 10 | -2.522 |
| product->ceo | 40 | 0.675 | 0.575 | 20 | 0.175 | 0.3 | 0.65 | 7v6 | FAIL | 11 | -1.287 |
| product->hq | 25 | 0.92 | 1.0 | 23 | 0.92 | 0.957 | 1.0 | 23v23 | FAIL | 2 | -0.466 |
| father->mother | 40 | 0.3 | 0.1 | 0 | 0.0 | None | None | 0v1 | FAIL | 38 | None |
