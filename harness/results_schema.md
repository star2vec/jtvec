# Results directory schema

Every results directory is created only through `jtvec.core.runctx.start_run`
and must contain, at minimum:

```
results/<YYYYMMDD-HHMMSS>-<run_name>/
  <config>.yaml          # copied in unconditionally by start_run (LAW)
  run.json               # run record, keys below
  raw_completions/       # one <cell>.jsonl per result cell (LAW: retained
                         #   on disk for every reported number; >= 20 per
                         #   headline cell before a claim can be verified)
  report.md              # optional; numbers rendered only via
                         #   jtvec.core.reporting (scope + sham mandatory)
```

`run.json` required keys (validated by `jtvec.validators.results_dirs`):

| key            | meaning                                                     |
|----------------|-------------------------------------------------------------|
| run_name       | short experiment name                                       |
| started        | ISO timestamp                                               |
| git_commit     | HEAD at launch; tree was clean (enforced) so this is exact  |
| prereg_path    | path to the prereg, or null only when post_hoc              |
| prereg_commit  | commit that introduced the prereg (predates the run)        |
| config_sha256  | hash of the config actually used                            |
| post_hoc       | true = labeled post-hoc forever (LAW)                       |

Raw completion records are one JSON object per line; include at least the
prompt, the completion, and the seed/draw they came from.
