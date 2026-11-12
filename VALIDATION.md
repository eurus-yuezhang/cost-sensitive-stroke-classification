# Validation record

## Passed

- `python -m unittest discover -s tests -v`: **13 tests passed** on the final code (2.816 seconds in the original check; 2.291 seconds in the pre-publication recheck).
- `python -m compileall -q stroke_tree.py examples tests`: Python syntax compilation passed.
- `python stroke_tree.py --help`: CLI argument parsing/help passed.
- `python examples/synthetic_smoke.py --output-dir /tmp/stroke_v2_synthetic_outputs`: a 240-row artificial-data workflow completed, including baseline comparison, validation selection, final refit, charts, JSON model bundle and run manifest.

The tests include an additional 160-row artificial end-to-end run with two repeated selections and 20 bootstrap resamples. These reduced counts exercise the optional paths; they are not a 30-repeat/1,000-bootstrap experiment.

## Regression coverage

1. The duplicate-value counterexample now finds threshold 15 and reduces cost from 20 to 10.
2. Twelve permutations of that fixture produce the same tree.
3. Exhaustive candidate enumeration does not silently cap at 256 or skip same-label boundaries.
4. Best-split choices agree with an independent brute-force cost calculation on ten small random fixtures.
5. Invalid numeric threshold inputs and invalid candidate limits fail explicitly.
6. Held-out extreme values/categories do not change training medians, vocabulary or feature order.
7. Preprocessing JSON state round-trips exactly.
8. Missing schema fields, infinities and all-missing training numeric columns are rejected.
9. Missing, fractional and non-binary labels are rejected before conversion or imputation.
10. Raw stratified splits are deterministic, disjoint and complete.
11. A complete model bundle reproduces raw-input predictions, including imputation and unseen categories.
12. Missing/NaN encoded split features fail instead of silently routing right.
13. Instrumented end-to-end fitting confirms that selection, refit and every repeated fold exclude the fixed test rows. The run manifest, outputs and consistent final feature schema are checked.

## Environment

Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; Matplotlib 3.10.8; scikit-learn 1.8.0. Tests use the standard-library unittest module. Headless execution used `MPLBACKEND=Agg`, writable `MPLCONFIGDIR` and `XDG_CACHE_HOME` settings. Initial font/cache warnings did not prevent execution; the final run used writable cache paths.

## Not verified

- No original/patient data was available or used. No historical scores were reproduced.
- No original Colab restoration is required or claimed.
- No fresh virtual-environment install, cross-platform run, dedicated lint/type checker, large-data runtime benchmark or independent clinical validation was performed.
- Direct dependency versions are pinned, but this is not a complete transitive dependency lock.
- The optional diagnostic's full 30 repeats/1,000 bootstrap samples have not been run on real data.
- A basic text scan found no obvious API/private-key patterns or the student identifier from the source report in the deliverable. This is not a comprehensive privacy or security audit.
- Dataset rights, course publication policy, desired code license and final author attribution still need review before publication.

## Scope

This record describes local software checks performed before publication. The original Space files and report were not edited. The source package contains only code, documentation and dependency/ignore configuration. Synthetic output records are outside the source package and excluded from the delivery ZIP. No software release or deployment is claimed.

