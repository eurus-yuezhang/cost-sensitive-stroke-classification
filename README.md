# Cost-Sensitive Stroke Classification

A corrected reconstruction of a coursework experiment: an interpretable decision tree that chooses splits by reducing `10 × FN + FP`, compared with majority-class and logistic-regression baselines.

**Status:** corrected reconstruction v2. The corrected code has passed synthetic software checks. The historical dataset and original executed notebook are unavailable, so real-data performance and the historical results have **not** been reproduced. This is an educational classification study, not a clinical screening system or a validated estimate of future stroke risk.

## What changed in this version

- Split raw rows first, using a deterministic stratified 60/20/20 split. Fit imputation medians and category vocabularies on training rows only. Fit a fresh preprocessor on train+validation for the final model, never on test rows.
- In optional repeated-selection diagnostics, fit preprocessing independently inside each training fold.
- Search all midpoints between distinct feature values. The previous label-boundary shortcut could miss useful splits when equal-valued rows were reordered. Approximate candidate limiting is no longer the default.
- Reject missing/fractional/non-binary labels before integer conversion. Reject missing feature columns and infinite numeric values. Unknown categorical values use fixed all-zero indicators; they do not change the trained schema.
- Save the tree together with preprocessing state, feature order, cost assumptions and decision threshold in a JSON model bundle. Check exact prediction equality after loading.
- Record the input CSV SHA-256, split row positions, runtime versions and selected configuration in a run manifest.

## Quick start

Tested with Python 3.12.14. Direct dependency versions are pinned; a clean-room install has not yet been verified.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/synthetic_smoke.py
```

For a headless machine, set `MPLBACKEND=Agg`. If its home directory is read-only, also set writable `MPLCONFIGDIR` and `XDG_CACHE_HOME` paths.

The synthetic demo creates artificial adult records in a temporary directory, runs the experiment, and saves clearly labelled software-check outputs to `synthetic_outputs/`. Its metrics are not project performance claims.

## Run with a separately obtained dataset

The historical report cites [Kaggle's Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). No dataset is bundled or downloaded automatically. Verify the current dataset license and provenance before use or redistribution. A newly obtained dataset need not be the exact historical file.

Expected raw columns: `age`, `avg_glucose_level`, `bmi`, `hypertension`, `heart_disease`, `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`, plus binary `stroke`. An `id` column is ignored by the model. Both classes need enough examples for stratified splitting; the tool raises an error rather than silently switching split methods.

```bash
python stroke_tree.py --csv /path/to/healthcare-dataset-stroke-data.csv \
  --output-dir stroke_tree_outputs --no-show
```

The CLI fits/selects the tree on training/validation data, refits on development data, then compares the final models on test data. Generated files include `test_model_summary.csv`, `grid_search_results.csv`, rule and confusion-matrix outputs, `model_bundle.json`, and `run_manifest.json`.

Use `--run-robustness --repeats 30 --bootstrap 1000` only for optional diagnostics. These repeatedly evaluate the same test set and must not be used to pick a model. They measure conditional selection sensitivity and test-sample uncertainty, not independent external validation or all sources of training uncertainty.

## Version and result provenance

1. **Historical coursework:** A2 Journal reports a final tree cost of 349, recall approximately 0.667 and precision approximately 0.160; logistic regression cost 310; majority baseline cost 510. These are historical report values, not results verified by this repository.
2. **September 2026 reconstruction:** the source was assembled from the Journal and prior ChatGPT conversation. Its notebook had no saved execution outputs.
3. **Corrected reconstruction v2:** this independent working copy fixes preprocessing and threshold generation, changes the raw-row split to stratification, and adds regression tests and a complete preprocessing/model bundle. Changed methods mean old numbers are not expected outputs. The source files in Space and the historical report were not edited.

The original work included ChatGPT-assisted code and explanations. This correction also used AI assistance. Algorithm implementation is from scratch rather than a scikit-learn tree wrapper; that does not mean every line was authored without assistance. Attribution and any course publication rules should be reviewed before publishing.

## Files

- `stroke_tree.py`: experiment, train-only preprocessing, custom tree and persistence.
- `tests/test_stroke_tree.py`: regression, validation, serialization and synthetic integration tests.
- `examples/synthetic_data.py`: reproducible artificial data generator.
- `examples/synthetic_smoke.py`: local end-to-end software check.
- `VALIDATION.md`: checks performed and remaining limits.
- `requirements.txt`: tested direct dependency versions.

## Limits and publication checklist

- The 10:1 error-cost ratio and risk-band cutoffs are experimental assumptions. Probabilities and bands are not clinically calibrated; no prospective or external clinical validation has been performed.
- Logistic regression had lower cost than the tree in the historical report. Interpretability and explicit trade-offs are the contribution, not a claim of superior predictive performance.
- Exhaustive threshold search prioritizes correctness and can be slower on larger datasets. Runtime on the original 5,110-row dataset remains unmeasured.
- A split from this previously studied dataset is not new independent clinical evidence. No test metric should guide model choices.
- Keep raw records, generated models/results, credentials, full chat logs and student identifiers out of a public repository. The ignore rules reduce accidental inclusion but are not a privacy guarantee.
- No open-source license has been assigned. Confirm ownership, attribution, course rules and the desired license before public release.

