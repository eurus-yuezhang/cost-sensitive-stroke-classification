import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import stroke_tree as st
from examples.synthetic_data import make_synthetic_stroke


class ThresholdRegressionTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.Series([10.] * 21 + [20.] * 20)
        self.y = pd.Series([0] * 20 + [1] * 21)

    def test_duplicate_boundary_regression(self):
        self.assertEqual(st.candidate_thresholds_numeric(self.x, self.y), [15.0])
        cut = st.best_threshold_for_feature_by_cost(self.x, self.y, min_leaf=20)
        self.assertEqual(cut["feature_threshold"], 15.0)
        self.assertEqual(cut["parent_cost"], 20)
        self.assertEqual(cut["left_cost"] + cut["right_cost"], 10)
        self.assertEqual(cut["delta_cost"], 10)

    def test_tree_is_invariant_to_row_permutation(self):
        X = pd.DataFrame({"x": self.x})
        expected = st.build_tree_by_cost(X, self.y, min_leaf=20)
        rng = np.random.default_rng(17)
        for _ in range(12):
            idx = rng.permutation(len(X))
            actual = st.build_tree_by_cost(
                X.iloc[idx].reset_index(drop=True),
                self.y.iloc[idx].reset_index(drop=True), min_leaf=20
            )
            self.assertEqual(actual, expected)

    def test_all_distinct_midpoints_are_searched_by_default(self):
        x = pd.Series(np.arange(400.))
        cuts = st.candidate_thresholds_numeric(x, pd.Series(np.zeros(400)))
        self.assertEqual(len(cuts), 399)
        self.assertEqual(cuts[0], 0.5)
        self.assertEqual(cuts[-1], 398.5)
        self.assertEqual(len(st.candidate_thresholds_numeric(x, x, max_candidates=5)), 5)

    def test_best_cut_matches_small_independent_exhaustive_search(self):
        rng = np.random.default_rng(32)
        for _ in range(10):
            x = pd.Series(rng.integers(10, 30, 80).astype(float))
            y = pd.Series(rng.integers(0, 2, 80))
            def loss(labels):
                probability = (int(labels.sum()) + 1) / (len(labels) + 2)
                return int((labels == 0).sum()) if probability >= 1 / 11 else 10 * int(labels.sum())
            best_loss, best_threshold = float("inf"), None
            unique = sorted(x.unique())
            for a, b in zip(unique[:-1], unique[1:]):
                threshold = (a + b) / 2
                left, right = y[x <= threshold], y[x > threshold]
                if min(len(left), len(right)) < 4:
                    continue
                value = loss(left) + loss(right)
                if value < best_loss:
                    best_loss, best_threshold = value, threshold
            actual = st.best_threshold_for_feature_by_cost(x, y, min_leaf=4)
            self.assertEqual(actual["feature_threshold"], best_threshold)
            self.assertEqual(actual["left_cost"] + actual["right_cost"], best_loss)

    def test_invalid_threshold_values_are_rejected(self):
        with self.assertRaises(ValueError):
            st.candidate_thresholds_numeric(pd.Series([1., np.nan]), pd.Series([0, 1]))
        with self.assertRaises(ValueError):
            st.candidate_thresholds_numeric(self.x, self.y, max_candidates=0)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.raw = make_synthetic_stroke(120)

    def test_heldout_values_do_not_change_statistics_or_vocabulary(self):
        train = self.raw.iloc[:10].copy()
        train["bmi"] = [20., 30., np.nan, 20., 30., 20., 30., 20., 30., 20.]
        train["work_type"] = "training-category"
        prep = st.StrokePreprocessor().fit(train)
        before = json.dumps(prep.to_dict(), sort_keys=True)
        heldout = self.raw.iloc[10:12].copy()
        heldout["bmi"] = [100000., np.nan]
        heldout["work_type"] = "heldout-only-category"
        transformed = prep.transform(heldout)
        self.assertEqual(prep.medians["bmi"], 20.)
        self.assertEqual(transformed.loc[1, "bmi"], 20.)
        self.assertEqual(before, json.dumps(prep.to_dict(), sort_keys=True))
        self.assertFalse(any("heldout-only" in name for name in transformed.columns))
        self.assertEqual(transformed.filter(like="work_type=").to_numpy().sum(), 0)
        self.assertEqual(list(transformed.columns), prep.feature_names)

    def test_preprocessing_state_roundtrip(self):
        prep = st.StrokePreprocessor().fit(self.raw)
        restored = st.StrokePreprocessor.from_dict(json.loads(json.dumps(prep.to_dict())))
        pd.testing.assert_frame_equal(prep.transform(self.raw), restored.transform(self.raw))

    def test_invalid_raw_inputs_are_rejected(self):
        prep = st.StrokePreprocessor().fit(self.raw)
        with self.assertRaises(ValueError):
            prep.transform(self.raw.drop(columns="age"))
        with self.assertRaises(ValueError):
            prep.transform(self.raw.assign(age=np.inf))
        with self.assertRaises(ValueError):
            st.StrokePreprocessor().fit(self.raw.assign(bmi=np.nan))

    def test_missing_fractional_and_nonbinary_labels_are_rejected(self):
        for value in [np.nan, 0.5, 2, np.inf]:
            bad = self.raw.copy()
            bad["stroke"] = bad["stroke"].astype(float)
            bad.loc[0, "stroke"] = value
            with self.assertRaises(ValueError):
                st.extract_target(bad)
        valid = st.extract_target(self.raw)
        np.testing.assert_array_equal(valid, self.raw["stroke"])

    def test_raw_split_is_deterministic_disjoint_complete_and_stratified(self):
        y = st.extract_target(self.raw)
        splits = st.split_row_positions(y)
        self.assertEqual([len(i) for i in splits], [72, 24, 24])
        self.assertEqual(sorted(np.concatenate(splits).tolist()), list(range(120)))
        self.assertEqual(len(set(splits[0]) & set(splits[1])), 0)
        for a, b in zip(splits, st.split_row_positions(y)):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(np.unique(y.iloc[a])), 2)


class PersistenceTests(unittest.TestCase):
    def test_bundle_roundtrip_covers_raw_imputation_and_unseen_category(self):
        raw = make_synthetic_stroke(120)
        prep = st.StrokePreprocessor().fit(raw.iloc[:90])
        tree = st.build_tree_by_cost(prep.transform(raw.iloc[:90]),
                                     st.extract_target(raw.iloc[:90]), min_leaf=5)
        heldout = raw.iloc[90:].copy()
        heldout["work_type"] = "unseen"
        with tempfile.TemporaryDirectory() as tmp:
            path = st.save_model_bundle(tree, prep, Path(tmp) / "model.json")
            bundle = st.load_model_bundle(path)
            actual = st.predict_raw(bundle, heldout)
            expected = st.predict_many(tree, prep.transform(heldout))
            np.testing.assert_array_equal(actual["p1"], expected)
            self.assertEqual(bundle["format_version"], 2)

    def test_nan_does_not_silently_route_right(self):
        tree = {"is_leaf": False, "feature": "x", "thr": 1.,
                "left": {"is_leaf": True, "proba": [1., 0.]},
                "right": {"is_leaf": True, "proba": [0., 1.]}}
        with self.assertRaises(ValueError):
            st.predict_one(tree, {"x": np.nan})
        with self.assertRaises(KeyError):
            st.predict_one(tree, {"wrong": 2.})


class EndToEndTests(unittest.TestCase):
    def test_full_synthetic_run_and_fold_local_preprocessing(self):
        raw = make_synthetic_stroke(160)
        y = st.extract_target(raw)
        train, val, test = st.split_row_positions(y)
        fitted_ids = []
        original = st.StrokePreprocessor.fit
        def recording_fit(self, frame):
            fitted_ids.append(set(frame["id"].tolist()))
            return original(self, frame)
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "synthetic.csv"
            raw.to_csv(csv, index=False)
            output = Path(tmp) / "output"
            with patch.object(st.StrokePreprocessor, "fit", recording_fit), contextlib.redirect_stdout(io.StringIO()):
                result = st.run_complete_analysis(
                    csv, output, run_robustness=True, n_repeats=2,
                    bootstrap_B=20, show_plots=False
                )
            # Selection, final refit, and each inner fold/refit never fit test data.
            self.assertEqual(fitted_ids[0], set(train))
            self.assertEqual(fitted_ids[1], set(train) | set(val))
            self.assertEqual(len(fitted_ids), 6)
            for ids in fitted_ids:
                self.assertFalse(ids & set(test))
            self.assertLess(len(fitted_ids[2]), len(fitted_ids[3]))
            self.assertEqual(result["reload_max_delta"], 0)
            for split in ["train", "val", "test"]:
                self.assertEqual(list(result[split][0].columns), result["preprocessor"].feature_names)
                st.predict_many(result["final_tree"], result[split][0])
            self.assertEqual(len(result["robustness"]["repeated_results"]), 2)
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertFalse(manifest["historical_results_reproduced"])
            self.assertFalse(manifest["test_used_for_selection"])
            self.assertEqual(len(manifest["dataset_sha256"]), 64)
            self.assertEqual(manifest["split_row_positions"]["test"], test.tolist())
            for name in ["model_bundle.json", "test_model_summary.csv",
                         "bootstrap_test_results.csv", "mycosttree_test_confusion.png"]:
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()

