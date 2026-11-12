"""Artificial data for software checks, not a health dataset or evidence."""
import numpy as np
import pandas as pd


def make_synthetic_stroke(n=240, seed=7):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 85, n)
    y = ((age > 60) & (rng.random(n) > 0.45)).astype(int)
    df = pd.DataFrame({
        "id": np.arange(n), "age": age,
        "avg_glucose_level": rng.integers(70, 200, n).astype(float),
        "bmi": rng.integers(19, 40, n).astype(float),
        "hypertension": rng.integers(0, 2, n),
        "heart_disease": rng.integers(0, 2, n),
        "gender": rng.choice(["Female", "Male"], n),
        "ever_married": rng.choice(["Yes", "No"], n),
        "work_type": rng.choice(["Private", "Self-employed"], n),
        "Residence_type": rng.choice(["Urban", "Rural"], n),
        "smoking_status": rng.choice(["never smoked", "formerly smoked", "Unknown"], n),
        "stroke": y,
    })
    df.loc[np.arange(0, n, 9), "bmi"] = np.nan
    return df

