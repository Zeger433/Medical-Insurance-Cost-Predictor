"""
predict.py — Prediction helper functions for Medical Insurance Cost Prediction.

Provides: load_artifacts, preprocess_input, predict_charges, get_cost_category
"""

import os
import pickle

import numpy as np


# ─── Artifact Loading ────────────────────────────────────────────────────────

def load_artifacts() -> tuple:
    """
    Load model.pkl, scaler.pkl, and features.pkl from disk.

    Returns
    -------
    (model, scaler, features) — fitted sklearn objects and feature name list.

    Raises
    ------
    FileNotFoundError if any artifact is missing with a clear message.
    """
    artifacts = {
        "model"    : "model.pkl",
        "scaler"   : "scaler.pkl",
        "features" : "features.pkl",
    }
    loaded = {}

    for key, filename in artifacts.items():
        path = os.path.join(filename)   # cross-platform path
        try:
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)  # deserialise from disk
        except FileNotFoundError:
            raise FileNotFoundError(
                f"[predict.py] Missing artifact: '{filename}'. "
                f"Please run 'python train.py' first to generate all .pkl files."
            )
        except Exception as e:
            raise RuntimeError(
                f"[predict.py] Failed to load '{filename}': {e}"
            )

    return loaded["model"], loaded["scaler"], loaded["features"]


# ─── Input Preprocessing ─────────────────────────────────────────────────────

def preprocess_input(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> np.ndarray:
    """
    Convert raw user inputs into a feature array matching the training schema.

    Encoding mirrors train.py exactly:
      sex    : male=1, female=0  (label encode)
      smoker : yes=1,  no=0      (label encode)
      region : one-hot into region_northwest, region_southeast,
               region_southwest — region_northeast is the dropped baseline
               (all three flags = 0 when region == 'northeast')

    Parameters
    ----------
    age      : int   — age of the beneficiary
    sex      : str   — 'male' or 'female'
    bmi      : float — Body Mass Index
    children : int   — number of dependents
    smoker   : str   — 'yes' or 'no'
    region   : str   — one of northeast / northwest / southeast / southwest

    Returns
    -------
    np.ndarray of shape (1, n_features) ready for scaling and prediction.
    """
    # ── Label Encode sex and smoker ──────────────────────────────────────────
    sex_enc    = 1 if str(sex).strip().lower()    == "male" else 0
    smoker_enc = 1 if str(smoker).strip().lower() == "yes"  else 0

    # ── One-Hot Encode region (drop_first baseline = northeast) ─────────────
    region_clean       = str(region).strip().lower()
    region_northwest   = 1 if region_clean == "northwest"  else 0
    region_southeast   = 1 if region_clean == "southeast"  else 0
    region_southwest   = 1 if region_clean == "southwest"  else 0
    # northeast maps to all three flags = 0 (baseline / dropped category)

    # ── Assemble in the exact column order produced by train.py ─────────────
    # Order: age, sex, bmi, children, smoker,
    #        region_northwest, region_southeast, region_southwest
    feature_vector = np.array([[
        int(age),
        sex_enc,
        float(bmi),
        int(children),
        smoker_enc,
        region_northwest,
        region_southeast,
        region_southwest,
    ]], dtype=float)

    return feature_vector


# ─── Main Prediction Function ─────────────────────────────────────────────────

def predict_charges(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> float:
    """
    End-to-end prediction: load artifacts → preprocess → scale → predict.

    Applies the same log1p transform reversal (np.expm1) that was used
    in train.py so the output is in original dollar units.

    Parameters
    ----------
    age, sex, bmi, children, smoker, region — raw user-supplied values.

    Returns
    -------
    float — predicted annual insurance charges in US dollars.
    """
    model, scaler, features = load_artifacts()  # load saved artifacts

    raw_input = preprocess_input(age, sex, bmi, children, smoker, region)

    # Validate column count against saved features list (fail-fast guard)
    if raw_input.shape[1] != len(features):
        raise ValueError(
            f"Feature mismatch: expected {len(features)} features, "
            f"got {raw_input.shape[1]}. Re-run train.py to regenerate artifacts."
        )

    scaled_input   = scaler.transform(raw_input)       # apply saved scaler
    log_prediction = model.predict(scaled_input)        # predict in log space
    charges        = float(np.expm1(log_prediction[0])) # reverse log1p

    return charges


# ─── Cost Category Helper ─────────────────────────────────────────────────────

def get_cost_category(charges: float) -> dict:
    """
    Map predicted charge amount to a human-readable cost category.

    Parameters
    ----------
    charges : float — predicted annual insurance cost in dollars.

    Returns
    -------
    dict with keys:
      category : str  — 'Low Cost', 'Moderate Cost', or 'High Cost'
      color    : str  — 'green', 'orange', or 'red'
      advice   : str  — personalised health advice string
    """
    if charges < 8_000:
        return {
            "category": "Low Cost",
            "color"   : "green",
            "advice"  : (
                "Your predicted insurance cost is low. "
                "Maintain a healthy lifestyle!"
            ),
        }
    elif charges <= 16_000:
        return {
            "category": "Moderate Cost",
            "color"   : "orange",
            "advice"  : (
                "Moderate insurance cost predicted. "
                "Consider reviewing your health habits."
            ),
        }
    else:
        return {
            "category": "High Cost",
            "color"   : "red",
            "advice"  : (
                "High insurance cost predicted. "
                "Consult a health advisor for cost reduction strategies."
            ),
        }
