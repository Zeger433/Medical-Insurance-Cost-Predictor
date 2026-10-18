"""
train.py — Full ML pipeline for Medical Insurance Cost Prediction.

Executes: Data Loading → Cleaning → Preprocessing → Training → Evaluation → Saving
Run standalone: python train.py
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ─── Directory Setup ────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join("assets", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)  # auto-create plots directory


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str = "insurance.csv") -> pd.DataFrame:
    """Load insurance CSV and print basic diagnostics."""
    print("\n" + "═" * 60)
    print("  STEP 1 — DATA LOADING")
    print("═" * 60)

    df = pd.read_csv(filepath)  # read the dataset from disk

    print(f"\nShape       : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nData Types  :\n{df.dtypes}")
    print(f"\nFirst 5 Rows:\n{df.head()}")
    print(f"\nBasic Statistics:\n{df.describe()}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check missing values, duplicates, and outliers.
    Save correlation heatmap and pairplot to assets/plots/.
    Returns cleaned DataFrame (duplicates removed, outliers retained).
    """
    print("\n" + "═" * 60)
    print("  STEP 2 — DATA CLEANING")
    print("═" * 60)

    # ── Missing Values ──────────────────────────────────────────────────────
    missing = df.isnull().sum()
    print(f"\nMissing Values per Column:\n{missing}")

    # ── Duplicate Rows ──────────────────────────────────────────────────────
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Row Count: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()  # remove duplicates if present
        print(f"  → {dup_count} duplicate(s) removed. New shape: {df.shape}")

    # ── Outlier Detection via IQR ───────────────────────────────────────────
    for col in ["bmi", "charges"]:
        Q1 = df[col].quantile(0.25)   # first quartile
        Q3 = df[col].quantile(0.75)   # third quartile
        IQR = Q3 - Q1                 # interquartile range
        lower = Q1 - 1.5 * IQR       # lower fence
        upper = Q3 + 1.5 * IQR       # upper fence
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"\nOutliers in '{col}': {len(outliers)} rows "
              f"(kept — genuine medical variation)")

    # ── Correlation Heatmap ─────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include=[np.number])  # only numeric cols
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",   # coolwarm palette as specified
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Heatmap — Numeric Features", fontsize=13, pad=12)
    plt.tight_layout()
    heatmap_path = os.path.join(PLOTS_DIR, "heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"\n  ✓ Heatmap saved → {heatmap_path}")

    # ── Pairplot ────────────────────────────────────────────────────────────
    # Encode smoker to numeric so pairplot can colour by it
    df_plot = df.copy()
    df_plot["smoker_enc"] = df_plot["smoker"].map({"no": 0, "yes": 1})
    pair_cols = ["age", "bmi", "charges", "smoker_enc"]
    pp = sns.pairplot(
        df_plot[pair_cols],
        hue="smoker_enc",
        palette={0: "steelblue", 1: "tomato"},
        plot_kws={"alpha": 0.5},
    )
    pp.fig.suptitle("Pairplot — Key Features (smoker encoded)", y=1.02)
    plt.tight_layout()
    pairplot_path = os.path.join(PLOTS_DIR, "pairplot.png")
    pp.savefig(pairplot_path, dpi=150)
    plt.close()
    print(f"  ✓ Pairplot saved → {pairplot_path}")

    # ── Cleaning Summary ────────────────────────────────────────────────────
    print("\n─── Cleaning Summary ───────────────────────────────────────────")
    print(f"  Missing values  : {missing.sum()} total")
    print(f"  Duplicates removed : {dup_count}")
    print(f"  Outliers        : retained (genuine medical variation)")
    print(f"  Final shape     : {df.shape}")
    print("────────────────────────────────────────────────────────────────")

    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, apply log1p to target, split 80/20, scale features.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_columns
    """
    print("\n" + "═" * 60)
    print("  STEP 3 — PREPROCESSING")
    print("═" * 60)

    df = df.copy()

    # ── Label Encoding ──────────────────────────────────────────────────────
    df["sex"]    = df["sex"].map({"female": 0, "male": 1})     # female=0, male=1
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})       # no=0, yes=1

    # ── One-Hot Encoding for region (drop_first removes region_northeast) ───
    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    print(f"\nEncoded DataFrame shape : {df.shape}")
    print(f"Columns : {list(df.columns)}")

    # ── Features / Target Split ─────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != "charges"]
    X = df[feature_cols]
    y_raw = df["charges"]

    # ── Log1p Transform on Target ───────────────────────────────────────────
    y = np.log1p(y_raw)  # normalise the right-skewed distribution

    # ── Train / Test Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Standard Scaling ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit only on train
    X_test_sc  = scaler.transform(X_test)        # transform test with same params

    print(f"\nTrain shape : {X_train_sc.shape}")
    print(f"Test  shape : {X_test_sc.shape}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, list(feature_cols)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_models(X_train, y_train) -> dict:
    """Train all 6 regression models and return fitted model dict."""
    print("\n" + "═" * 60)
    print("  STEP 4 — MODEL TRAINING")
    print("═" * 60)

    models = {
        "LinearRegression"        : LinearRegression(),
        "Ridge"                   : Ridge(alpha=1.0),
        "Lasso"                   : Lasso(alpha=0.1),
        "DecisionTree"            : DecisionTreeRegressor(random_state=42),
        "RandomForest"            : RandomForestRegressor(
                                        n_estimators=100, random_state=42),
        "GradientBoosting"        : GradientBoostingRegressor(
                                        n_estimators=200,
                                        learning_rate=0.1,
                                        max_depth=4,
                                        random_state=42),
    }

    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)   # fit each model on scaled training data
        fitted[name] = model
        print(f"  ✓ {name} trained")

    return fitted


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_models(fitted_models: dict, X_test, y_test) -> tuple:
    """
    Evaluate all models on original dollar scale (after inverse log transform).
    Prints comparison table, saves plots, returns (results_dict, best_model_name).
    """
    print("\n" + "═" * 60)
    print("  STEP 5 — EVALUATION")
    print("═" * 60)

    results = {}
    actual = np.expm1(y_test)  # convert log-scale actual values back to dollars

    for name, model in fitted_models.items():
        preds_log = model.predict(X_test)         # predict in log scale
        preds     = np.expm1(preds_log)           # invert log transform
        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae  = mean_absolute_error(actual, preds)
        r2   = r2_score(actual, preds)
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2,
                         "preds": preds, "preds_log": preds_log}

    # ── Comparison Table ────────────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  Model                   RMSE ($)    MAE ($)    R² Score    ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    for name, m in results.items():
        print(f"  ║  {name:<24s}  {m['RMSE']:>8,.0f}  {m['MAE']:>9,.0f}  "
              f"{m['R2']:>9.4f}    ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    # ── Best Model by R² ────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["R2"])
    best      = results[best_name]
    print(f"\n  🏆 Best Model: {best_name} | "
          f"R²: {best['R2']:.4f} | MAE: ${best['MAE']:,.2f}")

    # ── Plot a: Actual vs Predicted ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, best["preds"], alpha=0.5, color="steelblue",
               edgecolors="none", label="Predictions")
    lims = [min(actual.min(), best["preds"].min()),
            max(actual.max(), best["preds"].max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")  # diagonal
    ax.set_xlabel("Actual Charges ($)", fontsize=12)
    ax.set_ylabel("Predicted Charges ($)", fontsize=12)
    ax.set_title("Actual vs Predicted Insurance Charges", fontsize=13)
    ax.legend()
    plt.tight_layout()
    avp_path = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
    plt.savefig(avp_path, dpi=150)
    plt.close()
    print(f"\n  ✓ Plot saved → {avp_path}")

    # ── Plot b: Feature Importance (Random Forest) ──────────────────────────
    rf_model     = fitted_models["RandomForest"]
    importances  = rf_model.feature_importances_  # built-in RF attribute
    # feature names are loaded from caller scope via closure — avoid hardcoding
    feature_names = getattr(rf_model, "_feature_names_", None)
    # fallback: we pass feature names from outside when calling
    # (handled by saving to module-level variable below)
    _plot_feature_importance(importances, _FEATURE_COLS)

    # ── Plot c: Residuals ───────────────────────────────────────────────────
    residuals = actual.values - best["preds"]  # actual minus predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(best["preds"], residuals, alpha=0.5, color="steelblue",
               edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)  # zero line
    ax.set_xlabel("Predicted Values ($)", fontsize=12)
    ax.set_ylabel("Residuals ($)", fontsize=12)
    ax.set_title("Residuals Plot", fontsize=13)
    plt.tight_layout()
    res_path = os.path.join(PLOTS_DIR, "residuals.png")
    plt.savefig(res_path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved → {res_path}")

    return results, best_name


def _plot_feature_importance(importances: np.ndarray, feature_names: list):
    """Save horizontal bar chart of feature importances from Random Forest."""
    sorted_idx   = np.argsort(importances)           # ascending order
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sorted_names, sorted_vals, color="steelblue")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Feature Importance — Random Forest", fontsize=13)
    plt.tight_layout()
    fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved → {fi_path}")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVING ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, scaler: StandardScaler, feature_cols: list):
    """Pickle the best model, scaler, and feature column list to disk."""
    print("\n" + "═" * 60)
    print("  STEP 6 — SAVING ARTIFACTS")
    print("═" * 60)

    artifacts = {
        os.path.join("model.pkl")    : model,
        os.path.join("scaler.pkl")   : scaler,
        os.path.join("features.pkl") : feature_cols,
    }

    for path, obj in artifacts.items():
        try:
            with open(path, "wb") as f:
                pickle.dump(obj, f)  # serialise to disk
            print(f"  ✓ Saved {path}")
        except Exception as e:
            print(f"  ✗ Failed to save {path}: {e}")
            raise

    print("\n  ✅ All artifacts saved: model.pkl | scaler.pkl | features.pkl")


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

_FEATURE_COLS: list = []  # module-level placeholder populated during run

if __name__ == "__main__":
    # Step 1 — load
    df = load_data("insurance.csv")

    # Step 2 — clean
    df = clean_data(df)

    # Step 3 — preprocess
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df)
    _FEATURE_COLS = feature_cols   # make available to evaluation helpers

    # Step 4 — train
    fitted_models = train_models(X_train, y_train)

    # Step 5 — evaluate
    results, best_name = evaluate_models(fitted_models, X_test, y_test)

    # Step 6 — save
    best_model = fitted_models[best_name]
    save_artifacts(best_model, scaler, feature_cols)

    print("\n" + "═" * 60)
    print("  Pipeline complete.")
    print("═" * 60 + "\n")
