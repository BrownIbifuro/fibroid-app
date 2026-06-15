"""Load and clean the fibroid dataset from Excel."""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "TABULAR FIBROID DATA.xlsx"


def load_raw_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load raw Excel data and drop empty rows."""
    df = pd.read_excel(path)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def parse_blood_pressure(bp_series: pd.Series) -> pd.DataFrame:
    """Split 'BLOOD PRESSURE' column (e.g. '120/80') into systolic and diastolic."""
    split = bp_series.str.split("/", expand=True).astype(float)
    split.columns = ["systolic_bp", "diastolic_bp"]
    return split


def encode_symptoms(symptom_series: pd.Series) -> pd.DataFrame:
    """Create binary columns for each symptom keyword found in the SYMPTOMS column."""
    symptom_keywords = [
        "BLEEDING",
        "HEAVY PERIOD",
        "LOWER ABDOMINAL PAIN",
        "ABDOMINAL PAIN",
        "PAINFUL PERIOD",
        "FREQUENT URINATION",
        "CONSTIPATION",
        "PAIN DURING SEX",
        "BODY WEAKNESS",
    ]
    symptom_df = pd.DataFrame(index=symptom_series.index)
    upper = symptom_series.str.upper().fillna("NONE")
    for kw in symptom_keywords:
        col_name = "symptom_" + kw.lower().replace(" ", "_")
        symptom_df[col_name] = upper.str.contains(kw, na=False).astype(int)
    symptom_df["has_any_symptom"] = (upper != "NONE").astype(int)
    return symptom_df


def encode_target(target_series: pd.Series) -> pd.Series:
    """Encode target variable: 1 = has fibroid, 0 = no fibroid."""
    upper = target_series.str.upper().str.strip()
    return upper.apply(lambda x: 1 if "HAS" in str(x) else 0).rename("fibroid")


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target vector y from raw data."""
    bp = parse_blood_pressure(df["BLOOD PRESSURE"])
    symptoms = encode_symptoms(df["SYMPTOMS"])
    target = encode_target(df["FIBROID OCCURRENCE"])

    X = pd.concat(
        [
            df[["AGE", "HEIGHT", "WEIGHT"]].rename(
                columns={"AGE": "age", "HEIGHT": "height", "WEIGHT": "weight"}
            ),
            bp,
            symptoms,
        ],
        axis=1,
    )
    # Compute BMI
    X["bmi"] = X["weight"] / (X["height"] ** 2)

    return X, target


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to oversample the minority class. Only use on training data."""
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def _smote_oversample(
    X_class: pd.DataFrame,
    n_synthetic: int,
    k_neighbors: int = 5,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic samples via SMOTE-like KNN interpolation (numpy only)."""
    if rng is None:
        rng = np.random.default_rng(42)

    data = X_class.values
    n_samples = len(data)
    k = min(k_neighbors, n_samples - 1)

    # Compute pairwise distances for KNN
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(data)
    neighbors = nn.kneighbors(data, return_distance=False)[:, 1:]  # exclude self

    synthetic = np.empty((n_synthetic, data.shape[1]))
    for i in range(n_synthetic):
        idx = rng.integers(0, n_samples)
        neighbor_idx = neighbors[idx, rng.integers(0, k)]
        lam = rng.uniform(0, 1)
        synthetic[i] = data[idx] + lam * (data[neighbor_idx] - data[idx])

    return pd.DataFrame(synthetic, columns=X_class.columns)


def _bootstrap_with_noise(
    X_class: pd.DataFrame,
    n_synthetic: int,
    continuous_cols: list[str],
    noise_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Bootstrap resample with Gaussian noise on continuous features."""
    if rng is None:
        rng = np.random.default_rng(42)

    sampled_idx = rng.choice(X_class.index, size=n_synthetic, replace=True)
    X_syn = X_class.loc[sampled_idx].copy().reset_index(drop=True)

    for col in continuous_cols:
        if col in X_syn.columns:
            col_std = X_class[col].std()
            noise = rng.normal(0, noise_std * col_std, size=len(X_syn))
            X_syn[col] = X_syn[col] + noise

    return X_syn


def augment_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    target_per_class: int = 100,
    noise_std: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Augment dataset to target_per_class samples per class (balanced).

    Minority class: SMOTE-like KNN interpolation.
    Majority class: bootstrap resampling with Gaussian noise on continuous features.
    """
    rng = np.random.default_rng(random_state)

    continuous_cols = ["age", "height", "weight", "systolic_bp", "diastolic_bp"]
    binary_cols = [c for c in X.columns if c.startswith("symptom_") or c == "has_any_symptom"]

    clip_ranges = {
        "age": (15, 80),
        "height": (1.2, 2.1),
        "weight": (30, 200),
        "systolic_bp": (80, 200),
        "diastolic_bp": (50, 130),
    }

    minority_label = y.value_counts().idxmin()
    majority_label = y.value_counts().idxmax()
    n_minority = (y == minority_label).sum()
    n_majority = (y == majority_label).sum()

    parts_X = []
    parts_y = []

    # Keep all original data
    parts_X.append(X)
    parts_y.append(y)

    # --- Oversample minority class via SMOTE-like interpolation ---
    n_syn_min = target_per_class - n_minority
    if n_syn_min > 0:
        X_min = X[y == minority_label]
        X_syn_min = _smote_oversample(X_min, n_syn_min, k_neighbors=5, rng=rng)
        parts_X.append(X_syn_min)
        parts_y.append(pd.Series([minority_label] * n_syn_min, name=y.name))

    # --- Oversample majority class via bootstrap + noise ---
    n_syn_maj = target_per_class - n_majority
    if n_syn_maj > 0:
        X_maj = X[y == majority_label]
        X_syn_maj = _bootstrap_with_noise(X_maj, n_syn_maj, continuous_cols, noise_std, rng)
        parts_X.append(X_syn_maj)
        parts_y.append(pd.Series([majority_label] * n_syn_maj, name=y.name))

    X_aug = pd.concat(parts_X, ignore_index=True)
    y_aug = pd.concat(parts_y, ignore_index=True)

    # --- Post-processing ---
    for col, (lo, hi) in clip_ranges.items():
        if col in X_aug.columns:
            X_aug[col] = X_aug[col].clip(lo, hi)

    for col in binary_cols:
        if col in X_aug.columns:
            X_aug[col] = np.clip(np.round(X_aug[col]), 0, 1).astype(int)

    symptom_only = [c for c in binary_cols if c.startswith("symptom_")]
    if symptom_only and "has_any_symptom" in X_aug.columns:
        X_aug["has_any_symptom"] = (X_aug[symptom_only].sum(axis=1) > 0).astype(int)

    if "bmi" in X_aug.columns:
        X_aug["bmi"] = X_aug["weight"] / (X_aug["height"] ** 2)

    return X_aug, y_aug


def save_augmented_dataset(
    X: pd.DataFrame, y: pd.Series, path: str | Path | None = None
) -> Path:
    """Save augmented dataset to CSV."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "augmented_fibroid_data.csv"
    path = Path(path)
    df = pd.concat([X, y], axis=1)
    df.to_csv(path, index=False)
    print(f"Augmented dataset saved to {path} ({len(df)} rows)")
    return path


def load_augmented_dataset(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load augmented dataset from CSV."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "augmented_fibroid_data.csv"
    df = pd.read_csv(path)
    y = df["fibroid"]
    X = df.drop(columns=["fibroid"])
    return X, y


def load_dataset(path: str | Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Full pipeline: load, clean, feature-engineer."""
    df = load_raw_data(path)
    return build_features(df)
