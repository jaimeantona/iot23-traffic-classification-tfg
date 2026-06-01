"""Train and compare classification models on the five-class IoT-23 dataset.

The script evaluates Random Forest, LightGBM and a Multilayer Perceptron (MLP)
using the experimental procedure described in the TFG:
    - stratified train/validation/test split;
    - cross-validation on training data;
    - final evaluation on the held-out test set;
    - optional hyperparameter optimisation of the MLP with Optuna.

Expected dataset format:
    A semicolon-separated CSV containing a ``label`` column and numerical
    feature columns.

Example:
    python train_models_5class.py \
        --dataset data/dataset_balanced_100k_per_class_withproto_fixed_norm.csv \
        --output-dir results/5class
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_DATASET = Path("data/dataset_balanced_100k_per_class_withproto_fixed_norm.csv")
DEFAULT_OUTPUT_DIR = Path("results/5class")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to the input dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=20,
        help="Number of Optuna trials for MLP tuning (default: 20).",
    )
    return parser.parse_args()


def load_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and labels from a semicolon-separated CSV file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, sep=";")
    if "label" not in df.columns:
        raise ValueError("The input dataset must contain a 'label' column.")

    x = df.drop(columns=["label"])
    y = df["label"].astype(str)
    return x, y


def split_data(
    x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train, validation and test partitions (70/10/20)."""
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.125,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def save_importances(importances: pd.Series, output_path: Path) -> None:
    """Store the 15 most important features of a tree-based model."""
    importances.sort_values(ascending=False).head(15).to_csv(output_path)


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate the Random Forest model."""
    print("\n=== Random Forest ===")

    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1,
    )

    print(f"[RF] CV Accuracy: {scores['test_accuracy'].mean():.6f} +/- {scores['test_accuracy'].std():.6f}")
    print(f"[RF] CV F1-macro: {scores['test_f1_macro'].mean():.6f} +/- {scores['test_f1_macro'].std():.6f}")

    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(x_test)
    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"[RF] Training time: {train_time:.6f} s")
    print(f"[RF] Prediction time: {prediction_time:.6f} s")
    print(f"[RF] TEST Accuracy: {accuracy:.6f}")
    print(f"[RF] TEST F1-macro: {macro_f1:.6f}")
    print("\n[RF] TEST classification report:")
    print(classification_report(y_test, predictions))
    print("\n[RF] TEST confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    save_importances(importances, output_dir / "feature_importance_rf_5class.csv")

    return {
        "model": "RandomForest",
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        "test_accuracy": accuracy,
        "test_f1_macro": macro_f1,
        "train_time_s": train_time,
        "pred_time_s": prediction_time,
    }


def run_lightgbm_cv(x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Run the LightGBM cross-validation experiment."""
    print("\n=== LightGBM cross-validation ===")

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=-1,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        x,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=1,
    )

    print(f"[LGBM] CV Accuracy: {scores['test_accuracy'].mean():.6f} +/- {scores['test_accuracy'].std():.6f}")
    print(f"[LGBM] CV F1-macro: {scores['test_f1_macro'].mean():.6f} +/- {scores['test_f1_macro'].std():.6f}")

    return {
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
    }


def train_lightgbm(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate the LightGBM model."""
    print("\n=== LightGBM ===")

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=-1,
    )

    start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="multi_logloss",
    )
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(x_test)
    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"[LGBM] Training time: {train_time:.6f} s")
    print(f"[LGBM] Prediction time: {prediction_time:.6f} s")
    print(f"[LGBM] TEST Accuracy: {accuracy:.6f}")
    print(f"[LGBM] TEST F1-macro: {macro_f1:.6f}")
    print("\n[LGBM] TEST classification report:")
    print(classification_report(y_test, predictions))
    print("\n[LGBM] TEST confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    save_importances(importances, output_dir / "feature_importance_lgbm_5class.csv")

    return {
        "model": "LightGBM",
        "cv_accuracy_mean": np.nan,
        "cv_accuracy_std": np.nan,
        "cv_f1_macro_mean": np.nan,
        "cv_f1_macro_std": np.nan,
        "test_accuracy": accuracy,
        "test_f1_macro": macro_f1,
        "train_time_s": train_time,
        "pred_time_s": prediction_time,
    }


def tune_mlp_optuna(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    output_dir: Path,
    n_trials: int,
) -> dict[str, Any]:
    """Tune MLP hyperparameters by maximising validation macro F1-score."""
    print("\n=== Optuna tuning for MLP ===")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    def objective(trial: optuna.Trial) -> float:
        hidden_1 = trial.suggest_int("hidden1", 32, 256, step=32)
        use_second = trial.suggest_categorical("use_second", [True, False])
        hidden_layers = (
            (hidden_1, trial.suggest_int("hidden2", 16, 128, step=16))
            if use_second
            else (hidden_1,)
        )

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
            max_iter=trial.suggest_categorical("max_iter", [50, 100, 150, 200]),
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        model.fit(x_train_scaled, y_train_encoded)
        predictions = model.predict(x_val_scaled)
        return f1_score(y_val_encoded, predictions, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"[Optuna] Best macro F1: {study.best_value:.6f}")
    print(f"[Optuna] Best parameters: {study.best_params}")

    study.trials_dataframe().to_csv(output_dir / "optuna_mlp_trials.csv", index=False)
    pd.DataFrame(
        [{"best_value_macro_f1": study.best_value, **study.best_params}]
    ).to_csv(output_dir / "optuna_mlp_best_params.csv", index=False)

    return study.best_params


def train_mlp(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    best_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and evaluate the MLP model."""
    print("\n=== Multilayer Perceptron ===")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    if best_params is None:
        hidden_layers = (128, 64)
        alpha = 0.0001
        learning_rate_init = 0.001
        batch_size = 200
        max_iter = 100
    else:
        hidden_layers = (
            (best_params["hidden1"], best_params["hidden2"])
            if best_params["use_second"]
            else (best_params["hidden1"],)
        )
        alpha = best_params["alpha"]
        learning_rate_init = best_params["learning_rate_init"]
        batch_size = best_params["batch_size"]
        max_iter = best_params["max_iter"]

    def build_model() -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_accuracy: list[float] = []
    cv_f1_macro: list[float] = []

    for fold, (train_idx, validation_idx) in enumerate(cv.split(x_train, y_train), start=1):
        x_fold_train = x_train.iloc[train_idx]
        x_fold_val = x_train.iloc[validation_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[validation_idx]

        fold_scaler = StandardScaler()
        x_fold_train_scaled = fold_scaler.fit_transform(x_fold_train)
        x_fold_val_scaled = fold_scaler.transform(x_fold_val)

        fold_encoder = LabelEncoder()
        y_fold_train_encoded = fold_encoder.fit_transform(y_fold_train)
        y_fold_val_encoded = fold_encoder.transform(y_fold_val)

        fold_model = build_model()
        fold_model.fit(x_fold_train_scaled, y_fold_train_encoded)
        fold_predictions = fold_model.predict(x_fold_val_scaled)

        accuracy = accuracy_score(y_fold_val_encoded, fold_predictions)
        macro_f1 = f1_score(y_fold_val_encoded, fold_predictions, average="macro")
        cv_accuracy.append(accuracy)
        cv_f1_macro.append(macro_f1)
        print(f"[MLP-CV] Fold {fold}: Accuracy={accuracy:.6f}, F1-macro={macro_f1:.6f}")

    print(f"[MLP] CV Accuracy: {np.mean(cv_accuracy):.6f} +/- {np.std(cv_accuracy):.6f}")
    print(f"[MLP] CV F1-macro: {np.mean(cv_f1_macro):.6f} +/- {np.std(cv_f1_macro):.6f}")

    model = build_model()
    start = time.perf_counter()
    model.fit(x_train_scaled, y_train_encoded)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions_encoded = model.predict(x_test_scaled)
    prediction_time = time.perf_counter() - start
    predictions = encoder.inverse_transform(predictions_encoded)

    accuracy = accuracy_score(y_test_encoded, predictions_encoded)
    macro_f1 = f1_score(y_test_encoded, predictions_encoded, average="macro")

    print(f"[MLP] Hidden layers: {model.hidden_layer_sizes}")
    print(f"[MLP] Maximum epochs: {model.max_iter}")
    print(f"[MLP] Epochs used: {model.n_iter_}")
    print(f"[MLP] Final loss: {model.loss_:.6f}")
    print(f"[MLP] Training time: {train_time:.6f} s")
    print(f"[MLP] Prediction time: {prediction_time:.6f} s")
    print(f"[MLP] TEST Accuracy: {accuracy:.6f}")
    print(f"[MLP] TEST F1-macro: {macro_f1:.6f}")
    print("\n[MLP] TEST classification report:")
    print(classification_report(y_test, predictions))
    print("\n[MLP] TEST confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    loss_curve = pd.DataFrame(
        {"epoch": range(1, len(model.loss_curve_) + 1), "loss": model.loss_curve_}
    )
    loss_curve.to_csv(output_dir / "mlp_loss_curve.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_curve["epoch"], loss_curve["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mlp_loss_curve.png", dpi=300)
    plt.close()

    return {
        "model": "MLP",
        "cv_accuracy_mean": np.mean(cv_accuracy),
        "cv_accuracy_std": np.std(cv_accuracy),
        "cv_f1_macro_mean": np.mean(cv_f1_macro),
        "cv_f1_macro_std": np.std(cv_f1_macro),
        "test_accuracy": accuracy,
        "test_f1_macro": macro_f1,
        "train_time_s": train_time,
        "pred_time_s": prediction_time,
        "mlp_hidden_layers": str(model.hidden_layer_sizes),
        "mlp_max_iter": model.max_iter,
        "mlp_n_iter_used": model.n_iter_,
        "mlp_final_loss": model.loss_,
    }


def main() -> None:
    """Execute the complete five-class experiment."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    x, y = load_data(args.dataset)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

    print(f"Train: {x_train.shape} Val: {x_val.shape} Test: {x_test.shape}")
    print("\nTrain class distribution:\n", y_train.value_counts())
    print("\nTest class distribution:\n", y_test.value_counts())

    results: list[dict[str, Any]] = []
    results.append(train_random_forest(x_train, y_train, x_test, y_test, args.output_dir))

    lightgbm_cv = run_lightgbm_cv(
        pd.concat([x_train, x_val]),
        pd.concat([y_train, y_val]),
    )
    lightgbm_results = train_lightgbm(
        x_train, x_val, y_train, y_val, x_test, y_test, args.output_dir
    )
    lightgbm_results.update(lightgbm_cv)
    results.append(lightgbm_results)

    best_params = tune_mlp_optuna(
        x_train, x_val, y_train, y_val, args.output_dir, args.optuna_trials
    )
    results.append(
        train_mlp(x_train, y_train, x_test, y_test, args.output_dir, best_params)
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "model_comparison_5class.csv", index=False)

    print("\n=== FINAL MODEL COMPARISON ===")
    print(
        results_df[
            [
                "model",
                "cv_accuracy_mean",
                "cv_f1_macro_mean",
                "test_accuracy",
                "test_f1_macro",
                "train_time_s",
                "pred_time_s",
            ]
        ]
    )


if __name__ == "__main__":
    main()
