"""Train and compare classification models on the binary IoT-23 dataset.

The script evaluates Random Forest, LightGBM and a Multilayer Perceptron (MLP)
on the binary classification task: benign traffic versus attack traffic.

Expected dataset format:
    A semicolon-separated CSV containing a ``label`` column and numerical
    feature columns.

Example:
    python training/train_models_binary.py \
        --dataset data/dataset_binary_balanced_100k_per_class_withproto_fixed_norm.csv \
        --output-dir results/binary
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_DATASET = Path("data/dataset_binary_balanced_100k_per_class_withproto_fixed_norm.csv")
DEFAULT_OUTPUT_DIR = Path("results/binary")
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
    return parser.parse_args()


def load_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and labels from a semicolon-separated CSV file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path, sep=";")
    if "label" not in dataframe.columns:
        raise ValueError("The input dataset must contain a 'label' column.")

    features = dataframe.drop(columns=["label"])
    labels = dataframe["label"].astype(str)
    return features, labels


def split_data(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train, validation and test partitions (70/10/20)."""
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
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
    """Save the 15 most important features of a tree-based model."""
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
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=cross_validation,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1,
    )

    print(f"[RF] CV Accuracy: {scores['test_accuracy'].mean():.6f} +/- {scores['test_accuracy'].std():.6f}")
    print(f"[RF] CV F1-macro: {scores['test_f1_macro'].mean():.6f} +/- {scores['test_f1_macro'].std():.6f}")

    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(x_test)
    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"[RF] Training time: {training_time:.6f} s")
    print(f"[RF] Prediction time: {prediction_time:.6f} s")
    print(f"[RF] TEST Accuracy: {accuracy:.6f}")
    print(f"[RF] TEST F1-macro: {macro_f1:.6f}")
    print("\n[RF] TEST classification report:")
    print(classification_report(y_test, predictions))
    print("\n[RF] TEST confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    save_importances(importances, output_dir / "feature_importance_rf_binary.csv")

    return {
        "model": "Random Forest",
        "dataset": "binary",
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        "test_accuracy": accuracy,
        "test_f1_macro": macro_f1,
        "train_time_s": training_time,
        "pred_time_s": prediction_time,
    }


def cross_validate_lightgbm(features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """Run cross-validation for the LightGBM configuration."""
    print("\n=== LightGBM cross-validation ===")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=-1,
    )
    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        model,
        features,
        labels,
        cv=cross_validation,
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
        objective="binary",
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
        eval_metric="binary_logloss",
    )
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(x_test)
    prediction_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"[LGBM] Training time: {training_time:.6f} s")
    print(f"[LGBM] Prediction time: {prediction_time:.6f} s")
    print(f"[LGBM] TEST Accuracy: {accuracy:.6f}")
    print(f"[LGBM] TEST F1-macro: {macro_f1:.6f}")
    print("\n[LGBM] TEST classification report:")
    print(classification_report(y_test, predictions))
    print("\n[LGBM] TEST confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    save_importances(importances, output_dir / "feature_importance_lgbm_binary.csv")

    return {
        "model": "LightGBM",
        "dataset": "binary",
        "test_accuracy": accuracy,
        "test_f1_macro": macro_f1,
        "train_time_s": training_time,
        "pred_time_s": prediction_time,
    }


def build_mlp() -> MLPClassifier:
    """Return the MLP configuration used in the experiment."""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        batch_size=200,
        max_iter=100,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )


def train_mlp(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate the Multilayer Perceptron model."""
    print("\n=== Multilayer Perceptron ===")

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    validation_accuracies: list[float] = []
    validation_f1_scores: list[float] = []

    for fold, (train_indices, val_indices) in enumerate(
        cross_validation.split(x_train, y_train),
        start=1,
    ):
        fold_scaler = StandardScaler()
        x_fold_train = fold_scaler.fit_transform(x_train.iloc[train_indices])
        x_fold_val = fold_scaler.transform(x_train.iloc[val_indices])

        fold_encoder = LabelEncoder()
        y_fold_train = fold_encoder.fit_transform(y_train.iloc[train_indices])
        y_fold_val = fold_encoder.transform(y_train.iloc[val_indices])

        model = build_mlp()
        model.fit(x_fold_train, y_fold_train)
        fold_predictions = model.predict(x_fold_val)

        fold_accuracy = accuracy_score(y_fold_val, fold_predictions)
        fold_f1 = f1_score(y_fold_val, fold_predictions, average="macro")
        validation_accuracies.append(fold_accuracy)
        validation_f1_scores.append(fold_f1)
        print(f"[MLP-CV] Fold {fold}: Accuracy={fold_accuracy:.6f}, F1-macro={fold_f1:.6f}")

    print(f"[MLP] CV Accuracy: {np.mean(validation_accuracies):.6f} +/- {np.std(validation_accuracies):.6f}")
    print(f"[MLP] CV F1-macro: {np.mean(validation_f1_scores):.6f} +/- {np.std(validation_f1_scores):.6f}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    model = build_mlp()
    start = time.perf_counter()
    model.fit(x_train_scaled, y_train_encoded)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions_encoded = model.predict(x_test_scaled)
    prediction_time = time.perf_counter() - start
    predictions = encoder.inverse_transform(predictions_encoded)

    accuracy = accuracy_score(y_test_encoded, predictions_encoded)
    macro_f1 = f1_score(y_test_encoded, predictions_encoded, average="macro")

    print(f"[MLP] Training time: {training_time:.6f} s")
    print(f"[MLP] Prediction time: {prediction_time:.6f} s")
    print(f"[MLP] TEST Accuracy: {accuracy:.6f}")
    print(f"[MLP] TEST F1-macro: {macro_f1:.6f}")
    print("\n[MLP] TEST classification report:")
    print(classification_report(y_test, predictions))
    print("\n[MLP] TEST confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    loss_data = pd.DataFrame({
        "epoch": range(1, len(model.loss_curve_) + 1),
        "loss": model.loss_curve_,
    })
    loss_data.to_csv(output_dir / "mlp_loss_curve_binary.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("MLP Training Loss Curve - Binary Classification")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "mlp_loss_curve_binary.png", dpi=300)
    plt.close(figure)

    return {
        "model": "MLP",
        "dataset": "binary",
        "cv_accuracy_mean": np.mean(validation_accuracies),
        "cv_accuracy_std": np.std(validation_accuracies),
        "cv_f1_macro_mean": np.mean(validation_f1_scores),
        "cv_f1_macro_std": np.std(validation_f1_scores),
        "test_accuracy": accuracy,
        "test_f1_macro": macro_f1,
        "train_time_s": training_time,
        "pred_time_s": prediction_time,
        "mlp_hidden_layers": str(model.hidden_layer_sizes),
        "mlp_max_iter": model.max_iter,
        "mlp_n_iter_used": model.n_iter_,
        "mlp_final_loss": model.loss_,
    }


def main() -> None:
    """Run the complete binary classification experiment."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    features, labels = load_data(args.dataset)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(features, labels)

    print(f"Train: {x_train.shape} | Validation: {x_val.shape} | Test: {x_test.shape}")
    print("\nTrain class distribution:")
    print(y_train.value_counts())
    print("\nValidation class distribution:")
    print(y_val.value_counts())
    print("\nTest class distribution:")
    print(y_test.value_counts())

    results: list[dict[str, Any]] = []

    results.append(train_random_forest(x_train, y_train, x_test, y_test, args.output_dir))

    lgbm_cv_results = cross_validate_lightgbm(
        pd.concat([x_train, x_val]),
        pd.concat([y_train, y_val]),
    )
    lgbm_results = train_lightgbm(
        x_train,
        x_val,
        y_train,
        y_val,
        x_test,
        y_test,
        args.output_dir,
    )
    lgbm_results.update(lgbm_cv_results)
    results.append(lgbm_results)

    results.append(train_mlp(x_train, y_train, x_test, y_test, args.output_dir))

    comparison = pd.DataFrame(results)
    comparison.to_csv(args.output_dir / "model_comparison_binary.csv", index=False)

    columns = [
        "model",
        "cv_accuracy_mean",
        "cv_f1_macro_mean",
        "test_accuracy",
        "test_f1_macro",
        "train_time_s",
        "pred_time_s",
    ]
    print("\n=== FINAL MODEL COMPARISON - BINARY CLASSIFICATION ===")
    print(comparison[columns])


if __name__ == "__main__":
    main()
