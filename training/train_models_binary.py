import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


DATASET = "pipeline_hist/out/dataset_binary_balanced_100k_per_class_withproto_fixed_norm.csv"


def load_data():
    df = pd.read_csv(DATASET, sep=";")
    X = df.drop(columns=["label"])
    y = df["label"].astype(str)
    return X, y


def split_data(X, y):
    # 80/20 test. Después sacamos un 10% del total como validación.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.125,  # 0.125 de 0.8 = 0.1 del total
        stratify=y_train_full,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n=== Random Forest ===")

    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        rf,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1
    )

    print(f"\n[RF] CV Accuracy: {scores['test_accuracy'].mean():.6f} +/- {scores['test_accuracy'].std():.6f}")
    print(f"[RF] CV F1-macro:  {scores['test_f1_macro'].mean():.6f} +/- {scores['test_f1_macro'].std():.6f}")

    start_train = time.perf_counter()
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    preds = rf.predict(X_test)
    pred_time = time.perf_counter() - start_pred

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"\n[RF] Training time: {train_time:.6f} s")
    print(f"[RF] Prediction time: {pred_time:.6f} s")
    print(f"[RF] TEST Accuracy: {acc:.6f}")
    print(f"[RF] TEST F1-macro: {macro_f1:.6f}")

    print("\n[RF] TEST classification report:")
    print(classification_report(y_test, preds))

    print("\n[RF] TEST confusion matrix:")
    print(confusion_matrix(y_test, preds))

    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n[RF] Top 15 Feature Importances:")
    print(importances.head(15))

    importances.head(15).to_csv("feature_importance_rf_binary.csv")

    return {
        "model": "RandomForest",
        "dataset": "binary",
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        "test_accuracy": acc,
        "test_f1_macro": macro_f1,
        "train_time_s": train_time,
        "pred_time_s": pred_time
    }


def lightgbm_cv_light(X, y):
    print("\n=== LightGBM CV ===")

    lgb = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbosity=-1
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scores = cross_validate(
        lgb,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=1
    )

    print(f"[LGBM] CV Accuracy: {scores['test_accuracy'].mean():.6f} +/- {scores['test_accuracy'].std():.6f}")
    print(f"[LGBM] CV F1-macro:  {scores['test_f1_macro'].mean():.6f} +/- {scores['test_f1_macro'].std():.6f}")

    return {
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std()
    }


def train_lightgbm_fast(X_train, X_val, y_train, y_val, X_test, y_test):
    print("\n=== LightGBM ===")

    lgb = LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbosity=-1
    )

    start_train = time.perf_counter()
    lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss"
    )
    train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    preds = lgb.predict(X_test)
    pred_time = time.perf_counter() - start_pred

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"\n[LGBM] Training time: {train_time:.6f} s")
    print(f"[LGBM] Prediction time: {pred_time:.6f} s")
    print(f"[LGBM] TEST Accuracy: {acc:.6f}")
    print(f"[LGBM] TEST F1-macro: {macro_f1:.6f}")

    print("\n[LGBM] TEST classification report:")
    print(classification_report(y_test, preds))

    print("\n[LGBM] TEST confusion matrix:")
    print(confusion_matrix(y_test, preds))

    importances = pd.Series(lgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n[LGBM] Top 15 Feature Importances:")
    print(importances.head(15))

    importances.head(15).to_csv("feature_importance_lgbm_binary.csv")

    return {
        "model": "LightGBM",
        "dataset": "binary",
        "test_accuracy": acc,
        "test_f1_macro": macro_f1,
        "train_time_s": train_time,
        "pred_time_s": pred_time
    }


def train_mlp(X_train, y_train, X_test, y_test):
    print("\n=== MLP ===")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        batch_size=200,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
        Xtr = X_train.iloc[tr_idx]
        Xva = X_train.iloc[va_idx]
        ytr = y_train.iloc[tr_idx]
        yva = y_train.iloc[va_idx]

        scaler_f = StandardScaler()
        Xtr_scaled = scaler_f.fit_transform(Xtr)
        Xva_scaled = scaler_f.transform(Xva)

        le_f = LabelEncoder()
        ytr_enc = le_f.fit_transform(ytr)
        yva_enc = le_f.transform(yva)

        mlp_cv = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            batch_size=200,
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )

        mlp_cv.fit(Xtr_scaled, ytr_enc)
        preds_cv = mlp_cv.predict(Xva_scaled)

        acc_cv = accuracy_score(yva_enc, preds_cv)
        f1_cv = f1_score(yva_enc, preds_cv, average="macro")

        cv_accs.append(acc_cv)
        cv_f1s.append(f1_cv)

        print(f"[MLP-CV] Fold {fold}: Accuracy={acc_cv:.6f}, F1-macro={f1_cv:.6f}")

    print(f"\n[MLP] CV Accuracy: {np.mean(cv_accs):.6f} +/- {np.std(cv_accs):.6f}")
    print(f"[MLP] CV F1-macro: {np.mean(cv_f1s):.6f} +/- {np.std(cv_f1s):.6f}")

    start_train = time.perf_counter()
    mlp.fit(X_train_scaled, y_train_enc)
    train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    preds_enc = mlp.predict(X_test_scaled)
    pred_time = time.perf_counter() - start_pred

    acc = accuracy_score(y_test_enc, preds_enc)
    macro_f1 = f1_score(y_test_enc, preds_enc, average="macro")

    preds = le.inverse_transform(preds_enc)

    print(f"\n[MLP] Training time: {train_time:.6f} s")
    print(f"[MLP] Prediction time: {pred_time:.6f} s")
    print(f"[MLP] TEST Accuracy: {acc:.6f}")
    print(f"[MLP] TEST F1-macro: {macro_f1:.6f}")

    print("\n[MLP] TEST classification report:")
    print(classification_report(y_test, preds))

    print("\n[MLP] TEST confusion matrix:")
    print(confusion_matrix(y_test, preds))

    loss_df = pd.DataFrame({
        "epoch": range(1, len(mlp.loss_curve_) + 1),
        "loss": mlp.loss_curve_
    })
    loss_df.to_csv("mlp_loss_curve_binary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve - Binary")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mlp_loss_curve_binary.png", dpi=300)
    plt.close()

    return {
        "model": "MLP",
        "dataset": "binary",
        "cv_accuracy_mean": np.mean(cv_accs),
        "cv_accuracy_std": np.std(cv_accs),
        "cv_f1_macro_mean": np.mean(cv_f1s),
        "cv_f1_macro_std": np.std(cv_f1s),
        "test_accuracy": acc,
        "test_f1_macro": macro_f1,
        "train_time_s": train_time,
        "pred_time_s": pred_time,
        "mlp_hidden_layers": str(mlp.hidden_layer_sizes),
        "mlp_max_iter": mlp.max_iter,
        "mlp_n_iter_used": mlp.n_iter_,
        "mlp_final_loss": mlp.loss_
    }


def main():
    print("Loading binary dataset...")
    X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    print("\nTrain class distribution:")
    print(y_train.value_counts())

    print("\nVal class distribution:")
    print(y_val.value_counts())

    print("\nTest class distribution:")
    print(y_test.value_counts())

    all_results = []

    rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    all_results.append(rf_results)

    lgbm_cv_results = lightgbm_cv_light(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val])
    )

    lgbm_results = train_lightgbm_fast(X_train, X_val, y_train, y_val, X_test, y_test)

    lgbm_results["cv_accuracy_mean"] = lgbm_cv_results["cv_accuracy_mean"]
    lgbm_results["cv_accuracy_std"] = lgbm_cv_results["cv_accuracy_std"]
    lgbm_results["cv_f1_macro_mean"] = lgbm_cv_results["cv_f1_macro_mean"]
    lgbm_results["cv_f1_macro_std"] = lgbm_cv_results["cv_f1_macro_std"]

    all_results.append(lgbm_results)

    mlp_results = train_mlp(X_train, y_train, X_test, y_test)
    all_results.append(mlp_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("model_comparison_binary.csv", index=False)

    print("\n=== FINAL MODEL COMPARISON - BINARY ===")
    print(results_df[[
        "model",
        "cv_accuracy_mean",
        "cv_f1_macro_mean",
        "test_accuracy",
        "test_f1_macro",
        "train_time_s",
        "pred_time_s"
    ]])


if __name__ == "__main__":
    main()