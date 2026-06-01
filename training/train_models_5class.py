import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATASET = "pipeline_hist/out/dataset_balanced_100k_per_class_withproto_fixed_norm.csv"


def load_data():
    df = pd.read_csv(DATASET, sep=";")
    X = df.drop(columns=["label"])
    y = df["label"].astype(str)
    return X, y


def split_data(X, y):
    # 80/20 test + sacamos un 10% del train como validación para early stopping
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.125, stratify=y_train_full, random_state=42
        # 0.125 de 0.8 = 0.1 total -> 10% val
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

    importances.head(15).to_csv("feature_importance_rf_5class.csv")

    return {
        "model": "RandomForest",
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        "test_accuracy": acc,
        "test_f1_macro": macro_f1,
        "train_time_s": train_time,
        "pred_time_s": pred_time
    }


def train_lightgbm_fast(X_train, X_val, y_train, y_val, X_test, y_test):
    print("\n=== LightGBM (fast + early stopping) ===")

    lgb = LGBMClassifier(
        objective="multiclass",
        n_estimators=2000,          # alto, pero early stopping cortará
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,                   # importante en tu RAM: no uses -1
        verbosity=-1
    )

    start_train = time.perf_counter()
    lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[],
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

    # LightGBM sklearn API: early stopping vía fit_params cambia según versión.
    # Si tu versión soporta early_stopping_rounds, úsalo así:
    # lgb.fit(..., eval_set=[(X_val,y_val)], eval_metric="multi_logloss", early_stopping_rounds=50, verbose=False)


    print("\n[LGBM] TEST classification report:")
    print(classification_report(y_test, preds))

    print("\n[LGBM] TEST confusion matrix:")
    print(confusion_matrix(y_test, preds))

    importances = pd.Series(lgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\n[LGBM] Top 15 Feature Importances:")
    print(importances.head(15))

    importances.head(15).to_csv("feature_importance_lgbm_5class.csv")

    return {
        "model": "LightGBM",
        "cv_accuracy_mean": np.nan,
        "cv_accuracy_std": np.nan,
        "cv_f1_macro_mean": np.nan,
        "cv_f1_macro_std": np.nan,
        "test_accuracy": acc,
        "test_f1_macro": macro_f1,
        "train_time_s": train_time,
        "pred_time_s": pred_time
    }


def lightgbm_cv_light(X, y):
    # CV ligero para tener número comparable sin tardar media vida
    print("\n=== LightGBM (light CV, 3-fold, fewer trees) ===")

    lgb = LGBMClassifier(
        objective="multiclass",
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
        lgb, X, y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=1  # evita que joblib + lgbm se pisen
    )

    print(f"[LGBM-lightCV] CV Accuracy: {scores['test_accuracy'].mean():.6f} +/- {scores['test_accuracy'].std():.6f}")
    print(f"[LGBM-lightCV] CV F1-macro:  {scores['test_f1_macro'].mean():.6f} +/- {scores['test_f1_macro'].std():.6f}")

    return {
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std()
    }   

def tune_mlp_optuna(X_train, X_val, y_train, y_val, n_trials=20):
    print("\n=== Optuna tuning for MLP ===")

    # Escalado fijo train/val
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Labels a enteros
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    def objective(trial):
        hidden1 = trial.suggest_int("hidden1", 32, 256, step=32)
        use_second = trial.suggest_categorical("use_second", [True, False])

        if use_second:
            hidden2 = trial.suggest_int("hidden2", 16, 128, step=16)
            hidden_layers = (hidden1, hidden2)
        else:
            hidden_layers = (hidden1,)

        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        max_iter = trial.suggest_categorical("max_iter", [50, 100, 150, 200])

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )

        mlp.fit(X_train_scaled, y_train_enc)
        preds_val = mlp.predict(X_val_scaled)

        f1 = f1_score(y_val_enc, preds_val, average="macro")
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n[Optuna] Best trial:")
    print(" Best macro F1:", study.best_value)
    print(" Best params:", study.best_params)

    # Guardar resultados
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_mlp_trials.csv", index=False)

    best_df = pd.DataFrame([{
        "best_value_macro_f1": study.best_value,
        **study.best_params
    }])
    best_df.to_csv("optuna_mlp_best_params.csv", index=False)

    return study.best_params


def train_mlp(X_train, X_val, y_train, y_val, X_test, y_test, best_params=None):
    print("\n=== MLP ===")

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Codificar labels a enteros
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    # Si no hay parámetros de Optuna, usar configuración base
    if best_params is None:
        hidden_layers = (128, 64)
        alpha = 0.0001
        learning_rate_init = 0.001
        batch_size = 200
        max_iter = 100
    else:
        if best_params["use_second"]:
            hidden_layers = (best_params["hidden1"], best_params["hidden2"])
        else:
            hidden_layers = (best_params["hidden1"],)

        alpha = best_params["alpha"]
        learning_rate_init = best_params["learning_rate_init"]
        batch_size = best_params["batch_size"]
        max_iter = best_params["max_iter"]

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )

    # CV manual porque hay que escalar y codificar dentro de cada fold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
        Xtr_f = X_train.iloc[tr_idx]
        Xva_f = X_train.iloc[va_idx]
        ytr_f = y_train.iloc[tr_idx]
        yva_f = y_train.iloc[va_idx]

        scaler_f = StandardScaler()
        Xtr_f_scaled = scaler_f.fit_transform(Xtr_f)
        Xva_f_scaled = scaler_f.transform(Xva_f)

        le_f = LabelEncoder()
        ytr_f_enc = le_f.fit_transform(ytr_f)
        yva_f_enc = le_f.transform(yva_f)
        
        mlp_cv = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )

        mlp_cv.fit(Xtr_f_scaled, ytr_f_enc)
        preds_cv_enc = mlp_cv.predict(Xva_f_scaled)

        acc_cv = accuracy_score(yva_f_enc, preds_cv_enc)
        f1_cv = f1_score(yva_f_enc, preds_cv_enc, average="macro")

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

    print(f"\n[MLP] Hidden layers: {mlp.hidden_layer_sizes}")
    print(f"[MLP] Max epochs (max_iter): {mlp.max_iter}")
    print(f"[MLP] Actual epochs used (n_iter_): {mlp.n_iter_}")
    print(f"[MLP] Final loss: {mlp.loss_:.6f}")

    print(f"\n[MLP] Training time: {train_time:.6f} s")
    print(f"[MLP] Prediction time: {pred_time:.6f} s")
    print(f"[MLP] TEST Accuracy: {acc:.6f}")
    print(f"[MLP] TEST F1-macro: {macro_f1:.6f}")

    print("\n[MLP] TEST classification report:")
    print(classification_report(y_test, preds))

    print("\n[MLP] TEST confusion matrix:")
    print(confusion_matrix(y_test, preds))
    
    # Guardar curva de loss
    loss_df = pd.DataFrame({
        "epoch": range(1, len(mlp.loss_curve_) + 1),
        "loss": mlp.loss_curve_
    })
    loss_df.to_csv("mlp_loss_curve.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mlp_loss_curve.png", dpi=300)
    plt.close()

    return {
        "model": "MLP",
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
    print("Loading dataset...")
    X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("\nTrain class distribution:\n", y_train.value_counts())
    print("\nTest class distribution:\n", y_test.value_counts())

    all_results = []

    rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    all_results.append(rf_results)

    # CV ligero de LGBM para poder comparar también en CV
    lgbm_cv_results = lightgbm_cv_light(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    lgbm_results = train_lightgbm_fast(X_train, X_val, y_train, y_val, X_test, y_test)
    lgbm_results["cv_accuracy_mean"] = lgbm_cv_results["cv_accuracy_mean"]
    lgbm_results["cv_accuracy_std"] = lgbm_cv_results["cv_accuracy_std"]
    lgbm_results["cv_f1_macro_mean"] = lgbm_cv_results["cv_f1_macro_mean"]
    lgbm_results["cv_f1_macro_std"] = lgbm_cv_results["cv_f1_macro_std"]
    all_results.append(lgbm_results)

    best_params = tune_mlp_optuna(X_train, X_val, y_train, y_val, n_trials=20)

    mlp_results = train_mlp(
        X_train, X_val, y_train, y_val, X_test, y_test,
        best_params=best_params
    )
    all_results.append(mlp_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("model_comparison_5class.csv", index=False)

    print("\n=== FINAL MODEL COMPARISON ===")
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