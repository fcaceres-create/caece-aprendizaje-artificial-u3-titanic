"""Evaluación: hold-out estratificado + CV estratificada sobre el resto (mismo esquema que el notebook)."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import (ParameterGrid, StratifiedKFold, cross_validate, train_test_split,
                                     validation_curve)

from .features import armar_pipeline
from .models import CATALOGO, crear_modelo, prefijo_param

UMBRAL_BRECHA = 0.05


def dividir(X, y, holdout=0.2, seed=42):
    """Devuelve X_tr, X_ho, y_tr, y_ho (hold-out estratificado)."""
    return train_test_split(X, y, test_size=holdout, stratify=y, random_state=seed)


def esquema_cv(folds=10, seed=42):
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def pipeline_desde_config(cfg):
    """cfg = {"modelo", "params", "features": {...}, "eval": {"seed", ...}}"""
    modelo = crear_modelo(cfg["modelo"], cfg["params"], cfg["eval"]["seed"])
    return armar_pipeline(modelo, CATALOGO[cfg["modelo"]]["escalar"], **cfg["features"])


def base_mujeres(X):
    """Regla 'todas las mujeres sobreviven'."""
    return (X.Sex == "female").astype(int).to_numpy()


def evaluar(cfg, X, y):
    """Corre CV sobre el entrenamiento, entrena con todo el entrenamiento y mide en el hold-out.

    Devuelve (pipeline_entrenado, resultados).
    """
    ev = cfg["eval"]
    X_tr, X_ho, y_tr, y_ho = dividir(X, y, ev["holdout"], ev["seed"])
    pipe = pipeline_desde_config(cfg)
    r = cross_validate(pipe, X_tr, y_tr, cv=esquema_cv(ev["folds"], ev["seed"]),
                       scoring=["accuracy", "roc_auc", "f1"], return_train_score=True, n_jobs=-1)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_ho)
    proba = pipe.predict_proba(X_ho)[:, 1]
    fpr, tpr, _ = roc_curve(y_ho, proba)
    res = {
        "acc_train": r["train_accuracy"].mean(),
        "acc_cv": r["test_accuracy"].mean(),
        "acc_cv_std": r["test_accuracy"].std(),
        "auc_cv": r["test_roc_auc"].mean(),
        "f1_cv": r["test_f1"].mean(),
        "acc_ho": accuracy_score(y_ho, pred),
        "auc_ho": roc_auc_score(y_ho, proba),
        "f1_ho": f1_score(y_ho, pred),
        "base_ho": accuracy_score(y_ho, base_mujeres(X_ho)),
        "matriz": confusion_matrix(y_ho, pred),
        "roc": (fpr, tpr),
        "n_tr": len(X_tr), "n_ho": len(X_ho),
    }
    res["brecha"] = res["acc_train"] - res["acc_cv"]
    return pipe, res


def curva_validacion(cfg, X, y, param, valores):
    """Accuracy train vs CV al variar un hiperparámetro (solo sobre el entrenamiento)."""
    ev = cfg["eval"]
    X_tr, _, y_tr, _ = dividir(X, y, ev["holdout"], ev["seed"])
    tr, te = validation_curve(pipeline_desde_config(cfg), X_tr, y_tr, param_name=prefijo_param(cfg["modelo"]) + param,
                              param_range=valores, cv=esquema_cv(ev["folds"], ev["seed"]),
                              scoring="accuracy", n_jobs=-1)
    return pd.DataFrame({"valor": [str(v) for v in valores],
                         "train": tr.mean(1), "cv": te.mean(1), "cv_std": te.std(1)})


def busqueda_grilla(cfg, X, y, grilla):
    """Generador: recorre la grilla y va devolviendo (i, total, fila) para poder mostrar progreso."""
    ev = cfg["eval"]
    X_tr, _, y_tr, _ = dividir(X, y, ev["holdout"], ev["seed"])
    cv = esquema_cv(ev["folds"], ev["seed"])
    combinaciones = list(ParameterGrid(grilla))
    for i, comb in enumerate(combinaciones, 1):
        c = {**cfg, "params": {**cfg["params"], **comb}}
        r = cross_validate(pipeline_desde_config(c), X_tr, y_tr, cv=cv, scoring="accuracy",
                           return_train_score=True, n_jobs=-1)
        yield i, len(combinaciones), {**comb, "Acc train": r["train_score"].mean(),
                                      "Acc CV": r["test_score"].mean(), "± desv": r["test_score"].std()}


def importancia_permutacion(pipe, X_ho, y_ho, n_repeats=10, seed=42):
    """Sobre columnas ORIGINALES (se pasa el pipeline completo), igual que el notebook."""
    imp = permutation_importance(pipe, X_ho, y_ho, n_repeats=n_repeats, random_state=seed,
                                 scoring="accuracy", n_jobs=-1)
    return (pd.DataFrame({"media": imp.importances_mean, "desv": imp.importances_std}, index=X_ho.columns)
            .sort_values("media", ascending=False))


def validar_submission(sub, test):
    """Lista de problemas del archivo de envío (vacía = formato correcto)."""
    errores = []
    if list(sub.columns) != ["PassengerId", "Survived"]:
        errores.append(f"Columnas deben ser PassengerId,Survived (son {list(sub.columns)}).")
    if len(sub) != 418:
        errores.append(f"Debe tener 418 filas (tiene {len(sub)}).")
    if "PassengerId" in sub and not np.array_equal(sub.PassengerId.to_numpy(), test.PassengerId.to_numpy()):
        errores.append("Los PassengerId no coinciden (en valor u orden) con test.csv.")
    if "Survived" in sub:
        if sub.Survived.isna().any():
            errores.append("Survived tiene valores nulos.")
        elif not pd.api.types.is_integer_dtype(sub.Survived) or not set(sub.Survived.unique()) <= {0, 1}:
            errores.append("Survived debe ser entero 0/1.")
    return errores
