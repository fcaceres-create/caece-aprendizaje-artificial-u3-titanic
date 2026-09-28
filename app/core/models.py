"""Catálogo de modelos, sus hiperparámetros editables y las grillas chicas de búsqueda.

Cada hiperparámetro se describe con un dict:
  tipo "int"    → slider entero (min, max, paso)
  tipo "prof"   → slider entero donde 0 significa "sin límite" (None)
  tipo "opcion" → select_slider / selectbox con `opciones`
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

VALORES_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
LEARNING_RATES = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
MAX_FEATURES = ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]

CATALOGO = {
    "Regresión logística": {
        "escalar": True,
        "params": {
            "C": {"tipo": "opcion", "opciones": VALORES_C, "default": 1.0,
                  "ayuda": "Inversa de la regularización: C chico = modelo más simple."},
            "penalty": {"tipo": "opcion", "opciones": ["l2", "l1"], "default": "l2",
                        "ayuda": "l2 (Ridge) achica todos los coeficientes; l1 (Lasso) puede anular algunos."},
        },
    },
    "KNN": {
        "escalar": True,
        "params": {
            "n_neighbors": {"tipo": "int", "min": 1, "max": 50, "paso": 1, "default": 7,
                            "ayuda": "Cantidad de vecinos (k). k chico sobreajusta, k grande subajusta."},
            "weights": {"tipo": "opcion", "opciones": ["uniform", "distance"], "default": "uniform",
                        "ayuda": "uniform: todos los vecinos pesan igual; distance: los más cercanos pesan más."},
        },
    },
    "SVM": {
        "escalar": True,
        "params": {
            "C": {"tipo": "opcion", "opciones": VALORES_C, "default": 1.0,
                  "ayuda": "Penalización de errores: C grande = frontera más ajustada."},
            "kernel": {"tipo": "opcion", "opciones": ["rbf", "linear", "poly"], "default": "rbf",
                       "ayuda": "Forma de la frontera de decisión."},
            "gamma": {"tipo": "opcion", "opciones": ["scale", "auto", 0.001, 0.01, 0.1, 1.0], "default": "scale",
                      "ayuda": "Alcance de cada punto (rbf/poly): gamma grande = frontera más irregular."},
        },
    },
    "Árbol de decisión": {
        "escalar": False,
        "params": {
            "max_depth": {"tipo": "prof", "min": 0, "max": 20, "paso": 1, "default": 0,
                          "ayuda": "Profundidad máxima (0 = sin límite)."},
            "min_samples_leaf": {"tipo": "int", "min": 1, "max": 50, "paso": 1, "default": 1,
                                 "ayuda": "Mínimo de pasajeros en cada hoja."},
            "criterion": {"tipo": "opcion", "opciones": ["gini", "entropy", "log_loss"], "default": "gini",
                          "ayuda": "Medida de impureza para elegir los cortes."},
        },
    },
    "Random Forest": {
        "escalar": False,
        "params": {
            "n_estimators": {"tipo": "int", "min": 10, "max": 500, "paso": 10, "default": 300,
                             "ayuda": "Cantidad de árboles."},
            "max_depth": {"tipo": "prof", "min": 0, "max": 20, "paso": 1, "default": 4,
                          "ayuda": "Profundidad máxima de cada árbol (0 = sin límite)."},
            "max_features": {"tipo": "opcion", "opciones": MAX_FEATURES, "default": 0.5,
                             "ayuda": "Variables candidatas en cada corte (fracción o regla)."},
            "min_samples_leaf": {"tipo": "int", "min": 1, "max": 30, "paso": 1, "default": 3,
                                 "ayuda": "Mínimo de pasajeros en cada hoja."},
        },
    },
    "Gradient Boosting": {
        "escalar": False,
        "params": {
            "learning_rate": {"tipo": "opcion", "opciones": LEARNING_RATES, "default": 0.1,
                              "ayuda": "Cuánto corrige cada árbol nuevo."},
            "n_estimators": {"tipo": "int", "min": 10, "max": 500, "paso": 10, "default": 100,
                             "ayuda": "Cantidad de árboles (etapas)."},
            "max_depth": {"tipo": "int", "min": 1, "max": 8, "paso": 1, "default": 3,
                          "ayuda": "Profundidad de cada árbol."},
            "subsample": {"tipo": "opcion", "opciones": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "default": 1.0,
                          "ayuda": "Fracción de filas usada por cada árbol (<1 = boosting estocástico)."},
        },
    },
}

# Valores a recorrer en la curva de validación (por modelo y parámetro)
CURVAS = {
    "Regresión logística": {"C": VALORES_C},
    "KNN": {"n_neighbors": [1, 3, 5, 7, 9, 13, 17, 21, 31, 41, 50]},
    "SVM": {"C": VALORES_C[2:], "gamma": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "Árbol de decisión": {"max_depth": list(range(1, 16)), "min_samples_leaf": [1, 2, 3, 5, 8, 12, 20, 30, 50]},
    "Random Forest": {"max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15], "n_estimators": [10, 25, 50, 100, 200, 300],
                      "min_samples_leaf": [1, 2, 3, 5, 8, 12, 20, 30]},
    "Gradient Boosting": {"n_estimators": [10, 25, 50, 100, 200, 300], "max_depth": [1, 2, 3, 4, 5, 6],
                          "learning_rate": LEARNING_RATES},
}

# Grillas chicas (≤ 24 combinaciones) para la búsqueda
GRILLAS = {
    "Regresión logística": {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], "penalty": ["l2", "l1"]},
    "KNN": {"n_neighbors": [3, 5, 7, 9, 13, 17, 21, 31], "weights": ["uniform", "distance"]},
    "SVM": {"C": [0.1, 0.3, 1.0, 3.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
    "Árbol de decisión": {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 3, 5, 10]},
    "Random Forest": {"max_depth": [4, 6, 8, 0], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5]},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1],
                          "max_depth": [2, 3, 4], "subsample": [0.8, 1.0]},
}


def prefijo_param(nombre):
    """Prefijo para llegar al hiperparámetro dentro del Pipeline (el SVM va envuelto en el calibrador)."""
    return "model__estimator__" if nombre == "SVM" else "model__"


def params_por_defecto(nombre):
    return {p: d["default"] for p, d in CATALOGO[nombre]["params"].items()}


def crear_modelo(nombre, params, seed=42):
    """Instancia el estimador de scikit-learn a partir de los valores de la interfaz."""
    p = {**params_por_defecto(nombre), **params}
    if nombre == "Regresión logística":
        return LogisticRegression(C=p["C"], l1_ratio=1.0 if p["penalty"] == "l1" else 0.0,
                                  solver="liblinear", max_iter=2000)
    if nombre == "KNN":
        return KNeighborsClassifier(n_neighbors=int(p["n_neighbors"]), weights=p["weights"])
    if nombre == "SVM":
        # Probabilidades calibradas (reemplazo recomendado de SVC(probability=True))
        return CalibratedClassifierCV(SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], random_state=seed),
                                      ensemble=False)
    if nombre == "Árbol de decisión":
        return DecisionTreeClassifier(max_depth=int(p["max_depth"]) or None,
                                      min_samples_leaf=int(p["min_samples_leaf"]),
                                      criterion=p["criterion"], random_state=seed)
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]) or None,
                                      max_features=p["max_features"], min_samples_leaf=int(p["min_samples_leaf"]),
                                      random_state=seed, n_jobs=-1)
    if nombre == "Gradient Boosting":
        return GradientBoostingClassifier(learning_rate=p["learning_rate"], n_estimators=int(p["n_estimators"]),
                                          max_depth=int(p["max_depth"]), subsample=p["subsample"],
                                          random_state=seed)
    raise ValueError(f"Modelo desconocido: {nombre}")
