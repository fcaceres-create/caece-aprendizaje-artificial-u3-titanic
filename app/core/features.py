"""Ingeniería de variables del notebook, parametrizable.

Con los valores por defecto, `TitanicFeatures` produce exactamente lo mismo que la versión de
`titanic.ipynb` (lo verifica tests/test_features.py).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TITULOS_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
TITULOS_OK = {"Mr", "Mrs", "Miss", "Master"}

NUM = ["AgeImp", "LogFare", "FamilySize", "TicketGroup", "SibSp", "Parch",
       "HasCabin", "AgeMissing", "IsChild"]
CAT = ["Sex", "Pclass", "Title", "Embarked", "FamilyGroup", "Deck"]
VARIABLES = NUM + CAT

ORIGINALES = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
DERIVADAS = ["Title", "AgeImp", "AgeMissing", "IsChild", "FamilySize", "FamilyGroup",
             "TicketGroup", "HasCabin", "Deck", "LogFare"]

DESCRIPCIONES = {
    "Pclass": "Clase del pasaje (1ra, 2da, 3ra)",
    "Sex": "Sexo",
    "SibSp": "Hermanos/cónyuge a bordo",
    "Parch": "Padres/hijos a bordo",
    "Embarked": "Puerto de embarque (moda si falta)",
    "Title": "Título extraído del nombre (Mr, Mrs, Miss, Master, Rare)",
    "AgeImp": "Edad con faltantes imputados",
    "AgeMissing": "1 si la edad original faltaba",
    "IsChild": "1 si la edad imputada es menor al umbral de 'niño'",
    "FamilySize": "SibSp + Parch + 1",
    "FamilyGroup": "Solo / Chica / Grande según FamilySize",
    "TicketGroup": "Cuántos pasajeros comparten el ticket (tope 5)",
    "HasCabin": "1 si tiene cabina registrada",
    "Deck": "Cubierta: primera letra de la cabina (U = desconocida)",
    "LogFare": "log(1 + tarifa), faltantes → mediana de su clase",
}

IMPUTACIONES = {
    "titulo": "Mediana por título",
    "mediana": "Mediana global",
    "media": "Media global",
    "knn": "KNNImputer (k=5)",
}
_KNN_COLS = ["Age", "Pclass", "SibSp", "Parch", "LogFare", "SexNum"]


class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Crea las variables derivadas. fit() aprende sólo del conjunto que recibe."""

    def __init__(self, edad_nino=13, cortes_familia=(1, 4), imputacion_edad="titulo", variables=None):
        self.edad_nino = edad_nino
        self.cortes_familia = cortes_familia
        self.imputacion_edad = imputacion_edad
        self.variables = variables

    def fit(self, X, y=None):
        if self.imputacion_edad not in IMPUTACIONES:
            raise ValueError(f"Estrategia de imputación desconocida: {self.imputacion_edad}")
        X = self._basicas(X)
        self.age_med_title_ = X.groupby("Title").Age.median()
        self.age_med_ = X.Age.median()
        self.age_mean_ = X.Age.mean()
        self.fare_med_class_ = X.groupby("Pclass").Fare.median()
        self.embarked_mode_ = X.Embarked.mode()[0]
        self.ticket_counts_ = X.Ticket.value_counts()
        if self.imputacion_edad == "knn":
            M = self._matriz_knn(X)
            self.knn_mu_, self.knn_sd_ = M.mean(), M.std().replace(0, 1)
            self.knn_ = KNNImputer(n_neighbors=5).fit((M - self.knn_mu_) / self.knn_sd_)
        return self

    @staticmethod
    def _basicas(X):
        X = X.copy()
        t = X.Name.str.extract(r",\s*([^\.]+)\.")[0].str.strip().replace(TITULOS_MAP)
        X["Title"] = t.where(t.isin(TITULOS_OK), "Rare")
        return X

    def _fare(self, X):
        return X.Fare.fillna(X.Pclass.map(self.fare_med_class_))

    def _matriz_knn(self, X):
        return pd.DataFrame({
            "Age": X.Age, "Pclass": X.Pclass.astype(float), "SibSp": X.SibSp, "Parch": X.Parch,
            "LogFare": np.log1p(self._fare(X)), "SexNum": (X.Sex == "female").astype(float),
        }, index=X.index)[_KNN_COLS]

    def _imputar_edad(self, X):
        if self.imputacion_edad == "titulo":
            return X.Age.fillna(X.Title.map(self.age_med_title_)).fillna(self.age_med_)
        if self.imputacion_edad == "mediana":
            return X.Age.fillna(self.age_med_)
        if self.imputacion_edad == "media":
            return X.Age.fillna(self.age_mean_)
        Z = (self._matriz_knn(X) - self.knn_mu_) / self.knn_sd_
        edad = self.knn_.transform(Z)[:, 0] * self.knn_sd_["Age"] + self.knn_mu_["Age"]
        return pd.Series(edad, index=X.index).clip(lower=0.4)

    def columnas(self):
        """Variables de salida, en el orden del notebook."""
        activas = VARIABLES if self.variables is None else set(self.variables)
        return [v for v in VARIABLES if v in activas]

    def transform(self, X):
        X = self._basicas(X)
        c1, c2 = self.cortes_familia
        X["FamilySize"] = X.SibSp + X.Parch + 1
        X["FamilyGroup"] = pd.cut(X.FamilySize, [0, c1, c2, np.inf],
                                  labels=["Solo", "Chica", "Grande"]).astype(str)
        X["HasCabin"] = X.Cabin.notna().astype(int)
        # fillna antes de .str: funciona aunque Cabin venga toda vacía (p. ej. una sola fila del simulador)
        X["Deck"] = X.Cabin.fillna("U").astype(str).str[0].replace({"T": "U"})
        X["AgeMissing"] = X.Age.isna().astype(int)
        X["AgeImp"] = self._imputar_edad(X)
        X["IsChild"] = (X.AgeImp < self.edad_nino).astype(int)
        X["LogFare"] = np.log1p(self._fare(X))
        X["Embarked"] = X.Embarked.fillna(self.embarked_mode_)
        X["TicketGroup"] = X.Ticket.map(self.ticket_counts_).fillna(1).clip(upper=5)
        X["Pclass"] = X.Pclass.astype(str)          # se trata como categórica
        return X[self.columnas()]


def armar_pipeline(modelo, escalar=True, edad_nino=13, cortes_familia=(1, 4),
                   imputacion_edad="titulo", variables=None):
    feat = TitanicFeatures(edad_nino, tuple(cortes_familia), imputacion_edad,
                           None if variables is None else tuple(variables))
    cols = feat.columnas()
    num = [c for c in NUM if c in cols]
    cat = [c for c in CAT if c in cols]
    if not num and not cat:
        raise ValueError("No hay ninguna variable activa: activá al menos una en 'Ingeniería de variables'.")
    partes = []
    if num:
        partes.append(("num", StandardScaler() if escalar else "passthrough", num))
    if cat:
        partes.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return Pipeline([("feat", feat), ("prep", ColumnTransformer(partes)), ("model", modelo)])


def nombres_transformados(pipe):
    """Nombres de las columnas que ve el modelo (sin el prefijo num__/cat__)."""
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
