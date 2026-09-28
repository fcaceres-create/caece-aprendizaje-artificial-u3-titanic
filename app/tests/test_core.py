import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from core.data import cargar_test, cargar_train, separar_xy
from core.evaluation import evaluar, validar_submission
from core.features import IMPUTACIONES, NUM, CAT, TitanicFeatures, armar_pipeline
from core.models import CATALOGO, crear_modelo, params_por_defecto

# ---- Copia literal de la clase del notebook (titanic.ipynb), usada como referencia ----
TITULOS_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
TITULOS_OK = {"Mr", "Mrs", "Miss", "Master"}


class TitanicFeaturesNotebook(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = self._basicas(X)
        self.age_med_title_ = X.groupby("Title").Age.median()
        self.age_med_ = X.Age.median()
        self.fare_med_class_ = X.groupby("Pclass").Fare.median()
        self.embarked_mode_ = X.Embarked.mode()[0]
        self.ticket_counts_ = X.Ticket.value_counts()
        return self

    @staticmethod
    def _basicas(X):
        X = X.copy()
        t = X.Name.str.extract(r",\s*([^\.]+)\.")[0].str.strip().replace(TITULOS_MAP)
        X["Title"] = t.where(t.isin(TITULOS_OK), "Rare")
        return X

    def transform(self, X):
        X = self._basicas(X)
        X["FamilySize"] = X.SibSp + X.Parch + 1
        X["FamilyGroup"] = pd.cut(X.FamilySize, [0, 1, 4, 20],
                                  labels=["Solo", "Chica", "Grande"]).astype(str)
        X["HasCabin"] = X.Cabin.notna().astype(int)
        X["Deck"] = X.Cabin.str[0].fillna("U").replace({"T": "U"})
        X["AgeMissing"] = X.Age.isna().astype(int)
        X["AgeImp"] = X.Age.fillna(X.Title.map(self.age_med_title_)).fillna(self.age_med_)
        X["IsChild"] = (X.AgeImp < 13).astype(int)
        fare = X.Fare.fillna(X.Pclass.map(self.fare_med_class_))
        X["LogFare"] = np.log1p(fare)
        X["Embarked"] = X.Embarked.fillna(self.embarked_mode_)
        X["TicketGroup"] = X.Ticket.map(self.ticket_counts_).fillna(1).clip(upper=5)
        X["Pclass"] = X.Pclass.astype(str)
        return X[NUM + CAT]


X, y = separar_xy(cargar_train())
TEST = cargar_test()

CFG_DEFECTO = {
    "modelo": "Random Forest", "params": params_por_defecto("Random Forest"),
    "features": {"edad_nino": 13, "cortes_familia": (1, 4), "imputacion_edad": "titulo", "variables": None},
    "eval": {"folds": 10, "holdout": 0.2, "seed": 42},
}


@pytest.mark.parametrize("datos", [X, TEST], ids=["train", "test"])
def test_features_igual_al_notebook(datos):
    esperado = TitanicFeaturesNotebook().fit(X).transform(datos)
    obtenido = TitanicFeatures().fit(X).transform(datos)
    pd.testing.assert_frame_equal(obtenido, esperado)


@pytest.mark.parametrize("estrategia", list(IMPUTACIONES))
def test_imputaciones_sin_nulos(estrategia):
    tf = TitanicFeatures(imputacion_edad=estrategia).fit(X)
    assert tf.transform(X).isna().sum().sum() == 0
    assert tf.transform(TEST).isna().sum().sum() == 0


def test_subconjunto_de_variables():
    pipe = armar_pipeline(crear_modelo("Regresión logística", {}), True, variables=["Sex", "Pclass", "AgeImp"])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.75


def test_sin_variables_da_error_claro():
    with pytest.raises(ValueError, match="variable"):
        armar_pipeline(crear_modelo("KNN", {}), True, variables=[])


@pytest.mark.parametrize("nombre", list(CATALOGO))
def test_todos_los_modelos_entrenan(nombre):
    pipe = armar_pipeline(crear_modelo(nombre, {}), CATALOGO[nombre]["escalar"])
    pipe.fit(X, y)
    assert pipe.predict_proba(TEST).shape == (418, 2)


def test_random_forest_reproduce_notebook():
    """Referencia: output/tuning.csv (Acc CV 0,841) y output/holdout.csv (0,827 y base 0,777)."""
    _, r = evaluar(CFG_DEFECTO, X, y)
    assert r["acc_cv"] == pytest.approx(0.841, abs=0.01)
    assert r["acc_ho"] == pytest.approx(0.827, abs=0.015)
    assert r["base_ho"] == pytest.approx(0.777, abs=0.001)


def test_validar_submission():
    ok = pd.DataFrame({"PassengerId": TEST.PassengerId, "Survived": 0})
    assert validar_submission(ok, TEST) == []
    assert validar_submission(ok.iloc[:-1], TEST)
    assert validar_submission(ok.assign(Survived=0.5), TEST)
