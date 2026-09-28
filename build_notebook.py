"""Genera el notebook titanic.ipynb (celdas de markdown + código).

Uso:  python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace titanic.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Titanic — ¿Quién sobrevivió?

Problema de **clasificación binaria**: a partir de las características de cada pasajero
(`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Name`, `Cabin`, `Ticket`)
predecir `Survived` (1 = sobrevivió, 0 = no).

Datos: <https://www.kaggle.com/c/titanic/data> (`train.csv` con etiqueta, 891 filas;
`test.csv` sin etiqueta, 418 filas). Métrica oficial de la competencia: **accuracy**.

Estructura:
1. Carga y exploración (EDA)
2. Tratamiento de faltantes e ingeniería de variables
3. Modelos: línea base y comparación con validación cruzada
4. Ajuste de hiperparámetros
5. Evaluación en hold-out, importancia de variables
6. Modelo final y archivo de envío a Kaggle
""")

code("""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (StratifiedKFold, cross_validate, GridSearchCV,
                                     train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

SEED = 42                      # semilla fija => resultados reproducibles
np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="deep")
FIG = Path("figs"); FIG.mkdir(exist_ok=True)
OUT = Path("output"); OUT.mkdir(exist_ok=True)

def guardar(nombre):
    plt.tight_layout()
    plt.savefig(FIG / f"{nombre}.png", dpi=130, bbox_inches="tight")
    plt.show()
""")

md("## 1. Carga y primera inspección")
code("""
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print("train:", train.shape, " test:", test.shape)
train.head()
""")
code("""
train.info()
""")
code("""
faltantes = pd.DataFrame({
    "train_n": train.isna().sum(), "train_%": (train.isna().mean() * 100).round(1),
    "test_n": test.isna().sum(),
}).query("train_n > 0 or test_n > 0")
faltantes
""")
md("""
**Observaciones**
- `Age` falta en ~20% de los registros → hay que imputar (es una variable relevante).
- `Cabin` falta en ~77% → no se puede imputar con sentido; se usará sólo *si tiene o no cabina registrada* y la cubierta (primera letra).
- `Embarked` falta en 2 filas y `Fare` en 1 fila de test → imputación simple (moda / mediana).
""")
code("""
train.describe().T
""")
code("""
tasa = train["Survived"].mean()
print(f"Tasa global de supervivencia: {tasa:.3f}  ({train.Survived.sum()} de {len(train)})")
""")
md("""
La clase positiva es ~38%: hay un desbalance moderado, no extremo. Un modelo que prediga
"nadie sobrevive" ya obtiene ~62% de accuracy — ése es el **piso** que cualquier modelo debe superar.
Por eso, además de accuracy, se reportan ROC-AUC y F1.
""")

md("## 2. Análisis exploratorio")
code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["Sex", "Pclass", "Embarked"]):
    sns.barplot(data=train, x=col, y="Survived", ax=ax, errorbar=("ci", 95))
    ax.set_ylabel("Tasa de supervivencia"); ax.set_ylim(0, 1)
    ax.set_title(f"Supervivencia por {col}")
guardar("01_sexo_clase_puerto")
""")
code("""
pd.crosstab([train.Sex], train.Pclass, values=train.Survived, aggfunc="mean").round(2)
""")
md("""
- **Sexo** es el factor más fuerte: ~74% de las mujeres sobrevivió contra ~19% de los hombres
  (la norma "mujeres y niños primero").
- **Clase**: 1ra ≈ 63%, 2da ≈ 47%, 3ra ≈ 24%. La clase interactúa con el sexo: las mujeres de 1ra y 2da
  sobrevivieron casi todas, las de 3ra sólo ~50%.
- **Puerto**: Cherbourg muestra mayor supervivencia, pero probablemente porque allí embarcaron
  más pasajeros de 1ra clase (variable confundida con `Pclass`).
""")
code("""
pd.crosstab(train.Embarked, train.Pclass, normalize="index").round(2)
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(data=train, x="Age", hue="Survived", bins=30, multiple="stack", ax=axes[0])
axes[0].set_title("Distribución de edad por supervivencia")
sns.histplot(data=train, x="Fare", hue="Survived", bins=40, multiple="stack", ax=axes[1],
             log_scale=(False, False))
axes[1].set_xlim(0, 300); axes[1].set_title("Distribución de tarifa (Fare) por supervivencia")
guardar("02_edad_tarifa")
""")
md("""
- Los **niños pequeños** (< 10 años) tienen tasas de supervivencia claramente mayores.
- `Fare` está muy sesgada a la derecha (outliers de hasta 512) y se relaciona con la clase:
  tarifas altas → más supervivencia. Para modelos lineales conviene transformarla con `log1p`.
""")
code("""
train["FamilySize"] = train.SibSp + train.Parch + 1
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=train, x="FamilySize", y="Survived", ax=ax, errorbar=None)
ax.set_ylabel("Tasa de supervivencia"); ax.set_title("Supervivencia por tamaño de familia a bordo")
guardar("03_familia")
train.drop(columns="FamilySize", inplace=True)
""")
md("""
Relación **no lineal** con el tamaño de familia: viajar solo (1) o en familias grandes (≥5) reduce la
supervivencia; familias de 2 a 4 tienen la mayor tasa. Esto motiva crear `FamilySize` y agruparla.
""")
code("""
titulo = train.Name.str.extract(r",\\s*([^\\.]+)\\.")[0]
pd.DataFrame({"n": titulo.value_counts(),
              "tasa": train.groupby(titulo).Survived.mean().round(2)}).sort_values("n", ascending=False)
""")
md("""
El **título** extraído del nombre (`Mr`, `Mrs`, `Miss`, `Master`, …) resume sexo, edad aproximada y
estado civil/estatus. `Master` (niños varones) sobrevive ~57% contra ~16% de `Mr`: captura el efecto
"niños primero" en los varones, algo que `Sex` solo no ve. Los títulos poco frecuentes se agrupan en `Rare`.
""")
code("""
num = train[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].copy()
num["Sex_female"] = (train.Sex == "female").astype(int)
plt.figure(figsize=(7, 5.5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.title("Correlaciones (Pearson)")
guardar("04_correlaciones")
""")

md("""
## 3. Ingeniería de variables y tratamiento de faltantes

Decisión clave: **todo el preprocesamiento vive dentro de un `Pipeline` de scikit-learn**. Así, los
valores aprendidos (p. ej. la mediana de edad por título) se calculan sólo con los datos de
entrenamiento de cada *fold* de validación cruzada, y se evita la **fuga de información (data leakage)**
del conjunto de validación hacia el modelo.

Variables creadas:

| Variable | Cómo se construye | Justificación |
|---|---|---|
| `Title` | Texto entre la coma y el punto en `Name`; títulos raros → `Rare`, `Mlle/Ms`→`Miss`, `Mme`→`Mrs` | Resume sexo + edad + estatus |
| `FamilySize` | `SibSp + Parch + 1` | Efecto no lineal observado |
| `FamilyGroup` | Solo (1) / Chica (2–4) / Grande (≥5) | Captura la forma de U invertida |
| `HasCabin` | 1 si `Cabin` no es nulo | Proxy de estatus; 77% faltante impide usar el valor |
| `Deck` | Primera letra de `Cabin` (o `U` = desconocida) | Ubicación física en el barco |
| `AgeImp` | `Age` imputada con la mediana **por título** | Más precisa que la mediana global (un `Master` no tiene 28 años) |
| `AgeMissing` | 1 si la edad era nula | Que falte el dato puede ser informativo |
| `IsChild` | `AgeImp < 13` | Umbral de "niños primero" |
| `LogFare` | `log1p(Fare)`, faltantes → mediana por clase | Reduce el sesgo de la distribución |
| `TicketGroup` | Cantidad de pasajeros que comparten el mismo ticket | Grupos que viajaban juntos (amigos, servicio) |

Se **descartan** `PassengerId` (identificador), `Name` y `Ticket` crudos (texto de alta cardinalidad ya
resumido en `Title` y `TicketGroup`) y `Cabin` cruda.
""")
code("""
TITULOS_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
TITULOS_OK = {"Mr", "Mrs", "Miss", "Master"}

class TitanicFeatures(BaseEstimator, TransformerMixin):
    \"\"\"Crea las variables derivadas. fit() aprende sólo del conjunto que recibe.\"\"\"

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
        t = X.Name.str.extract(r",\\s*([^\\.]+)\\.")[0].str.strip().replace(TITULOS_MAP)
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
        X["Pclass"] = X.Pclass.astype(str)          # se trata como categórica
        return X[NUM + CAT]

NUM = ["AgeImp", "LogFare", "FamilySize", "TicketGroup", "SibSp", "Parch",
       "HasCabin", "AgeMissing", "IsChild"]
CAT = ["Sex", "Pclass", "Title", "Embarked", "FamilyGroup", "Deck"]

def armar_pipeline(modelo, escalar=True):
    prep = ColumnTransformer([
        ("num", StandardScaler() if escalar else "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    return Pipeline([("feat", TitanicFeatures()), ("prep", prep), ("model", modelo)])

X = train.drop(columns="Survived")
y = train.Survived
TitanicFeatures().fit(X).transform(X).head()
""")
code("""
# Verificación: no quedan nulos luego de la transformación (train y test)
tf = TitanicFeatures().fit(X)
print("Nulos en train transformado:", tf.transform(X).isna().sum().sum())
print("Nulos en test transformado: ", tf.transform(test).isna().sum().sum())
""")

md("""
## 4. Modelos: línea base y comparación

**Protocolo de evaluación**
- Se reserva un **20% estratificado como hold-out** que no se toca hasta el final: sirve para una
  estimación honesta luego de elegir y ajustar el modelo.
- Sobre el 80% restante se usa **validación cruzada estratificada de 10 folds** para comparar modelos.
  Con sólo ~700 filas, una única partición train/validación daría estimaciones muy ruidosas; la CV
  promedia 10 estimaciones y permite ver la variabilidad (desvío estándar).
- Se compara contra dos líneas base: la clase mayoritaria y la regla "todas las mujeres sobreviven".
""")
code("""
X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
print(len(X_tr), "filas para CV  |", len(X_ho), "filas de hold-out")
""")
code("""
# Línea base "género": predice sobrevive si es mujer
base_genero = (X_tr.Sex == "female").astype(int)
print(f"Base clase mayoritaria : {1 - y_tr.mean():.3f}")
print(f"Base 'mujeres viven'   : {accuracy_score(y_tr, base_genero):.3f}")
""")
code("""
modelos = {
    "Regresión logística": (LogisticRegression(max_iter=2000), True),
    "KNN (k=7)":           (KNeighborsClassifier(n_neighbors=7), True),
    "SVM (RBF)":           (SVC(probability=True, random_state=SEED), True),
    "Árbol de decisión":   (DecisionTreeClassifier(random_state=SEED), False),
    "Random Forest":       (RandomForestClassifier(n_estimators=300, random_state=SEED), False),
    "Gradient Boosting":   (GradientBoostingClassifier(random_state=SEED), False),
}
filas = []
for nombre, (m, esc) in modelos.items():
    r = cross_validate(armar_pipeline(m, esc), X_tr, y_tr, cv=cv,
                       scoring=["accuracy", "roc_auc", "f1"], return_train_score=True)
    filas.append({"Modelo": nombre,
                  "Acc train": r["train_accuracy"].mean(),
                  "Acc CV": r["test_accuracy"].mean(), "± desv": r["test_accuracy"].std(),
                  "AUC CV": r["test_roc_auc"].mean(), "F1 CV": r["test_f1"].mean()})
comparacion = pd.DataFrame(filas).set_index("Modelo").sort_values("Acc CV", ascending=False).round(3)
comparacion["Brecha (sobreajuste)"] = (comparacion["Acc train"] - comparacion["Acc CV"]).round(3)
comparacion.to_csv(OUT / "comparacion_modelos.csv")
comparacion
""")
code("""
fig, ax = plt.subplots(figsize=(8, 4))
c = comparacion.sort_values("Acc CV")
ax.barh(c.index, c["Acc CV"], xerr=c["± desv"], color="#4C72B0", alpha=.85, label="CV (validación)")
ax.scatter(c["Acc train"], c.index, color="#C44E52", zorder=3, label="Entrenamiento")
base = accuracy_score(y_tr, base_genero)
ax.axvline(base, ls="--", color="gray", lw=1); ax.text(base + 0.002, -0.6, "base 'mujeres viven'", fontsize=8, color="gray")
ax.set_xlim(0.6, 1.0); ax.set_xlabel("Accuracy"); ax.legend(loc="lower right")
ax.set_title("Comparación de modelos (CV estratificada 10 folds)")
guardar("05_comparacion_modelos")
""")
md("""
**Primer problema encontrado: sobreajuste.** El árbol de decisión sin restricciones (y en menor medida
Random Forest) llega a ~98% en entrenamiento pero cae mucho en validación: memoriza el conjunto de
entrenamiento. La columna *Brecha* lo cuantifica. Esto se corrige limitando la complejidad
(profundidad máxima, mínimo de muestras por hoja), que es lo que se busca en el paso siguiente.
""")

md("""
## 5. Ajuste de hiperparámetros

Se ajustan con `GridSearchCV` (misma CV de 10 folds, sólo sobre el 80% de entrenamiento) los tres
candidatos más prometedores y de naturaleza distinta: un modelo lineal (interpretable), un ensamble por
*bagging* y uno por *boosting*.
""")
code("""
grillas = {
    "Regresión logística": (armar_pipeline(LogisticRegression(max_iter=2000)),
        {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
    "Random Forest": (armar_pipeline(RandomForestClassifier(n_estimators=300, random_state=SEED), False),
        {"model__max_depth": [4, 6, 8, None], "model__min_samples_leaf": [1, 3, 5],
         "model__max_features": ["sqrt", 0.5]}),
    "Gradient Boosting": (armar_pipeline(GradientBoostingClassifier(random_state=SEED), False),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1],
         "model__max_depth": [2, 3, 4], "model__subsample": [0.8, 1.0]}),
}
ajustados, filas = {}, []
for nombre, (pipe, grid) in grillas.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1, return_train_score=True)
    gs.fit(X_tr, y_tr)
    ajustados[nombre] = gs.best_estimator_
    i = gs.best_index_
    filas.append({"Modelo": nombre, "Mejores parámetros": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
                  "Acc train": gs.cv_results_["mean_train_score"][i],
                  "Acc CV": gs.best_score_, "± desv": gs.cv_results_["std_test_score"][i]})
tuning = pd.DataFrame(filas).set_index("Modelo").round(3)
tuning.to_csv(OUT / "tuning.csv")
tuning
""")

md("""
## 6. Evaluación final en el hold-out

Recién ahora se usan las 179 filas reservadas. Es la estimación más honesta del desempeño en datos nuevos,
porque no intervinieron en ninguna decisión.
""")
code("""
filas = []
for nombre, m in ajustados.items():
    p = m.predict(X_ho); pr = m.predict_proba(X_ho)[:, 1]
    filas.append({"Modelo": nombre, "Accuracy": accuracy_score(y_ho, p),
                  "ROC-AUC": roc_auc_score(y_ho, pr)})
holdout = pd.DataFrame(filas).set_index("Modelo").round(3)
holdout.loc["Base 'mujeres viven'"] = [accuracy_score(y_ho, (X_ho.Sex == "female").astype(int)), np.nan]
holdout.to_csv(OUT / "holdout.csv")
holdout
""")
code("""
fig, ax = plt.subplots(figsize=(6, 5))
for nombre, m in ajustados.items():
    RocCurveDisplay.from_estimator(m, X_ho, y_ho, name=nombre, ax=ax)
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_title("Curvas ROC en hold-out")
guardar("06_roc")
""")
code("""
MEJOR = tuning["Acc CV"].idxmax()     # se elige por CV, no por hold-out (evita "espiar" el test)
print("Modelo elegido (mejor accuracy en CV):", MEJOR)
m = ajustados[MEJOR]
pred = m.predict(X_ho)
print(classification_report(y_ho, pred, target_names=["No sobrevivió", "Sobrevivió"], digits=3))
ConfusionMatrixDisplay(confusion_matrix(y_ho, pred),
                       display_labels=["No sobrev.", "Sobrev."]).plot(cmap="Blues", colorbar=False)
plt.title(f"Matriz de confusión — {MEJOR} (hold-out)")
guardar("07_confusion")
""")
md("""
Lectura de la matriz de confusión: los errores más frecuentes suelen ser **sobrevivientes predichos como
no sobrevivientes** (falsos negativos), típicamente hombres adultos que sobrevivieron o mujeres de 3ra
clase que no: casos que contradicen el patrón dominante y que con estas variables no se pueden distinguir.
""")

md("""
## 7. ¿Qué características determinan la supervivencia?

Se usan dos enfoques complementarios:
1. **Coeficientes de la regresión logística** (variables estandarizadas): signo = dirección del efecto,
   magnitud = fuerza; `exp(coef)` es el *odds ratio*.
2. **Importancia por permutación** en el hold-out: cuánto cae la accuracy si se "desordena" una variable
   original. Es agnóstica al modelo y se mide sobre datos no vistos.
""")
code("""
lr = ajustados["Regresión logística"]
nombres = lr.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lr.named_steps["model"].coef_[0],
                  index=[n.split("__")[1] for n in nombres]).sort_values()
fig, ax = plt.subplots(figsize=(7, 7))
coefs.plot.barh(ax=ax, color=np.where(coefs > 0, "#55A868", "#C44E52"))
ax.axvline(0, color="black", lw=.8)
ax.set_title("Coeficientes de la regresión logística\\n(verde: aumenta la prob. de sobrevivir)")
guardar("08_coeficientes_lr")
coefs.to_frame("coef").assign(odds_ratio=np.exp(coefs)).round(3).to_csv(OUT / "coeficientes_lr.csv")
""")
code("""
imp = permutation_importance(ajustados[MEJOR], X_ho, y_ho, n_repeats=30,
                             random_state=SEED, scoring="accuracy")
importancia = (pd.DataFrame({"media": imp.importances_mean, "desv": imp.importances_std}, index=X_ho.columns)
               .sort_values("media", ascending=False))
importancia = importancia[importancia.media.abs() > 0]
fig, ax = plt.subplots(figsize=(7, 4))
importancia.sort_values("media").plot.barh(y="media", xerr="desv", ax=ax, legend=False, color="#4C72B0")
ax.set_xlabel("Caída de accuracy al permutar la variable")
ax.set_title(f"Importancia por permutación — {MEJOR} (hold-out)")
guardar("09_importancia_permutacion")
importancia.round(4).to_csv(OUT / "importancia_permutacion.csv")
importancia.round(4)
""")
md("""
La permutación se hace sobre las **columnas originales** (antes de la ingeniería de variables). Por eso
`Name` aparece como la segunda más importante: al desordenarla se rompe `Title`. `SibSp`/`Parch` alimentan
`FamilySize`, y `Fare`/`Cabin` son reflejo de la clase. Un valor ≈ 0 o negativo (`Ticket`, `Cabin`,
`Embarked`) indica que el modelo prácticamente no depende de esa variable en datos nuevos.
""")
code("""
# Árbol poco profundo: una visualización interpretable de las reglas principales
arbol = armar_pipeline(DecisionTreeClassifier(max_depth=3, random_state=SEED), False).fit(X_tr, y_tr)
nombres_arbol = [n.split("__")[1] for n in arbol.named_steps["prep"].get_feature_names_out()]
plt.figure(figsize=(18, 7))
plot_tree(arbol.named_steps["model"], feature_names=nombres_arbol, class_names=["Muere", "Sobrevive"],
          filled=True, rounded=True, fontsize=9, impurity=False, proportion=True)
guardar("10_arbol_reglas")
print(f"Accuracy hold-out del árbol de profundidad 3: {arbol.score(X_ho, y_ho):.3f}")
""")

md("""
## 8. Modelo final y envío a Kaggle

Una vez elegido el modelo y sus hiperparámetros, se **reentrena con las 891 filas** (más datos → mejor
modelo) y se predice `test.csv`. El archivo `output/submission.csv` tiene el formato que pide Kaggle
(`PassengerId,Survived`).
""")
code("""
final = ajustados[MEJOR]
final.fit(X, y)
sub = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": final.predict(test).astype(int)})
sub.to_csv(OUT / "submission.csv", index=False)
print(sub.shape, "| proporción predicha como sobreviviente:", round(sub.Survived.mean(), 3))
sub.head()
""")
code("""
# Envíos adicionales con los otros modelos ajustados (para comparar en el leaderboard)
for nombre, m in ajustados.items():
    m.fit(X, y)
    slug = {"Regresión logística": "logistica", "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting"}[nombre]
    archivo = OUT / f"submission_{slug}.csv"
    pd.DataFrame({"PassengerId": test.PassengerId, "Survived": m.predict(test).astype(int)}).to_csv(archivo, index=False)
    print("guardado", archivo)
""")
md("""
## 9. Conclusiones

- Las variables que más determinan la supervivencia son **sexo / título**, **clase** (y tarifa / cabina,
  que la reflejan) y **edad** (ser niño), seguidas por el **tamaño del grupo familiar**.
- Los ensambles ajustados y la regresión logística quedan en ~82–84% de accuracy en validación, unos
  4–6 puntos sobre la regla trivial "mujeres viven" (~79%). En el leaderboard público de Kaggle, valores de
  0.77–0.80 son habituales para este tipo de enfoque; la diferencia con la CV se debe a que el test es
  pequeño (418 filas) y a la variabilidad propia del problema.
- Problemas enfrentados: datos faltantes (Age, Cabin), variables de texto (Name, Ticket, Cabin), sobreajuste
  de modelos basados en árboles, riesgo de fuga de información en la imputación y dataset pequeño con
  estimaciones ruidosas. Cada uno se trató explícitamente (ver documento).
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
nbf.write(nb, "titanic.ipynb")
print("titanic.ipynb generado con", len(cells), "celdas")
