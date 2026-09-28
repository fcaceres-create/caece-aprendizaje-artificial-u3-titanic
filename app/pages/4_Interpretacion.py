import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.tree import plot_tree

from core import state
from core.features import nombres_transformados

st.title("Interpretación")
cfg, pipe, res, es_defecto = state.modelo_vigente()
state.aviso_modelo(cfg, es_defecto)
modelo = pipe.named_steps["model"]
nombres = nombres_transformados(pipe)
cfg_json = state.a_json(cfg)

pestanias = ["Importancia por permutación", "Importancia según el modelo"]
if cfg["modelo"] in ("Árbol de decisión", "Random Forest"):
    pestanias.append("Árbol")
tabs = st.tabs(pestanias)

# ---------------------------------------------------------------- permutación
with tabs[0]:
    st.markdown("Cuánto **cae la accuracy en el hold-out** al desordenar al azar cada columna original. "
                "No depende del tipo de modelo y se mide sobre datos que el modelo no vio.")
    n_rep = st.slider("Repeticiones", 5, 30, 10, step=5,
                      help="Cada columna se permuta varias veces; más repeticiones = estimación más estable.")
    with st.spinner("Calculando importancia por permutación…"):
        imp = state.permutacion(cfg_json, n_rep)
    imp = imp[imp.media.abs() > 0].sort_values("media")
    fig = px.bar(imp.reset_index(names="Columna"), x="media", y="Columna", error_x="desv", orientation="h",
                 labels={"media": "Caída de accuracy al permutar"}, color_discrete_sequence=["#2a9d8f"])
    st.plotly_chart(fig, theme="streamlit")
    st.caption("La permutación se hace sobre las **columnas originales**, antes de la ingeniería de variables. "
               "Por eso `Name` suele aparecer muy arriba: al desordenarla se rompe `Title` (Mr, Mrs, Master…). "
               "`SibSp`/`Parch` alimentan `FamilySize`, y `Fare`/`Cabin` reflejan la clase. Un valor ≈ 0 o negativo "
               "indica que el modelo prácticamente no depende de esa columna en datos nuevos.")

# ---------------------------------------------------------------- importancia propia del modelo
with tabs[1]:
    if hasattr(modelo, "coef_"):
        coefs = pd.DataFrame({"Variable": nombres, "Coeficiente": modelo.coef_[0]})
        coefs["Odds ratio"] = np.exp(coefs.Coeficiente)
        coefs["Efecto"] = np.where(coefs.Coeficiente > 0, "Aumenta la prob. de sobrevivir", "La disminuye")
        coefs = coefs.sort_values("Coeficiente")
        fig = px.bar(coefs, x="Coeficiente", y="Variable", color="Efecto", orientation="h",
                     hover_data={"Odds ratio": ":.2f"}, height=max(400, 22 * len(coefs)),
                     color_discrete_map={"Aumenta la prob. de sobrevivir": "#2a9d8f", "La disminuye": "#e76f51"})
        st.plotly_chart(fig, theme="streamlit")
        st.caption("Coeficientes de la regresión logística. Las variables numéricas están estandarizadas, así que "
                   "la magnitud es comparable. El **odds ratio** = exp(coef): cuánto se multiplican las chances de "
                   "sobrevivir por cada desvío estándar (o al tener esa categoría). Con penalización l1 algunos "
                   "coeficientes quedan exactamente en 0.")
        st.dataframe(coefs.drop(columns="Efecto").sort_values("Coeficiente", ascending=False), hide_index=True,
                     column_config={"Coeficiente": st.column_config.NumberColumn(format="%.3f"),
                                    "Odds ratio": st.column_config.NumberColumn(format="%.3f")})
    elif hasattr(modelo, "feature_importances_"):
        fi = (pd.DataFrame({"Variable": nombres, "Importancia": modelo.feature_importances_})
              .sort_values("Importancia"))
        fi = fi[fi.Importancia > 0]
        fig = px.bar(fi, x="Importancia", y="Variable", orientation="h", height=max(400, 22 * len(fi)),
                     color_discrete_sequence=["#264653"])
        st.plotly_chart(fig, theme="streamlit")
        st.caption("`feature_importances_`: cuánto reduce la impureza cada variable (ya transformada, con one-hot) "
                   "sumando todos los cortes del árbol o del ensamble. Se calcula sobre el entrenamiento y tiende a "
                   "favorecer variables con muchos valores distintos; por eso conviene contrastarla con la "
                   "importancia por permutación.")
    else:
        st.info(f"**{cfg['modelo']}** no tiene coeficientes ni `feature_importances_`: no es un modelo que se "
                "pueda leer directamente. Usá la importancia por permutación, que funciona con cualquier modelo.",
                icon=":material/info:")

# ---------------------------------------------------------------- árbol
if len(tabs) > 2:
    with tabs[2]:
        if cfg["modelo"] == "Random Forest":
            n = len(modelo.estimators_)
            i = st.slider("Árbol del bosque", 0, n - 1, 0)
            arbol = modelo.estimators_[i]
            st.caption(f"El bosque tiene {n} árboles; cada uno ve una muestra distinta de filas y de variables. "
                       "La predicción final es el promedio de todos, así que ningún árbol individual la explica sola.")
        else:
            arbol = modelo
        prof_real = arbol.get_depth()
        prof = st.slider("Profundidad visible", 1, max(1, min(prof_real, 6)), min(3, prof_real),
                         help=f"El árbol tiene profundidad {prof_real}. Se muestran solo los primeros niveles.")
        fig, ax = plt.subplots(figsize=(4 * 2 ** min(prof, 4) / 2 + 6, 2.2 * prof + 2))
        plot_tree(arbol, feature_names=nombres, class_names=["Muere", "Sobrevive"], filled=True, rounded=True,
                  max_depth=prof, impurity=False, proportion=True, fontsize=9, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Cada caja muestra la regla de corte, la proporción de pasajeros que llega a ese nodo y la "
                   "distribución Muere/Sobrevive. El color indica la clase mayoritaria (más intenso = más puro). "
                   "Los nodos `…` siguen debajo pero están ocultos por la profundidad visible.")
