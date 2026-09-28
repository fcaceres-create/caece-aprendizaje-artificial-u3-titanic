import pandas as pd
import plotly.express as px
import streamlit as st

from core import state
from core.features import DERIVADAS, DESCRIPCIONES, IMPUTACIONES, ORIGINALES, TitanicFeatures

st.title("Ingeniería de variables")
st.caption("Elegí qué variables ve el modelo y cómo se construyen. Los cambios se aplican en la próxima corrida "
           "de **Entrenar modelos**. Todo esto se ajusta dentro del Pipeline, solo con los datos de entrenamiento.")

with st.sidebar:
    st.header("Parámetros de las variables")
    st.selectbox("Imputación de la edad", list(IMPUTACIONES), **state.widget("p_imputacion"), format_func=IMPUTACIONES.get,
                 help="Cómo se completa `Age` cuando falta (≈20% de los pasajeros). Genera `AgeImp`.")
    st.slider("Umbral de 'niño' (años)", 1, 18, **state.widget("p_edad_nino"),
              help="IsChild = 1 si la edad imputada es menor a este valor.")
    st.slider("Cortes de FamilyGroup", 1, 10, **state.widget("p_cortes"),
              help="Solo: tamaño ≤ primer corte · Chica: ≤ segundo corte · Grande: el resto.")
    st.button("Restaurar valores del notebook", on_click=state.restaurar_features, icon=":material/restart_alt:")

c1, c2 = st.session_state["p_cortes"]
if c1 >= c2:
    st.error("Los cortes de FamilyGroup deben ser distintos (el primero menor que el segundo).")
    st.stop()

a, b = st.columns(2)
with a:
    st.subheader("Variables originales")
    for v in ORIGINALES:
        st.checkbox(f"`{v}` — {DESCRIPCIONES[v]}", **state.widget(f"p_var_{v}"))
with b:
    st.subheader("Variables derivadas")
    for v in DERIVADAS:
        st.checkbox(f"`{v}` — {DESCRIPCIONES[v]}", **state.widget(f"p_var_{v}"))

cfg = state.config_features()
activas = cfg["variables"]
if activas == []:
    st.error("Desactivaste todas las variables: el modelo no tendría nada con qué aprender. Activá al menos una.")
    st.stop()
st.success(f"Variables activas: {state.describir_variables({'features': cfg})}", icon=":material/check_circle:")

X, _ = state.xy()
tf = TitanicFeatures(cfg["edad_nino"], tuple(cfg["cortes_familia"]), cfg["imputacion_edad"], activas)
tf.fit(X)
Xt = tf.transform(X)

t1, t2, t3 = st.tabs(["Vista previa", "Nulos resultantes", "Efecto de la imputación de edad"])
with t1:
    st.dataframe(Xt.head(15))
    st.caption(f"Primeras 15 filas transformadas ({Xt.shape[0]} filas × {Xt.shape[1]} columnas, antes del "
               "escalado y del one-hot que hace el Pipeline).")
with t2:
    nulos = Xt.isna().sum().rename("Nulos").to_frame()
    if nulos.Nulos.sum() == 0:
        st.success("No quedan valores nulos: todas las imputaciones funcionaron.", icon=":material/check:")
    st.dataframe(nulos.T)
with t3:
    tf_edad = TitanicFeatures(cfg["edad_nino"], tuple(cfg["cortes_familia"]), cfg["imputacion_edad"],
                              ["AgeImp", "AgeMissing"]).fit(X)
    e = tf_edad.transform(X)
    e["Origen"] = e.AgeMissing.map({0: "Edad original", 1: "Edad imputada"})
    fig = px.histogram(e, x="AgeImp", color="Origen", nbins=40, barmode="stack", labels={"AgeImp": "Edad"},
                       color_discrete_map={"Edad original": "#8ab6d6", "Edad imputada": "#e76f51"})
    st.plotly_chart(fig, theme="streamlit")
    st.caption(f"Distribución de la edad después de imputar con **{IMPUTACIONES[cfg['imputacion_edad']]}**. "
               "Con la mediana o la media global todos los faltantes caen en un solo valor (un pico artificial); "
               "por título o con KNN se reparten según el perfil del pasajero (un `Master` es un niño).")
    imputadas = e[e.AgeMissing == 1].AgeImp
    st.dataframe(pd.DataFrame({"Edades imputadas": [imputadas.nunique(), imputadas.mean(), imputadas.min(),
                                                     imputadas.max()]},
                              index=["Valores distintos", "Media", "Mínimo", "Máximo"]).T.round(1))
