import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import state
from core.features import TitanicFeatures

st.title("Explorar datos")
st.caption("Los filtros de la barra lateral solo afectan esta exploración, nunca los datos con los que se entrenan los modelos.")


@st.cache_data(show_spinner=False)
def datos_exploracion():
    df = state.train_df().copy()
    df["Title"] = TitanicFeatures._basicas(df)["Title"]
    df["FamilySize"] = df.SibSp + df.Parch + 1
    df["Deck"] = df.Cabin.str[0].fillna("U").replace({"T": "U"})
    df["HasCabin"] = np.where(df.Cabin.notna(), "Sí", "No")
    df["Puerto"] = df.Embarked.map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}).fillna("Desconocido")
    df["Rango de edad"] = pd.cut(df.Age, [0, 12, 18, 30, 45, 60, 81],
                                 labels=["0–12", "13–18", "19–30", "31–45", "46–60", "61+"]).astype(str)
    df["Rango de edad"] = df["Rango de edad"].replace("nan", "Desconocida")
    df["Cuartil de tarifa"] = pd.qcut(df.Fare, 4, labels=["Q1 (baja)", "Q2", "Q3", "Q4 (alta)"]).astype(str)
    df["Resultado"] = np.where(df.Survived == 1, "Sobrevivió", "No sobrevivió")
    return df


df = datos_exploracion()

# ---------------------------------------------------------------- filtros
with st.sidebar:
    st.header("Filtros")
    st.session_state.setdefault("p_f_sexo", ["male", "female"])
    st.session_state.setdefault("p_f_clase", [1, 2, 3])
    st.session_state.setdefault("p_f_puerto", sorted(df.Puerto.unique()))
    st.session_state.setdefault("p_f_edad", (0, 80))
    st.session_state.setdefault("p_f_edad_nan", True)
    st.session_state.setdefault("p_f_familia", (1, 11))
    sexo = st.multiselect("Sexo", ["male", "female"], **state.widget("p_f_sexo"),
                          format_func={"male": "Hombre", "female": "Mujer"}.get)
    clase = st.multiselect("Clase", [1, 2, 3], **state.widget("p_f_clase"), format_func=lambda c: f"{c}ª clase")
    puerto = st.multiselect("Puerto", sorted(df.Puerto.unique()), **state.widget("p_f_puerto"))
    edad = st.slider("Rango de edad", 0, 80, **state.widget("p_f_edad"))
    edad_nan = st.checkbox("Incluir edad desconocida", **state.widget("p_f_edad_nan"))
    familia = st.slider("Tamaño de familia", 1, 11, **state.widget("p_f_familia"))

mask = (df.Sex.isin(sexo) & df.Pclass.isin(clase) & df.Puerto.isin(puerto)
        & df.FamilySize.between(*familia)
        & (df.Age.between(*edad) | (edad_nan & df.Age.isna())))
f = df[mask]

if f.empty:
    st.warning("Ningún pasajero cumple los filtros. Ampliá la selección en la barra lateral.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pasajeros", f"{len(f)} de {len(df)}")
c2.metric("Tasa de supervivencia", f"{f.Survived.mean():.1%}",
          delta=f"{f.Survived.mean() - df.Survived.mean():+.1%} vs total", delta_color="off")
c3.metric("Mujeres", f"{(f.Sex == 'female').mean():.1%}")
c4.metric("Edad media", f"{f.Age.mean():.1f}" if f.Age.notna().any() else "—")

t1, t2, t3, t4 = st.tabs(["Supervivencia por variable", "Edad y tarifa", "Sexo × clase", "Faltantes"])

# ---------------------------------------------------------------- tasa por variable
with t1:
    opciones = {"Sexo": "Sex", "Clase": "Pclass", "Puerto": "Puerto", "Título": "Title",
                "Tamaño de familia": "FamilySize", "Hermanos/cónyuge (SibSp)": "SibSp",
                "Padres/hijos (Parch)": "Parch", "Cubierta": "Deck", "Tiene cabina": "HasCabin",
                "Rango de edad": "Rango de edad", "Cuartil de tarifa": "Cuartil de tarifa"}
    st.session_state.setdefault("p_f_var", "Sexo")
    nombre = st.selectbox("Variable", list(opciones), **state.widget("p_f_var"))
    col = opciones[nombre]
    g = (f.groupby(col).Survived.agg(["mean", "size"]).reset_index()
         .rename(columns={"mean": "Tasa de supervivencia", "size": "Pasajeros"}))
    g[col] = g[col].astype(str)
    fig = px.bar(g, x=col, y="Tasa de supervivencia", text_auto=".0%", hover_data=["Pasajeros"],
                 labels={col: nombre}, color_discrete_sequence=[state.COLORES["Sobrevivió"]])
    fig.add_hline(y=f.Survived.mean(), line_dash="dash", annotation_text="promedio filtrado")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, theme="streamlit")
    st.caption(f"Proporción de sobrevivientes en cada valor de **{nombre}** (pasá el mouse para ver cuántos "
               "pasajeros hay en cada barra). La línea punteada es la tasa promedio de los pasajeros filtrados: "
               "las barras muy por encima o por debajo indican una variable con poder predictivo.")

# ---------------------------------------------------------------- histogramas
with t2:
    a, b = st.columns(2)
    with a:
        fig = px.histogram(f.dropna(subset=["Age"]), x="Age", color="Resultado", nbins=40, barmode="overlay",
                           opacity=0.7, color_discrete_map=state.COLORES, labels={"Age": "Edad"})
        st.plotly_chart(fig, theme="streamlit")
        st.caption("Distribución de edades según el resultado. Los niños pequeños sobreviven en mayor proporción "
                   "(\"mujeres y niños primero\"); las edades faltantes no se muestran.")
    with b:
        st.session_state.setdefault("p_f_log", True)
        log = st.toggle("Escala logarítmica", **state.widget("p_f_log"))
        x, etiqueta = ("LogFare", "log(1 + tarifa)") if log else ("Fare", "Tarifa (£)")
        fig = px.histogram(f.assign(LogFare=np.log1p(f.Fare)), x=x, color="Resultado", nbins=40,
                           barmode="overlay", opacity=0.7, color_discrete_map=state.COLORES, labels={x: etiqueta})
        st.plotly_chart(fig, theme="streamlit")
        st.caption("Distribución de la tarifa pagada. Las tarifas altas (1ª clase) concentran más sobrevivientes. "
                   "La escala logarítmica evita que unos pocos pasajes muy caros aplasten el gráfico.")

# ---------------------------------------------------------------- tabla cruzada
with t3:
    tasa = f.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="mean")
    n = f.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="size")
    tasa.index = tasa.index.map({"male": "Hombre", "female": "Mujer"})
    tasa.columns = [f"{c}ª clase" for c in tasa.columns]
    fig = px.imshow(tasa, text_auto=".0%", color_continuous_scale="Tealgrn", zmin=0, zmax=1, aspect="auto",
                    labels={"color": "Supervivencia"})
    st.plotly_chart(fig, theme="streamlit")
    st.caption("Tasa de supervivencia por sexo y clase. La combinación explica gran parte del problema: "
               "casi todas las mujeres de 1ª y 2ª clase sobreviven, mientras que los hombres de 2ª y 3ª muy pocos.")
    n.index = n.index.map({"male": "Hombre", "female": "Mujer"})
    n.columns = [f"{c}ª clase" for c in n.columns]
    st.dataframe(n.rename_axis("Pasajeros"))

# ---------------------------------------------------------------- faltantes
with t4:
    orig = f[["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
              "Ticket", "Fare", "Cabin", "Embarked"]]
    falt = orig.isna().mean().sort_values(ascending=False)
    a, b = st.columns([1, 2])
    with a:
        fig = px.bar(falt[falt > 0].rename("Faltantes").reset_index(), x="index", y="Faltantes", text_auto=".1%",
                     labels={"index": "Columna"}, color_discrete_sequence=[state.COLORES["No sobrevivió"]])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, theme="streamlit")
        st.caption("Porcentaje de valores faltantes por columna (solo las que tienen alguno).")
    with b:
        fig = px.imshow(orig.isna().T.astype(int), color_continuous_scale=["#dddddd", "#e76f51"], aspect="auto",
                        labels={"x": "Pasajero (fila)", "y": "Columna", "color": "Falta"})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, theme="streamlit")
        st.caption("Mapa de faltantes: cada columna del gráfico es un pasajero y en naranja se marcan los datos "
                   "ausentes. `Cabin` falta en ~77% de los casos y `Age` en ~20%, por eso se crean `HasCabin`, "
                   "`AgeMissing` y se imputa la edad.")
