import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import state

st.title("¿Hubiera sobrevivido?")
cfg, pipe, res, es_defecto = state.modelo_vigente()
state.aviso_modelo(cfg, es_defecto)

TITULOS = {"Mr": "Mr (hombre adulto)", "Mrs": "Mrs (mujer casada)", "Miss": "Miss (mujer soltera / niña)",
           "Master": "Master (niño varón)", "Rare": "Otro (Dr, Rev, Col, Countess…)"}
PUERTOS = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
DEFECTOS = {"p_s_sexo": "female", "p_s_titulo": "Miss", "p_s_edad": 25, "p_s_edad_nan": False, "p_s_clase": 3,
            "p_s_tarifa": 8.0, "p_s_puerto": "S", "p_s_sibsp": 0, "p_s_parch": 0, "p_s_cabina": False,
            "p_s_cubierta": "C", "p_s_grupo": 1}
for k, v in DEFECTOS.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------- formulario
with st.container(border=True):
    a, b, c = st.columns(3)
    with a:
        st.radio("Sexo", ["female", "male"], **state.widget("p_s_sexo"), horizontal=True,
                 format_func={"female": "Mujer", "male": "Hombre"}.get)
        st.selectbox("Título", list(TITULOS), **state.widget("p_s_titulo"), format_func=TITULOS.get,
                     help="El modelo lo extrae del nombre: aporta información de sexo, edad y estado civil.")
        st.slider("Edad", 0, 80, **state.widget("p_s_edad"), disabled=st.session_state["p_s_edad_nan"])
        st.checkbox("Edad desconocida", **state.widget("p_s_edad_nan"), help="Como el ~20% de los pasajeros reales.")
    with b:
        st.segmented_control("Clase", [1, 2, 3], **state.widget("p_s_clase"), format_func=lambda c: f"{c}ª")
        st.number_input("Tarifa (£)", 0.0, 520.0, step=1.0, **state.widget("p_s_tarifa"),
                        help="Referencia: mediana 1ª ≈ 60 £, 2ª ≈ 14 £, 3ª ≈ 8 £.")
        st.selectbox("Puerto de embarque", list(PUERTOS), **state.widget("p_s_puerto"), format_func=PUERTOS.get)
    with c:
        st.number_input("Hermanos / cónyuge a bordo", 0, 8, **state.widget("p_s_sibsp"))
        st.number_input("Padres / hijos a bordo", 0, 6, **state.widget("p_s_parch"))
        st.slider("Pasajeros con el mismo pasaje", 1, 5, **state.widget("p_s_grupo"), help="Genera TicketGroup (tope 5).")
        st.checkbox("Tenía cabina asignada", **state.widget("p_s_cabina"))
        if st.session_state["p_s_cabina"]:
            st.selectbox("Cubierta", list("ABCDEFG"), **state.widget("p_s_cubierta"))

ss = st.session_state
clase = ss["p_s_clase"] or 3
edad = np.nan if ss["p_s_edad_nan"] else ss["p_s_edad"]

avisos = []
if ss["p_s_sexo"] == "male" and ss["p_s_titulo"] in ("Mrs", "Miss"):
    avisos.append(f"El título **{ss['p_s_titulo']}** corresponde a una mujer.")
if ss["p_s_sexo"] == "female" and ss["p_s_titulo"] in ("Mr", "Master"):
    avisos.append(f"El título **{ss['p_s_titulo']}** corresponde a un hombre.")
if ss["p_s_titulo"] == "Master" and not ss["p_s_edad_nan"] and ss["p_s_edad"] >= 15:
    avisos.append("**Master** se usaba para niños varones; en los datos casi todos tienen menos de 13 años.")
if ss["p_s_titulo"] == "Mr" and not ss["p_s_edad_nan"] and ss["p_s_edad"] < 12:
    avisos.append("Un varón de esa edad normalmente figura como **Master**, no **Mr**.")
for m in avisos:
    st.warning(m + " El modelo igual va a predecir, pero es un pasajero que no se parece a los datos reales.",
               icon=":material/warning:")


# ---------------------------------------------------------------- fila compatible con el Pipeline
def ticket_con_grupo(n):
    """Un ticket del entrenamiento compartido por n pasajeros (así TicketGroup = n)."""
    conteos = pipe.named_steps["feat"].ticket_counts_
    candidatos = conteos[conteos.clip(upper=5) == n]
    return candidatos.index[0] if len(candidatos) else "SIMULADO-0"


def pasajero(**cambios):
    p = {"sexo": ss["p_s_sexo"], "titulo": ss["p_s_titulo"], "edad": edad, "clase": clase,
         "tarifa": ss["p_s_tarifa"], "puerto": ss["p_s_puerto"], "sibsp": ss["p_s_sibsp"], "parch": ss["p_s_parch"],
         "cabina": ss["p_s_cabina"], "cubierta": ss["p_s_cubierta"], "grupo": ss["p_s_grupo"], **cambios}
    titulo = "Dr" if p["titulo"] == "Rare" else p["titulo"]
    return {"PassengerId": 0, "Pclass": p["clase"], "Name": f"Pasajero, {titulo}. Simulado", "Sex": p["sexo"],
            "Age": p["edad"], "SibSp": p["sibsp"], "Parch": p["parch"], "Ticket": ticket_con_grupo(p["grupo"]),
            "Fare": p["tarifa"], "Cabin": f"{p['cubierta']}1" if p["cabina"] else np.nan, "Embarked": p["puerto"]}


def probabilidades(filas):
    return pipe.predict_proba(pd.DataFrame(filas))[:, 1]


prob = probabilidades([pasajero()])[0]

# ---------------------------------------------------------------- resultado
izq, der = st.columns([1, 2])
with izq:
    color = state.COLORES["Sobrevivió"] if prob >= 0.5 else state.COLORES["No sobrevivió"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".0f"},
        title={"text": "Probabilidad de sobrevivir"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "threshold": {"line": {"color": "gray", "width": 3}, "value": 50}}))
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
    st.plotly_chart(fig, theme="streamlit")
    if prob >= 0.5:
        st.success("El modelo predice que **sobrevive**.", icon=":material/sailing:")
    else:
        st.error("El modelo predice que **no sobrevive**.", icon=":material/tsunami:")

with der:
    st.markdown("**¿Qué pasa si cambio una sola cosa?** (el resto queda fijo, incluido el título)")
    t1, t2, t3, t4 = st.tabs(["Edad", "Tarifa", "Clase", "Sexo"])
    with t1:
        edades = np.arange(0, 81)
        g = pd.DataFrame({"Edad": edades, "Probabilidad": probabilidades([pasajero(edad=e) for e in edades])})
        fig = px.line(g, x="Edad", y="Probabilidad", color_discrete_sequence=["#2a9d8f"])
        if not np.isnan(edad):
            fig.add_vline(x=edad, line_dash="dot", annotation_text="actual")
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, theme="streamlit")
        st.caption(f"Con edad desconocida la probabilidad sería {probabilidades([pasajero(edad=np.nan)])[0]:.0%} "
                   "(el modelo imputa la edad y además usa la marca AgeMissing).")
    with t2:
        tarifas = np.linspace(0, 300, 61)
        g = pd.DataFrame({"Tarifa": tarifas, "Probabilidad": probabilidades([pasajero(tarifa=t) for t in tarifas])})
        fig = px.line(g, x="Tarifa", y="Probabilidad", color_discrete_sequence=["#264653"])
        fig.add_vline(x=min(ss["p_s_tarifa"], 300), line_dash="dot", annotation_text="actual")
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, theme="streamlit")
    with t3:
        g = pd.DataFrame({"Clase": ["1ª", "2ª", "3ª"],
                          "Probabilidad": probabilidades([pasajero(clase=c) for c in (1, 2, 3)])})
        fig = px.bar(g, x="Clase", y="Probabilidad", text_auto=".0%", color_discrete_sequence=["#2a9d8f"])
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, theme="streamlit")
    with t4:
        g = pd.DataFrame({"Sexo": ["Mujer", "Hombre"],
                          "Probabilidad": probabilidades([pasajero(sexo=s) for s in ("female", "male")])})
        fig = px.bar(g, x="Sexo", y="Probabilidad", text_auto=".0%", color_discrete_sequence=["#2a9d8f"])
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, theme="streamlit")
        st.caption("Solo cambia la columna `Sex`; el título queda igual. Cambiá también el título para ver el "
                   "efecto completo del sexo.")

with st.expander("Fila que recibe el modelo"):
    st.caption("El Pipeline espera las columnas originales de Kaggle: el nombre, el ticket y la cabina se generan "
               "a partir del formulario para que produzcan el título, TicketGroup y Deck elegidos.")
    st.dataframe(pd.DataFrame([pasajero()]), hide_index=True)
