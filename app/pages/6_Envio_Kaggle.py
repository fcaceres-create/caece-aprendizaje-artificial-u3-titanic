import json

import pandas as pd
import plotly.express as px
import streamlit as st

from core import state
from core.data import DATA
from core.evaluation import validar_submission

st.title("Envío a Kaggle")
cfg_json = st.session_state.get("ultimo")
es_defecto = cfg_json is None
cfg_json = cfg_json or state.a_json(state.config_defecto())
cfg = json.loads(cfg_json)
state.aviso_modelo(cfg, es_defecto)

st.markdown("Para el envío se **reentrena el modelo vigente con las 891 filas** (más datos → mejor modelo): "
            "ya no hace falta reservar el hold-out, porque la evaluación honesta ya se hizo. Después se predice "
            "`test.csv` y se valida el formato antes de habilitar la descarga.")

if st.button("Entrenar con los 891 pasajeros y predecir", type="primary", icon=":material/rocket_launch:"):
    st.session_state["envio"] = cfg_json

if st.session_state.get("envio") != cfg_json:
    st.stop()

test = state.test_df()
with st.spinner("Entrenando con todos los datos…"):
    final = state.entrenar_completo(cfg_json)
sub = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": final.predict(test).astype(int)})

errores = validar_submission(sub, test)
if errores:
    st.error("El archivo no tiene el formato que pide Kaggle:\n\n" + "\n".join(f"- {e}" for e in errores))
    st.stop()

st.success("Formato validado: 418 filas, columnas `PassengerId,Survived`, IDs idénticos a `test.csv`, "
           "valores enteros 0/1.", icon=":material/verified:")
_, r = state.entrenar(cfg_json)
genero = pd.read_csv(DATA / "gender_submission.csv")
c1, c2, c3 = st.columns(3)
c1.metric("Predichos como sobrevivientes", f"{sub.Survived.mean():.1%}", help="En el entrenamiento sobrevivió el 38,4%.")
c2.metric("Coincidencia con 'mujeres viven'", f"{(sub.Survived == genero.Survived).mean():.1%}",
          help="Proporción de predicciones iguales a gender_submission.csv (la base de Kaggle).")
c3.metric("Accuracy esperada (hold-out)", f"{r['acc_ho']:.3f}",
          help="Estimación del puntaje público; Kaggle usa otra muestra, así que puede variar ±0,02–0,03.")

a, b = st.columns([1, 1])
with a:
    st.download_button("Descargar submission.csv", sub.to_csv(index=False).encode("utf-8"), "submission.csv",
                       "text/csv", type="primary", icon=":material/download:")
    st.dataframe(sub.head(10), hide_index=True)
with b:
    g = (sub.merge(test[["PassengerId", "Sex", "Pclass"]])
         .assign(Sexo=lambda d: d.Sex.map({"male": "Hombre", "female": "Mujer"}), Clase=lambda d: d.Pclass.astype(str) + "ª")
         .groupby(["Sexo", "Clase"]).Survived.mean().reset_index())
    fig = px.bar(g, x="Clase", y="Survived", color="Sexo", barmode="group", text_auto=".0%",
                 labels={"Survived": "Predichos como sobrevivientes"},
                 color_discrete_map={"Mujer": "#2a9d8f", "Hombre": "#e76f51"})
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, theme="streamlit")
    st.caption("Proporción de pasajeros de test que el modelo predice como sobrevivientes, por sexo y clase.")
