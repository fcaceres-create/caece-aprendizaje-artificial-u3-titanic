"""Laboratorio Titanic — punto de entrada.  Ejecutar con:  streamlit run app/app.py"""
import streamlit as st

from core import state

st.set_page_config(page_title="Laboratorio Titanic", page_icon="🚢", layout="wide")
state.inicializar()


def inicio():
    st.title("🚢 Laboratorio Titanic")
    st.markdown(
        "Un laboratorio para experimentar con el problema **Titanic** de Kaggle: ¿qué pasajeros "
        "sobrevivieron? Cambiá variables, modelos e hiperparámetros y mirá el efecto en las métricas.")
    X, y = state.xy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pasajeros (train)", len(X))
    c2.metric("Sobrevivieron", f"{y.mean():.1%}")
    c3.metric("Pasajeros (test)", len(state.test_df()))
    c4.metric("Base 'mujeres viven'", f"{((X.Sex == 'female').astype(int) == y).mean():.1%}",
              help="Accuracy de predecir que sobreviven todas las mujeres y ningún hombre (891 filas).")
    st.subheader("Recorrido sugerido")
    st.markdown(
        "1. **Explorar datos** — quién sobrevivió según sexo, clase, edad, familia…\n"
        "2. **Ingeniería de variables** — elegí qué variables usa el modelo y cómo se imputa la edad.\n"
        "3. **Entrenar modelos** — el *playground*: modelos, hiperparámetros, sobreajuste, historial.\n"
        "4. **Interpretación** — qué variables pesan en el modelo entrenado.\n"
        "5. **¿Hubiera sobrevivido?** — simulá un pasajero y mirá su probabilidad.\n"
        "6. **Envío a Kaggle** — generá `submission.csv` con el modelo actual.")
    with st.expander("¿Cómo se evalúa? (sin fuga de información)"):
        st.markdown(
            "- Se reserva un **hold-out estratificado** (20% por defecto) que no interviene en ninguna decisión.\n"
            "- Sobre el resto se hace **validación cruzada estratificada** (10 folds por defecto).\n"
            "- Todo el preprocesamiento (imputaciones, escalado, one-hot) está dentro de un `Pipeline`, "
            "así que en cada fold se ajusta **solo** con los datos de entrenamiento de ese fold.\n"
            "- Semilla fija (42) → resultados reproducibles. Con la configuración por defecto se obtiene "
            "lo mismo que en el notebook: accuracy CV 0,841 y hold-out 0,827.")


paginas = [
    st.Page(inicio, title="Inicio", icon=":material/home:", default=True),
    st.Page("pages/1_Explorar_datos.py", title="Explorar datos", icon=":material/query_stats:"),
    st.Page("pages/2_Ingenieria_de_variables.py", title="Ingeniería de variables", icon=":material/tune:"),
    st.Page("pages/3_Entrenar_modelos.py", title="Entrenar modelos", icon=":material/model_training:"),
    st.Page("pages/4_Interpretacion.py", title="Interpretación", icon=":material/insights:"),
    st.Page("pages/5_Simulador.py", title="¿Hubiera sobrevivido?", icon=":material/person_search:"),
    st.Page("pages/6_Envio_Kaggle.py", title="Envío a Kaggle", icon=":material/upload_file:"),
]
st.navigation(paginas).run()
