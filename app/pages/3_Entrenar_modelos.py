import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import state
from core.evaluation import UMBRAL_BRECHA, busqueda_grilla
from core.models import CATALOGO, CURVAS, GRILLAS

st.title("Entrenar y comparar modelos")


def control(modelo, param, spec):
    """Dibuja el widget de un hiperparámetro según su tipo."""
    clave, ayuda = f"p_hp_{modelo}_{param}", spec["ayuda"]
    if spec["tipo"] in ("int", "prof"):
        etiqueta = f"{param} (0 = sin límite)" if spec["tipo"] == "prof" else param
        st.slider(etiqueta, spec["min"], spec["max"], step=spec["paso"], **state.widget(clave), help=ayuda)
    elif all(isinstance(o, (int, float)) for o in spec["opciones"]):
        st.select_slider(param, spec["opciones"], **state.widget(clave), help=ayuda)
    else:
        st.selectbox(param, spec["opciones"], **state.widget(clave), help=ayuda, format_func=str)


def aplicar_params(modelo, params):
    for p, v in params.items():
        st.session_state[f"p_hp_{modelo}_{p}"] = v


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Modelo")
    modelo = st.selectbox("Algoritmo", list(CATALOGO), **state.widget("p_modelo"))
    st.caption("Variables escaladas (StandardScaler)." if CATALOGO[modelo]["escalar"]
               else "Sin escalado (los árboles no lo necesitan).")
    for param, spec in CATALOGO[modelo]["params"].items():
        control(modelo, param, spec)
    st.button("Hiperparámetros por defecto", on_click=state.restaurar_hiperparametros, args=(modelo,),
              icon=":material/restart_alt:")
    st.header("Evaluación")
    st.slider("Folds de validación cruzada", 3, 10, **state.widget("p_folds"))
    st.slider("% de hold-out", 10, 40, step=5, **state.widget("p_holdout"))
    st.number_input("Semilla", 0, 10_000, step=1, **state.widget("p_seed"))
    entrenar = st.button("Entrenar", type="primary", icon=":material/play_arrow:", width="stretch")

cfg = state.config_actual()
cfg_json = state.a_json(cfg)
if cfg["features"]["variables"] == []:
    st.error("No hay variables activas. Activá al menos una en **Ingeniería de variables**.")
    st.stop()

if entrenar:
    try:
        with st.spinner(f"Entrenando {modelo} con validación cruzada de {cfg['eval']['folds']} folds…"):
            t0 = time.perf_counter()
            _, res = state.entrenar(cfg_json)
        state.registrar(cfg, res)
        st.session_state["ultimo"] = cfg_json
        st.toast(f"Listo en {time.perf_counter() - t0:.1f} s", icon=":material/check:")
    except ValueError as e:
        st.error(f"No se pudo entrenar: {e}")

t_res, t_hist, t_curva, t_grilla = st.tabs(["Resultados", "Historial de experimentos", "Curva de validación",
                                            "Búsqueda en grilla"])

# ---------------------------------------------------------------- resultados
with t_res:
    ultimo = st.session_state.get("ultimo")
    if ultimo is None:
        st.info("Elegí el modelo y sus hiperparámetros en la barra lateral y presioná **Entrenar**. "
                "Por defecto está cargado el Random Forest ajustado del notebook.", icon=":material/arrow_back:")
    else:
        if ultimo != cfg_json:
            st.warning("La configuración cambió desde el último entrenamiento: se muestran los resultados anteriores. "
                       "Presioná **Entrenar** para actualizarlos.", icon=":material/sync_problem:")
        ucfg = json.loads(ultimo)
        _, r = state.entrenar(ultimo)
        st.subheader(f"{ucfg['modelo']}")
        st.caption(f"{state.describir_params(ucfg)} · {r['n_tr']} filas para CV ({ucfg['eval']['folds']} folds) · "
                   f"{r['n_ho']} de hold-out")
        c = st.columns(6)
        c[0].metric("Acc entrenamiento", f"{r['acc_train']:.3f}", help="Media de la accuracy de entrenamiento en cada fold.")
        c[1].metric("Acc CV", f"{r['acc_cv']:.3f}", f"± {r['acc_cv_std']:.3f}", delta_color="off",
                    delta_arrow="off", help="Media y desvío de la accuracy de validación en los folds.")
        c[2].metric("Acc hold-out", f"{r['acc_ho']:.3f}", f"{r['acc_ho'] - r['base_ho']:+.3f} vs base",
                    help="Accuracy en el hold-out, comparada con la regla 'mujeres viven' en el mismo hold-out.")
        c[3].metric("Brecha train − CV", f"{r['brecha']:.3f}",
                    "sobreajuste" if r["brecha"] > UMBRAL_BRECHA else "OK",
                    delta_color="inverse" if r["brecha"] > UMBRAL_BRECHA else "normal", delta_arrow="off",
                    help=f"Si supera {UMBRAL_BRECHA} el modelo memoriza el entrenamiento más de lo que generaliza.")
        c[4].metric("ROC-AUC hold-out", f"{r['auc_ho']:.3f}", f"CV {r['auc_cv']:.3f}", delta_color="off",
                    delta_arrow="off")
        c[5].metric("F1 hold-out", f"{r['f1_ho']:.3f}", f"CV {r['f1_cv']:.3f}", delta_color="off", delta_arrow="off")
        if r["brecha"] > UMBRAL_BRECHA:
            st.error(f"**Sobreajuste:** la accuracy de entrenamiento supera a la de validación por {r['brecha']:.3f} "
                     f"(> {UMBRAL_BRECHA}). Probá limitar la complejidad (menos profundidad, más muestras por hoja, "
                     "más regularización).", icon=":material/warning:")

        a, b, c3 = st.columns(3)
        with a:
            comp = pd.DataFrame({"Conjunto": ["Entrenamiento", "Validación (CV)", "Hold-out", "Base 'mujeres viven'"],
                                 "Accuracy": [r["acc_train"], r["acc_cv"], r["acc_ho"], r["base_ho"]]})
            fig = px.bar(comp, x="Conjunto", y="Accuracy", text_auto=".3f", color="Conjunto",
                         color_discrete_sequence=["#8ab6d6", "#2a9d8f", "#264653", "#adb5bd"])
            fig.update_layout(showlegend=False, yaxis_range=[0.5, 1], title="Accuracy por conjunto")
            st.plotly_chart(fig, theme="streamlit")
        with b:
            m = pd.DataFrame(r["matriz"], index=["Real: no sobrevivió", "Real: sobrevivió"],
                             columns=["Pred: no sobrevivió", "Pred: sobrevivió"])
            fig = px.imshow(m, text_auto=True, color_continuous_scale="Tealgrn", aspect="auto",
                            title="Matriz de confusión (hold-out)")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, theme="streamlit")
        with c3:
            fpr, tpr = r["roc"]
            fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {r['auc_ho']:.3f}",
                                       line=dict(color="#2a9d8f", width=3)))
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
            fig.update_layout(title="Curva ROC (hold-out)", xaxis_title="Tasa de falsos positivos",
                              yaxis_title="Tasa de verdaderos positivos", legend=dict(x=0.45, y=0.08))
            st.plotly_chart(fig, theme="streamlit")
        with st.expander("Configuración completa de esta corrida"):
            st.json(ucfg)

# ---------------------------------------------------------------- historial
with t_hist:
    hist = st.session_state["historial"]
    if not hist:
        st.info("Cada vez que entrenás, la corrida queda registrada acá para compararla con las demás.")
    else:
        df = state.historial_df()
        solo_fav = st.toggle("Mostrar solo favoritas")
        vista = df[df["★"]] if solo_fav else df
        editado = st.data_editor(
            vista.drop(columns="cfg").assign(Borrar=False), hide_index=True, key="editor_hist",
            disabled=[c for c in vista.columns if c not in ("★",)],
            column_config={"★": st.column_config.CheckboxColumn("★", help="Marcar como favorita"),
                           "Borrar": st.column_config.CheckboxColumn("Borrar"),
                           **{c: st.column_config.NumberColumn(format="%.3f")
                              for c in ["Acc train", "Acc CV", "± desv", "Acc hold-out", "AUC hold-out",
                                        "F1 hold-out", "Brecha"]}})
        favs = dict(zip(editado["#"], editado["★"]))
        for h in hist:
            h["★"] = bool(favs.get(h["#"], h["★"]))
        a, b, c = st.columns(3)
        borrar = editado.loc[editado.Borrar, "#"].tolist()
        if a.button(f"Borrar seleccionadas ({len(borrar)})", disabled=not borrar, icon=":material/delete:"):
            st.session_state["historial"] = [h for h in hist if h["#"] not in borrar]
            st.rerun()
        b.download_button("Descargar CSV", df.drop(columns="cfg").to_csv(index=False).encode("utf-8"),
                          "experimentos_titanic.csv", "text/csv", icon=":material/download:")
        elegido = c.selectbox("Usar corrida como modelo vigente", df["#"], index=len(df) - 1,
                              format_func=lambda i: f"#{i} — {df.set_index('#').loc[i, 'Modelo']}")
        if c.button("Usar esta corrida", icon=":material/done:"):
            st.session_state["ultimo"] = df.set_index("#").loc[elegido, "cfg"]
            st.rerun()

        g = vista.assign(Corrida=vista["#"].map(lambda i: f"#{i}") + " " + vista.Modelo +
                         vista["★"].map({True: " ★", False: ""}))
        fig = go.Figure()
        fig.add_bar(x=g.Corrida, y=g["Acc CV"], error_y=dict(type="data", array=g["± desv"]), name="CV",
                    marker_color="#2a9d8f")
        fig.add_scatter(x=g.Corrida, y=g["Acc train"], mode="markers", name="Entrenamiento",
                        marker=dict(symbol="diamond", size=11, color="#e76f51"))
        fig.add_scatter(x=g.Corrida, y=g["Acc hold-out"], mode="markers", name="Hold-out",
                        marker=dict(symbol="circle", size=11, color="#264653"))
        fig.update_layout(title="Comparación de corridas", yaxis_title="Accuracy", yaxis_range=[0.6, 1.0])
        st.plotly_chart(fig, theme="streamlit")
        st.caption("Barras: accuracy de validación cruzada (con su desvío). Si el rombo (entrenamiento) queda muy por "
                   "encima de la barra, esa corrida sobreajusta.")

# ---------------------------------------------------------------- curva de validación
with t_curva:
    st.markdown(f"Cómo cambia la accuracy de **{modelo}** al variar un hiperparámetro, dejando el resto como está "
                "en la barra lateral. Se calcula con validación cruzada sobre el conjunto de entrenamiento.")
    param = st.selectbox("Hiperparámetro", list(CURVAS[modelo]), key="curva_param")
    valores = tuple(CURVAS[modelo][param])
    st.caption(f"Valores: {', '.join(map(str, valores))}")
    if st.button("Calcular curva", icon=":material/show_chart:"):
        st.session_state["curva_pedida"] = (cfg_json, param)
    if st.session_state.get("curva_pedida") == (cfg_json, param):
        with st.spinner("Calculando…"):
            cv = state.curva(cfg_json, param, valores)
        fig = go.Figure()
        fig.add_scatter(x=cv.valor, y=cv.train, name="Entrenamiento", mode="lines+markers",
                        line=dict(color="#e76f51"))
        fig.add_scatter(x=cv.valor, y=cv.cv + cv.cv_std, mode="lines", line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=cv.valor, y=cv.cv - cv.cv_std, mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(42,157,143,0.2)", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=cv.valor, y=cv.cv, name="Validación (CV)", mode="lines+markers",
                        line=dict(color="#2a9d8f", width=3))
        fig.update_layout(xaxis_title=param, yaxis_title="Accuracy", xaxis_type="category")
        st.plotly_chart(fig, theme="streamlit")
        mejor = cv.loc[cv.cv.idxmax()]
        st.caption(f"Mejor valor en CV: **{param} = {mejor.valor}** (accuracy {mejor.cv:.3f}). Donde la curva de "
                   "entrenamiento sube pero la de validación baja hay **sobreajuste**; si las dos están bajas, "
                   "**subajuste**.")

# ---------------------------------------------------------------- búsqueda en grilla
with t_grilla:
    grilla = GRILLAS[modelo]
    total = 1
    for v in grilla.values():
        total *= len(v)
    st.markdown(f"Prueba todas las combinaciones de esta grilla chica para **{modelo}** "
                f"({total} combinaciones × {cfg['eval']['folds']} folds), con el resto de la configuración actual.")
    if modelo in ("Random Forest", "Gradient Boosting", "SVM"):
        st.caption("En un servidor chico (como Streamlit Cloud) puede tardar algunos minutos. Para acelerar, bajá la "
                   "cantidad de folds o de árboles (`n_estimators`) en la barra lateral.")
    st.json({k: [str(x) for x in v] for k, v in grilla.items()}, expanded=False)
    clave = f"grilla_{cfg_json}"
    if st.button("Lanzar búsqueda", icon=":material/grid_on:"):
        barra = st.progress(0.0, text="Empezando…")
        filas = []
        for i, n, fila in busqueda_grilla(cfg, *state.xy(), grilla):
            filas.append(fila)
            barra.progress(i / n, text=f"Combinación {i} de {n}")
        barra.empty()
        st.session_state[clave] = pd.DataFrame(filas).sort_values("Acc CV", ascending=False)
    if clave in st.session_state:
        res = st.session_state[clave]
        mejor = res.iloc[0]
        mejores = {k: (mejor[k].item() if hasattr(mejor[k], "item") else mejor[k]) for k in grilla}
        st.success(f"Mejor combinación: {mejores} → accuracy CV {mejor['Acc CV']:.3f} ± {mejor['± desv']:.3f}",
                   icon=":material/emoji_events:")
        st.button("Aplicar estos hiperparámetros", on_click=aplicar_params, args=(modelo, mejores),
                  icon=":material/input:", help="Los copia a la barra lateral; después presioná Entrenar.")
        res = res.assign(Brecha=res["Acc train"] - res["Acc CV"])
        st.dataframe(res.style.format({c: "{:.3f}" for c in ["Acc train", "Acc CV", "± desv", "Brecha"]}),
                     hide_index=True)
