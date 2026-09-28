"""Estado compartido entre páginas y funciones cacheadas.

Convención: las claves "p_*" de st.session_state guardan la configuración y NO son widgets (Streamlit
borra el estado de los widgets que no se dibujan en la página actual). Cada control usa su propia clave
"w_p_*", que se sincroniza con la canónica mediante `widget()`. La configuración se pasa a las funciones cacheadas como
JSON ordenado para que sea hasheable.
"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from .data import cargar_test, cargar_train, separar_xy
from .evaluation import curva_validacion, dividir, evaluar, importancia_permutacion, pipeline_desde_config
from .features import VARIABLES
from .models import CATALOGO, params_por_defecto

MODELO_DEFECTO = "Random Forest"
FEATURES_DEFECTO = {"p_edad_nino": 13, "p_cortes": (1, 4), "p_imputacion": "titulo"}
EVAL_DEFECTO = {"p_folds": 10, "p_holdout": 20, "p_seed": 42}
COLORES = {"Sobrevivió": "#2a9d8f", "No sobrevivió": "#e76f51"}


# ---------------------------------------------------------------- datos
@st.cache_data(show_spinner=False)
def train_df():
    return cargar_train()


@st.cache_data(show_spinner=False)
def test_df():
    return cargar_test()


def xy():
    return separar_xy(train_df())


# ---------------------------------------------------------------- estado
def _clave_hp(modelo, param):
    return f"p_hp_{modelo}_{param}"


def restaurar_features():
    for k, v in FEATURES_DEFECTO.items():
        st.session_state[k] = v
    for v in VARIABLES:
        st.session_state[f"p_var_{v}"] = True


def restaurar_hiperparametros(modelo):
    for p, v in params_por_defecto(modelo).items():
        st.session_state[_clave_hp(modelo, p)] = v


def inicializar():
    ss = st.session_state
    if ss.get("_inicializado"):
        return
    restaurar_features()
    for k, v in EVAL_DEFECTO.items():
        ss[k] = v
    ss["p_modelo"] = MODELO_DEFECTO
    for m in CATALOGO:
        restaurar_hiperparametros(m)
    ss["historial"] = []
    ss["ultimo"] = None          # JSON de la última configuración entrenada
    ss["_inicializado"] = True


def _copiar(clave):
    st.session_state[clave] = st.session_state["w_" + clave]


def widget(clave):
    """kwargs para un widget ligado a la clave persistente `clave`:  st.slider(..., **widget("p_folds"))."""
    st.session_state["w_" + clave] = st.session_state[clave]
    return {"key": "w_" + clave, "on_change": _copiar, "args": (clave,)}


# ---------------------------------------------------------------- configuración
def config_features():
    ss = st.session_state
    activas = [v for v in VARIABLES if ss.get(f"p_var_{v}", True)]
    return {"edad_nino": ss["p_edad_nino"], "cortes_familia": list(ss["p_cortes"]),
            "imputacion_edad": ss["p_imputacion"], "variables": None if len(activas) == len(VARIABLES) else activas}


def config_actual():
    ss = st.session_state
    m = ss["p_modelo"]
    return {"modelo": m,
            "params": {p: ss[_clave_hp(m, p)] for p in CATALOGO[m]["params"]},
            "features": config_features(),
            "eval": {"folds": ss["p_folds"], "holdout": ss["p_holdout"] / 100, "seed": ss["p_seed"]}}


def config_defecto():
    return {"modelo": MODELO_DEFECTO, "params": params_por_defecto(MODELO_DEFECTO),
            "features": {"edad_nino": 13, "cortes_familia": [1, 4], "imputacion_edad": "titulo", "variables": None},
            "eval": {"folds": 10, "holdout": 0.2, "seed": 42}}


def a_json(cfg):
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def describir_params(cfg):
    return ", ".join(f"{k}={v}" for k, v in cfg["params"].items())


def describir_variables(cfg):
    v = cfg["features"]["variables"]
    return f"todas ({len(VARIABLES)})" if v is None else f"{len(v)}: " + ", ".join(v)


# ---------------------------------------------------------------- cálculo (cacheado)
@st.cache_resource(show_spinner=False, max_entries=40)
def entrenar(cfg_json):
    """(pipeline entrenado sobre el 80%, resultados). Compartido entre sesiones con la misma config."""
    X, y = xy()
    return evaluar(json.loads(cfg_json), X, y)


@st.cache_resource(show_spinner=False, max_entries=10)
def entrenar_completo(cfg_json):
    """Mismo pipeline, entrenado con las 891 filas (para el envío a Kaggle)."""
    X, y = xy()
    return pipeline_desde_config(json.loads(cfg_json)).fit(X, y)


@st.cache_data(show_spinner=False, max_entries=40)
def curva(cfg_json, param, valores):
    X, y = xy()
    return curva_validacion(json.loads(cfg_json), X, y, param, list(valores))


@st.cache_data(show_spinner=False, max_entries=20)
def permutacion(cfg_json, n_repeats):
    cfg = json.loads(cfg_json)
    pipe, _ = entrenar(cfg_json)
    X, y = xy()
    _, X_ho, _, y_ho = dividir(X, y, cfg["eval"]["holdout"], cfg["eval"]["seed"])
    return importancia_permutacion(pipe, X_ho, y_ho, n_repeats, cfg["eval"]["seed"])


def modelo_vigente():
    """(cfg, pipeline, resultados, es_defecto). Si no se entrenó nada, usa el Random Forest del notebook."""
    cfg_json = st.session_state.get("ultimo")
    es_defecto = cfg_json is None
    if es_defecto:
        cfg_json = a_json(config_defecto())
    with st.spinner("Entrenando el modelo…"):
        pipe, res = entrenar(cfg_json)
    return json.loads(cfg_json), pipe, res, es_defecto


def aviso_modelo(cfg, es_defecto):
    if es_defecto:
        st.info("Todavía no entrenaste ningún modelo en **Entrenar modelos**: se usa el Random Forest "
                "ajustado del notebook (max_depth=4, max_features=0.5, min_samples_leaf=3, 300 árboles).",
                icon=":material/info:")
    else:
        st.caption(f"Modelo vigente: **{cfg['modelo']}** ({describir_params(cfg)}) · "
                   f"variables: {describir_variables(cfg)}")


# ---------------------------------------------------------------- historial
def registrar(cfg, res):
    hist = st.session_state["historial"]
    cfg_json = a_json(cfg)
    if any(h["cfg"] == cfg_json for h in hist):
        return
    hist.append({
        "#": (hist[-1]["#"] + 1) if hist else 1, "★": False,
        "Hora": datetime.now().strftime("%H:%M:%S"),
        "Modelo": cfg["modelo"], "Parámetros": describir_params(cfg),
        "Variables": describir_variables(cfg), "Imputación edad": cfg["features"]["imputacion_edad"],
        "Folds": cfg["eval"]["folds"], "Hold-out": f"{cfg['eval']['holdout']:.0%}", "Semilla": cfg["eval"]["seed"],
        "Acc train": res["acc_train"], "Acc CV": res["acc_cv"], "± desv": res["acc_cv_std"],
        "Acc hold-out": res["acc_ho"], "AUC hold-out": res["auc_ho"], "F1 hold-out": res["f1_ho"],
        "Brecha": res["brecha"], "cfg": cfg_json,
    })


def historial_df():
    return pd.DataFrame(st.session_state["historial"])
