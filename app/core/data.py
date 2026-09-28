"""Carga de datos. Las rutas son relativas a este archivo, así funciona igual local y en Streamlit Cloud."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"


def cargar_train() -> pd.DataFrame:
    return pd.read_csv(DATA / "train.csv")


def cargar_test() -> pd.DataFrame:
    return pd.read_csv(DATA / "test.csv")


def separar_xy(train: pd.DataFrame):
    return train.drop(columns="Survived"), train.Survived
