import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# -------------------------
# CSV robusto
# -------------------------
def read_csv_smart(path: str):
    seps = [";", ",", "\t", "|"]
    encs = ["utf-8", "latin1", "cp1252"]

    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1] > 1:
                    return df, sep, enc
            except Exception:
                pass

    # fallback: autodetect + skip bad lines
    df = pd.read_csv(path, sep=None, encoding="latin1", engine="python", on_bad_lines="skip")
    return df, None, "latin1"


# -------------------------
# Tu preprocesamiento (idéntico al training)
# -------------------------
def preprocesar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols_a_eliminar = []
    fugas_data = [
        "codigo", "amie", "fex_", "puntuacion", "inev", "imat", "ilyl", "icn", "ies",
        "ihis", "ifil", "ied", "ifis", "iqui", "ibio", "nl_"
    ]

    for col in df.columns:
        if col == "Target":
            continue
        if any(key in col.lower() for key in fugas_data):
            cols_a_eliminar.append(col)
        elif df[col].dtype == "object" and df[col].nunique(dropna=True) > 100 and "id_" not in col.lower():
            cols_a_eliminar.append(col)

    df = df.drop(columns=list(set(cols_a_eliminar)), errors="ignore")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")
    for col in ["id_prov", "id_cant", "grado"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df = pd.get_dummies(df, drop_first=True)
    df = df.fillna(0)
    return df.astype("float32")


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones y heatmaps (prov/cant) desde un SEST CSV.")
    parser.add_argument("--input", required=True, help="Ruta al CSV SEST (crudo).")
    parser.add_argument("--prefix", default="SEST", help="Prefijo para archivos de salida.")
    parser.add_argument("--artifacts", default="artifacts", help="Carpeta con preprocess_bundle.pkl y best_model_bundle.pkl")
    parser.add_argument("--outdir", default="outputs", help="Carpeta de salida")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prep_pkl = artifacts_dir / "preprocess_bundle.pkl"
    model_pkl = artifacts_dir / "best_model_bundle.pkl"

    # 1) Cargar artefactos
    prep = joblib.load(prep_pkl)
    mb = joblib.load(model_pkl)
    model = mb["model"]
    thr = float(mb["threshold"])

    print("✅ Artefactos cargados")
    print(" - threshold:", thr)
    print(" - PCA k:", len(prep["pc_cols"]))

    # 2) Leer SEST robusto
    df_raw, sep, enc = read_csv_smart(args.input)
    print(f"✅ CSV leído (sep={sep}, enc={enc}) shape={df_raw.shape}")

    # 3) IDs para mapa (si existen)
    if "id_prov" in df_raw.columns:
        df_raw["CODPRO"] = df_raw["id_prov"].astype(str).str.zfill(2)
    if "id_cant" in df_raw.columns:
        df_raw["DPA_CANTON"] = df_raw["id_cant"].astype(str).str.zfill(4)

    # 4) Preprocesar + alinear columnas
    X = preprocesar_features(df_raw)
    X = X.reindex(columns=prep["model_input_cols"], fill_value=0)

    # 5) scaler + pca
    X_scaled = prep["scaler"].transform(X)
    X_pca = prep["pca"].t_
