import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, json, joblib
from datetime import datetime

# =========================
# RUTAS
# =========================
archivos_train = [
    "ineval_serestudiante2018_2019_2021noviembre.csv",
    "ineval_serestudiante2020_2021_2023diciembre.csv",
    "ineval_serestudiante2022_2023_2023diciembre.csv",
    "SEST24_Micro_50545_20241216_CSV.csv"
]
archivo_test = "SEST25_micro_50578_20251215_CSV.csv"

ID_COLS = ["codigo", "amie", "ciclo", "grado", "id_prov", "id_cant"]

# =========================
# LECTURA ROBUSTA
# =========================
def read_csv_smart(path):
    for sep in [";", ",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                return df, sep
        except Exception:
            pass
    raise ValueError(f"No pude leer {path} con separadores comunes (; , tab |).")

# =========================
# 1) Construir stats nl_* por fila
# =========================
def nl_stats(df, excluir_nl_inev=True):
    nl_cols = [c for c in df.columns if c.startswith("nl_")]
    if excluir_nl_inev and "nl_inev" in nl_cols:
        nl_cols.remove("nl_inev")

    if len(nl_cols) == 0:
        raise ValueError("No encontré columnas nl_* para etiquetar.")

    tmp = df[nl_cols].copy()
    for c in nl_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    valid = tmp.notna().sum(axis=1)                           # # áreas disponibles (4 o 8)
    n0 = (tmp.eq(0)).sum(axis=1)                              # # insuficientes
    low = ((tmp.le(1)) & tmp.notna()).sum(axis=1)             # # bajas (0 o 1)
    low_prop = (low / valid.replace(0, np.nan)).fillna(0.0)   # proporción bajas

    return n0, low_prop, valid

# =========================
# 2) Calibrar umbral para que Train ≈ 30% riesgo
#    Riesgo = (n0>=1) OR (low_prop >= t)
# =========================
def calibrar_umbral_por_prevalencia(n0, low_prop, target_rate=0.30):
    base = (n0 >= 1).mean()  # ya etiquetados por tener al menos un 0

    # Si solo con "cualquier 0" ya pasas el target, no añadimos la regla por proporción
    if base >= target_rate:
        return 1.01, base  # t>1 => low_prop nunca activa, solo manda n0>=1

    # Necesitamos agregar más riesgo desde los que NO tienen 0
    mask_no0 = (n0 == 0)
    lp = low_prop[mask_no0].replace([np.inf, -np.inf], np.nan).dropna()

    # fracción adicional requerida dentro del grupo no0
    p_add = (target_rate - base) / (1 - base)  # entre 0 y 1
    p_add = float(np.clip(p_add, 0.0, 1.0))

    # Queremos que (low_prop >= t) ocurra en aproximadamente p_add del grupo no0
    # => t es el cuantíl (1 - p_add)
    q = 1 - p_add
    t = float(lp.quantile(q))

    achieved = ((n0 >= 1) | (low_prop >= t)).mean()
    return t, achieved

# =========================
# 3) Preprocesar features (sin nl_* y sin fugas)
# =========================
def preprocesar_features(df):
    df = df.copy()

    # eliminar columnas de fuga + nl_*
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

    # categóricas -> dummies
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")
    for col in ["id_prov", "id_cant", "grado"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df = pd.get_dummies(df, drop_first=True)
    df = df.fillna(0)
    return df.astype("float32")

# =========================
# Cargar y concatenar TRAIN/TEST crudos
# =========================
raw_trains = []
train_ids = []

print("Cargando TRAIN...")
for f in archivos_train:
    df, sep = read_csv_smart(f)
    raw_trains.append(df)
    train_ids.append(df[[c for c in ID_COLS if c in df.columns]].copy())

raw_train = pd.concat(raw_trains, ignore_index=True)
train_ids = pd.concat(train_ids, ignore_index=True)

print("Cargando TEST...")
raw_test, sep_test = read_csv_smart(archivo_test)
test_ids = raw_test[[c for c in ID_COLS if c in raw_test.columns]].copy()

# =========================
# Etiquetado con regla realista
# =========================
TARGET_RATE = 0.30

n0_train, low_prop_train, valid_train = nl_stats(raw_train, excluir_nl_inev=True)
t, achieved_train = calibrar_umbral_por_prevalencia(n0_train, low_prop_train, target_rate=TARGET_RATE)

raw_train["Target"] = ((n0_train >= 1) | (low_prop_train >= t)).astype(int)

n0_test, low_prop_test, valid_test = nl_stats(raw_test, excluir_nl_inev=True)
raw_test["Target"] = ((n0_test >= 1) | (low_prop_test >= t)).astype(int)

print("\n=== Etiquetado ===")
print(f"Umbral low_prop (train-calibrado): t = {t:.4f}")
print("Riesgo train (esperado ~30%):", raw_train["Target"].mean())
print("Riesgo test  :", raw_test["Target"].mean())

# =========================
# Preprocesamiento features (sin SMOTE)
# =========================
df_train = preprocesar_features(raw_train)
df_test  = preprocesar_features(raw_test)

y_train = df_train["Target"].astype(int)
X_train = df_train.drop(columns=["Target"])

y_test = df_test["Target"].astype(int)
X_test = df_test.drop(columns=["Target"])

# alinear columnas train -> test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nShapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

# =========================
# Escalado + PCA 95% varianza
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

k = X_train_pca.shape[1]
pc_cols = [f"PC{i+1}" for i in range(k)]

print(f"\n✅ PCA eligió {k} componentes para {pca.explained_variance_ratio_.sum()*100:.2f}% varianza")

# =========================
# Guardar ARTEFACTO de PREPROCESAMIENTO (scaler + pca + columnas)
# =========================
os.makedirs("artifacts", exist_ok=True)

pc_cols = [f"PC{i+1}" for i in range(k)]  # ya lo tienes abajo, aquí lo usamos también

preprocess_bundle = {
    # para reconstruir X exactamente igual (después de get_dummies)
    "model_input_cols": list(X_train.columns),

    # transformadores fit
    "scaler": scaler,
    "pca": pca,

    # nombres de salida PCA
    "pc_cols": pc_cols,

    # config útil (reproducibilidad)
    "id_cols": ID_COLS,
    "target_rate": TARGET_RATE,
    "low_prop_threshold_t": float(t),
    "exclude_nl_inev": True,
    "created_at": datetime.now().isoformat(),
}

joblib.dump(preprocess_bundle, "artifacts/preprocess_bundle.pkl", compress=3)

# metadata liviana (opcional, no pesa y ayuda a tus compañeros)
meta = {
    "model_input_cols_count": len(preprocess_bundle["model_input_cols"]),
    "pca_components_k": k,
    "pca_explained_var_pct": float(pca.explained_variance_ratio_.sum() * 100),
    "target_rate": TARGET_RATE,
    "low_prop_threshold_t": float(t),
    "created_at": preprocess_bundle["created_at"],
}
with open("artifacts/preprocess_bundle.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n✅ Artefacto guardado:")
print(" - artifacts/preprocess_bundle.pkl")
print(" - artifacts/preprocess_bundle.json")

X_test_pca  = pca.transform(X_test_scaled)


print(f"\n✅ PCA eligió {k} componentes para {pca.explained_variance_ratio_.sum()*100:.2f}% varianza")

# =========================
# Guardar datasets finales
# =========================

train_out = pd.DataFrame(X_train_pca, columns=pc_cols)
for c in train_ids.columns:
    train_out[c] = train_ids[c].values
train_out["Target"] = y_train.values

test_out = pd.DataFrame(X_test_pca, columns=pc_cols)
for c in test_ids.columns:
    test_out[c] = test_ids[c].values
test_out["Target"] = y_test.values

train_out.to_csv("train_18_24_pca95_target30_NO_SMOTE.csv", index=False)
test_out.to_csv("test_25_pca95_target30_NO_SMOTE.csv", index=False)

print("\n✅ Listo:")
print(" - train_18_24_pca95_target30_NO_SMOTE.csv")
print(" - test_25_pca95_target30_NO_SMOTE.csv")
