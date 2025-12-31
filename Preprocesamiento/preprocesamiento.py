import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
dir_script = os.path.dirname(os.path.abspath(__file__))
dir_proyecto = os.path.dirname(dir_script) 

def r(nombre):
    return os.path.join(dir_proyecto, nombre)

archivos_train = [
    r('ineval_serestudiante2018_2019_2021noviembre.csv'),
    r('ineval_serestudiante2020_2021_2023diciembre.csv'),
    r('ineval_serestudiante2022_2023_2023diciembre.csv'),
    r('SEST24_Micro_50545_20241216_CSV.csv')
]
archivo_test = r('SEST25_micro_50578_20251215_CSV.csv')

def procesar_datos(df):
    if df is None or df.empty: return None
    df = df.copy()

    # 1. Identificar columna de puntaje
    posibles_puntos = ['p_puntuacion_calculada_global', 'inev']
    col_puntos = next((c for c in posibles_puntos if c in df.columns), None)
    if col_puntos is None: return None

    # 2. Crear Target y limpiar Nulos
    df[col_puntos] = pd.to_numeric(df[col_puntos], errors='coerce')
    df = df.dropna(subset=[col_puntos])
    umbral = df[col_puntos].median()
    df['Target'] = (df[col_puntos] >= umbral).astype(int)

    # 3. ELIMINACIÓN AGRESIVA
    cols_a_eliminar = []
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > 50:
            cols_a_eliminar.append(col)
        elif any(key in col.lower() for key in ['id_', 'codigo', 'amie', 'fex_', 'puntuacion', 'inev', 'imat', 'ilyl', 'icn', 'ies', 'ihis', 'ifil', 'ied', 'ifis', 'iqui', 'ibio']):
            cols_a_eliminar.append(col)

    cols_a_eliminar = [c for c in set(cols_a_eliminar) if c != 'Target' and c in df.columns]
    df = df.drop(columns=cols_a_eliminar)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    df = pd.get_dummies(df, drop_first=True)
    return df.astype('float32')

# --- 2. EJECUCIÓN: ENTRENAMIENTO ---
print("Cargando y procesando entrenamiento (2018-2024)...")
dfs_train_processed = []
dfs_train_raw_ids = []

for f in archivos_train:
    sep = ';' if 'ineval' in f or 'SEST25' in f else ','
    temp_raw = pd.read_csv(f, sep=sep, low_memory=False)
    if len(temp_raw.columns) < 2:
        temp_raw = pd.read_csv(f, sep=',' if sep==';' else ';', low_memory=False)
    
    procesado = procesar_datos(temp_raw)
    if procesado is not None:
        dfs_train_processed.append(procesado)
        columnas_id = ['codigo', 'amie', 'ciclo', 'grado']
        ids_rescatados = temp_raw.loc[procesado.index, [c for c in columnas_id if c in temp_raw.columns]]
        dfs_train_raw_ids.append(ids_rescatados)

df_train = pd.concat(dfs_train_processed, ignore_index=True).fillna(0)
df_train_raw = pd.concat(dfs_train_raw_ids, ignore_index=True)

# --- 3. EJECUCIÓN: PRUEBA (2025) ---
print("Procesando prueba (2025)...")
sep_test = ';' if 'SEST25' in archivo_test else ','
df_test_raw_full = pd.read_csv(archivo_test, sep=sep_test, low_memory=False)
if len(df_test_raw_full.columns) < 2:
    df_test_raw_full = pd.read_csv(archivo_test, sep=',', low_memory=False)

df_test = procesar_datos(df_test_raw_full).fillna(0)

# Rescatar IDs de prueba usando el índice de 'df_test' para que coincidan las longitudes
columnas_id = ['codigo', 'amie', 'ciclo', 'grado']
df_test_raw_ids = df_test_raw_full.loc[df_test.index, [c for c in columnas_id if c in df_test_raw_full.columns]]

# Alineación de columnas
df_train, df_test = df_train.align(df_test, join='inner', axis=1)

# --- 4. PCA Y ESCALADO ---
X_train = df_train.drop('Target', axis=1)
y_train = df_train['Target']
X_test = df_test.drop('Target', axis=1)
y_test = df_test['Target']

scaler = StandardScaler()
X_train_pca = PCA(n_components=6).fit_transform(scaler.fit_transform(X_train))
X_test_pca = PCA(n_components=6).fit_transform(scaler.transform(X_test))

# --- 5. GUARDAR RESULTADOS FINALES ---
nombres_pc = [f'PC{i+1}' for i in range(6)]

# Dataset Entrenamiento
df_train_final = pd.DataFrame(X_train_pca, columns=nombres_pc)
for col in ['codigo', 'amie', 'ciclo', 'grado']:
    if col in df_train_raw.columns:
        df_train_final[col] = df_train_raw[col].values
df_train_final['Target'] = y_train.values

# Dataset Prueba (CORREGIDO)
df_test_final = pd.DataFrame(X_test_pca, columns=nombres_pc)
for col in ['codigo', 'amie', 'ciclo', 'grado']:
    if col in df_test_raw_ids.columns:
        # Ahora usamos 'df_test_raw_ids' que ya tiene la longitud correcta (47308)
        df_test_final[col] = df_test_raw_ids[col].values
df_test_final['Target'] = y_test.values

df_train_final.to_csv('train_18_24_pca_final.csv', index=False)
df_test_final.to_csv('test_25_pca_final.csv', index=False)

print(f"¡Éxito total! Archivos generados.")
print(f"Entrenamiento: {df_train_final.shape} | Prueba: {df_test_final.shape}")