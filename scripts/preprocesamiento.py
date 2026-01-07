import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
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

    # 1. TARGET PEDAGÓGICO: Riesgo (1) = Niveles Insuficiente(0) y Elemental(1)
    cols_logro = [c for c in df.columns if c.startswith('nl_')]
    
    if cols_logro:
        # CORRECCIÓN: Convertimos las columnas de logro a numérico explícitamente
        # 'coerce' transforma textos no numéricos en NaN para evitar errores
        for col in cols_logro:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ahora que son números, calculamos el Riesgo (Clase 1)
        df['Target'] = df[cols_logro].apply(
            lambda row: 1 if any(val <= 1 for val in row if pd.notnull(val)) else 0, 
            axis=1
        )
    else:
        # Fallback: Usar puntaje global con umbral de 700 pts (Punto 172)
        col_puntos = next((c for c in ['p_puntuacion_calculada_global', 'inev'] if c in df.columns), None)
        if col_puntos:
            df[col_puntos] = pd.to_numeric(df[col_puntos], errors='coerce')
            df = df.dropna(subset=[col_puntos])
            df['Target'] = (df[col_puntos] < 700).astype(int)
        else:
            return None

    # 2. LIMPIEZA QUIRÚRGICA (Preservamos IDs Geográficos para Mapas de Calor)
    cols_a_eliminar = []
    # Definimos fugas de datos (Punto 176)
    fugas_data = ['codigo', 'amie', 'fex_', 'puntuacion', 'inev', 'imat', 'ilyl', 'icn', 'ies', 
                  'ihis', 'ifil', 'ied', 'ifis', 'iqui', 'ibio', 'nl_']
    
    for col in df.columns:
        if col == 'Target': continue
        # Mantenemos variables geográficas para los reportes finales [cite: 178]
        if any(key in col.lower() for key in fugas_data):
            cols_a_eliminar.append(col)
        # Eliminar strings con demasiada variedad (ruido)
        elif df[col].dtype == 'object' and df[col].nunique() > 100 and 'id_' not in col.lower():
            cols_a_eliminar.append(col)

    df = df.drop(columns=list(set(cols_a_eliminar)))

    # 3. ENCODING Y CATEGORÍAS
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    # Aseguramos que los IDs y el grado sean categóricos para OneHot [cite: 168, 179]
    for col in ['id_prov', 'id_cant', 'grado']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    df = pd.get_dummies(df, drop_first=True)
    return df.astype('float32')

# --- 2. EJECUCIÓN: CARGA Y ALINEACIÓN ---
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
        # Rescatamos metadatos para el reporte final (incluyendo IDs geográficos)
        columnas_id = ['codigo', 'amie', 'ciclo', 'grado', 'id_prov', 'id_cant']
        ids_rescatados = temp_raw.loc[procesado.index, [c for c in columnas_id if c in temp_raw.columns]]
        dfs_train_raw_ids.append(ids_rescatados)

df_train = pd.concat(dfs_train_processed, ignore_index=True).fillna(0)
df_train_raw = pd.concat(dfs_train_raw_ids, ignore_index=True)

print("Procesando prueba (2025)...")
sep_test = ';' if 'SEST25' in archivo_test else ','
df_test_raw_full = pd.read_csv(archivo_test, sep=sep_test, low_memory=False)
df_test = procesar_datos(df_test_raw_full).fillna(0)

# Rescatar metadatos de prueba
columnas_id = ['codigo', 'amie', 'ciclo', 'grado', 'id_prov', 'id_cant']
df_test_raw_ids = df_test_raw_full.loc[df_test.index, [c for c in columnas_id if c in df_test_raw_full.columns]]

# Alinear columnas entre train y test
df_train, df_test = df_train.align(df_test, join='inner', axis=1)

# --- 3. ESCALADO Y BALANCEO (SMOTE) ---
X_train = df_train.drop('Target', axis=1)
y_train = df_train['Target']
X_test = df_test.drop('Target', axis=1)
y_test = df_test['Target']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE para mejorar el Recall de la clase minoritaria (estudiantes en riesgo)
print(f"Distribución antes de SMOTE: {np.bincount(y_train.astype(int))}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"Distribución después de SMOTE: {np.bincount(y_train_res.astype(int))}")

# --- 4. PCA (15 COMPONENTES SOBRE DATOS BALANCEADOS) ---
n_comp = min(15, X_train_res.shape[1])
pca = PCA(n_components=n_comp) 
X_train_pca = pca.fit_transform(X_train_res)
X_test_pca = pca.transform(X_test_scaled)

print(f"Varianza total explicada con {n_comp} componentes: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# --- 5. GUARDAR DATASETS FINALES ---
nombres_pc = [f'PC{i+1}' for i in range(n_comp)]

# Dataset Entrenamiento Final
df_train_final = pd.DataFrame(X_train_pca, columns=nombres_pc)
# Nota: SMOTE crea filas nuevas, por lo que no podemos mapear los IDs originales 
# uno a uno fácilmente. El entrenamiento usa solo los componentes.
df_train_final['Target'] = y_train_res.values

# Dataset Prueba Final (Mantenemos IDs para validación y mapas)
df_test_final = pd.DataFrame(X_test_pca, columns=nombres_pc)
for col in ['codigo', 'amie', 'ciclo', 'grado', 'id_prov', 'id_cant']:
    if col in df_test_raw_ids.columns:
        df_test_final[col] = df_test_raw_ids[col].values
df_test_final['Target'] = y_test.values

df_train_final.to_csv('train_preprocesado_balanceado.csv', index=False)
df_test_final.to_csv('test_preprocesado_2025.csv', index=False)

print("¡Proceso completado! Los datasets están listos para el entrenamiento de los 5 modelos.")