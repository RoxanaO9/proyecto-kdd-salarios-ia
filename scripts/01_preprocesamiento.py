# FASE 1 Y 2: SELECCIÓN, PREPROCESAMIENTO Y LIMPIEZA

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Carga de Datos 
print("\n Cargando datos: ")
try:
    df = pd.read_csv('salaries.csv')
    print(f" Datos cargados correctamente. Dimensiones: {df.shape}")
except FileNotFoundError:
    print(" Error: El archivo 'salaries.csv' no se encontró en el directorio.")
    exit()

print(f"\nColumnas disponibles:\n{df.columns.tolist()}")

# 2. Preprocesamiento y Limpieza 
print("\n Realizando preprocesamiento y limpieza: ")

# 2.1 Verificar valores nulos
print(f"\nValores nulos por columna:\n{df.isnull().sum()}")

# 2.2 Verificar y eliminar duplicados
duplicates = df.duplicated().sum()
print(f"\nRegistros duplicados encontrados: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f" Registros duplicados eliminados. Nuevas dimensiones: {df.shape}")

# 2.3 Estadísticas completas y Outliers
print(f"\nEstadísticas descriptivas completas:")
print(df.describe(include='all'))

# 2.4 Detección de outliers (sin eliminación)
Q1 = df['salary_in_usd'].quantile(0.25)
Q3 = df['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['salary_in_usd'] < lower_bound) | (df['salary_in_usd'] > upper_bound)]

print(f"\nOutliers detectados en salario: {len(outliers)} registros")
print("NOTA: Se conservan los outliers por representar salarios reales de cargos ejecutivos.")
print(f"Ejemplos: {outliers['job_title'].value_counts().head(5)}")

# 3. Transformación de Datos 
print("\n Transformando datos para el modelado: ")

# 3.1 Crear copia para transformaciones
df_encoded = df.copy()

# 3.2 One-Hot Encoding para variables categóricas
categorical_features = ['job_title', 'employment_type', 'experience_level',
    'company_size', 'employee_residence', 'company_location']

print(f"Codificando variables categóricas: {categorical_features}")
for col in categorical_features:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df_encoded = pd.concat([df_encoded, dummies], axis=1)

print(f" One-Hot Encoding completado. Nuevas dimensiones: {df_encoded.shape}")

# 3.3 Crear categorías de salario para análisis
df_encoded['salary_category'] = pd.cut(df_encoded['salary_in_usd'],
    bins=[0, 70000, 120000, 200000, float('inf')],
    labels=['Bajo (<70k)', 'Medio (70-120k)',
        'Alto (120-200k)', 'Ejecutivo (>200k)'])

# 3.4 Escalar variables numéricas para modelos
scaler = StandardScaler()
numeric_features = ['salary_in_usd', 'remote_ratio', 'work_year']
df_encoded_scaled = df_encoded.copy()
df_encoded_scaled[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

print(" Variables numéricas escaladas.")

# 3.5 Guardar datos procesados
df_encoded_scaled.to_csv('salarios_procesados.csv', index=False)
print("\n Archivo 'salarios_procesados.csv' guardado.")
