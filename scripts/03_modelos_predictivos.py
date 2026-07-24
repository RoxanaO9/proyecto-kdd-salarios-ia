# FASE 4.3: MODELADO PREDICTIVO

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Carga de Datos Procesados 
print("\n Cargando datos procesados: ")
try:
    df_encoded = pd.read_csv('salarios_procesados.csv')
    print(f" Datos procesados cargados. Dimensiones: {df_encoded.shape}")
except FileNotFoundError:
    print(" Error: No se encontró 'salarios_procesados.csv'. Ejecute '01_preprocesamiento.py' primero.")
    exit()

# 2. Preparación de Datos 
print("\n Preparando datos para los modelos: ")

# 2.1 Seleccionar características
feature_cols = [col for col in df_encoded.columns if col.startswith('experience_level_') or
    col.startswith('employment_type_') or
    col.startswith('company_size_') or
    col.startswith('job_title_') or
    col.startswith('employee_residence_') or
    col.startswith('company_location_')]

# Incluir también variables numéricas (que ya fueron escaladas)
feature_cols += ['remote_ratio', 'work_year']

# Asegurar que todas existen
feature_cols = [col for col in feature_cols if col in df_encoded.columns]

X = df_encoded[feature_cols]
y = df_encoded['salary_in_usd'] # Usamos el salario original, no el escalado

# 2.2 Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f" Tamaño de entrenamiento: {len(X_train)}")
print(f" Tamaño de prueba: {len(X_test)}")

# 3. Entrenamiento y Evaluación de Modelos 
print("\n Entrenando y evaluando modelos: ")

# Modelo 1: Regresión Lineal
print("\n--- MODELO 1: REGRESIÓN LINEAL ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred_lr):,.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_lr)):,.2f}")

# Modelo 2: Árbol de Decisión
print("\n--- MODELO 2: ÁRBOL DE DECISIÓN ---")
dt = DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred_dt):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred_dt):,.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_dt)):,.2f}")

# Modelo 3: Random Forest
print("\n--- MODELO 3: RANDOM FOREST ---")
rf = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred_rf):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, y_pred_rf):,.2f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_rf)):,.2f}")

# 4. Comparación de Modelos
print("\n Comparando Modelos: ")
model_comparison = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Árbol de Decisión', 'Random Forest'],
    'R²': [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_dt), r2_score(y_test, y_pred_rf)],
    'MAE': [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_dt), mean_absolute_error(y_test, y_pred_rf)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_lr)), np.sqrt(mean_squared_error(y_test, y_pred_dt)), np.sqrt(mean_squared_error(y_test, y_pred_rf))]
})
print(model_comparison.round(2))

# 5. Importancia de Características (Random Forest) 
print("\n TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES (Random Forest): ")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)
print(feature_importance)
