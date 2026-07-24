# 📊 Análisis de Salarios en el Sector de Inteligencia Artificial

## 📌 Descripción del Proyecto

Este proyecto aplica la metodología **KDD (Knowledge Discovery in Databases)** para analizar un conjunto de datos de salarios de profesionales del sector de Inteligencia Artificial, Machine Learning y Ciencia de Datos a nivel mundial.

El objetivo principal es identificar patrones, factores influyentes en la remuneración y desarrollar modelos predictivos que permitan estimar salarios en función de variables como nivel de experiencia, país de residencia, modalidad de trabajo y tamaño de la empresa.

---

## 🎯 Objetivos

### Objetivo General
Aplicar la metodología KDD para analizar un conjunto de datos de salarios del sector de IA, identificando patrones y desarrollando modelos predictivos.

### Objetivos Específicos
- Seleccionar y comprender un conjunto de datos real de salarios en IA
- Realizar preprocesamiento y limpieza de datos
- Transformar variables categóricas mediante codificación
- Aplicar análisis exploratorio con visualizaciones estadísticas
- Construir modelos predictivos (Regresión Lineal, Árbol de Decisión, Random Forest)
- Implementar clustering con K-Means para identificar perfiles
- Interpretar resultados y generar conclusiones útiles

---

## 🛠️ Tecnologías Utilizadas

| Herramienta | Propósito |
|-------------|-----------|
| **Python 3.10+** | Lenguaje de programación principal |
| **Pandas** | Manipulación y limpieza de datos |
| **NumPy** | Operaciones numéricas y cálculos |
| **Matplotlib** | Visualizaciones y gráficos |
| **Seaborn** | Visualizaciones estadísticas avanzadas |
| **Scikit-learn** | Modelado predictivo y clustering |

### Librerías Específicas
- `StandardScaler` – Escalado de variables numéricas
- `OneHotEncoder` – Codificación de variables categóricas
- `LinearRegression` – Modelo de regresión lineal
- `DecisionTreeRegressor` – Árbol de decisión
- `RandomForestRegressor` – Random Forest
- `KMeans` – Algoritmo de clustering
- `PCA` – Análisis de Componentes Principales

---

## 📁 Estructura del Proyecto
proyecto-kdd-salarios-ia/
│
├── 📄 README.md # Documentación principal
├── 📄 requirements.txt # Dependencias del proyecto
│
├── 📁 notebooks/
│ └── proyecto_kdd_salarios.ipynb # Notebook principal
│
├── 📁 data/
│ └── salaries.csv # Dataset utilizado
│
├── 📁 scripts/
│ ├── 01_preprocesamiento.py # Limpieza y transformación
│ ├── 02_analisis_exploratorio.py # EDA y visualizaciones
│ ├── 03_modelos_predictivos.py # Modelos de regresión
│ └── 04_clustering.py # K-Means y PCA
│
├── 📁 results/
│ ├── 📁 images/
│ │ ├── correlacion.png
│ │ ├── analisis_salarios_completo.png
│ │ ├── metodo_codo_mejorado.png
│ │ └── clustering_mejorado.png
│ ├── resultados_modelos.csv
│ └── cluster_analysis.csv
│
└── 📁 docs/
└── informe_actividad_formativa.docx


## 📊 Metodología KDD Aplicada

### Fase 1: Selección de Datos
- Dataset: `salaries.csv`
- 73,148 registros iniciales, 11 variables
- Variables: experiencia, tipo de empleo, cargo, salario, país, modalidad remota, tamaño de empresa

### Fase 2: Preprocesamiento y Limpieza
- Verificación de valores nulos: **0 valores faltantes**
- Eliminación de duplicados: **39,124 registros eliminados**
- Dataset final: **34,024 registros únicos**
- Detección de outliers: **755 registros identificados y conservados**

### Fase 3: Transformación de Datos
- One-Hot Encoding para variables categóricas
- Creación de variable `salary_category`
- Escalado de variables numéricas con StandardScaler

### Fase 4: Minería de Datos
- **Análisis Exploratorio:** 9 visualizaciones estadísticas
- **Modelado Predictivo:** 3 modelos de regresión
- **Clustering:** K-Means con 4 clústeres

### Fase 5: Interpretación
- Análisis de resultados y generación de conclusiones

---

## 📈 Resultados Principales

### Modelos Predictivos

| Modelo | R² | MAE | RMSE |
|--------|-----|-----|------|
| Regresión Lineal | 0.2751 | $47,118.38 | $64,243.59 |
| Árbol de Decisión | 0.2512 | $47,729.19 | $65,292.11 |
| Random Forest | 0.2594 | $47,420.65 | $64,935.83 |

### Clústeres Identificados

| Cluster | Salario Promedio | Remote Ratio | Tamaño | % |
|---------|-----------------|--------------|--------|---|
| 0 | $155,525 | 27.18% | 21,907 | 64.4% |
| 1 | $137,672 | 22.94% | 10,864 | 31.9% |
| 2 | $87,812 | 65.52% | 203 | 0.6% |
| 3 | $199,993 | 23.62% | 1,050 | 3.1% |

### Factores Clave Identificados

1. **País de residencia** – EE.UU. presenta los salarios más altos
2. **Nivel de experiencia** – Senior y Executive tienen mayor remuneración
3. **Título del puesto** – Roles de liderazgo y especialización técnica
4. **Tamaño de empresa** – Empresas medianas (M) pagan mejor en promedio
