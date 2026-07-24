# FASE 3 Y 4.1 / 4.2: ANÁLISIS EXPLORATORIO Y VISUALIZACIÓN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# 1. Carga de Datos Procesados 
print("\n Cargando datos procesados: ")
try:
    df = pd.read_csv('salarios_procesados.csv')
    print(f" Datos procesados cargados. Dimensiones: {df.shape}")
except FileNotFoundError:
    print(" Error: No se encontró 'salarios_procesados.csv'. Ejecute '01_preprocesamiento.py' primero.")
    exit()

# 2. Matriz de Correlación 
print("\n Generando Matriz de Correlación: ")

# Seleccionar columnas numéricas relevantes (incluyendo algunas codificadas)
numeric_cols = ['salary_in_usd', 'remote_ratio', 'work_year']
df_numeric = df[numeric_cols].copy()

# Agregar variables codificadas relevantes
label_enc = LabelEncoder()
for col in ['experience_level', 'company_size', 'employment_type']:
    # Asegurarse de que la columna existe antes de codificar
    if col in df.columns:
        df_numeric[col + '_enc'] = label_enc.fit_transform(df[col].astype(str))

plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matriz de Correlación - Variables Numéricas')
plt.tight_layout()
plt.savefig('correlacion.png', dpi=300, bbox_inches='tight')
plt.show()
print(" Matriz de correlación guardada como 'correlacion.png'")

# 3. Gráficos Avanzados 
print("\n Generando Gráficos Avanzados: ")
fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 1. Distribución de salarios
ax1 = axes[0, 0]
df['salary_in_usd'].hist(bins=50, ax=ax1, color='steelblue', edgecolor='black')
ax1.axvline(df['salary_in_usd'].mean(), color='red', linestyle='--', label=f'Media: ${df["salary_in_usd"].mean():,.0f}')
ax1.axvline(df['salary_in_usd'].median(), color='green', linestyle='--', label=f'Mediana: ${df["salary_in_usd"].median():,.0f}')
ax1.set_title('Distribución de Salarios (USD)')
ax1.set_xlabel('Salario (USD)')
ax1.set_ylabel('Frecuencia')
ax1.legend()

# 2. Salario por nivel de experiencia
ax2 = axes[0, 1]
order = ['EN', 'MI', 'SE', 'EX']
sns.boxplot(x='experience_level', y='salary_in_usd', data=df, order=order, ax=ax2, palette='Set2')
ax2.set_title('Salario por Nivel de Experiencia')
ax2.set_xlabel('Nivel de Experiencia')
ax2.set_ylabel('Salario (USD)')

# 3. Top 15 trabajos mejor pagados
ax3 = axes[0, 2]
top_jobs = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(15)
top_jobs.plot(kind='barh', ax=ax3, color='coral')
ax3.set_title('Top 15 Trabajos Mejor Pagados')
ax3.set_xlabel('Salario Promedio (USD)')
ax3.set_ylabel('Título del Trabajo')

# 4. Salario por país (top 15)
ax4 = axes[1, 0]
top_countries = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).head(15)
top_countries.plot(kind='barh', ax=ax4, color='lightgreen')
ax4.set_title('Salario Promedio por País (Top 15)')
ax4.set_xlabel('Salario Promedio (USD)')
ax4.set_ylabel('País')

# 5. Salario por tamaño de empresa
ax5 = axes[1, 1]
sns.boxplot(x='company_size', y='salary_in_usd', data=df, ax=ax5, palette='Set3')
ax5.set_title('Salario por Tamaño de Empresa')
ax5.set_xlabel('Tamaño de Empresa')
ax5.set_ylabel('Salario (USD)')

# 6. Evolución temporal de salarios
ax6 = axes[1, 2]
yearly_stats = df.groupby('work_year')['salary_in_usd'].agg(['mean', 'median', 'std']).reset_index()
ax6.plot(yearly_stats['work_year'], yearly_stats['mean'], marker='o', linewidth=2, label='Media')
ax6.plot(yearly_stats['work_year'], yearly_stats['median'], marker='s', linewidth=2, label='Mediana')
ax6.fill_between(yearly_stats['work_year'],
    yearly_stats['mean'] - yearly_stats['std'],
    yearly_stats['mean'] + yearly_stats['std'],
    alpha=0.2, label='±1 Desviación')
ax6.set_title('Evolución del Salario por Año')
ax6.set_xlabel('Año')
ax6.set_ylabel('Salario (USD)')
ax6.legend()

# 7. Relación remoto vs salario
ax7 = axes[2, 0]
df.boxplot(column='salary_in_usd', by='remote_ratio', ax=ax7)
ax7.set_title('Salario vs Modalidad Remota')
ax7.set_xlabel('Porcentaje Remoto')
ax7.set_ylabel('Salario (USD)')

# 8. Top 5 trabajos por nivel de experiencia
ax8 = axes[2, 1]
exp_titles = df.groupby(['experience_level', 'job_title'])['salary_in_usd'].mean().reset_index()
top_exp = exp_titles.sort_values(['experience_level', 'salary_in_usd'], ascending=[True, False])
top_exp = top_exp.groupby('experience_level').head(5)
sns.barplot(x='salary_in_usd', y='job_title', hue='experience_level', data=top_exp, ax=ax8)
ax8.set_title('Top 5 Trabajos por Nivel de Experiencia')
ax8.set_xlabel('Salario Promedio (USD)')
ax8.set_ylabel('Título del Trabajo')
ax8.legend(title='Nivel')

# 9. Distribución por categoría de salario
ax9 = axes[2, 2]
df['salary_category'].value_counts().plot(kind='pie', ax=ax9, autopct='%1.1f%%', explode=[0, 0.05, 0.1, 0.15])
ax9.set_title('Distribución por Categoría Salarial')
ax9.set_ylabel('')

plt.tight_layout()
plt.savefig('analisis_salarios_completo.png', dpi=300, bbox_inches='tight')
plt.show()

print(" Gráficos completos guardados como 'analisis_salarios_completo.png'")
