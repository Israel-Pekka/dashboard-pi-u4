import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Configuración inicial de Streamlit
# ==========================================
st.set_page_config(page_title="Dashboard Agroindustrial", layout="wide")
st.title("🌱 Optimización Inferencial del Rendimiento Agrícola")
st.markdown("Dashboard interactivo para analizar la influencia del riego y fertilizante, y evaluar la viabilidad del nuevo bio-estimulante.")

# ==========================================
# FASE 1: Ingesta Determinista con DuckDB
# ==========================================
@st.cache_data
def load_data():
    # Conexión a DuckDB en memoria
    con = duckdb.connect(database=':memory:')
    
    # Fijar la semilla determinista como indica el paso 1.1
    con.execute("SELECT setseed(0.42);")
    
    # Crear la tabla simulando las condiciones exactas (Pasos 1.2 y 1.3)
    query_creacion = """
    CREATE TABLE parcelas AS
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        round(5.0 + (0.035 * volumen_riego) + (0.020 * fertilizante_kg) + 
              (CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END) + 
              (-0.8 + (random() * 1.6)), 2) AS rendimiento_ton
    FROM (
        SELECT 
            (range + 1) AS parcela_id,
            round(100.0 + (random() * 200.0), 2) AS volumen_riego,
            round(50.0 + (random() * 100.0), 2) AS fertilizante_kg,
            CASE WHEN random() < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento
        FROM range(100)
    );
    """
    con.execute(query_creacion)
    
    # Extraer a pandas DataFrame
    df = con.execute("SELECT * FROM parcelas").df()
    con.close()
    return df

df = load_data()

# Filtro lateral interactivo (Paso 4.2 y Fase 5)
st.sidebar.header("Filtros Dinámicos")
tratamiento_filtro = st.sidebar.multiselect(
    "Selecciona el Tipo de Tratamiento:",
    options=df["tipo_tratamiento"].unique(),
    default=df["tipo_tratamiento"].unique()
)

df_filtrado = df[df["tipo_tratamiento"].isin(tratamiento_filtro)]

if df_filtrado.empty:
    st.warning("Selecciona al menos un tratamiento en el menú lateral.")
    st.stop()

# ==========================================
# FASE 2: Correlación y Regresión (OLS)
# ==========================================
# Definir variables independientes (X) y dependiente (Y)
X = df_filtrado[['volumen_riego', 'fertilizante_kg']]
Y = df_filtrado['rendimiento_ton']
X_con_constante = sm.add_constant(X) # Agregar intercepto (Beta 0)

# Ajustar Modelo OLS
modelo_ols = sm.OLS(Y, X_con_constante).fit()
r2_ajustado = modelo_ols.rsquared_adj
residuos = modelo_ols.resid

# ==========================================
# FASE 3: Prueba de Hipótesis Inferencial
# ==========================================
df_A = df[df['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
df_B = df[df['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']

# t-Student de dos muestras independientes
t_stat, p_value = stats.ttest_ind(df_B, df_A, alternative='greater')

# ==========================================
# FASE 5: Tarjetas KPI de Resumen
# ==========================================
st.markdown("### 📊 Indicadores Clave de Rendimiento (KPIs)")
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric("Rendimiento Promedio Global", f"{df_filtrado['rendimiento_ton'].mean():.2f} ton/ha")
kpi2.metric("R² Ajustado del Modelo", f"{r2_ajustado * 100:.2f}%", help="Porcentaje de variabilidad explicada por riego y fertilizante.")
kpi3.metric("P-Valor (Prueba Tratamiento B)", f"{p_value:.4f}", 
            delta="Significativo (Adopción viable)" if p_value < 0.05 else "No significativo", delta_color="normal" if p_value < 0.05 else "inverse")

st.divider()

# ==========================================
# FASE 4: Visualizaciones del Dashboard
# ==========================================
# Cuadrícula 2x2
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Gráfica 1: Mapa de Calor de Correlación
with col1:
    st.subheader("1. Mapa de Calor de Correlación")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    # Seleccionamos solo las variables numéricas
    vars_numericas = df_filtrado[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']]
    sns.heatmap(vars_numericas.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax1, vmin=-1, vmax=1)
    st.pyplot(fig1)

# Gráfica 2: Dispersión con Línea OLS
with col2:
    st.subheader("2. Regresión: Riego vs Rendimiento")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.regplot(data=df_filtrado, x='volumen_riego', y='rendimiento_ton', 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax2)
    ax2.set_xlabel("Volumen de Riego (m³/ha)")
    ax2.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig2)

# Gráfica 3: Diagrama de Cajas (Tratamientos)
with col3:
    st.subheader("3. Comparativo de Tratamientos")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_filtrado, x='tipo_tratamiento', y='rendimiento_ton', palette="Set2", ax=ax3)
    ax3.set_xlabel("Tipo de Tratamiento")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig3)

# Gráfica 4: Histograma de Residuos
with col4:
    st.subheader("4. Diagnóstico de Errores (Residuos)")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(residuos, kde=True, color="purple", ax=ax4)
    ax4.set_xlabel("Error (Real - Predicho)")
    ax4.set_ylabel("Frecuencia")
    st.pyplot(fig4)

# Mostrar tabla cruda opcionalmente (Para auditoría de la fase 1)
with st.expander("Ver Datos de Telemetría (DuckDB)"):
    st.dataframe(df_filtrado)