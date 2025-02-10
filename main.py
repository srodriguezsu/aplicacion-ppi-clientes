import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import streamlit as st

def load_data(file, url):
    if file is not None:
        return pd.read_csv(file)
    elif url:
        return pd.read_csv(url)
    return None

def process_dataframe(df):
    df.dropna(subset=['Edad', 'Ingreso_Anual_USD', 'Latitud', 'Longitud'], inplace=True)
    return df

def correlacion_edad_ingreso(df, segmentar_por=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD', hue=df[segmentar_por] if segmentar_por else None, ax=ax)
    ax.set_title(f'Correlación entre Edad e Ingreso Anual {"por " + segmentar_por if segmentar_por else ""}')
    st.pyplot(fig)

def mapa_ubicacion(df, filtro=None, valor=None):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitud, df.Latitud))
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural\
/ne_110m_admin_0_countries.zip")
    if filtro and valor:
        gdf = gdf[gdf[filtro] == valor]
    fig, ax = plt.subplots(figsize=(10, 6))
    world.plot(ax=ax, color='lightgrey')
    gdf.plot(ax=ax, markersize=5, alpha=0.5, color='red')
    ax.set_title(f'Mapa de Ubicación de Clientes {" - " + filtro + ": " + str(valor) if filtro else ""}')
    st.pyplot(fig)

def cluster_frecuencia(df):
    """
    Realiza un análisis de clúster según la frecuencia de compra.
    Convierte la variable categórica en valores numéricos antes de aplicar clustering.
    """
    # Mapeo de categorías a valores numéricos
    categoria_a_numero = {'Baja': 0, 'Media': 1, 'Alta': 2}
    df['Frecuencia_Compra_Num'] = df['Frecuencia_Compra'].map(categoria_a_numero)

    # Filtrar valores nulos después de la conversión
    data = df[['Frecuencia_Compra_Num']].dropna()

    try:
        # Aplicar clustering jerárquico
        linkage_matrix = linkage(data, method='ward')

        # Graficar dendrograma
        fig, ax = plt.subplots(figsize=(8, 6))
        dendrogram(linkage_matrix, labels=df.index, leaf_rotation=90, ax=ax)
        ax.set_title('Clúster de Frecuencia de Compra')

        # Mostrar en Streamlit
        st.pyplot(fig)
    except ValueError as e:
        st.error(f"Error en el clustering: {e}")

def grafico_barras(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    df.groupby(['Género', 'Frecuencia_Compra']).size().unstack().plot(kind='bar', ax=ax)
    ax.set_title('Distribución de Clientes por Género y Frecuencia de Compra')
    st.pyplot(fig)

def mapa_calor_ingresos(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(x=df.Longitud, y=df.Latitud, weights=df.Ingreso_Anual_USD, cmap="Reds", fill=True, ax=ax)
    ax.set_title('Mapa de Calor de Ingresos')
    st.pyplot(fig)

def distancia_altos_ingresos(df, segmentar_por=None):
    if segmentar_por:
        grupos = df.groupby(segmentar_por)
        resultados = {grupo: squareform(pdist(datos[['Latitud', 'Longitud']].dropna())) for grupo, datos in grupos}
        return resultados
    return squareform(pdist(df[['Latitud', 'Longitud']].dropna()))

st.title("Análisis de Datos de Clientes")
st.subheader("Carga de datos")
file = st.file_uploader("Sube un archivo CSV", type=["csv"])
csv_url = st.text_input("O ingresa la URL de un CSV")
df = load_data(file, csv_url)
if df is not None:
    df = process_dataframe(df)
    st.write("Vista previa de los datos:")
    st.dataframe(df)
    
    st.subheader("Mapa de Ubicación")
    filtro = st.selectbox("Filtrar mapa por", ["Ninguno", "Género", "Frecuencia_Compra"])
    valor = st.selectbox("Valor", df[filtro].unique()) if filtro != "Ninguno" else None
    mapa_ubicacion(df, filtro if filtro != "Ninguno" else None, valor)
    
    st.subheader("Correlación Edad-Ingreso")
    segmentar_por = st.sidebar.selectbox("Segmentar correlación por", ["Ninguno", "Género", "Frecuencia_Compra"])
    correlacion_edad_ingreso(df, segmentar_por if segmentar_por != "Ninguno" else None)
    
    st.subheader("Clúster de Frecuencia de Compra")
    cluster_frecuencia(df)
    
    st.subheader("Mapa de Calor de Ingresos")
    mapa_calor_ingresos(df)
    
    st.subheader("Gráfico de Barras")
    grafico_barras(df)
    
    st.subheader("Distancias entre Compradores de Altos Ingresos")
    segmentar_dist = st.sidebar.selectbox("Segmentar distancias por", ["Ninguno", "Género", "Frecuencia_Compra"])
    distancias = distancia_altos_ingresos(df, segmentar_dist if segmentar_dist != "Ninguno" else None)
    st.write(distancias)
