import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import streamlit as st


def process_dataframe(df):
    """
    Procesa un DataFrame rellenando valores nulos en columnas numéricas con la media
    y en columnas categóricas con la moda.

    Parámetros:
        df: DataFrame original con valores nulos.

    Retorna:
        DataFrame procesado con los valores interpolados.
    """
    df_og = df.copy()

    # Iterar sobre las columnas del DataFrame
    for column in df.columns:
        if df[column].dtype == 'object':  # Si la columna es categórica
            # Llenar valores nulos con la moda
            mode_value = df[column].mode().iloc[0]
            df[column] = df[column].fillna(mode_value)
        else:  # Si la columna es numérica
            # Llenar valores nulos con la media
            mean_value = df[column].mean()
            df[column] = df[column].fillna(mean_value)

    return df


def load_data(file, url):
    if file is not None:
        return pd.read_csv(file)
    elif url:
        return pd.read_csv(url)
    return None

def correlacion_edad_ingreso(df, segmentar_por=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD', hue=df[segmentar_por] if segmentar_por else None, ax=ax)
    ax.set_title(f'Correlación entre Edad e Ingreso Anual {"por " + segmentar_por if segmentar_por else ""}')
    st.pyplot(fig)

def mapa_ubicacion(df, filtro=None, valor=None):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitud, df.Latitud))
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural\
/ne_110m_admin_0_countries.zip")
    world = world[world["CONTINENT"] == "South America"]

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
    """
    Genera un mapa de calor mostrando la relación entre ingresos y ubicación geográfica 
    sobre un mapa de Sudamérica utilizando Geopandas.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame con 'Longitud', 'Latitud' e 'Ingreso_Anual_USD'.
    """
    
    # Cargar el mapa de Sudamérica usando geopandas
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
    world = world[world["CONTINENT"] == "South America"]
    
    # Crear un GeoDataFrame con las coordenadas de los clientes
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitud, df.Latitud))

    # Establecer un sistema de coordenadas proyectadas (puedes elegir otro si es necesario)
    gdf = gdf.set_crs("EPSG:4326")  # WGS84 (lat/lon)

    # Crear una figura para la visualización
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Dibujar el mapa base de Sudamérica
    world.plot(ax=ax, color='lightgray')

    # Añadir el mapa de calor (kdeplot) sobre el mapa de Sudamérica
    sns.kdeplot(
        x=gdf.Longitud, y=gdf.Latitud, 
        weights=gdf.Ingreso_Anual_USD, cmap="Reds", fill=True, 
        ax=ax, alpha=0.6
    )

    ax.set_title('Mapa de Calor de Ingresos sobre Sudamérica')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    
    # Mostrar el gráfico en Streamlit
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
    segmentar_por = st.selectbox("Segmentar correlación por", ["Ninguno", "Género", "Frecuencia_Compra"])
    correlacion_edad_ingreso(df, segmentar_por if segmentar_por != "Ninguno" else None)
    
    st.subheader("Clúster de Frecuencia de Compra")
    cluster_frecuencia(df)
    
    st.subheader("Mapa de Calor de Ingresos")
    mapa_calor_ingresos(df)
    
    st.subheader("Gráfico de Barras")
    grafico_barras(df)
    
    st.subheader("Distancias entre Compradores de Altos Ingresos")
    segmentar_dist = st.selectbox("Segmentar distancias por", ["Ninguno", "Género", "Frecuencia_Compra"])
    distancias = distancia_altos_ingresos(df, segmentar_dist if segmentar_dist != "Ninguno" else None)
    st.write(distancias)
