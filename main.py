import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import streamlit as st

def load_geodata():
    """Carga los datos de límites de países desde Natural Earth."""
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    return world

def process_dataframe(df):
    """Convierte el DataFrame en un GeoDataFrame y llena valores nulos."""
    df = df.dropna(subset=['Latitud', 'Longitud'])
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitud, df.Latitud))
    return df

def mapa_ubicacion(df, world, filtro=None, valor=None):
    """Genera un mapa con la ubicación de los clientes."""
    gdf = df.copy()
    if filtro and valor:
        gdf = gdf[gdf[filtro] == valor]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    world.plot(ax=ax, color='lightgrey')
    gdf.plot(ax=ax, markersize=10, color='red', alpha=0.6)
    ax.set_title(f'Mapa de Ubicación de Clientes {"- " + filtro + ": " + str(valor) if filtro else ""}')
    st.pyplot(fig)

def distancia_altos_ingresos(df):
    """Calcula distancias geográficas entre los compradores con mayores ingresos."""
    top_income = df.nlargest(10, 'Ingreso_Anual_USD')  # Tomar los 10 clientes con mayores ingresos
    coords = np.vstack((top_income.geometry.x, top_income.geometry.y)).T
    dist_matrix = squareform(pdist(coords))
    return dist_matrix

st.title("Análisis de Datos de Clientes con GeoPandas")

# Carga de datos
file = st.file_uploader("Sube un archivo CSV", type=["csv"])
csv_url = st.text_input("O ingresa la URL de un CSV")

world = load_geodata()
df = None

if file is not None:
    df = pd.read_csv(file)
    st.success("Archivo CSV cargado con éxito.")
elif csv_url:
    try:
        df = pd.read_csv(csv_url)
        st.success("CSV cargado desde la URL con éxito.")
    except Exception as e:
        st.error(f"Error al cargar el CSV desde la URL: {e}")

if df is not None:
    df = process_dataframe(df)
    st.write("Vista previa de los datos:")
    st.dataframe(df)
    
    # Mapa de ubicación
    st.subheader("Mapa de Ubicación")
    filtro = st.selectbox("Filtrar mapa por", [None, "Género", "Frecuencia_Compra"])
    valor = st.selectbox("Valor", df[filtro].unique()) if filtro else None
    mapa_ubicacion(df, world, filtro, valor)
    
    # Distancias entre compradores con altos ingresos
    st.subheader("Distancias entre Compradores de Altos Ingresos")
    distancias = distancia_altos_ingresos(df)
    st.write(distancias)
