import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import streamlit as st

def get_ranges(df):
    # Definir bins y labels para Edad
    bins_edad = [10, 20, 30, 40, 50, 60, 70]
    labels_edad = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70']
    df['Edad_Rango'] = pd.cut(df['Edad'], bins=bins_edad, labels=labels_edad, right=False)
    
    # Definir bins y labels para Ingreso Anual (de 0 a 100k en saltos de 10k)
    bins_ingreso = list(range(0, 110000, 10000))  
    labels_ingreso = ['0-10000', '10000-20000', '20000-30000', '30000-40000', 
                      '40000-50000', '50000-60000', '60000-70000', '70000-80000', 
                      '80000-90000', '90000-100000']

    df['Ingreso_Rango'] = pd.cut(df['Ingreso_Anual_USD'], bins=bins_ingreso, labels=labels_ingreso, right=False)


    return df


def group_for_categorica(df, column, group_columns):
    df_filled = df.copy()
    other_columns = list(df.columns)
    other_columns.remove(column)
    not_null_rows = df_filled.copy()
    not_null_rows.dropna(inplace=True)
    

    modas = not_null_rows.groupby(group_columns, observed=True)[column].agg(pd.Series.mode )
    modas = modas.explode().groupby(level=group_columns).first()
    return modas


def group_for_numericas(df, column, group_columns):
    df_filled = df.copy()
    other_columns = list(df.columns)
    other_columns.remove(column)
    not_null_rows = df_filled.copy()
    not_null_rows.dropna(inplace=True)
    not_null_rows = get_ranges(not_null_rows)
    
    mean = not_null_rows.groupby(group_columns, observed=True)[column].agg(pd.Series.mean )
    return mean


def fill_categorica(df, column, criteria):
    """
    Rellena valores nulos en una columna categórica basada en la moda de grupos.

    Parámetros:
        df (pd.DataFrame): DataFrame original.
        column (str): Nombre de la columna categórica a rellenar.
        criteria (list): Columnas usadas para definir los grupos.

    Retorna:
        pd.DataFrame: DataFrame con los valores categóricos rellenados.
    """
    grouped = group_for_categorica(df, column, criteria)
    mode = df[column].mode().iloc[0]
    # Usamos transform para aplicar la moda de los grupos a cada fila con NaN
    df[column] = df[column].fillna(df[criteria].apply(lambda row: grouped.get(tuple(row), mode), axis=1))
    
    return df


def fill_numericas(df, column, criteria):
    """
    Rellena valores nulos en una columna categórica basada en la moda de grupos.

    Parámetros:
        df (pd.DataFrame): DataFrame original.
        column (str): Nombre de la columna categórica a rellenar.
        criteria (list): Columnas usadas para definir los grupos.

    Retorna:
        pd.DataFrame: DataFrame con los valores categóricos rellenados.
    """
    grouped = group_for_numericas(df, column, criteria)
    mean = df[column].mean()
    # Usamos transform para aplicar la moda de los grupos a cada fila con NaN
    df[column] = df[column].fillna(df[criteria].apply(lambda row: grouped.get(tuple(row), mean), axis=1))
    
    return df

def process_dataframe(df):
    """
    Procesa un DataFrame rellenando valores nulos en columnas categóricas y numéricas
    basándose en grupos relevantes.

    Parámetros:
        df: DataFrame original con valores nulos.

    Retorna:
        DataFrame procesado con los valores interpolados.
    """
    df_og = df.copy()
    df = get_ranges(df)

    # Llenar valores categóricos según grupos relevantes
    df = fill_categorica(df, 'Nombre', ['Género', 'Frecuencia_Compra'])  # Frecuencia puede estar relacionada con el tipo de cliente.
    df = fill_numericas(df, 'Edad', ['Ingreso_Rango', 'Frecuencia_Compra'])  # La edad puede estar correlacionada con ingreso y frecuencia de compra.
    df = fill_categorica(df, 'Género', ['Nombre', 'Ingreso_Rango'])  # El género podría estar relacionado con algunos nombres y el nivel de ingresos.
    df = fill_numericas(df, 'Ingreso_Anual_USD', ['Edad', 'Frecuencia_Compra'])  # Relacionamos ingreso con edad y frecuencia de compra.
    df = fill_numericas(df, 'Historial_Compras', ['Edad', 'Ingreso_Anual_USD', 'Frecuencia_Compra'])  # Consideramos ingresos y edad para predecir historial de compras.
    df = fill_categorica(df, 'Frecuencia_Compra', ['Ingreso_Rango', 'Historial_Compras'])  # La frecuencia de compra puede depender del ingreso y compras previas.
    df = fill_numericas(df, 'Latitud', ['Ingreso_Anual_USD', 'Frecuencia_Compra'])  # Los clientes con mayores ingresos pueden vivir en zonas específicas.
    df = fill_numericas(df, 'Longitud', ['Ingreso_Anual_USD', 'Frecuencia_Compra'])  # Similar a la latitud, ajustamos por ubicación e ingresos.

    df = get_ranges(df)
    df.dropna()
    # Llenar valores numéricos según grupos relevantes
    df = get_ranges(df)
    
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
    world = world[world["CONTINENT"].isin(["South America", "Central America", "America])]

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
