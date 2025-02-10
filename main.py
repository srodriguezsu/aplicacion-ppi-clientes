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
    
    mode_value = df[column].mode().iloc[0]  # Gets the first mode if multiple exist

    # Usamos transform para aplicar la moda de los grupos a cada fila con NaN
    df[column] = df[column].fillna(df[criteria].apply(lambda row: grouped.get(tuple(row), mode_value), axis=1))
    
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
    mean_value = df[column].mean()
    # Usamos transform para aplicar la moda de los grupos a cada fila con NaN
    df[column] = df[column].fillna(df[criteria].apply(lambda row: grouped.get(tuple(row), mean_value), axis=1))
    
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
    # Llenar valores numéricos según grupos relevantes
    df = get_ranges(df)
    
    return df

def correlacion_edad_ingreso(df, segmentar_por=None):
    """
    Genera un gráfico de dispersión y calcula la correlación entre edad e ingreso anual.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame con los datos de clientes.
    - segmentar_por (str, opcional): Columna para segmentar (ej. 'Género', 'Frecuencia_Compra').
    
    Retorna:
    - Matriz de correlación y gráfico.
    """
    plt.figure(figsize=(8, 6))
    
    if segmentar_por:
        sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD', hue=df[segmentar_por])
    else:
        sns.scatterplot(data=df, x='Edad', y='Ingreso_Anual_USD')
    
    plt.title(f'Correlación entre Edad e Ingreso Anual {"por " + segmentar_por if segmentar_por else ""}')
    plt.show()
    
    return df[['Edad', 'Ingreso_Anual_USD']].corr()


def mapa_ubicacion(df, filtro=None, valor=None):
    """
    Genera un mapa de ubicaciones de clientes.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame con latitud y longitud.
    - filtro (str, opcional): Columna para filtrar (ej. 'Género', 'Frecuencia_Compra').
    - valor (varios, opcional): Valor de la columna filtro a mostrar.
    
    Retorna:
    - Mapa de ubicaciones.
    """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitud, df.Latitud))
    
    if filtro and valor:
        gdf = gdf[gdf[filtro] == valor]

    gdf.plot(markersize=5, figsize=(8, 6), alpha=0.5)
    plt.title(f'Mapa de Ubicación de Clientes {" - " + filtro + ": " + str(valor) if filtro else ""}')
    plt.show()

def mapa_personalizado(df, filtros):
    """
    Genera un mapa filtrado por hasta 4 variables con sus respectivos rangos.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame con datos.
    - filtros (dict): Diccionario con {columna: (min, max)}
    
    Retorna:
    - Mapa de ubicación con filtros aplicados.
    """
    for col, (min_val, max_val) in filtros.items():
        df = df[(df[col] >= min_val) & (df[col] <= max_val)]
    
    mapa_ubicacion(df)


def cluster_frecuencia(df):
    """
    Realiza un análisis de clúster según la frecuencia de compra.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame con la columna 'Frecuencia_Compra'.
    
    Retorna:
    - Dendrograma de clustering.
    """
    data = df[['Frecuencia_Compra']].dropna()
    linkage_matrix = linkage(data, method='ward')
    
    plt.figure(figsize=(8, 6))
    dendrogram(linkage_matrix, labels=df.index, leaf_rotation=90)
    plt.title('Clúster de Frecuencia de Compra')
    plt.show()

def grafico_barras(df):
    """
    Genera un gráfico de barras mostrando la cantidad de clientes por género y frecuencia de compra.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame con 'Género' y 'Frecuencia_Compra'.
    
    Retorna:
    - Gráfico de barras.
    """
    df.groupby(['Género', 'Frecuencia_Compra']).size().unstack().plot(kind='bar', figsize=(8, 6))
    plt.title('Distribución de Clientes por Género y Frecuencia de Compra')
    plt.xlabel('Género')
    plt.ylabel('Cantidad de Clientes')
    plt.show()

def mapa_calor_ingresos(df):
    """
    Genera un mapa de calor mostrando la relación entre ingreso anual y ubicación geográfica.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame con 'Latitud', 'Longitud' e 'Ingreso_Anual_USD'.
    
    Retorna:
    - Mapa de calor.
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=df.Longitud, y=df.Latitud, weights=df.Ingreso_Anual_USD, cmap="Reds", fill=True)
    plt.title('Mapa de Calor de Ingresos')
    plt.show()


def distancia_altos_ingresos(df, segmentar_por=None):
    """
    Calcula la distancia entre compradores con mayores ingresos.
    
    Parámetros:
    - df (pd.DataFrame): DataFrame con 'Ingreso_Anual_USD', 'Latitud' y 'Longitud'.
    - segmentar_por (str, opcional): Segmentación por género o frecuencia de compra.
    
    Retorna:
    - Matriz de distancias.
    """
    if segmentar_por:
        grupos = df.groupby(segmentar_por)
        resultados = {}
        for grupo, datos in grupos:
            coords = datos[['Latitud', 'Longitud']].dropna()
            dist_matrix = squareform(pdist(coords))
            resultados[grupo] = dist_matrix
        return resultados
    else:
        coords = df[['Latitud', 'Longitud']].dropna()
        return squareform(pdist(coords))




st.title("Análisis de Datos de Clientes")

st.subheader("Carga de datos")

# Subir archivo
file = st.file_uploader("Sube un archivo CSV", type=["csv"])

# Ingresar URL
csv_url = st.text_input("O ingresa la URL de un CSV")

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

# Mostrar los datos si se cargaron correctamente
if df is not None:
    df = process_dataframe(df)
    st.write("Vista previa de los datos:")
    st.dataframe(df)

# Mapa de ubicación
st.subheader("Mapa de Ubicación")
filtro = st.selectbox("Filtrar mapa por", ["Ninguno", "Género", "Frecuencia_Compra"])
valor = st.selectbox("Valor", df[filtro].unique()) if filtro != "Ninguno" else None
mapa_ubicacion(df, filtro if filtro != "Ninguno" else None, valor)

# Correlación Edad-Ingreso
st.subheader("Correlación Edad-Ingreso")
segmentar_por = st.sidebar.selectbox("Segmentar correlación por", ["Ninguno", "Género", "Frecuencia_Compra"])
correlacion_edad_ingreso(df, segmentar_por if segmentar_por != "Ninguno" else None)

# Clúster de Frecuencia de Compra
st.subheader("Clúster de Frecuencia de Compra")
cluster_frecuencia(df)

# Mapa de Calor de Ingresos
st.subheader("Mapa de Calor de Ingresos")
mapa_calor_ingresos(df)

# Gráfico de Barras
st.subheader("Gráfico de Barras")
grafico_barras(df)

# Distancias entre Compradores de Altos Ingresos
st.subheader("Distancias entre Compradores de Altos Ingresos")
segmentar_dist = st.sidebar.selectbox("Segmentar distancias por", ["Ninguno", "Género", "Frecuencia_Compra"])
distancias = distancia_altos_ingresos(df, segmentar_dist if segmentar_dist != "Ninguno" else None)
st.write(distancias)

    
