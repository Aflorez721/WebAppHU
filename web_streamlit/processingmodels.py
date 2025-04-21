#%%
# ===========================================
# 1. Configuración del Entorno
# ===========================================
try:
    import pandas as pd
    import numpy as np

    import os

    from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    from joblib import Parallel, delayed, dump, load
    import matplotlib.pyplot as plt
    import seaborn as sns

    from ds_utils import *

    import warnings
    warnings.filterwarnings('ignore')
    
except Exception as e:
    print(f"Error importing libraries: {e}")

def generate_forecast_and_metrics(input_file):
    """
    Procesa el archivo Excel, genera pronósticos y calcula métricas de desempeño.
    Devuelve los nombres de los archivos generados.
    """
    # Leer el archivo Excel
    df = pd.read_excel(input_file)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['codigoarticulo', 'fecha']).reset_index(drop=True)

    # Aplicar la función de manejo de valores atípicos para cada codigoarticulo
    df_cleaned = df.groupby('codigoarticulo').apply(lambda group: handle_outliers(group, 'cantidad'))
    df_cleaned = mediana_rolling(df_cleaned)

    # Eliminar el índice adicional creado por groupby.apply
    df_cleaned = df_cleaned.reset_index(drop=True)
    df_cleaned.sort_values(['codigoarticulo', 'fecha'], inplace=True)
    df_cleaned['cantidad1'] = df_cleaned['cantidad1'].interpolate(method='backfill')
    df_cleaned['margen'] = df_cleaned['margen'].interpolate(method='backfill')

    # Crear las características
    df_feat = create_features(df_cleaned)

    # Definir las características y la variable objetivo
    features = [
        'stock',
        'margen',
        #'tendencia',
        'dummy_pandemia',
        # 'cantidad_lag_1',
        # 'cantidad_lag_2',
        # 'cantidad_lag_3',
        'stock_cuadrado',
        'margen_cuadrado',
        'stock_cubo',
        'margen_cubo',
        'stock_raiz',
        'margen_raiz',
        'stock_log',
        'margen_log',
        'mediana'
    ] + [col for col in df_feat.columns if 'mes_' in col]

    target = 'cantidad1'

    # Definir el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features)
        ]
    )

    # Definir TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=2)

    # Obtener todos los códigos de artículo únicos
    articulos = df_feat['codigoarticulo'].unique()

    # Procesar en paralelo
    resultados_finales = Parallel(n_jobs=-1)(
        delayed(process_articulo)(articulo, df_feat, features, target, preprocessor, tscv)
        for articulo in articulos
    )

    # Convertir los resultados a un DataFrame
    df_resultados_finales = pd.DataFrame(resultados_finales)

    #=======================================================================================================================
    # Pronóstico de variables exógenas
    #=======================================================================================================================
    # Obtener todos los códigos de artículo únicos
    articulos = df_feat['codigoarticulo'].unique()

    # Número de meses a pronosticar
    k = 6

    # Procesar en paralelo para cada artículo
    resultados_pronostico = Parallel(n_jobs=-1)(
        delayed(pronosticar_variables_exog)(articulo, df_feat, k) for articulo in articulos
    )

    # Convertir los resultados a un DataFrame
    df_pronostico = pd.DataFrame(resultados_pronostico)

    # Crear un DataFrame para almacenar los pronósticos con fechas
    pronosticos_con_fechas = []

    for index, row in df_pronostico.iterrows():
        codigo_articulo = row['codigoarticulo']
        ultima_fecha = df_feat[df_feat['codigoarticulo'] == codigo_articulo]['fecha'].max()
        print(ultima_fecha)
        fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(months=0), periods=k, freq='MS')
        print(fechas_futuras)
        for i in range(k):
            pronosticos_con_fechas.append({
                'codigoarticulo': codigo_articulo,
                'fecha': fechas_futuras[i],
                'stock_pronostico': row['stock_pronostico'][i],
                'margen_pronostico': row['margen_pronostico'][i],
                'mediana_pronostico': row['mediana_pronostico'][i]
            })

    df_pronostico_fechas = pd.DataFrame(pronosticos_con_fechas)

    #-- Creando las variables necesarias para los pronósticos
    df_pronostico_fechas.rename(columns={'stock_pronostico': 'stock', 'margen_pronostico': 'margen', 'mediana_pronostico': 'mediana'}, inplace=True)
    df_features_tmp = create_features(df_pronostico_fechas)
    df_features_pronostico = check_and_create_missing_month_columns(df_features_tmp)

    #-- Aplicar la función de bandas de suavizado
    df_features_pronostico = smoothing_bands(df_features_pronostico, alpha=0.1)

    # forecast con control de pronosticos si stock en cero reemplazar por mediana  y pronostico pasa banda superior se reemplanza con mediana

    pronosticos = []


    for codigo in df_features_pronostico['codigoarticulo'].unique():

        df_temp = df_features_pronostico[df_features_pronostico['codigoarticulo'] == codigo].copy()

        X_nuevo = df_temp[features]

        try:
            predicciones = cargar_modelo_y_predecir(codigo, X_nuevo)

            for i in range(len(predicciones)):
                final_value = df_temp['mediana'].iloc[i] if df_temp['stock'].iloc[i] <= 0 else predicciones[i]

                pronosticos.append({
                    'codigoarticulo': codigo,
                    'fecha': df_temp['fecha'].iloc[i],
                    'cantidad_pronostico': final_value
                })

        except Exception as e:
            print(f"Error al predecir para el artículo {codigo}: {e}")
            continue


    # Convertir la lista de pronósticos en un DataFrame
    df_pronosticos = pd.DataFrame(pronosticos)

    # 
    new_data_temp = df_features_pronostico[['codigoarticulo', 'fecha', 'stock', 'margen', 'mediana', 'upper_bound']]

    # merge with predictions
    df_pronosticos = df_pronosticos.merge(new_data_temp, on=['codigoarticulo', 'fecha'], how='left')


    df_pronosticos['cantidad_pronostico'] = np.where(df_pronosticos['cantidad_pronostico'] < 0, 0, df_pronosticos['cantidad_pronostico'])
    df_pronosticos['cantidad_pronostico'] = np.where(df_pronosticos['cantidad_pronostico'] > df_pronosticos['upper_bound'], df_pronosticos['mediana'], df_pronosticos['cantidad_pronostico'])
    df_pronosticos['cantidad_pronostico'] = df_pronosticos['cantidad_pronostico'].round().astype(int)



    # Guardar los resultados en archivos
    forecast_file = "output_forecast.xlsx"
    metrics_file = "output_metrics.xlsx"

    df_pronosticos.to_excel(forecast_file, index=False)
    #-- Almacenamiento de las metricas de desempeño en los diferentes modelos
    df_resultados_finales.to_excel(metrics_file, index=False)
    ##return forecast_file, metrics_file
    return forecast_file, metrics_file


# %%
# forecast_file, metrics_file = generate_forecast_and_metrics(input_file = r"C:\Users\Alberto Florez\OneDrive\Documentos\GitHub\ujueta_project\ds_backstage\data\Ejercicio Forecast ALL.xlsx")

# %%
