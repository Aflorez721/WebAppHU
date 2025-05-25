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
    from typing import Dict, Union

    from functools import partial
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mape, mase, mse, smape

except Exception as e:
    print(f"Error importing libraries: {e}")

# ===========================================
# 1. Definición de Funciones
# ===========================================

# Función para identificar y manejar valores atípicos
def handle_outliers(data, column):
    """
    Identifica valores atípicos usando el método IQR y crea una nueva columna sin ellos.
    Luego, imputa los valores eliminados mediante interpolación.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5* IQR
    upper_bound = Q3 + 1.5 * IQR

    # Crear una nueva columna con valores atípicos marcados como NaN
    #data[f'{column}1'] = data[column].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)
    #-- Solo los atípicos superiores
    data[f'{column}1'] = data[column].apply(lambda x: x if  x <= upper_bound else np.nan)
    # Imputar valores NaN mediante interpolación
    data[f'{column}1'] = data[f'{column}1'].interpolate(method='linear')

    return data

# Función para calcular la mediana móvil
def mediana_rolling(df, window=6):
    """
    Calcula la mediana móvil de una columna específica en un DataFrame.
    """
    for col in df.columns:
        df['mediana'] = df['cantidad1'].rolling(window=window, min_periods=1).median()

    return df

#def smoothing_bands(df, alpha):
#    """
#    Creation of smoothing bands for the forecast to control the forecast
#
#    """
#    df['lower_bound'] = df['mediana'] - (df['mediana'] * alpha)
#    df['upper_bound'] = df['mediana'] + (df['mediana'] * alpha)
#    return df

def smoothing_bands_iqr(df, iqr_multiplier=1.5):
    """
    Creates statistically robust smoothing bands for forecasts using IQR (Interquartile Range).
    
    Parameters:
        df (pd.DataFrame): Input data containing 'mediana' column
        iqr_multiplier (float): Multiplier for IQR range (default: 1.5)
        
    Returns:
        pd.DataFrame: Original DataFrame with added 'lower_bound' and 'upper_bound' columns
    """
    # Calculate IQR statistics
    Q1 = df['mediana'].quantile(0.25)
    Q3 = df['mediana'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Create bounds
    df['lower_bound'] = df['mediana'] - (iqr_multiplier * IQR)
    df['upper_bound'] = df['mediana'] + (iqr_multiplier * IQR)
    
    # Ensure non-negative bounds for positive data
    if (df['mediana'] >= 0).all():
        df['lower_bound'] = df['lower_bound'].clip(lower=0)
    
    return df

# Función de creación de caracteristicas
def create_features(data, pandemia_start='2020-03-01', pandemia_end='2021-12-31', lags=[1,2,3]):
    df = data.copy()
    df['mes'] = df['fecha'].dt.month
    #df['tendencia'] = np.arange(len(df)) # Tendencia lineal
    df['tendencia'] = 0
    df.loc[df.index[-6:], 'tendencia'] = np.arange(6) # Últimos 6 meses
    pandemia_start = pd.to_datetime(pandemia_start)
    pandemia_end = pd.to_datetime(pandemia_end)
    df['dummy_pandemia'] = ((df['fecha'] >= pandemia_start) & (df['fecha'] <= pandemia_end)).astype(int)

    #-- Variables Lag
    # for lag in lags:
    #     df[f'cantidad_lag_{lag}'] = df.groupby('codigoarticulo')['cantidad'].shift(lag)
    #df = df.dropna().reset_index(drop=True)

    #-- Variables potencia
    df['stock_cuadrado'] = df['stock'] ** 2
    df['margen_cuadrado'] = df['margen'] ** 2
    df['stock_cubo'] = df['stock'] ** 3
    df['margen_cubo'] = df['margen']**3
    df['stock_raiz'] = np.sqrt(df['stock'])
    df['margen_raiz'] = np.sqrt(df['margen'])
    

    #-- Variables logaritmicas
    df['stock_log'] = np.log(df['stock'] + 1)
    df['margen_log'] = np.log(df['margen'] + 1)

    df = pd.get_dummies(df, columns=['mes'], drop_first=True)
    return df

def check_and_create_missing_month_columns(df):
    """
    Verifica que el DataFrame tenga las columnas 'mes_1' a 'mes_12',
    y crea las que falten con valor False.

    Args:
        df (pd.DataFrame): El DataFrame a verificar.

    Returns:
        pd.DataFrame: El DataFrame con las columnas de mes completas.
    """
    mes_columns = [f'mes_{i}' for i in range(1, 13)]

    for mes_col in mes_columns:
        if mes_col not in df.columns:
            df[mes_col] = False  # Crea la columna con valor False
    return df


def guardar_mejor_modelo(modelo, codigoarticulo):
    """
    Guarda el modelo entrenado en un archivo .joblib en la carpeta modelos_guardados.
    """
    if not os.path.exists('modelos_guardados'):
        os.makedirs('modelos_guardados')

    path = f'modelos_guardados/modelo_{codigoarticulo}.joblib'
    dump(modelo, path)
    print(f'Modelo guardado: {path}')

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred)
    non_zero = denominator != 0
    return np.mean(2 * diff[non_zero] / denominator[non_zero]) * 100

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    smape_val = smape(y_test, y_pred)
    return {'rmse': rmse, 'smape': smape_val, 'predictions': y_pred}

def process_articulo(articulo, data, features, target, preprocessor, tscv):
    df_art = data[data['codigoarticulo'] == articulo].copy().reset_index(drop=True)

    
    if len(df_art) < 18:
        # Si no hay suficientes datos, devolver un mensaje
        
        #-- Guardar el modelo
        return {'codigoarticulo': articulo, 'Mensaje': 'Datos insuficientes'}

    train_size = int(len(df_art) * 0.85)
    train_df, test_df = df_art.iloc[:train_size], df_art.iloc[train_size:]

    X_train_art, y_train_art = train_df[features], train_df[target]
    X_test_art, y_test_art = test_df[features], test_df[target]

    modelos = {
        'Regresión Lineal': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Ridge Regression': GridSearchCV(Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Ridge())
        ]), {'regressor__alpha': [0.1,0.5, 1.0, 5, 10.0,50, 100.0]}, cv=tscv, scoring='neg_root_mean_squared_error'),
        'Lasso Regression': GridSearchCV(Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Lasso())
        ]), {'regressor__alpha': [0.001, 0.01, 0.1, 0.5,0.7, 1.0]}, cv=tscv, scoring='neg_root_mean_squared_error'),
        'ElasticNet Regression': GridSearchCV(Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet())
        ]), {'regressor__alpha': [0.01, 0.1,0.2,0.5,0.8, 1.0], 'regressor__l1_ratio': [0.1,0.2,0.35, 0.5,0.75, 0.8]}, cv=tscv, scoring='neg_root_mean_squared_error'),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42))
        ])
    }

    resultados_art = {'codigoarticulo': articulo}

    mejor_modelo = None
    mejor_rmse = float('inf')

    for nombre, modelo in modelos.items():
        try:
            if isinstance(modelo, GridSearchCV):
                modelo.fit(X_train_art, y_train_art)
                y_pred = modelo.predict(X_test_art)
            else:
                y_pred = evaluate_model(modelo, X_train_art, y_train_art, X_test_art, y_test_art)['predictions']

            rmse = np.sqrt(mean_squared_error(y_test_art, y_pred))
            smape_val = smape(y_test_art, y_pred)

            resultados_art[f'{nombre}_RMSE'] = rmse
            resultados_art[f'{nombre}_SMAPE'] = smape_val
            # Opcional: Almacenar las predicciones
            resultados_art[f'{nombre}_Predicciones'] = y_pred.tolist()


            if rmse < mejor_rmse:
                mejor_rmse = rmse
                mejor_modelo = modelo


        except Exception as e:
            resultados_art[f'{nombre}_Error'] = str(e)

    if mejor_modelo:
        guardar_mejor_modelo(mejor_modelo, articulo)

    return resultados_art

def cargar_modelo_y_predecir(codigoarticulo, X_nuevo):
    """
    Carga el modelo guardado y predice para nuevos valores de X.
    """
    path = f'modelos_guardados/modelo_{codigoarticulo}.joblib'
    if not os.path.exists(path):
        raise FileNotFoundError(f'No se encontró un modelo guardado para el código de artículo {codigoarticulo}')

    modelo = load(path)
    return modelo.predict(X_nuevo)

def pronosticar_variables_exog(articulo, data, k):
    """
    Pronostica k meses hacia adelante las variables stock y margen utilizando modelos de series de tiempo univariantes.
    """
    df_art = data[data['codigoarticulo'] == articulo].copy().reset_index(drop=True)
    resultados = {'codigoarticulo': articulo}

    for variable in ['stock', 'margen', 'mediana']:
        if df_art[variable].isna().sum() > 0:
            df_art[variable].fillna(method='ffill', inplace=True)

        # Group data by month and calculate the mean stock for each month
        monthly_stock = df_art.groupby(df_art['fecha'].dt.month)[variable].mean()

        if variable == 'stock':
            # Create a list to store the forecasted values
            predicciones = []
            for i in range(1, k + 1):
                # Get the month for the next k periods
                month = (df_art['fecha'].max().month + i) % 12
                if month == 0:
                    month = 12
                # Predict using the average stock of the corresponding month
                predicciones.append(monthly_stock.get(month, monthly_stock.mean()))

            predicciones = np.where(np.array(predicciones) < 0, 0, np.array(predicciones))
            predicciones = np.round(predicciones).astype(int)
            resultados[f'{variable}_pronostico'] = predicciones.tolist()

        else:
            # Check if enough data for ExponentialSmoothing
            if len(df_art) >= 24:  # Check if there are at least 24 months of data
                modelo = ExponentialSmoothing(df_art[variable], trend='add', seasonal='add', seasonal_periods=12).fit()
                predicciones = modelo.forecast(k)
            else:
                # If not enough data, use a simple average for forecasting
                predicciones = np.repeat(df_art[variable].mean(), k)

            predicciones = np.where(predicciones < 0, 0, predicciones)
            predicciones = np.round(predicciones).astype(int)
            resultados[f'{variable}_pronostico'] = predicciones.tolist()

    return resultados

# prompt: hacer una función para gráficar los valores de cantidad en el DataFrame df_feat y la variable cantidad_pronostico en el Dataframe  df_pronosticos para cada articulo

def plot_cantidad_vs_pronostico(df1, df2, articulo):
    """
    Grafica los valores de cantidad en df_feat y cantidad_pronostico en df_pronosticos para un artículo dado.
    """
    # Filtrar los DataFrames para el artículo específico
    df_feat_articulo = df1[df1['codigoarticulo'] == articulo].sort_values('fecha')
    df_pronosticos_articulo = df2[df2['codigoarticulo'] == articulo].sort_values('fecha')


    # Crear el gráfico
    plt.figure(figsize=(14, 6))
    plt.plot(df_feat_articulo['fecha'], df_feat_articulo['cantidad1'], label='Cantidad Real')
    plt.plot(df_pronosticos_articulo['fecha'], df_pronosticos_articulo['cantidad_pronostico'], label='Cantidad Pronosticada')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad')
    plt.title(f'Cantidad vs. Pronóstico para Artículo {articulo}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_cantidad_vs_mediana(df, articulo):
    """
    Grafica los valores de cantidad en df_feat y cantidad_pronostico en df_pronosticos para un artículo dado.
    """
    # Filtrar los DataFrames para el artículo específico
    df_feat_articulo = df[df['codigoarticulo'] == articulo].sort_values('fecha')
    df_pronosticos_articulo = df[df['codigoarticulo'] == articulo].sort_values('fecha')


    # Crear el gráfico
    plt.figure(figsize=(14, 6))
    plt.plot(df_feat_articulo['fecha'], df_feat_articulo['cantidad1'], label='Cantidad Real')
    plt.plot(df_pronosticos_articulo['fecha'], df_pronosticos_articulo['mediana'], label='Cantidad Pronosticada')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad')
    plt.title(f'Cantidad vs. Pronóstico para Artículo {articulo}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



### Revisar si se puede mejorar la función plot_modelos_articulos
def plot_modelos_articulos(df, articulo_ejemplo):
  # Filtrar los resultados para el artículo seleccionado
  resultado_articulo = df_resultados_finales[df_resultados_finales[['codigoarticulo'] == articulo_ejemplo]]

  # Verificar si hay predicciones disponibles
  modelos_disponibles = ['Regresión Lineal', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression', 'Random Forest', 'XGBoost']
  predicciones = {}
  for modelo in modelos_disponibles:
      pred_col = f'{modelo}_Predicciones'
      if pred_col in resultado_articulo.columns:
          predicciones[modelo] = resultado_articulo.iloc[0][pred_col]

  # Verificar que haya al menos un modelo con predicciones
  if not predicciones:
      print(f"No hay predicciones disponibles para el artículo {articulo_ejemplo}.")
  else:
      # Obtener los datos reales
      df_art_ejemplo = df_feat[df_feat['codigoarticulo'] == articulo_ejemplo].copy().reset_index(drop=True)
      train_size_art = int(len(df_art_ejemplo) * 0.85)
      test_df_art = df_art_ejemplo.iloc[train_size_art:].copy()
      y_true = test_df_art[target].values

      # Crear DataFrame para visualizar las predicciones de diferentes modelos
      df_plot = pd.DataFrame({
          'fecha': test_df_art['fecha'],
          'Real': y_true
      })

      for modelo, preds in predicciones.items():
          df_plot[modelo] = preds

      # Graficar todas las predicciones
      plt.figure(figsize=(14,8))
      plt.plot(df_plot['fecha'], df_plot['Real'], label='Real', marker='o')

      for modelo in modelos_disponibles:
          if modelo in predicciones:
              plt.plot(df_plot['fecha'], df_plot[modelo], label=f'Predicción {modelo}', marker='x')

      plt.xlabel('Fecha')
      plt.ylabel('Cantidad')
      plt.title(f'Pronóstico de Cantidad para {articulo_ejemplo}')
      plt.legend()
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()

###-- Función para crear el DataFrame de entrada para los modelos de series de tiempo
# low Dda

#def create_sma_models(
#    data: pd.DataFrame,
#    window_size: int = 3,
#    forecast_periods: int = 1,
#    min_data_points: int = 5
#) -> Dict[str, Union[pd.DataFrame, dict]]:
#    """
#    Creates SMA models for each unique 'codigoarticulo' in the dataset.
#    
#    Parameters:
#    -----------
#    data : pd.DataFrame
#        Input dataframe containing columns: 'codigoarticulo', 'fecha', 'cantidad'
#    window_size : int, optional
#        Number of periods to include in the moving average calculation (default: 3)
#    forecast_periods : int, optional
#        Number of future periods to forecast (default: 1)
#    min_data_points : int, optional
#        Minimum data points required to build a model (default: 5)
#    
#    Returns:
#    --------
#    dict
#        A dictionary containing:
#        - 'models': Dictionary of SMA models (one per codigoarticulo)
#        - 'forecasts': DataFrame with forecasts for each article
#        - 'stats': Summary statistics for each model
#    """
#    
#    # Validate input data
#    required_cols = {'codigoarticulo', 'fecha', 'cantidad'}
#    if not required_cols.issubset(data.columns):
#        raise ValueError(f"Input data must contain columns: {required_cols}")
#    
#    # Convert fecha to datetime if not already
#    data['fecha'] = pd.to_datetime(data['fecha'])
#    
#    # Sort data by fecha for each article
#    data = data.sort_values(['codigoarticulo', 'fecha'])
#    
#    # Initialize output containers
#    models = {}
#    forecasts = []
#    stats = []
#    
#    # Process each article separately
#    for codigo, group in data.groupby('codigoarticulo'):
#        # Check if we have enough data
#        if len(group) < min_data_points:
#            print(f"Warning: Not enough data for article {codigo} (only {len(group)} points)")
#            continue
#        
#        # Create time series
#        ts = group.set_index('fecha')['cantidad']
#        
#        # Calculate SMA
#        sma = ts.rolling(window=window_size, min_periods=1).mean()
#        
#        # Store model
#        models[codigo] = {
#            'window_size': window_size,
#            'last_values': ts.tail(window_size).values,
#            'last_date': ts.index[-1],
#            'n_observations': len(ts)
#        }
#        
#        # Generate forecasts
#        last_sma = sma.iloc[-1]
#        forecast_dates = pd.date_range(
#            start=ts.index[-1] + pd.DateOffset(months=1),
#            periods=forecast_periods,
#            freq='MS'
#        )
#        
#        for i, date in enumerate(forecast_dates, 1):
#            forecasts.append({
#                'codigoarticulo': codigo,
#                'fecha': date,
#                'cantidad_pronostico': last_sma,
#                'forecast_period': i,
#                'model': 'SMA',
#                'window_size': window_size
#            })
#        
#        # Calculate statistics
#        stats.append({
#            'codigoarticulo': codigo,
#            'mean': ts.mean(),
#            'std': ts.std(),
#            'min': ts.min(),
#            'max': ts.max(),
#            'last_value': ts.iloc[-1],
#            'sma_value': last_sma,
#            'n_observations': len(ts)
#        })
#    
#    # Convert forecasts to DataFrame
#    forecast_df = pd.DataFrame(forecasts)
#    
#    # Convert stats to DataFrame
#    stats_df = pd.DataFrame(stats)
#    
#    return {
#        'models': models,
#        'forecasts': forecast_df,
#        'stats': stats_df
#    }


from typing import Dict, Union
from pandas.tseries.offsets import MonthBegin

def create_sma_models(
    data: pd.DataFrame,
    window_size: int = 3,
    forecast_periods: int = 1,
    min_data_points: int = 5,
    freq: str = 'MS'  # Month Start by default
) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Creates SMA models for each unique 'codigoarticulo' with MONTHLY forecasting.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe containing columns: 'codigoarticulo', 'fecha', 'cantidad'
    window_size : int, optional
        Number of MONTHLY periods for moving average (default: 3)
    forecast_periods : int, optional
        Number of future MONTHS to forecast (default: 1)
    min_data_points : int, optional
        Minimum monthly data points required (default: 5)
    freq : str, optional
        Frequency for date range ('MS' for month start, 'M' for month end)
    
    Returns:
    --------
    dict
        A dictionary containing:
        - 'models': Dictionary of SMA models
        - 'forecasts': DataFrame with monthly forecasts
        - 'stats': Summary statistics
    """
    
    # Validate input data
    required_cols = {'codigoarticulo', 'fecha', 'cantidad'}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"Input data must contain columns: {required_cols}")
    
    # Convert and ensure monthly data
    data['fecha'] = pd.to_datetime(data['fecha'])
    data = data.sort_values(['codigoarticulo', 'fecha'])
    
    # Initialize outputs
    models = {}
    forecasts = []
    stats = []
    
    for codigo, group in data.groupby('codigoarticulo'):
        # Check data sufficiency
        if len(group) < min_data_points:
            print(f"Warning: Insufficient monthly data for {codigo} ({len(group)} < {min_data_points})")
            continue
        
        # Resample to monthly frequency if needed
        ts = (group.set_index('fecha')['cantidad']
                .resample('MS').mean())  # or .sum() depending on your needs
        
        # Calculate SMA
        sma = ts.rolling(window=window_size, min_periods=1).mean()
        
        # Store model
        models[codigo] = {
            'window_size': window_size,
            'last_values': ts.tail(window_size).values,
            'last_date': ts.index[-1],
            'n_observations': len(ts),
            'frequency': 'monthly'
        }
        
        # Generate monthly forecasts
        forecast_dates = pd.date_range(
            start=ts.index[-1] + MonthBegin(1),
            periods=forecast_periods,
            freq=freq
        )
        
        forecasts.extend({
            'codigoarticulo': codigo,
            'fecha': date,
            'cantidad_pronostico': last_sma,
            'forecast_period': i,
            'model': 'SMA',
            'window_size': window_size,
            'frequency': 'monthly'
        } for i, (date, last_sma) in enumerate(zip(forecast_dates, [sma.iloc[-1]]*forecast_periods), 1))
        
        # Calculate statistics
        stats.append({
            'codigoarticulo': codigo,
            'mean': ts.mean(),
            'std': ts.std(),
            'min': ts.min(),
            'max': ts.max(),
            'last_value': ts.iloc[-1],
            'sma_value': sma.iloc[-1],
            'n_observations': len(ts),
            'window_size': window_size
        })
    
    return {
        'models': models,
        'forecasts': pd.DataFrame(forecasts),
        'stats': pd.DataFrame(stats)
    }