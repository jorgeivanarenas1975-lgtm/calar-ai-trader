import pandas as pd
import numpy as np

def analizar_velas_1h(df):
    """
    Recibe un DataFrame con columnas: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    correspondientes a la temporalidad de 1 hora.
    """
    # 1. Determinar si cada vela es verde (1) o roja (-1)
    df['color_vela'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))
    
    # 2. Calcular la EMA de 50 periodos como filtro de tendencia
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 3. Calcular el volumen promedio (SMA de volumen de 20 periodos) para confirmar fuerza
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    # 4. Identificar patrones de 2 velas consecutivas
    df['dos_verdes'] = (df['color_vela'] == 1) & (df['color_vela'].shift(1) == 1)
    df['dos_rojas'] = (df['color_vela'] == -1) & (df['color_vela'].shift(1) == -1)
    
    # 5. Filtros de calidad añadidos:
    # - Señal CALL: 2 velas verdes + Precio por encima de la EMA 50 + Volumen de la última vela mayor al promedio
    df['senal_call'] = df['dos_verdes'] & (df['close'] > df['ema_50']) & (df['volume'] > df['vol_ma'])
    
    # - Señal PUT: 2 velas rojas + Precio por debajo de la EMA 50 + Volumen de la última vela mayor al promedio
    df['senal_put'] = df['dos_rojas'] & (df['close'] < df['ema_50']) & (df['volume'] > df['vol_ma'])
    
    return df

if __name__ == "__main__":
    print("Calar AI Trader V6 - Estrategia 1H optimizada iniciada correctamente.")
