import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Calar AI Trader V6", layout="centered")

st.title("🤖 Calar AI Trader V6")
st.markdown("Estrategia de trading en temporalidad de **1 Hora (1H)** con análisis de dos velas consecutivas, filtro de tendencia (EMA 50) y volumen institucional.")

def analizar_velas_1h(df):
    df['color_vela'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    
    df['dos_verdes'] = (df['color_vela'] == 1) & (df['color_vela'].shift(1) == 1)
    df['dos_rojas'] = (df['color_vela'] == -1) & (df['color_vela'].shift(1) == -1)
    
    df['senal_call'] = df['dos_verdes'] & (df['close'] > df['ema_50']) & (df['volume'] > df['vol_ma'])
    df['senal_put'] = df['dos_rojas'] & (df['close'] < df['ema_50']) & (df['volume'] > df['vol_ma'])
    
    return df

st.info("Sube o conecta tus datos de velas en formato CSV para evaluar las señales en tiempo real.")

archivo_subido = st.file_uploader("Cargar archivo de datos (CSV con columnas: open, high, low, close, volume)", type=["csv"])

if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.write("Vista previa de los datos cargados:", df.tail())
    
    df_procesado = analizar_velas_1h(df)
    ultima = df_procesado.iloc[-1]
    
    st.subheader("Resultado del último análisis (1H):")
    if ultima['senal_call']:
        st.success("¡SEÑAL CALL DETECTADA! 🟢 (2 Velas verdes + EMA 50 + Volumen alcista)")
    elif ultima['senal_put']:
        st.error("¡SEÑAL PUT DETECTADA! 🔴 (2 Velas rojas + EMA 50 + Volumen bajista)")
    else:
        st.warning("Sin señales claras en la última vela de 1H. El mercado está en espera.")
else:
    st.markdown("---")
    st.write("ℹ️ *Esperando archivo CSV con datos del mercado para ejecutar el análisis.*")
