import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Calar AI Trader V7", page_icon="📊", layout="wide")
st.title("📊 Calar AI Trader V7")
st.caption("Escáner educativo de acciones: 1H + 5M | análisis técnico | sin ejecución de operaciones")

DEFAULT = "NVDA,AVGO,META,AMD,AMZN,GOOGL,MSFT,AAPL,TSLA,MU,CRWD,ANET,PLTR,ORCL,APP,ARM,SNOW,SMCI,TSM,SPY,QQQ"

def clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"])

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    pc = df.Close.shift(1)
    tr = pd.concat([(df.High-df.Low),
                    (df.High-pc).abs(),
                    (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df, n=14):
    up = df.High.diff()
    dn = -df.Low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df,n).replace(0,np.nan)
    pdi = 100*plus.rolling(n).mean()/a
    mdi = 100*minus.rolling(n).mean()/a
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.rolling(n).mean()

@st.cache_data(ttl=1500, show_spinner=False)
def market_context():
    try:
        vals=[]
        for t in ["SPY","QQQ"]:
            x=clean(yf.download(t,period="20d",interval="1h",auto_adjust=False,progress=False))
            e=x.Close.ewm(span=20,adjust=False).mean()
            vals.append(float(x.Close.iloc[-1]) > float(e.iloc[-1]))
        return "ALCISTA" if all(vals) else ("BAJISTA" if not any(vals) else "NEUTRAL")
    except:
        return "NEUTRAL"

@st.cache_data(ttl=1500, show_spinner=False)
def analyze(ticker, market):
    try:
        h=clean(yf.download(ticker,period="60d",interval="1h",auto_adjust=False,progress=False))
        m=clean(yf.download(ticker,period="10d",interval="5m",auto_adjust=False,progress=False))
        if len(h)<60: return None

        h["EMA20"]=h.Close.ewm(span=20,adjust=False).mean()
        h["EMA50"]=h.Close.ewm(span=50,adjust=False).mean()
        h["EMA200"]=h.Close.ewm(span=200,adjust=False).mean()
        h["RSI"]=rsi(h.Close)
        h["ATR"]=atr(h)
        h["ADX"]=adx(h)
        h["VMA20"]=h.Volume.rolling(20).mean()
        h["RelVol"]=h.Volume/h.VMA20.replace(0,np.nan)

        last=h.iloc[-1]
        pair=h.iloc[-2:]
        green2=bool((pair.Close>pair.Open).all())
        red2=bool((pair.Close<pair.Open).all())

        bull=bear=0
        rb=[]; rs=[]

        if green2: bull+=20; rb.append("2 velas 1H verdes")
        if red2: bear+=20; rs.append("2 velas 1H rojas")
        if last.Close>last.EMA20: bull+=10; rb.append("precio > EMA20")
        else: bear+=10; rs.append("precio < EMA20")
        if last.EMA20>last.EMA50: bull+=15; rb.append("EMA20 > EMA50")
        else: bear+=15; rs.append("EMA20 < EMA50")
        if last.EMA50>last.EMA200: bull+=10; rb.append("EMA50 > EMA200")
        else: bear+=10; rs.append("EMA50 < EMA200")

        if pd.notna(last.RSI):
            if 50<=last.RSI<=70: bull+=10; rb.append("RSI alcista")
            if 30<=last.RSI<=50: bear+=10; rs.append("RSI bajista")

        if pd.notna(last.ADX) and last.ADX>=20:
            if last.EMA20>last.EMA50: bull+=10; rb.append("ADX confirma")
            else: bear+=10; rs.append("ADX confirma")

        if pd.notna(last.RelVol) and last.RelVol>=1.2:
            if last.Close>=last.Open: bull+=10; rb.append("volumen relativo alto")
            else: bear+=10; rs.append("volumen relativo alto")

        hi=h.High.iloc[-21:-1].max()
        lo=h.Low.iloc[-21:-1].min()
        if last.Close>hi: bull+=10; rb.append("ruptura máximo 20H")
        if last.Close<lo: bear+=10; rs.append("ruptura mínimo 20H")

        if market=="ALCISTA": bull+=5
        if market=="BAJISTA": bear+=5

        c5="—"
        if len(m)>=20:
            e5=m.Close.ewm(span=20,adjust=False).mean()
            c5="ALCISTA" if float(m.Close.iloc[-1])>float(e5.iloc[-1]) else "BAJISTA"

        if bull>=65 and bull>=bear+10: state="ALCISTA"
        elif bear>=65 and bear>=bull+10: state="BAJISTA"
        elif max(bull,bear)>=50: state="ESPERAR"
        else: state="NO OPERAR"

        return {
            "Ticker":ticker,"Precio":round(float(last.Close),2),"Estado":state,
            "Score alcista":min(bull,100),"Score bajista":min(bear,100),
            "RSI":round(float(last.RSI),1) if pd.notna(last.RSI) else np.nan,
            "ADX":round(float(last.ADX),1) if pd.notna(last.ADX) else np.nan,
            "RelVol":round(float(last.RelVol),2) if pd.notna(last.RelVol) else np.nan,
            "2x1H":"VERDE/VERDE" if green2 else ("ROJA/ROJA" if red2 else "MIXTA"),
            "5M":c5,
            "EMA20/50":"↑" if last.EMA20>last.EMA50 else "↓",
            "Ruptura":"↑20H" if last.Close>hi else ("↓20H" if last.Close<lo else "—"),
            "Lectura alcista":", ".join(rb[:5]),
            "Lectura bajista":", ".join(rs[:5])
        }
    except Exception:
        return None

st.sidebar.header("⚙️ Configuración")
raw=st.sidebar.text_area("Símbolos separados por coma",DEFAULT,height=150)
threshold=st.sidebar.slider("Score mínimo",40,90,65)
scan=st.sidebar.button("🔎 ESCANEAR AHORA",use_container_width=True)

if "df" not in st.session_state or scan:
    tickers=list(dict.fromkeys([x.strip().upper() for x in raw.split(",") if x.strip()]))
    market=market_context()
    rows=[]
    progress=st.progress(0)
    for i,t in enumerate(tickers):
        r=analyze(t,market)
        if r: rows.append(r)
        progress.progress((i+1)/len(tickers))
    progress.empty()
    st.session_state.df=pd.DataFrame(rows)
    st.session_state.market=market
    st.session_state.when=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

df=st.session_state.get("df",pd.DataFrame())
market=st.session_state.get("market","—")

a,b,c=st.columns(3)
a.metric("Contexto SPY / QQQ",market)
b.metric("Símbolos analizados",len(df))
c.metric("Actualizado",st.session_state.get("when","—"))

if df.empty:
    st.info("Pulsa «ESCANEAR AHORA» para comenzar.")
else:
    st.subheader("📊 Ranking completo")
    st.dataframe(df.sort_values(["Score alcista","Score bajista"],ascending=False),
                 use_container_width=True,hide_index=True)

    x=df[df["Score alcista"]>=threshold].sort_values("Score alcista",ascending=False)
    y=df[df["Score bajista"]>=threshold].sort_values("Score bajista",ascending=False)

    col1,col2=st.columns(2)
    with col1:
        st.subheader("🟢 TOP ALCISTA")
        st.dataframe(x[["Ticker","Precio","Score alcista","RSI","ADX","RelVol","2x1H","5M","Ruptura"]].head(15),
                     use_container_width=True,hide_index=True)
    with col2:
        st.subheader("🔴 TOP BAJISTA")
        st.dataframe(y[["Ticker","Precio","Score bajista","RSI","ADX","RelVol","2x1H","5M","Ruptura"]].head(15),
                     use_container_width=True,hide_index=True)

    st.download_button("⬇️ Descargar CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       "calar_ai_trader_v7.csv","text/csv")

st.divider()
st.caption("V7: herramienta educativa de análisis técnico. No ejecuta operaciones ni constituye asesoría financiera.")
