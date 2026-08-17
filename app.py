
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Calar AI Market Analyzer", page_icon="📈", layout="wide")

DEFAULT_TICKERS = "NVDA,CRWD,AMD,SNOW,AMZN,META,MSFT,GOOGL,TSLA,MU,AVGO,PLTR"
BENCHMARKS = ["SPY", "QQQ"]

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_indicators(df):
    x = df.copy()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA40"] = x["Close"].ewm(span=40, adjust=False).mean()
    x["EMA200"] = x["Close"].ewm(span=200, adjust=False).mean()
    x["RSI"] = rsi(x["Close"])
    x["ATR"] = atr(x)
    x["VOL20"] = x["Volume"].rolling(20).mean()
    x["RVOL"] = x["Volume"] / x["VOL20"]
    x["ROC20"] = x["Close"].pct_change(20) * 100
    x["HIGH20"] = x["High"].rolling(20).max().shift(1)
    x["LOW20"] = x["Low"].rolling(20).min().shift(1)
    return x.dropna()

@st.cache_data(ttl=300)
def get_history(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def score_equity(df, benchmark=None):
    x = add_indicators(df)
    if len(x) < 30:
        return None
    r = x.iloc[-1]
    prev = x.iloc[-2]
    score = 50.0
    reasons = []
    risks = []

    # Trend: 30 points
    if r.Close > r.EMA20:
        score += 7; reasons.append("Precio sobre EMA20")
    else:
        score -= 7; risks.append("Precio bajo EMA20")
    if r.EMA20 > r.EMA40:
        score += 8; reasons.append("EMA20 sobre EMA40")
    else:
        score -= 8; risks.append("EMA20 bajo EMA40")
    if r.Close > r.EMA200:
        score += 10; reasons.append("Precio sobre EMA200")
    else:
        score -= 10; risks.append("Precio bajo EMA200")

    # Momentum: 25
    if 55 <= r.RSI <= 70:
        score += 10; reasons.append(f"RSI favorable ({r.RSI:.1f})")
    elif 45 <= r.RSI < 55:
        score += 2
    elif r.RSI > 75:
        score -= 7; risks.append("RSI muy elevado")
    elif r.RSI < 35:
        score += 2; reasons.append("RSI deprimido; posible rebote")
    else:
        score -= 4

    if r.ROC20 > 5:
        score += 8; reasons.append(f"Momentum 20D positivo ({r.ROC20:.1f}%)")
    elif r.ROC20 < -5:
        score -= 8; risks.append(f"Momentum 20D negativo ({r.ROC20:.1f}%)")
    else:
        score += 1

    # Volume / breakout: 20
    if r.RVOL >= 1.5:
        score += 8; reasons.append(f"Volumen relativo alto ({r.RVOL:.2f}x)")
    elif r.RVOL >= 1.0:
        score += 3
    if r.Close > r.HIGH20:
        score += 12; reasons.append("Ruptura de máximo de 20 sesiones")
    elif r.Close < r.LOW20:
        score -= 12; risks.append("Ruptura de mínimo de 20 sesiones")

    # Benchmark relative direction
    if benchmark is not None and len(benchmark) > 30:
        b = add_indicators(benchmark).iloc[-1]
        if b.Close > b.EMA20:
            score += 5; reasons.append("Mercado de referencia sobre EMA20")
        else:
            score -= 5; risks.append("Mercado de referencia débil")

    score = max(0, min(100, score))
    if score >= 80:
        signal = "FUERTE ALCISTA"
    elif score >= 65:
        signal = "ALCISTA"
    elif score <= 35:
        signal = "BAJISTA"
    else:
        signal = "NEUTRAL"

    return {
        "Score": round(score,1),
        "Signal": signal,
        "Price": float(r.Close),
        "RSI": float(r.RSI),
        "RVOL": float(r.RVOL),
        "ROC20": float(r.ROC20),
        "ATR": float(r.ATR),
        "EMA20": float(r.EMA20),
        "EMA40": float(r.EMA40),
        "EMA200": float(r.EMA200),
        "Reasons": reasons,
        "Risks": risks,
        "Data": x
    }

def get_option_chain(ticker, max_expirations=6):
    tk = yf.Ticker(ticker)
    expirations = list(tk.options or [])[:max_expirations]
    rows = []
    for exp in expirations:
        try:
            chain = tk.option_chain(exp)
            for typ, table in [("CALL", chain.calls), ("PUT", chain.puts)]:
                if table is None or table.empty:
                    continue
                t = table.copy()
                t["type"] = typ
                t["expiration"] = exp
                rows.append(t)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def option_candidates(chain, spot, direction, min_dte=7, max_dte=60):
    if chain.empty:
        return pd.DataFrame()
    c = chain.copy()
    c["expiration"] = pd.to_datetime(c["expiration"])
    today = pd.Timestamp.now().normalize()
    c["DTE"] = (c["expiration"] - today).dt.days
    c = c[(c["DTE"] >= min_dte) & (c["DTE"] <= max_dte)].copy()
    wanted = "CALL" if direction == "CALL" else "PUT"
    c = c[c["type"] == wanted].copy()
    if c.empty:
        return c

    c["mid"] = (c["bid"].fillna(0) + c["ask"].fillna(0)) / 2
    c["spread_pct"] = np.where(c["mid"] > 0, (c["ask"]-c["bid"]) / c["mid"] * 100, np.nan)
    c["distance_pct"] = (c["strike"] - spot).abs() / spot * 100
    # Liquidity-first heuristic; not a prediction of profit.
    c["liq_score"] = (
        np.log1p(c["volume"].fillna(0)) * 8 +
        np.log1p(c["openInterest"].fillna(0)) * 5 -
        c["spread_pct"].clip(0, 100) * 0.5
    )
    c["delta_abs"] = c["delta"].abs() if "delta" in c.columns else np.nan
    c["contract_score"] = c["liq_score"] + np.where(
        c["delta_abs"].between(.45,.70, inclusive="both"), 12, 0
    ) + np.where(c["distance_pct"] <= 7, 8, 0)
    cols = ["contractSymbol","expiration","DTE","strike","bid","ask","mid","volume","openInterest","impliedVolatility","delta","gamma","theta","vega","spread_pct","contract_score"]
    cols = [z for z in cols if z in c.columns]
    return c.sort_values("contract_score", ascending=False)[cols].head(15)

st.title("📈 Calar AI Market Analyzer")
st.caption("Analizador educativo de acciones y opciones. No ejecuta órdenes.")

with st.sidebar:
    st.header("Configuración")
    tickers_text = st.text_area("Acciones", DEFAULT_TICKERS, height=100)
    period = st.selectbox("Histórico", ["6mo","1y","2y","5y"], index=1)
    interval = st.selectbox("Intervalo", ["1d","1h"], index=0)
    run = st.button("🔎 Analizar mercado", type="primary", use_container_width=True)

if run:
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
    bench = {}
    for b in BENCHMARKS:
        try: bench[b] = get_history(b, period, interval)
        except Exception: pass
    benchmark = bench.get("QQQ", bench.get("SPY"))
    results = []
    details = {}
    for t in tickers:
        try:
            df = get_history(t, period, interval)
            res = score_equity(df, benchmark)
            if res:
                results.append({k: res[k] for k in ["Score","Signal","Price","RSI","RVOL","ROC20"]} | {"Ticker":t})
                details[t] = res
        except Exception as e:
            st.warning(f"{t}: no se pudo analizar ({e})")

    if not results:
        st.error("No se obtuvieron datos. Revisa tu conexión o los símbolos.")
        st.stop()

    table = pd.DataFrame(results).sort_values("Score", ascending=False)
    st.subheader("Ranking")
    st.dataframe(table[["Ticker","Score","Signal","Price","RSI","RVOL","ROC20"]], use_container_width=True, hide_index=True)

    best = table.iloc[0]["Ticker"]
    st.subheader(f"🥇 Mejor puntuación: {best}")
    d = details[best]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Score", f'{d["Score"]:.1f}/100')
    c2.metric("Precio", f'${d["Price"]:.2f}')
    c3.metric("RSI", f'{d["RSI"]:.1f}')
    c4.metric("RVOL", f'{d["RVOL"]:.2f}x')
    c5.metric("ROC20", f'{d["ROC20"]:.1f}%')

    st.write("**Factores favorables**")
    for z in d["Reasons"]: st.write("• " + z)
    st.write("**Riesgos / invalidaciones**")
    for z in d["Risks"]: st.write("• " + z)

    st.subheader("Gráfico")
    chart = d["Data"][["Close","EMA20","EMA40","EMA200"]].tail(250)
    st.line_chart(chart)

    st.subheader("🧩 Contratos de opciones candidatos (solo análisis)")
    direction = "CALL" if d["Score"] >= 65 else "PUT"
    st.info(f"Sesgo del modelo: {direction}. La selección de contrato prioriza liquidez, spread y delta; no garantiza rentabilidad.")
    chain = get_option_chain(best)
    cand = option_candidates(chain, d["Price"], direction)
    if cand.empty:
        st.warning("No se pudo obtener una cadena de opciones compatible desde el proveedor.")
    else:
        st.dataframe(cand, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar ranking CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"ranking_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

else:
    st.info("Configura las acciones y pulsa **Analizar mercado**.")
    st.markdown("""
### Qué hace esta V1
- Escanea una lista de acciones.
- Calcula tendencia, momentum, RSI, volumen relativo y rupturas.
- Produce un score 0–100.
- Muestra razones y riesgos.
- Consulta cadenas de opciones disponibles.
- Filtra candidatos por DTE, liquidez, spread y delta.
- No envía órdenes ni conecta con una cuenta de broker.

### Importante
Los datos de Yahoo Finance pueden tener retrasos, limitaciones o cambios de disponibilidad. Para un sistema profesional intradía conviene sustituir el proveedor por un feed de mercado contratado.
""")
