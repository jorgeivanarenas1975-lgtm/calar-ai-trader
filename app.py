
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
import math

st.set_page_config(page_title="Calar AI Trader V5", page_icon="📈", layout="wide")

# =========================
# CONFIG
# =========================
DEFAULT_TICKERS = (
    "NVDA,AVGO,META,AMD,AMZN,GOOGL,MSFT,AAPL,TSLA,MU,CRWD,ANET,"
    "PLTR,ORCL,APP,ARM,SNOW,ZBRA,SMCI,TSM,SPY,QQQ"
)
AUTO_REFRESH_MIN = 30

# =========================
# DATA HELPERS
# =========================
@st.cache_data(ttl=1500, show_spinner=False)
def get_data(ticker):
    t = ticker.upper().strip()
    h1 = yf.download(t, period="30d", interval="1h", auto_adjust=False, progress=False, threads=False)
    m5 = yf.download(t, period="10d", interval="5m", auto_adjust=False, progress=False, threads=False)
    return clean_ohlcv(h1), clean_ohlcv(m5)

def clean_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = {c: c.title() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    needed = ["Open","High","Low","Close","Volume"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    return df[needed].dropna()

def indicators(df):
    x = df.copy()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        x["High"]-x["Low"],
        (x["High"]-x["Close"].shift()).abs(),
        (x["Low"]-x["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    x["VolAvg20"] = x["Volume"].rolling(20).mean()
    return x

def candle_direction(row):
    if row["Close"] > row["Open"]:
        return "GREEN"
    if row["Close"] < row["Open"]:
        return "RED"
    return "DOJI"

def qqq_trend():
    q1, q5 = get_data("QQQ")
    if q1.empty:
        return None
    q = indicators(q1)
    r = q.iloc[-1]
    return {
        "bull": r["Close"] > r["EMA20"] > r["EMA50"],
        "bear": r["Close"] < r["EMA20"] < r["EMA50"],
        "rsi": float(r["RSI"]) if pd.notna(r["RSI"]) else np.nan
    }

def five_min_confirmation(m5, side):
    if m5.empty or len(m5) < 20:
        return False, "Sin datos 5M suficientes"
    x = indicators(m5)
    last = x.iloc[-1]
    prev = x.iloc[-2]
    recent_high = x["High"].iloc[-8:-2].max()
    recent_low = x["Low"].iloc[-8:-2].min()
    if side == "CALL":
        ok = last["Close"] > recent_high and last["Close"] > last["EMA20"]
        return bool(ok), "Ruptura 5M confirmada" if ok else "Esperando ruptura 5M"
    else:
        ok = last["Close"] < recent_low and last["Close"] < last["EMA20"]
        return bool(ok), "Ruptura 5M confirmada" if ok else "Esperando ruptura 5M"

def analyze(ticker, qqq):
    h1, m5 = get_data(ticker)
    if h1.empty or len(h1) < 60:
        return {"ticker": ticker, "status": "SIN DATOS", "side": "—"}

    x = indicators(h1)
    # IMPORTANT: only closed candles are used. Last candle can be incomplete,
    # so use the two candles immediately before the latest bar.
    c1 = x.iloc[-3]
    c2 = x.iloc[-2]
    last = x.iloc[-2]

    d1, d2 = candle_direction(c1), candle_direction(c2)
    price = float(last["Close"])
    atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.02
    ema20, ema50 = float(last["EMA20"]), float(last["EMA50"])
    rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else 50
    vol_ok = bool(last["Volume"] >= last["VolAvg20"]) if pd.notna(last["VolAvg20"]) else False

    call_candle = d1 == "GREEN" and d2 == "GREEN"
    put_candle = d1 == "RED" and d2 == "RED"

    call_trend = price > ema20 > ema50
    put_trend = price < ema20 < ema50
    call_rsi = 55 <= rsi <= 70
    put_rsi = 30 <= rsi <= 45
    call_qqq = bool(qqq and qqq["bull"])
    put_qqq = bool(qqq and qqq["bear"])

    call_5m, call_5m_txt = five_min_confirmation(m5, "CALL")
    put_5m, put_5m_txt = five_min_confirmation(m5, "PUT")

    # Score is secondary. The two-candle rule is mandatory.
    call_points = sum([
        call_candle, call_trend, call_rsi, vol_ok, call_qqq, call_5m,
        float(c2["Close"]) > float(c1["High"])
    ])
    put_points = sum([
        put_candle, put_trend, put_rsi, vol_ok, put_qqq, put_5m,
        float(c2["Close"]) < float(c1["Low"])
    ])

    # Strict signal: mandatory 1H candle pair + trend + QQQ + 5M confirmation.
    call_signal = call_candle and call_trend and call_qqq and call_5m
    put_signal = put_candle and put_trend and put_qqq and put_5m

    side = "CALL" if call_signal else ("PUT" if put_signal else "NO OPERAR")
    points = call_points if call_signal else (put_points if put_signal else max(call_points, put_points))
    confidence = min(98, round(55 + points * 6))

    # Practical target/stop based on ATR; not a guarantee.
    if side == "CALL":
        target = price + 1.5 * atr
        stop = price - 1.0 * atr
        setup = "🟢 CALL"
    elif side == "PUT":
        target = price - 1.5 * atr
        stop = price + 1.0 * atr
        setup = "🔴 PUT"
    else:
        target = np.nan
        stop = np.nan
        setup = "⏸ NO OPERAR"

    return {
        "ticker": ticker,
        "status": "OK",
        "side": side,
        "setup": setup,
        "price": price,
        "candle1": d1,
        "candle2": d2,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi,
        "volume_ok": vol_ok,
        "qqq_ok": call_qqq if call_signal else put_qqq,
        "confirm5m": call_5m if call_signal else put_5m,
        "points": points,
        "confidence": confidence if side != "NO OPERAR" else 0,
        "target": target,
        "stop": stop,
        "atr": atr,
        "call_points": call_points,
        "put_points": put_points,
        "five_txt": call_5m_txt if side != "PUT" else put_5m_txt,
        "last_time": str(last.name),
        "h1": x,
        "m5": m5
    }

# =========================
# OPTIONS
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def option_contracts(ticker):
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return pd.DataFrame()
        today = pd.Timestamp.now(tz=None).normalize()
        rows = []
        # Prefer expirations 21–60 days out.
        selected = []
        for e in expiries:
            d = pd.Timestamp(e)
            days = (d - today).days
            if 21 <= days <= 60:
                selected.append((e, days))
        if not selected:
            selected = [(e, (pd.Timestamp(e)-today).days) for e in expiries[:5]]
        for e, days in selected[:6]:
            chain = tk.option_chain(e)
            for typ, df in [("CALL", chain.calls), ("PUT", chain.puts)]:
                if df is None or df.empty:
                    continue
                z = df.copy()
                z["type"] = typ
                z["expiry"] = e
                z["dte"] = days
                rows.append(z)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def suggest_option(ticker, side, stock_price, target_price):
    if side not in ("CALL","PUT"):
        return None
    df = option_contracts(ticker)
    if df.empty:
        return None
    z = df[df["type"] == side].copy()
    if z.empty:
        return None

    z["dist"] = (z["strike"] - stock_price).abs()
    # Prefer near-ATM, 21–45 DTE, usable liquidity and nonzero bid/ask.
    z["dte_pref"] = (z["dte"] - 30).abs()
    z["liq"] = z["openInterest"].fillna(0) + z["volume"].fillna(0)
    z["spread"] = (z["ask"] - z["bid"]).replace([np.inf,-np.inf], np.nan)

    # Basic liquidity filter.
    candidates = z[(z["dte"] >= 21) & (z["dte"] <= 45) & (z["liq"] >= 10)].copy()
    if candidates.empty:
        candidates = z.copy()
    candidates["score"] = (
        candidates["dist"] / max(stock_price, 1)
        + candidates["dte_pref"] / 100
        - np.log1p(candidates["liq"]) / 100
    )
    row = candidates.sort_values("score").iloc[0]
    mid = np.nan
    if pd.notna(row.get("bid")) and pd.notna(row.get("ask")):
        mid = (float(row["bid"]) + float(row["ask"])) / 2
    elif pd.notna(row.get("lastPrice")):
        mid = float(row["lastPrice"])

    # For a simple educational estimate, show stock target and contract BE.
    strike = float(row["strike"])
    be = strike + mid if side == "CALL" and pd.notna(mid) else (
        strike - mid if side == "PUT" and pd.notna(mid) else np.nan
    )
    return {
        "expiry": row["expiry"], "dte": int(row["dte"]), "strike": strike,
        "bid": row.get("bid", np.nan), "ask": row.get("ask", np.nan),
        "mid": mid, "breakeven": be, "open_interest": row.get("openInterest", 0),
        "volume": row.get("volume", 0)
    }

# =========================
# UI
# =========================
st.title("📈 Calar AI Trader V5")
st.caption("Motor de confirmación 1H + 5M • CALL/PUT • filtro de mercado • sugerencia de contrato")

st.info(
    "REGLA V5: CALL exige 2 velas verdes consecutivas cerradas en 1H. "
    "PUT exige 2 velas rojas consecutivas cerradas en 1H. "
    "Después exige tendencia EMA20/EMA50, confirmación de QQQ y ruptura en 5M. "
    "Si no se cumplen, el sistema dice NO OPERAR."
)

with st.sidebar:
    st.header("⚙️ Configuración")
    tickers_text = st.text_area("Acciones a revisar", DEFAULT_TICKERS, height=150)
    tickers = [x.strip().upper() for x in tickers_text.replace(";", ",").split(",") if x.strip()]
    st.write(f"**{len(tickers)} símbolos**")
    if st.button("🔄 Actualizar ahora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.write("**Revisión automática:** cada 30 minutos")
    st.write("Usa siempre velas 1H cerradas; no toma decisiones con la vela 1H todavía abierta.")

# Automatic refresh every 30 minutes.
# The user can also force a refresh with the sidebar button.
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30 * 60 * 1000, key="calar_v5_refresh")
except Exception:
    pass

if "results" not in st.session_state or st.button("▶️ Ejecutar análisis"):
    with st.spinner("Analizando mercado 1H + 5M..."):
        qqq = qqq_trend()
        results = []
        for t in tickers:
            try:
                results.append(analyze(t, qqq))
            except Exception as e:
                results.append({"ticker": t, "status": f"ERROR: {e}", "side": "—"})
        st.session_state.results = results
else:
    results = st.session_state.results

valid = [r for r in results if r.get("status") == "OK"]

if valid:
    table = []
    for r in valid:
        table.append({
            "Ticker": r["ticker"],
            "Señal V5": r["setup"],
            "Precio": round(r["price"], 2),
            "1H": f'{r["candle1"]} + {r["candle2"]}',
            "EMA20/50": "OK" if ((r["side"]=="CALL" and r["ema20"]>r["ema50"]) or (r["side"]=="PUT" and r["ema20"]<r["ema50"])) else "NO",
            "RSI": round(r["rsi"], 1),
            "Volumen": "OK" if r["volume_ok"] else "NO",
            "QQQ": "OK" if r["qqq_ok"] else "NO",
            "5M": "OK" if r["confirm5m"] else "ESPERAR",
            "Score": r["points"],
            "Confianza setup": f'{r["confidence"]}/100' if r["side"] != "NO OPERAR" else "—",
        })
    df_table = pd.DataFrame(table)
    # Put confirmed signals first.
    df_table["_sort"] = df_table["Señal V5"].map(lambda x: 0 if "CALL" in x or "PUT" in x else 1)
    df_table = df_table.sort_values(["_sort","Score"], ascending=[True,False]).drop(columns="_sort")
    st.subheader("🔎 Escáner V5")
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    confirmed = [r for r in valid if r["side"] in ("CALL","PUT")]
    if confirmed:
        st.subheader("🏆 Señales confirmadas")
        for r in sorted(confirmed, key=lambda a: a["confidence"], reverse=True)[:5]:
            with st.container(border=True):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Setup", r["setup"])
                c2.metric("Confianza del setup", f'{r["confidence"]}/100')
                c3.metric("Precio acción", f'${r["price"]:.2f}')
                c4.metric("Objetivo acción", f'${r["target"]:.2f}')
                st.write(
                    f'**{r["ticker"]}** — 1H: {r["candle1"]} + {r["candle2"]} | '
                    f'RSI {r["rsi"]:.1f} | EMA20 {r["ema20"]:.2f} / EMA50 {r["ema50"]:.2f} | '
                    f'5M: {r["five_txt"]}'
                )
                st.write(f'**Stop técnico aproximado:** ${r["stop"]:.2f}  •  **ATR 1H:** ${r["atr"]:.2f}')
                opt = suggest_option(r["ticker"], r["side"], r["price"], r["target"])
                if opt:
                    st.write(
                        f'**Contrato orientativo:** {r["side"]} strike **${opt["strike"]:.0f}**, '
                        f'expira **{opt["expiry"]}** ({opt["dte"]} DTE). '
                        f'Bid/Ask: ${opt["bid"]:.2f} / ${opt["ask"]:.2f}.'
                    )
                    if pd.notna(opt["breakeven"]):
                        st.write(f'**Break-even aproximado al vencimiento:** ${opt["breakeven"]:.2f}')
                else:
                    st.warning("No se pudo obtener una cadena de opciones suficientemente líquida para sugerir contrato.")

                st.caption(
                    "El strike es una referencia técnica, no una garantía de ganancia. "
                    "La prima puede caer aunque la acción suba por IV, theta, spread y tiempo."
                )
    else:
        st.warning("V5 no encuentra ninguna señal que cumpla TODOS los filtros. Esto es intencional: NO FORZAR una operación.")

st.divider()
st.subheader("📌 Cómo interpretar V5")
st.markdown("""
- **CALL confirmado:** 2 velas 1H verdes cerradas + tendencia alcista + QQQ alcista + ruptura 5M.
- **PUT confirmado:** 2 velas 1H rojas cerradas + tendencia bajista + QQQ bajista + ruptura 5M.
- **NO OPERAR:** falta una confirmación. V5 no inventa una señal para llenar el ranking.
- **Objetivo/stop:** se calculan con ATR 1H para crear un escenario de salida; no son predicciones.
- **Revisión:** puedes pulsar **Actualizar ahora** o dejar la página abierta para refrescar cada 30 minutos si tu versión de Streamlit soporta el fragmento automático.
""")

st.caption(
    f"Calar AI Trader V5 • Datos proporcionados por Yahoo Finance mediante yfinance • "
    f"Última ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
