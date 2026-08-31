import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math

st.set_page_config(page_title="Calar AI Trader V6", page_icon="📈", layout="wide")

DEFAULT_TICKERS = "NVDA,AVGO,META,AMD,AMZN,GOOGL,MSFT,AAPL,TSLA,MU,CRWD,ANET,PLTR,ORCL,APP,ARM,SNOW,ZBRA,SMCI,TSM,SPY,QQQ"
AUTO_REFRESH_MIN = 30
RISK_FREE = 0.04

@st.cache_data(ttl=1500, show_spinner=False)
def get_data(ticker):
    t = ticker.upper().strip()
    h1 = yf.download(t, period="60d", interval="1h", auto_adjust=False, progress=False, threads=False)
    m5 = yf.download(t, period="10d", interval="5m", auto_adjust=False, progress=False, threads=False)
    return clean_ohlcv(h1), clean_ohlcv(m5)

def clean_ohlcv(df):
    if df is None or df.empty: return pd.DataFrame()
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x.columns=[str(c).title() for c in x.columns]
    need=["Open","High","Low","Close","Volume"]
    if any(c not in x.columns for c in need): return pd.DataFrame()
    return x[need].dropna()

def add_indicators(df):
    x=df.copy()
    x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean()
    x["EMA50"]=x.Close.ewm(span=50,adjust=False).mean()
    x["EMA200"]=x.Close.ewm(span=200,adjust=False).mean()
    delta=x.Close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/14,adjust=False).mean(); al=loss.ewm(alpha=1/14,adjust=False).mean()
    rs=ag/al.replace(0,np.nan); x["RSI"]=100-(100/(1+rs))
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1)
    x["TR"]=tr; x["ATR"]=tr.rolling(14).mean()
    x["ATR_PCT"]=100*x.ATR/x.Close
    x["VolAvg20"]=x.Volume.rolling(20).mean(); x["RelVol"]=x.Volume/x.VolAvg20.replace(0,np.nan)
    # ADX
    up=x.High.diff(); down=-x.Low.diff()
    plus_dm=np.where((up>down)&(up>0),up,0.0); minus_dm=np.where((down>up)&(down>0),down,0.0)
    atr14=x.ATR.replace(0,np.nan)
    pdi=100*pd.Series(plus_dm,index=x.index).rolling(14).mean()/atr14
    mdi=100*pd.Series(minus_dm,index=x.index).rolling(14).mean()/atr14
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x["ADX"]=dx.rolling(14).mean(); x["+DI"]=pdi; x["-DI"]=mdi
    # MACD
    e12=x.Close.ewm(span=12,adjust=False).mean(); e26=x.Close.ewm(span=26,adjust=False).mean()
    x["MACD"]=e12-e26; x["MACD_SIG"]=x.MACD.ewm(span=9,adjust=False).mean(); x["MACD_H"]=x.MACD-x.MACD_SIG
    # VWAP proxy (session-aware for intraday data)
    typical=(x.High+x.Low+x.Close)/3
    dates=pd.Series(x.index.date,index=x.index)
    pv=typical*x.Volume
    x["VWAP"]=pv.groupby(dates).cumsum()/x.Volume.groupby(dates).cumsum().replace(0,np.nan)
    return x

def direction(r):
    if r.Close>r.Open:return "GREEN"
    if r.Close<r.Open:return "RED"
    return "DOJI"

def market_regime():
    out={}
    for t in ["SPY","QQQ"]:
        h,_=get_data(t)
        if h.empty: out[t]=None; continue
        x=add_indicators(h); r=x.iloc[-2]
        out[t]={"price":float(r.Close),"bull":bool(r.Close>r.EMA20>r.EMA50),"bear":bool(r.Close<r.EMA20<r.EMA50),"rsi":float(r.RSI) if pd.notna(r.RSI) else 50,"adx":float(r.ADX) if pd.notna(r.ADX) else 0}
    return out

def five_min_confirmation(m5, side):
    if m5.empty or len(m5)<30:return False,"Sin datos 5M suficientes"
    x=add_indicators(m5); last=x.iloc[-2]
    recent=x.iloc[-10:-2]
    if side=="CALL":
        breakout=float(last.Close)>float(recent.High.max()) and float(last.Close)>float(last.EMA20) and float(last.Close)>float(last.VWAP)
        vol=float(last.RelVol)>=1.0 if pd.notna(last.RelVol) else False
        ok=breakout and vol
        return ok, "Ruptura 5M + VWAP + volumen" if ok else "Esperando ruptura 5M/VWAP/volumen"
    breakout=float(last.Close)<float(recent.Low.min()) and float(last.Close)<float(last.EMA20) and float(last.Close)<float(last.VWAP)
    vol=float(last.RelVol)>=1.0 if pd.notna(last.RelVol) else False
    ok=breakout and vol
    return ok, "Ruptura 5M + VWAP + volumen" if ok else "Esperando ruptura 5M/VWAP/volumen"

def earnings_risk(ticker):
    try:
        cal=yf.Ticker(ticker).calendar
        if cal is None or len(cal)==0:return None
        dates=[]
        if isinstance(cal,pd.DataFrame):
            for c in cal.columns:
                if "earnings" in str(c).lower(): dates += list(pd.to_datetime(cal[c],errors="coerce").dropna())
            for idx in cal.index:
                if "earnings" in str(idx).lower(): dates += list(pd.to_datetime(cal.loc[idx],errors="coerce").dropna())
        elif isinstance(cal,dict):
            for k,v in cal.items():
                if "earnings" in str(k).lower():
                    vals=v if isinstance(v,(list,tuple,pd.Series,np.ndarray)) else [v]
                    dates += list(pd.to_datetime(vals,errors="coerce"))
        dates=[d.tz_localize(None) if getattr(d,"tzinfo",None) else d for d in dates if pd.notna(d)]
        if not dates:return None
        today=pd.Timestamp.now().normalize()
        nearest=min(dates,key=lambda d:abs((d.normalize()-today).days))
        return int((nearest.normalize()-today).days)
    except Exception:return None

def analyze(ticker, markets, avoid_earnings=True):
    h1,m5=get_data(ticker)
    if h1.empty or len(h1)<80:return {"ticker":ticker,"status":"SIN DATOS","side":"—"}
    x=add_indicators(h1)
    c1=x.iloc[-3]; c2=x.iloc[-2]; r=x.iloc[-2]
    d1,d2=direction(c1),direction(c2); price=float(r.Close)
    atr=float(r.ATR) if pd.notna(r.ATR) else price*.02
    ema20,ema50=float(r.EMA20),float(r.EMA50); rsi=float(r.RSI) if pd.notna(r.RSI) else 50
    adx=float(r.ADX) if pd.notna(r.ADX) else 0; relvol=float(r.RelVol) if pd.notna(r.RelVol) else 0
    macd=float(r.MACD_H) if pd.notna(r.MACD_H) else 0
    swing_high=float(x.High.iloc[-21:-2].max()); swing_low=float(x.Low.iloc[-21:-2].min())
    call_pair=d1=="GREEN" and d2=="GREEN"; put_pair=d1=="RED" and d2=="RED"
    call_trend=price>ema20>ema50; put_trend=price<ema20<ema50
    call_rsi=52<=rsi<=70; put_rsi=30<=rsi<=48
    call_mom=macd>0; put_mom=macd<0
    call_adx=adx>=18 and float(r.PlusDI)>float(r.MinusDI) if pd.notna(r.PlusDI) and pd.notna(r.MinusDI) else False
    put_adx=adx>=18 and float(r.MinusDI)>float(r.PlusDI) if pd.notna(r.PlusDI) and pd.notna(r.MinusDI) else False
    volume_ok=relvol>=1.0
    spy=markets.get("SPY"); qqq=markets.get("QQQ")
    call_market=bool(spy and qqq and spy["bull"] and qqq["bull"])
    put_market=bool(spy and qqq and spy["bear"] and qqq["bear"])
    call5,call5txt=five_min_confirmation(m5,"CALL"); put5,put5txt=five_min_confirmation(m5,"PUT")
    # Do not chase: reject if already > 1 ATR beyond EMA20.
    not_chasing_call=price <= ema20 + 1.0*atr; not_chasing_put=price >= ema20 - 1.0*atr
    # Support/resistance proximity: prefer calls above recent resistance and puts below recent support.
    call_break=price>swing_high; put_break=price<swing_low
    earn=earnings_risk(ticker) if avoid_earnings else None
    earnings_block=earn is not None and abs(earn)<=2
    # Score: mandatory pair + multi-factor confluence.
    call_checks={"2 velas 1H":call_pair,"tendencia":call_trend,"SPY+QQQ":call_market,"5M":call5,"volumen":volume_ok,"RSI":call_rsi,"ADX":call_adx,"MACD":call_mom,"no persecución":not_chasing_call,"ruptura 1H":call_break}
    put_checks={"2 velas 1H":put_pair,"tendencia":put_trend,"SPY+QQQ":put_market,"5M":put5,"volumen":volume_ok,"RSI":put_rsi,"ADX":put_adx,"MACD":put_mom,"no persecución":not_chasing_put,"ruptura 1H":put_break}
    call_score=round(100*sum(call_checks.values())/len(call_checks)); put_score=round(100*sum(put_checks.values())/len(put_checks))
    mandatory_call=call_pair and call_trend and call_market and call5
    mandatory_put=put_pair and put_trend and put_market and put5
    call_signal=mandatory_call and call_score>=80 and not earnings_block
    put_signal=mandatory_put and put_score>=80 and not earnings_block
    if call_signal and put_signal: side="CALL" if call_score>=put_score else "PUT"
    elif call_signal: side="CALL"
    elif put_signal: side="PUT"
    else: side="NO OPERAR"
    score=call_score if side=="CALL" else put_score if side=="PUT" else max(call_score,put_score)
    if side=="CALL": target=price+2*atr; stop=price-atr
    elif side=="PUT": target=price-2*atr; stop=price+atr
    else: target=stop=np.nan
    return {"ticker":ticker,"status":"OK","side":side,"price":price,"candle1":d1,"candle2":d2,"ema20":ema20,"ema50":ema50,"rsi":rsi,"adx":adx,"relvol":relvol,"macd_hist":macd,"call_score":call_score,"put_score":put_score,"score":score,"confidence":score if side!="NO OPERAR" else 0,"target":target,"stop":stop,"atr":atr,"call_checks":call_checks,"put_checks":put_checks,"five_txt":call5txt if side!="PUT" else put5txt,"earnings_days":earn,"earnings_block":earnings_block,"h1":x,"m5":m5}

def norm_cdf(x): return .5*(1+math.erf(x/math.sqrt(2)))
def bs_delta(S,K,T,sigma,side):
    if not all(np.isfinite([S,K,T,sigma])) or T<=0 or sigma<=0:return np.nan
    d1=(math.log(S/K)+(RISK_FREE+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    return norm_cdf(d1) if side=="CALL" else norm_cdf(d1)-1

def bs_price(S,K,T,sigma,side):
    if not all(np.isfinite([S,K,T,sigma])) or T<=0 or sigma<=0:return np.nan
    d1=(math.log(S/K)+(RISK_FREE+.5*sigma*sigma)*T)/(sigma*math.sqrt(T)); d2=d1-sigma*math.sqrt(T)
    if side=="CALL":return S*norm_cdf(d1)-K*math.exp(-RISK_FREE*T)*norm_cdf(d2)
    return K*math.exp(-RISK_FREE*T)*norm_cdf(-d2)-S*norm_cdf(-d1)

@st.cache_data(ttl=900,show_spinner=False)
def option_contracts(ticker):
    try:
        tk=yf.Ticker(ticker); exps=tk.options
        if not exps:return pd.DataFrame()
        today=pd.Timestamp.now().normalize(); rows=[]
        for e in exps:
            d=pd.Timestamp(e); days=(d-today).days
            if 21<=days<=60:
                ch=tk.option_chain(e)
                for typ,z in [("CALL",ch.calls),("PUT",ch.puts)]:
                    if z is not None and not z.empty:
                        q=z.copy(); q["type"]=typ; q["expiry"]=e; q["dte"]=days; rows.append(q)
        return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    except Exception:return pd.DataFrame()

def suggest_option(ticker,side,S,target):
    z=option_contracts(ticker)
    if z.empty:return None
    z=z[z.type==side].copy()
    if z.empty:return None
    z["liq"]=z.openInterest.fillna(0)+z.volume.fillna(0)
    z["spread_pct"]=((z.ask-z.bid)/((z.ask+z.bid)/2).replace(0,np.nan))*100
    z=z[(z.dte>=21)&(z.dte<=45)&(z.liq>=20)&(z.spread_pct<=15)].copy()
    if z.empty:return None
    today=pd.Timestamp.now().normalize()
    deltas=[]
    for _,row in z.iterrows():
        iv=float(row.impliedVolatility) if pd.notna(row.get("impliedVolatility")) else np.nan
        T=max((pd.Timestamp(row.expiry)-today).days/365,1/365)
        d=bs_delta(S,float(row.strike),T,iv,side)
        deltas.append(d)
    z["delta_est"]=deltas
    z["delta_dist"]=(z.delta_est.abs()-0.50).abs()
    z["atm_dist"]=(z.strike-S).abs()/S
    z["dte_dist"]=(z.dte-30).abs()/30
    z["score_select"]=z.delta_dist+z.atm_dist+0.35*z.dte_dist+0.002*z.spread_pct
    row=z.sort_values("score_select").iloc[0]
    strike=float(row.strike); iv=float(row.impliedVolatility) if pd.notna(row.get("impliedVolatility")) else np.nan
    bid=float(row.bid) if pd.notna(row.get("bid")) else np.nan; ask=float(row.ask) if pd.notna(row.get("ask")) else np.nan
    mid=(bid+ask)/2 if np.isfinite(bid) and np.isfinite(ask) else float(row.lastPrice) if pd.notna(row.get("lastPrice")) else np.nan
    T=max(int(row.dte)/365,1/365); delta=float(row.delta_est) if pd.notna(row.delta_est) else np.nan
    opt_target=bs_price(target,strike,T,iv,side) if np.isfinite(iv) else np.nan
    be=strike+mid if side=="CALL" else strike-mid if np.isfinite(mid) else np.nan
    return {"expiry":row.expiry,"dte":int(row.dte),"strike":strike,"bid":bid,"ask":ask,"mid":mid,"iv":iv,"delta":delta,"breakeven":be,"target_premium":opt_target,"oi":int(row.openInterest) if pd.notna(row.openInterest) else 0,"volume":int(row.volume) if pd.notna(row.volume) else 0,"spread_pct":float(row.spread_pct)}

# UI
st.title("📈 Calar AI Trader V6")
st.caption("Confluencia 1H + 5M • régimen SPY/QQQ • volumen • ADX • MACD • ATR • control de persecución • opciones")
st.info("REGLA HEREDADA: CALL exige 2 velas verdes consecutivas cerradas en 1H. PUT exige 2 velas rojas consecutivas cerradas en 1H. Se mantienen EMA20/EMA50, SPY+QQQ y ruptura 5M. V6 añade filtros de tendencia, momentum, volumen, volatilidad, estructura y liquidez de opciones. Si no hay confluencia suficiente: NO OPERAR.")
with st.sidebar:
    st.header("⚙️ Configuración")
    txt=st.text_area("Acciones a revisar",DEFAULT_TICKERS,height=170)
    tickers=[s.strip().upper() for s in txt.replace(";",",").split(",") if s.strip()]
    avoid=st.checkbox("🚫 Bloquear ±2 días alrededor de earnings",True)
    min_score=st.slider("Puntaje mínimo V6",70,95,80)
    if st.button("🔄 Actualizar ahora",use_container_width=True): st.cache_data.clear(); st.rerun()
    st.divider(); st.write("**Revisión automática:** cada 30 minutos")
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=AUTO_REFRESH_MIN*60*1000,key="v6_refresh")
except Exception: pass

if "v6_results" not in st.session_state or st.button("▶️ Ejecutar análisis"):
    with st.spinner("Analizando régimen, tendencia, volumen, estructura y opciones..."):
        mk=market_regime(); res=[]
        for t in tickers:
            try:
                r=analyze(t,mk,avoid); 
                # Apply UI threshold without changing raw scores.
                if r.get("side") in ("CALL","PUT") and r.get("score",0)<min_score:r["side"]="NO OPERAR"; r["confidence"]=0
                res.append(r)
            except Exception as e: res.append({"ticker":t,"status":f"ERROR: {e}","side":"—"})
        st.session_state.v6_results=res; st.session_state.v6_market=mk
else: res=st.session_state.v6_results; mk=st.session_state.get("v6_market",{})

if mk:
    a,b=st.columns(2)
    for col,t in zip((a,b),("SPY","QQQ")):
        m=mk.get(t)
        if m: col.metric(t,"🟢 ALCISTA" if m["bull"] else "🔴 BAJISTA" if m["bear"] else "🟡 MIXTO",f'RSI {m["rsi"]:.1f} • ADX {m["adx"]:.1f}')
valid=[r for r in res if r.get("status")=="OK"]
if valid:
    rows=[]
    for r in valid:
        rows.append({"Ticker":r["ticker"],"Señal V6":"🟢 CALL" if r["side"]=="CALL" else "🔴 PUT" if r["side"]=="PUT" else "⏸ NO OPERAR","Precio":round(r["price"],2),"1H":f'{r["candle1"]} + {r["candle2"]}',"Score":r["score"],"RSI":round(r["rsi"],1),"ADX":round(r["adx"],1),"RelVol":round(r["relvol"],2),"5M":"OK" if (r["five_txt"].startswith("Ruptura")) else "ESPERAR","Earnings":"BLOQUEADO" if r["earnings_block"] else (str(r["earnings_days"])+"d" if r["earnings_days"] is not None else "—")})
    tab=pd.DataFrame(rows).sort_values(["Señal V6","Score"],ascending=[True,False]); st.subheader("🔎 Escáner V6"); st.dataframe(tab,use_container_width=True,hide_index=True)
    confirmed=[r for r in valid if r["side"] in ("CALL","PUT")]
    if confirmed:
        st.subheader("🏆 Operaciones con máxima confluencia")
        for r in sorted(confirmed,key=lambda x:x["score"],reverse=True)[:5]:
            with st.container(border=True):
                c1,c2,c3,c4=st.columns(4); c1.metric("Setup",r["side"]); c2.metric("Score V6",f'{r["score"]}/100'); c3.metric("Precio",f'${r["price"]:.2f}'); c4.metric("Objetivo acción",f'${r["target"]:.2f}')
                checks=r["call_checks"] if r["side"]=="CALL" else r["put_checks"]
                good=sum(checks.values()); st.write(f'**{r["ticker"]}** — {good}/{len(checks)} filtros positivos. ' + " • ".join([f'{k}: {"✓" if v else "—"}' for k,v in checks.items()]))
                st.write(f'**Stop técnico:** ${r["stop"]:.2f} • **ATR 1H:** ${r["atr"]:.2f} • **RSI:** {r["rsi"]:.1f} • **ADX:** {r["adx"]:.1f} • **RelVol:** {r["relvol"]:.2f}x')
                opt=suggest_option(r["ticker"],r["side"],r["price"],r["target"])
                if opt:
                    st.write(f'**Strike orientativo:** ${opt["strike"]:.0f} • vencimiento **{opt["expiry"]}** ({opt["dte"]} DTE) • delta estimada **{opt["delta"]:.2f}** • IV **{opt["iv"]*100:.1f}%** • spread **{opt["spread_pct"]:.1f}%**')
                    st.write(f'**Prima mid:** ${opt["mid"]:.2f} • **Break-even al vencimiento:** ${opt["breakeven"]:.2f} • **Prima teórica al objetivo de la acción:** ${opt["target_premium"]:.2f}')
                    if opt["iv"]>=0.70: st.warning("IV alta: la prima puede caer aunque la dirección sea correcta. Evitar perseguir la entrada.")
                else: st.warning("No hay contrato suficientemente líquido que cumpla los filtros de V6.")
    else: st.success("V6 no encuentra una operación con suficiente confluencia. NO OPERAR es una salida válida.")

st.divider(); st.subheader("🧠 Filosofía V6")
st.markdown("""
**V6 conserva el núcleo de V5 y agrega disciplina de selección:**
- **Precio primero:** estructura, ruptura y confirmación; no se anticipa una vela 1H todavía abierta.
- **Tendencia:** EMA20/EMA50 + régimen simultáneo de SPY y QQQ.
- **Fuerza:** ADX/+DI/-DI, MACD y RSI como filtros, no como señales aisladas.
- **Participación:** volumen relativo para validar rupturas.
- **No perseguir:** evita entradas demasiado alejadas de EMA20 en relación con ATR.
- **Riesgo:** objetivo 2×ATR y stop 1×ATR como escenario técnico; no garantía.
- **Opciones:** busca 21–45 DTE, delta cercana a 0.50, liquidez y spread razonable; muestra IV y una estimación teórica de prima al objetivo.
- **Catalizador:** bloquea ±2 días de earnings cuando el calendario está disponible.
- **Resultado:** menos señales, pero más selectivas. Si falta confluencia, muestra **NO OPERAR**.

**Importante:** ningún sistema técnico garantiza ganancias. La puntuación es una medida de confluencia, no una probabilidad estadística de éxito. La estimación de prima usa un modelo simplificado y puede diferir mucho del mercado real por IV, theta, spread y cambios bruscos de precio.
""")
