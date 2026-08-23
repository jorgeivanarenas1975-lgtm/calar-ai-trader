
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Calar AI Trader V4.1", page_icon="📈", layout="wide")
st.markdown("<style>.block-container{padding:.7rem}.stButton button{width:100%;min-height:2.7rem}</style>", unsafe_allow_html=True)

SP500_URL="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK="AAPL,AMD,AMZN,AVGO,BAC,CRWD,GOOGL,META,MSFT,MU,NVDA,PLTR,SNOW,TSLA,ZBRA"

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500():
    try:
        t=next(x for x in pd.read_html(SP500_URL) if "Symbol" in x.columns)
        return sorted(set(t["Symbol"].astype(str).str.replace(".","-",regex=False))),"Wikipedia / S&P 500"
    except Exception:
        return sorted(set(FALLBACK.split(","))),"Fallback"

def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))

def ind(df):
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    needed=["Open","High","Low","Close","Volume"]
    if any(c not in x for c in needed): return pd.DataFrame()
    for c in needed: x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["Open","High","Low","Close"])
    for n in [10,20,40,100,200]:
        x[f"PM{n}"]=x.Close.rolling(n).mean()
        x[f"EMA{n}"]=x.Close.ewm(span=n,adjust=False).mean()
    x["RSI"]=rsi(x.Close)
    prev=x.Close.shift(1)
    tr=pd.concat([x.High-x.Low,(x.High-prev).abs(),(x.Low-prev).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14).mean()
    x["RVOL"]=x.Volume/x.Volume.rolling(20).mean()
    x["ROC5"]=x.Close.pct_change(5)*100
    x["ROC20"]=x.Close.pct_change(20)*100
    x["H20"]=x.High.rolling(20).max().shift(1)
    x["L20"]=x.Low.rolling(20).min().shift(1)
    x["RET1"]=x.Close.pct_change()*100
    return x.dropna()

def candles(df):
    if len(df)<5:return []
    x=df
    def v(i):
        r=x.iloc[i]; o,h,l,c=map(float,[r.Open,r.High,r.Low,r.Close])
        body=abs(c-o); rng=max(h-l,1e-9)
        return o,h,l,c,body,rng,h-max(o,c),min(o,c)-l
    o,h,l,c,b,r,u,d=v(-1); out=[]
    if d>=2*b and u<=.6*b and b/r<=.45: out.append("Martillo")
    if u>=2*b and d<=.6*b and b/r<=.45: out.append("Estrella fugaz")
    o1,h1,l1,c1,b1,r1,u1,d1=v(-2)
    if c1<o1 and c>o and c>=o1 and o<=c1: out.append("Envolvente alcista")
    if c1>o1 and c<o and o>=c1 and c<=o1: out.append("Envolvente bajista")
    if b/r<=.10: out.append("Doji")
    return list(dict.fromkeys(out))

@st.cache_data(ttl=300,show_spinner=False)
def batch(symbols,period="6mo",interval="1d"):
    out={}
    for i in range(0,len(symbols),80):
        part=symbols[i:i+80]
        try:
            d=yf.download(part,period=period,interval=interval,auto_adjust=False,progress=False,group_by="ticker",threads=True)
            if isinstance(d.columns,pd.MultiIndex):
                for t in part:
                    try:
                        z=d[t].dropna()
                        if not z.empty: out[t]=z
                    except: pass
            elif len(part)==1 and not d.empty: out[part[0]]=d.dropna()
        except: pass
        time.sleep(.05)
    return out

@st.cache_data(ttl=300,show_spinner=False)
def optchain(t):
    try:
        tk=yf.Ticker(t); rows=[]
        for e in list(tk.options or [])[:10]:
            try:
                q=tk.option_chain(e)
                for typ,z in [("CALL",q.calls),("PUT",q.puts)]:
                    if z is not None and not z.empty:
                        a=z.copy(); a["type"]=typ; a["expiration"]=e; rows.append(a)
            except: pass
        return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    except: return pd.DataFrame()

def score(df,bench=None,intra=None):
    x=ind(df)
    if len(x)<40:return None
    r=x.iloc[-1]; call=50.; put=50.; ce=[]; pe=[]
    # Trend
    for ok,n,a,b in [
        (r.Close>r.PM10,6,"Precio > PM10","Precio < PM10"),
        (r.PM10>r.PM20,8,"PM10 > PM20","PM10 < PM20"),
        (r.PM20>r.PM40,7,"PM20 > PM40","PM20 < PM40"),
        (r.Close>r.PM100,6,"Precio > PM100","Precio < PM100"),
        (r.Close>r.PM200,6,"Precio > PM200","Precio < PM200")]:
        (ce if ok else pe).append(a if ok else b); call+=n if ok else 0; put+=0 if ok else n
    # Recent momentum gets strong weight
    if r.ROC5>2: call+=7; ce.append(f"ROC5 +{r.ROC5:.1f}%")
    if r.ROC5<-2: put+=7; pe.append(f"ROC5 {r.ROC5:.1f}%")
    if r.ROC20>5: call+=7; ce.append(f"ROC20 +{r.ROC20:.1f}%")
    if r.ROC20<-5: put+=7; pe.append(f"ROC20 {r.ROC20:.1f}%")
    if 55<=r.RSI<=70: call+=6; ce.append(f"RSI {r.RSI:.1f}")
    if 30<=r.RSI<=45: put+=6; pe.append(f"RSI {r.RSI:.1f}")
    if r.RSI>75: call-=5; ce.append("RSI sobrecomprado: penalización")
    if r.RSI<25: put-=5; pe.append("RSI sobrevendido: penalización")
    # Volume
    if r.RVOL>=1.5:
        if r.Close>r.PM10: call+=7; ce.append(f"RVOL {r.RVOL:.2f}x")
        if r.Close<r.PM10: put+=7; pe.append(f"RVOL {r.RVOL:.2f}x")
    # Breakout / breakdown
    if r.Close>r.H20: call+=10; ce.append("Ruptura máximo 20D")
    if r.Close<r.L20: put+=10; pe.append("Ruptura mínimo 20D")
    # Benchmark
    if bench is not None and not bench.empty:
        b=ind(bench)
        if not b.empty:
            q=b.iloc[-1]
            if q.Close>q.PM20: call+=5; ce.append("QQQ > PM20")
            else: put+=5; pe.append("QQQ < PM20")
    # Candles
    for p in candles(x):
        if p in ["Martillo","Envolvente alcista"]: call+=4; ce.append(p)
        elif p in ["Estrella fugaz","Envolvente bajista"]: put+=4; pe.append(p)
    # Intraday confirmation: critical, not optional for an ENTRY
    intraday_ok=False
    if intra is not None and not intra.empty:
        y=ind(intra)
        if not y.empty:
            q=y.iloc[-1]
            intraday_ok=True
            if q.Close>q.PM10: call+=8; ce.append("Intraday > PM10")
            else: put+=8; pe.append("Intraday < PM10")
            if q.PM10>q.PM20: call+=7; ce.append("Intraday PM10 > PM20")
            else: put+=7; pe.append("Intraday PM10 < PM20")
            if q.RVOL>=1.3:
                if q.Close>q.PM10: call+=5; ce.append(f"Intraday RVOL {q.RVOL:.2f}x")
                elif q.Close<q.PM10: put+=5; pe.append(f"Intraday RVOL {q.RVOL:.2f}x")
            for p in candles(y):
                if p in ["Martillo","Envolvente alcista"]: call+=3; ce.append("Intraday "+p)
                elif p in ["Estrella fugaz","Envolvente bajista"]: put+=3; pe.append("Intraday "+p)
    call=float(np.clip(call,0,100)); put=float(np.clip(put,0,100))
    gap=call-put
    if call>=80 and gap>=15 and intraday_ok: signal="🟢 CALL — GATILLO CONFIRMADO"
    elif put>=80 and gap<=-15 and intraday_ok: signal="🔴 PUT — GATILLO CONFIRMADO"
    elif call>=70 and gap>=10: signal="🟢 CALL — VIGILAR"
    elif put>=70 and gap<=-10: signal="🔴 PUT — VIGILAR"
    else: signal="🟡 NO OPERAR"
    return dict(call=call,put=put,signal=signal,price=float(r.Close),rsi=float(r.RSI),rvol=float(r.RVOL),atr=float(r.ATR),data=x,ce=ce,pe=pe)

def option_candidates(ch,spot,direction):
    if ch.empty:return pd.DataFrame()
    z=ch.copy(); z["expiration"]=pd.to_datetime(z.expiration)
    z["DTE"]=(z.expiration-pd.Timestamp.now().normalize()).dt.days
    z=z[(z.DTE>=14)&(z.DTE<=60)&(z.type==direction)].copy()
    if z.empty:return z
    z["mid"]=(z.bid.fillna(0)+z.ask.fillna(0))/2
    z["spread_pct"]=np.where(z.mid>0,(z.ask-z.bid)/z.mid*100,np.nan)
    z["dist_pct"]=(z.strike-spot).abs()/spot*100
    delta=z.delta.abs() if "delta" in z else pd.Series(np.nan,index=z.index)
    z["liq_score"]=np.log1p(z.volume.fillna(0))*7+np.log1p(z.openInterest.fillna(0))*5-z.spread_pct.clip(0,100)*.7
    z["contract_score"]=z.liq_score+np.where(delta.between(.45,.65),15,0)+np.where(z.dist_pct<=7,8,0)
    cols=["contractSymbol","expiration","DTE","strike","bid","ask","mid","volume","openInterest","impliedVolatility","delta","gamma","theta","vega","spread_pct","contract_score"]
    return z.sort_values("contract_score",ascending=False)[[c for c in cols if c in z]].head(15)

def target_model(price,atr,direction):
    # Educational scenario targets, not guaranteed outcomes.
    move=max(atr*1.5,price*0.015)
    if direction=="CALL":
        entry=price; target=price+move; stop=price-max(atr*1.0,price*.01)
    else:
        entry=price; target=price-move; stop=price+max(atr*1.0,price*.01)
    return entry,target,stop

def select_contract(q,spot,direction):
    if q.empty:return None
    # Prefer delta 0.45-0.65, DTE 21-45, tight spread, reasonable distance.
    z=q.copy()
    z["delta_abs"]=z.delta.abs() if "delta" in z else np.nan
    z["fit"]=np.where(z.delta_abs.between(.45,.65),20,0)+np.where(z.DTE.between(21,45),15,0)+np.where(z.dist_pct<=7,10,0)-z.spread_pct.fillna(100).clip(0,100)
    return z.sort_values("fit",ascending=False).iloc[0]

def contract_scenario(row,price,target,stop,direction):
    if row is None:return None
    premium=float(row.mid) if pd.notna(row.mid) else np.nan
    delta=float(row.delta) if "delta" in row and pd.notna(row.delta) else np.nan
    if not np.isfinite(premium) or premium<=0:return None
    if not np.isfinite(delta): return dict(entry=premium,target=np.nan,stop=np.nan,roi=np.nan)
    # Linear delta approximation for a scenario only.
    if direction=="CALL":
        target_p=premium+max(0,target-price)*abs(delta)
        stop_p=max(0,premium-max(0,price-stop)*abs(delta))
    else:
        target_p=premium+max(0,price-target)*abs(delta)
        stop_p=max(0,premium-max(0,stop-price)*abs(delta))
    roi=(target_p-premium)/premium*100 if premium else np.nan
    return dict(entry=premium,target=target_p,stop=stop_p,roi=roi)

# ---------------- UI ----------------
syms,source=get_sp500()
st.title("📈 CALAR AI TRADER V4.1")
st.caption("Motor de confirmación: tendencia + momentum + estructura + volumen + intradía + opciones")

with st.sidebar:
    full=st.checkbox("Analizar S&P 500 completo",True)
    if full: selected=syms
    else:
        txt=st.text_area("Tickers","NVDA,CRWD,AMD,SNOW,AMZN,META,MSFT,GOOGL,TSLA,MU,AVGO,PLTR,ZBRA")
        selected=[x.strip().upper() for x in txt.split(",") if x.strip()]
    period=st.selectbox("Histórico diario",["6mo","1y","2y"],1)
    topn=st.slider("Confirmación intradía Top N",20,100,50,10)
    st.divider()
    st.markdown("**TradingView:** usa la importación CSV si quieres contrastar datos exportados de tu gráfico. TradingView no ofrece una API general de datos de cuenta; los webhooks sirven para alertas, no para extraer toda la cuenta.")
    tv=st.file_uploader("CSV de TradingView (opcional)",type=["csv"])

scan=st.button("🔄 ESCANEAR V4.1",type="primary",use_container_width=True)

if scan:
    bench=batch(["QQQ"],period,"1d").get("QQQ",pd.DataFrame())
    daily=batch(selected,period,"1d")
    rows=[]; details={}
    prog=st.progress(0)
    for i,t in enumerate(selected):
        try:
            d=daily.get(t)
            if d is not None and not d.empty:
                s=score(d,bench)
                if s:
                    rows.append([t,s["call"],s["put"],s["signal"],s["price"],s["rsi"],s["rvol"]])
                    details[t]=s
        except: pass
        prog.progress((i+1)/max(1,len(selected)))
    prog.empty()
    rank=pd.DataFrame(rows,columns=["Ticker","CALL","PUT","Señal","Precio","RSI","RVOL"])
    rank["Score"]=rank[["CALL","PUT"]].max(axis=1)
    rank=rank.sort_values(["Score","CALL","PUT"],ascending=False)

    finalists=rank.head(min(topn,len(rank))).Ticker.tolist()
    intra=batch(finalists,"5d","5m")
    for t in finalists:
        if t in details and t in intra:
            try:
                s=score(daily[t],bench,intra[t]); details[t]=s
                rank.loc[rank.Ticker==t,["CALL","PUT","Señal","Precio","RSI","RVOL"]]=[s["call"],s["put"],s["signal"],s["price"],s["rsi"],s["rvol"]]
            except: pass
    rank["Score"]=rank[["CALL","PUT"]].max(axis=1)
    rank=rank.sort_values(["Score","CALL","PUT"],ascending=False)

    st.subheader("🏆 Ranking V4.1")
    st.dataframe(rank.head(100),use_container_width=True,hide_index=True)
    a,b,c=st.columns(3)
    with a:
        st.subheader("🟢 CALL")
        st.dataframe(rank[rank.Señal.str.contains("CALL")].head(10),use_container_width=True,hide_index=True)
    with b:
        st.subheader("🔴 PUT")
        st.dataframe(rank[rank.Señal.str.contains("PUT")].head(10),use_container_width=True,hide_index=True)
    with c:
        st.subheader("🟡 NO OPERAR")
        st.dataframe(rank[rank.Señal.str.contains("NO OPERAR")].head(10),use_container_width=True,hide_index=True)

    actionable=rank[rank.Señal.str.contains("GATILLO CONFIRMADO")]
    best=actionable.iloc[0].Ticker if not actionable.empty else rank.iloc[0].Ticker
    s=details[best]
    direction="CALL" if s["call"]>s["put"] else "PUT"
    st.subheader(f"🎯 {best} — {s['signal']}")
    m=st.columns(5)
    m[0].metric("CALL",f"{s['call']:.0f}/100"); m[1].metric("PUT",f"{s['put']:.0f}/100")
    m[2].metric("Precio",f"${s['price']:.2f}"); m[3].metric("RSI",f"{s['rsi']:.1f}"); m[4].metric("RVOL",f"{s['rvol']:.2f}x")
    with st.expander("🧠 Evidencia alcista",True):
        for x in s["ce"]: st.write("✅",x)
    with st.expander("🧠 Evidencia bajista",True):
        for x in s["pe"]: st.write("🔻",x)
    st.line_chart(s["data"][["Close","PM10","PM20","PM40","PM100","PM200"]].tail(180))

    q=option_candidates(optchain(best),s["price"],direction)
    st.subheader(f"🎯 Contrato candidato para simulación — {direction}")
    if q.empty:
        st.warning("No se encontró una cadena compatible.")
    else:
        chosen=select_contract(q,s["price"],direction)
        entry,target,stop=target_model(s["price"],s["atr"],direction)
        scenario=contract_scenario(chosen,s["price"],target,stop,direction)
        show=q.copy()
        st.dataframe(show,use_container_width=True,hide_index=True)
        if chosen is not None:
            st.markdown("### 📌 Escenario del contrato")
            cc=st.columns(6)
            cc[0].metric("Strike",f"{chosen.strike:g}")
            cc[1].metric("DTE",f"{int(chosen.DTE)}")
            cc[2].metric("Delta",f"{chosen.delta:.2f}" if pd.notna(chosen.delta) else "N/D")
            cc[3].metric("Prima mid",f"${chosen.mid:.2f}")
            cc[4].metric("Objetivo acción",f"${target:.2f}")
            cc[5].metric("Stop acción",f"${stop:.2f}")
            if scenario:
                st.info(f"Prima objetivo aproximada (modelo lineal de Delta): **${scenario['target']:.2f}** · prima stop aproximada: **${scenario['stop']:.2f}** · retorno teórico del escenario: **{scenario['roi']:.1f}%**.")
            st.caption("Estos objetivos son escenarios matemáticos, no una garantía de ganancia. La aproximación lineal por Delta no modela IV, gamma, theta, spread ni cambios de volatilidad.")
    st.download_button("⬇️ Descargar ranking",rank.to_csv(index=False).encode(),f"calar_v41_{datetime.now():%Y%m%d_%H%M}.csv","text/csv")
else:
    st.info("Pulsa ESCANEAR V4.1 para ejecutar el análisis.")
    st.markdown("""
### V4.1 — Correcciones clave
- No convierte una tendencia de fondo en una entrada automática.
- El **gatillo intradía** es obligatorio para una señal “GATILLO CONFIRMADO”.
- Penaliza sobrecompra/sobreventa.
- Aumenta el peso de momentum y estructura reciente.
- CALL y PUT son independientes.
- Incluye **NO OPERAR** cuando no hay ventaja clara.
- Selecciona un contrato candidato por DTE, Delta, liquidez, spread y distancia.
- Calcula un **escenario de objetivo/stop de la acción** y una aproximación de prima usando Delta.
- Permite importar un CSV exportado de TradingView para contraste manual.
""")
    st.warning("Esta aplicación es un analizador/escenario educativo. Los datos de Yahoo Finance pueden tener retrasos y las opciones cambian rápidamente.")
