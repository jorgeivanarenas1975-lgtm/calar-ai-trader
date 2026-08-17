import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Calar AI Trader V3",page_icon="📈",layout="wide")
st.markdown("<style>.block-container{padding:.7rem}.stButton button{width:100%;min-height:2.7rem}</style>",unsafe_allow_html=True)
DEFAULT="NVDA,CRWD,AMD,SNOW,AMZN,META,MSFT,GOOGL,TSLA,MU,AVGO,PLTR"

def rsi(s,n=14):
    d=s.diff(); u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    v=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+u/v.replace(0,np.nan))

def ind(df):
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    for c in ["Open","High","Low","Close","Volume"]: x[c]=pd.to_numeric(x[c],errors="coerce")
    x["EMA20"]=x.Close.ewm(span=20,adjust=False).mean(); x["EMA40"]=x.Close.ewm(span=40,adjust=False).mean(); x["EMA200"]=x.Close.ewm(span=200,adjust=False).mean()
    x["RSI"]=rsi(x.Close); p=x.Close.shift(1)
    tr=pd.concat([x.High-x.Low,(x.High-p).abs(),(x.Low-p).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14).mean(); x["RVOL"]=x.Volume/x.Volume.rolling(20).mean()
    x["ROC20"]=x.Close.pct_change(20)*100; x["ROC5"]=x.Close.pct_change(5)*100
    x["H20"]=x.High.rolling(20).max().shift(1); x["L20"]=x.Low.rolling(20).min().shift(1)
    return x.dropna()

@st.cache_data(ttl=300,show_spinner=False)
def hist(t,p,i):
    d=yf.download(t,period=p,interval=i,auto_adjust=False,progress=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna()

def score(df,bench):
    x=ind(df)
    if len(x)<30:return None
    r=x.iloc[-1]; call=50.; put=50.; cg=[]; pg=[]
    tests=[(r.Close>r.EMA20,8,"Precio > EMA20","Precio < EMA20"),
           (r.EMA20>r.EMA40,10,"EMA20 > EMA40","EMA20 < EMA40"),
           (r.Close>r.EMA200,12,"Precio > EMA200","Precio < EMA200")]
    for ok,n,a,b in tests:
        if ok: call+=n; cg.append(a)
        else: put+=n; pg.append(b)
    if 55<=r.RSI<=70: call+=9; cg.append(f"RSI alcista {r.RSI:.1f}")
    if r.RSI<=45: put+=9; pg.append(f"RSI bajista {r.RSI:.1f}")
    if r.ROC20>5: call+=9; cg.append(f"Momentum 20D +{r.ROC20:.1f}%")
    if r.ROC20<-5: put+=9; pg.append(f"Momentum 20D {r.ROC20:.1f}%")
    if r.ROC5>2: call+=5; cg.append(f"Momentum 5D +{r.ROC5:.1f}%")
    if r.ROC5<-2: put+=5; pg.append(f"Momentum 5D {r.ROC5:.1f}%")
    if r.RVOL>=1.5:
        if r.Close>r.EMA20: call+=7; cg.append(f"RVOL {r.RVOL:.2f}x")
        else: put+=7; pg.append(f"RVOL {r.RVOL:.2f}x")
    if r.Close>r.H20: call+=12; cg.append("Ruptura máximo 20D")
    if r.Close<r.L20: put+=12; pg.append("Ruptura mínimo 20D")
    if bench is not None:
        b=ind(bench).iloc[-1]
        if b.Close>b.EMA20: call+=5; cg.append("QQQ > EMA20")
        else: put+=5; pg.append("QQQ < EMA20")
    call=float(np.clip(call,0,100)); put=float(np.clip(put,0,100))
    sig="🟢 CALL CONFIRMADA" if call>=75 and call-put>=12 else "🔴 PUT CONFIRMADA" if put>=75 and put-call>=12 else "🟢 SESGO CALL" if call>=65 and call>put else "🔴 SESGO PUT" if put>=65 and put>call else "🟡 ESPERAR"
    return dict(call=call,put=put,signal=sig,price=float(r.Close),rsi=float(r.RSI),rvol=float(r.RVOL),data=x,cg=cg,pg=pg)

@st.cache_data(ttl=300,show_spinner=False)
def chain(t):
    tk=yf.Ticker(t); out=[]
    for e in list(tk.options or [])[:10]:
        try:
            c=tk.option_chain(e)
            for typ,z in [("CALL",c.calls),("PUT",c.puts)]:
                if z is not None and not z.empty:
                    q=z.copy(); q["type"]=typ; q["expiration"]=e; out.append(q)
        except: pass
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def contracts(ch,spot,direction):
    if ch.empty:return pd.DataFrame()
    z=ch.copy(); z["expiration"]=pd.to_datetime(z.expiration); z["DTE"]=(z.expiration-pd.Timestamp.now().normalize()).dt.days
    z=z[(z.DTE>=7)&(z.DTE<=60)&(z.type==direction)].copy()
    if z.empty:return z
    z["mid"]=(z.bid.fillna(0)+z.ask.fillna(0))/2
    z["spread_pct"]=np.where(z.mid>0,(z.ask-z.bid)/z.mid*100,np.nan)
    z["dist_pct"]=(z.strike-spot).abs()/spot*100
    delta=z.delta.abs() if "delta" in z else pd.Series(np.nan,index=z.index)
    z["contract_score"]=np.log1p(z.volume.fillna(0))*8+np.log1p(z.openInterest.fillna(0))*5-z.spread_pct.clip(0,100)*.5+np.where(delta.between(.45,.70),12,0)+np.where(z.dist_pct<=7,8,0)
    cols=["contractSymbol","expiration","DTE","strike","bid","ask","mid","volume","openInterest","impliedVolatility","delta","gamma","theta","vega","spread_pct","contract_score"]
    return z.sort_values("contract_score",ascending=False)[[c for c in cols if c in z]].head(10)

st.title("📈 CALAR AI TRADER V3")
st.caption("CALL + PUT independientes · confirmación · opciones · actualización manual cada 30 min")
with st.sidebar:
    tickers=st.text_area("Universo",DEFAULT)
    period=st.selectbox("Histórico",["6mo","1y","2y"],1)
    interval=st.selectbox("Marco",["1d","1h"],0)
go=st.button("🔄 ANALIZAR MERCADO AHORA",type="primary",use_container_width=True)

if go:
    syms=[s.strip().upper() for s in tickers.split(",") if s.strip()]; b=hist("QQQ",period,interval); rows=[]; det={}; bar=st.progress(0)
    for i,t in enumerate(syms):
        try:
            d=score(hist(t,period,interval),b)
            if d: rows.append([t,d["call"],d["put"],d["signal"],d["price"],d["rsi"],d["rvol"]]); det[t]=d
        except: pass
        bar.progress((i+1)/len(syms))
    bar.empty()
    if not rows: st.error("No se obtuvieron datos."); st.stop()
    rank=pd.DataFrame(rows,columns=["Ticker","CALL","PUT","Señal","Precio","RSI","RVOL"]).sort_values(["CALL","PUT"],ascending=False)
    st.subheader("🏆 Ranking CALL vs PUT"); st.dataframe(rank,use_container_width=True,hide_index=True)
    a,b=st.columns(2)
    with a:
        st.subheader("🟢 CALL confirmadas"); st.dataframe(rank[(rank.CALL>=75)&(rank.CALL-rank.PUT>=12)][["Ticker","CALL","PUT","Señal"]],use_container_width=True,hide_index=True)
    with b:
        st.subheader("🔴 PUT confirmadas"); st.dataframe(rank[(rank.PUT>=75)&(rank.PUT-rank.CALL>=12)][["Ticker","CALL","PUT","Señal"]],use_container_width=True,hide_index=True)
    best=rank.iloc[0].Ticker; d=det[best]; st.subheader(f"🔎 {best} — {d['signal']}")
    c1,c2,c3,c4=st.columns(4); c1.metric("CALL",f"{d['call']:.0f}/100"); c2.metric("PUT",f"{d['put']:.0f}/100"); c3.metric("RSI",f"{d['rsi']:.1f}"); c4.metric("RVOL",f"{d['rvol']:.2f}x")
    with st.expander("🧠 Evidencia CALL",True):
        for x in d["cg"]: st.write("✅",x)
    with st.expander("🧠 Evidencia PUT",True):
        for x in d["pg"]: st.write("🔻",x)
    st.line_chart(d["data"][["Close","EMA20","EMA40","EMA200"]].tail(180))
    direction="CALL" if d["call"]>=d["put"] else "PUT"
    st.subheader(f"🎯 Contratos candidatos — {direction}")
    q=contracts(chain(best),d["price"],direction)
    if q.empty: st.warning("No se encontró una cadena compatible.")
    else: st.dataframe(q,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Descargar ranking CSV",rank.to_csv(index=False).encode(),f"calar_v3_{datetime.now():%Y%m%d_%H%M}.csv","text/csv")
else:
    st.info("Pulsa ANALIZAR MERCADO AHORA. No necesitas enviar fotos.")
    st.markdown("### V3 incluye\n- CALL Score y PUT Score independientes\n- Confirmación de dirección\n- Rankings separados CALL/PUT\n- Momentum 5D/20D, RSI, RVOL, EMA20/40/200\n- Rupturas de 20 sesiones y filtro QQQ\n- Cadena de opciones con DTE, liquidez, spread y griegas disponibles\n\n**Nota:** esta versión actualiza manualmente al pulsar el botón. Automatización cada 30 minutos y datos profesionales en tiempo real son la siguiente etapa.")
