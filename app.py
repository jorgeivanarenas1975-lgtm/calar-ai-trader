import io
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Calar AI Trader V4", page_icon="📈", layout="wide")
st.markdown("<style>.block-container{padding:.7rem}.stButton button{width:100%;min-height:2.7rem}</style>", unsafe_allow_html=True)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FALLBACK = "AOS,ABT,ABBV,ACN,ADBE,AMD,AES,AFL,A,APD,AKAM,ALB,ARE,ALGN,ALLE,LNT,ALL,GOOGL,GOOG,MO,AMZN,AMCR,AMTM,AEE,AEP,AXP,AIG,AMT,AWK,AMP,AME,AMGN,APH,ADI,ANSS,APTV,ACGL,ADM,ANET,AJG,AIZ,T,ATO,ADSK,ADP,AZO,AVB,AVY,AXON,BKR,BALL,BAC,BK,BAX,BDX,BBY,TECH,BIIB,BLK,BX,BKNG,BA,BSX,BMY,BR,BRO,BF.B,BLDR,BG,BXP,CHRW,CDNS,CZR,CPT,CARR,CAT,CBOE,CBRE,CDW,CE,CVS,CVX,CMG,CB,CINF,CTAS,CSCO,C,CFG,CLX,CME,CMCSA,CAG,COP,ED,STZ,CEG,COO,CPRT,GLW,CPAY,CTVA,CSGP,COST,CTRA,CRWD,CROX,CCI,CSX,CMI,CVS,DHR,DRI,DVA,DAY,DECK,DE,DAL,DVAX,DG,DLTR,D,DPZ,DOV,DOW,DHI,DTE,DUK,DD,EMN,ETN,EBAY,ECL,EIX,EW,EA,ELV,EMR,ENPH,ETR,EFX,EQIX,EQR,ESS,EL,EG,EVRG,ES,EXC,EXPE,EXR,XOM,FFIV,FDS,FANG,FAST,FRT,FDX,FIS,FITB,FSLR,FE,FMC,FOX,FOXA,GRMN,IT,GE,GEHC,GEN,GNRC,GD,GIS,GM,GPC,GILD,GPN,GL,GS,HAL,HAS,HCA,HSIC,HSY,HES,HPE,HLT,HOLX,HD,HON,HRL,HST,HWM,HPQ,HUBB,HUM,HUN,IBM,IFF,ILMN,INCY,IR,INTC,ICE,IEX,IDXX,ITW,INCY,IRM,JBHT,JBL,JKHY,J,JNJ,JCI,JPM,JNPR,K,KDP,KEY,KEYS,KMB,KVUE,KMI,KHC,KR,LHX,LH,LRCX,LVS,LDOS,LEN,LII,LLY,LIN,LMT,L,LOW,LULU,LYB,MTB,MRO,MPC,MKTX,MAR,MET,MTD,MGM,MCHP,MU,MSCI,MA,MLM,MAS,MKSI,MOH,TAP,MDLZ,MKR,MPWR,MNST,MO,MS,MOS,MOTV,MSFT,MSCI,NDAQ,NTAP,NFLX,NEM,NWSA,NWS,NEE,NKE,NI,NRG,NDSN,NSC,NTRS,NOC,NCLH,NRDA,NVDA,NVR,NXPI,ORLY,OXY,ODFL,OMC,ON,OKE,ORCL,OTIS,PCAR,PKG,PLTR,PANW,PARA,PH,PAYX,PYPL,PENN,PEP,PFE,PCG,PM,PSX,PNR,PNW,POOL,PPG,PPL,PFG,PG,PGR,PLD,PRU,PEG,PTC,PSA,PHM,PWR,QCOM,QRVO,PX,RJF,O,RCL,REG,REGN,RF,RSG,ROK,ROL,ROP,ROST,RCL,SPGI,SLB,STX,SBAC,SMCI,SRE,NOW,SHW,SPG,SWK,SNA,SO,SOS,SBNY,STLD,SBUX,STT,STX,SYF,SNPS,SYY,TROW,TTWO,TPR,TRGP,TGT,TEL,TDY,TFX,TER,TSLA,TXN,TXT,TMO,TJX,TSCO,TT,TDG,TRV,TRMB,TFC,TYL,TSN,USB,UBER,UDR,ULTA,UNP,UAL,UPS,URI,UNH,UHS,VLO,VTR,VRSN,VRSK,VZ,VRTX,VTRS,VICI,V,WAB,WMT,DIS,WBD,WM,WAT,WEC,WFC,WELL,WST,WDC,WY,WRB,WYNN,XEL,XYL,YUM,ZBRA,ZBH,ZTS"

# ---------------------------
# Universe
# ---------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500():
    try:
        tables = pd.read_html(SP500_URL)
        t = next(x for x in tables if "Symbol" in x.columns)
        syms = t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return sorted(set(syms)), "Wikipedia / S&P 500 constituents"
    except Exception:
        return sorted(set(FALLBACK.split(","))), "Fallback universe"

# ---------------------------
# Indicators / candle patterns
# ---------------------------
def rsi(s, n=14):
    d = s.diff()
    u = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    v = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + u/v.replace(0, np.nan))

def add_indicators(df):
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in x: return pd.DataFrame()
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Open","High","Low","Close"])
    # PM = moving averages; include the user's PM10/20/40/100/200 concept.
    for n in [10,20,40,100,200]:
        x[f"PM{n}"] = x.Close.rolling(n).mean()
        x[f"EMA{n}"] = x.Close.ewm(span=n, adjust=False).mean()
    x["RSI"] = rsi(x.Close)
    p = x.Close.shift(1)
    tr = pd.concat([x.High-x.Low, (x.High-p).abs(), (x.Low-p).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    x["RVOL"] = x.Volume / x.Volume.rolling(20).mean()
    x["ROC5"] = x.Close.pct_change(5)*100
    x["ROC20"] = x.Close.pct_change(20)*100
    x["H20"] = x.High.rolling(20).max().shift(1)
    x["L20"] = x.Low.rolling(20).min().shift(1)
    return x.dropna()

def candle_patterns(df):
    if len(df) < 5: return []
    x = df.copy()
    rows=[]
    def vals(i):
        r=x.iloc[i]; o,h,l,c=[float(r[k]) for k in ["Open","High","Low","Close"]]
        body=abs(c-o); rng=max(h-l,1e-9); up=h-max(o,c); dn=min(o,c)-l
        return o,h,l,c,body,rng,up,dn
    o,h,l,c,b,r,u,d=vals(-1)
    # Hammer / shooting star
    if d >= 2*b and u <= 0.6*b and b/r <= .45: rows.append("Martillo")
    if u >= 2*b and d <= 0.6*b and b/r <= .45: rows.append("Estrella fugaz")
    o1,h1,l1,c1,b1,r1,u1,d1=vals(-2)
    # Engulfing bodies
    if c1<o1 and c>o and c>=o1 and o<=c1: rows.append("Envolvente alcista")
    if c1>o1 and c<o and o>=c1 and c<=o1: rows.append("Envolvente bajista")
    # Doji
    if b/r <= .10: rows.append("Doji")
    if len(x)>=4:
        a=vals(-3); m=vals(-2); z=vals(-1)
        ao,ah,al,ac,ab,ar,au,ad=a
        mo,mh,ml,mc,mb,mr,mu,md=m
        zo,zh,zl,zc,zb,zr,zu,zd=z
        if ac<ao and mb/mr<=.35 and zc>zo and zc >= (ao+ac)/2: rows.append("Estrella de la mañana")
        if ac>ao and mb/mr<=.35 and zc<zo and zc <= (ao+ac)/2: rows.append("Estrella de la tarde")
        if ac>ao and mc>mo and zc>zo and ac<mc<zc and (ab/ar)>.5 and (mb/mr)>.5 and (zb/zr)>.5:
            rows.append("Tres soldados blancos")
        if ac<ao and mc<mo and zc<zo and ac>mc>zc and (ab/ar)>.5 and (mb/mr)>.5 and (zb/zr)>.5:
            rows.append("Tres cuervos negros")
    return list(dict.fromkeys(rows))

def benchmark_score(bench):
    x=add_indicators(bench)
    if x.empty: return 0, []
    r=x.iloc[-1]; score=0; ev=[]
    if r.Close>r.PM20: score+=5; ev.append("QQQ > PM20")
    else: score-=5; ev.append("QQQ < PM20")
    if r.PM10>r.PM20: score+=5; ev.append("QQQ PM10 > PM20")
    else: score-=5; ev.append("QQQ PM10 < PM20")
    return score,ev

def score_symbol(df, bench=None, intraday=None):
    x=add_indicators(df)
    if len(x)<40: return None
    r=x.iloc[-1]; call=50.; put=50.; cg=[]; pg=[]
    # Trend / PM10 PM20 PM40 PM100 PM200
    tests=[
        (r.Close>r.PM10,7,"Precio > PM10","Precio < PM10"),
        (r.PM10>r.PM20,8,"PM10 > PM20","PM10 < PM20"),
        (r.PM20>r.PM40,7,"PM20 > PM40","PM20 < PM40"),
        (r.Close>r.PM100,8,"Precio > PM100","Precio < PM100"),
        (r.Close>r.PM200,8,"Precio > PM200","Precio < PM200"),
    ]
    for ok,n,a,b in tests:
        if ok: call+=n; cg.append(a)
        else: put+=n; pg.append(b)
    if 55<=r.RSI<=70: call+=6; cg.append(f"RSI alcista {r.RSI:.1f}")
    if 30<=r.RSI<=45: put+=6; pg.append(f"RSI bajista {r.RSI:.1f}")
    if r.ROC20>5: call+=7; cg.append(f"Momentum 20D +{r.ROC20:.1f}%")
    if r.ROC20<-5: put+=7; pg.append(f"Momentum 20D {r.ROC20:.1f}%")
    if r.ROC5>2: call+=5; cg.append(f"Momentum 5D +{r.ROC5:.1f}%")
    if r.ROC5<-2: put+=5; pg.append(f"Momentum 5D {r.ROC5:.1f}%")
    if r.RVOL>=1.5:
        if r.Close>r.PM20: call+=6; cg.append(f"RVOL {r.RVOL:.2f}x")
        elif r.Close<r.PM20: put+=6; pg.append(f"RVOL {r.RVOL:.2f}x")
    if r.Close>r.H20: call+=10; cg.append("Ruptura máximo 20D")
    if r.Close<r.L20: put+=10; pg.append("Ruptura mínimo 20D")
    if bench is not None:
        bs,bev=benchmark_score(bench)
        if bs>0: call+=min(bs,10); cg.extend(bev)
        elif bs<0: put+=min(abs(bs),10); pg.extend(bev)
    patterns=candle_patterns(x)
    for p in patterns:
        if p in ["Martillo","Envolvente alcista","Estrella de la mañana","Tres soldados blancos"]:
            call+=5; cg.append(f"Vela: {p}")
        elif p in ["Estrella fugaz","Envolvente bajista","Estrella de la tarde","Tres cuervos negros"]:
            put+=5; pg.append(f"Vela: {p}")
        elif p=="Doji":
            cg.append("Vela: Doji / indecisión")
            pg.append("Vela: Doji / indecisión")
    # Intraday confirmation is deliberately strong, so a bearish options-only signal cannot override bullish price action.
    intraday_evidence=[]
    if intraday is not None and not intraday.empty:
        y=add_indicators(intraday)
        if not y.empty:
            q=y.iloc[-1]
            if q.Close>q.PM10: call+=5; cg.append("Intraday precio > PM10")
            else: put+=5; pg.append("Intraday precio < PM10")
            if q.PM10>q.PM20: call+=5; cg.append("Intraday PM10 > PM20")
            else: put+=5; pg.append("Intraday PM10 < PM20")
            if q.RVOL>=1.3:
                if q.Close>q.PM10: call+=4; cg.append(f"Intraday RVOL {q.RVOL:.2f}x")
                elif q.Close<q.PM10: put+=4; pg.append(f"Intraday RVOL {q.RVOL:.2f}x")
            ip=candle_patterns(y)
            for p in ip:
                if p in ["Martillo","Envolvente alcista","Estrella de la mañana","Tres soldados blancos"]: call+=3; cg.append(f"Intraday vela: {p}")
                if p in ["Estrella fugaz","Envolvente bajista","Estrella de la tarde","Tres cuervos negros"]: put+=3; pg.append(f"Intraday vela: {p}")
    call=float(np.clip(call,0,100)); put=float(np.clip(put,0,100))
    gap=call-put
    if call>=75 and gap>=12: sig="🟢 CALL CONFIRMADA"
    elif put>=75 and gap<=-12: sig="🔴 PUT CONFIRMADA"
    elif call>=65 and gap>=8: sig="🟢 SESGO CALL"
    elif put>=65 and gap<=-8: sig="🔴 SESGO PUT"
    else: sig="🟡 ESPERAR / CONFLICTO"
    direction="CALL" if call>put else "PUT" if put>call else "NEUTRAL"
    confidence=max(call,put)
    return dict(call=call,put=put,signal=sig,direction=direction,confidence=confidence,price=float(r.Close),rsi=float(r.RSI),rvol=float(r.RVOL),data=x,cg=cg,pg=pg,patterns=patterns)

# ---------------------------
# Data download: batch daily, intraday only for finalists
# ---------------------------
@st.cache_data(ttl=300, show_spinner=False)
def batch_download(symbols, period="6mo", interval="1d"):
    out={}
    chunk=80
    for i in range(0,len(symbols),chunk):
        part=symbols[i:i+chunk]
        try:
            d=yf.download(part,period=period,interval=interval,auto_adjust=False,progress=False,group_by="ticker",threads=True)
            if isinstance(d.columns,pd.MultiIndex):
                # yfinance typically returns Ticker/field.
                for t in part:
                    try:
                        z=d[t].dropna()
                        if not z.empty: out[t]=z
                    except Exception: pass
            elif len(part)==1:
                out[part[0]]=d.dropna()
        except Exception: pass
        time.sleep(.1)
    return out

@st.cache_data(ttl=180, show_spinner=False)
def get_intraday(symbols):
    return batch_download(symbols,period="5d",interval="5m")

@st.cache_data(ttl=300, show_spinner=False)
def option_chain(t):
    try:
        tk=yf.Ticker(t); out=[]
        for e in list(tk.options or [])[:8]:
            try:
                c=tk.option_chain(e)
                for typ,z in [("CALL",c.calls),("PUT",c.puts)]:
                    if z is not None and not z.empty:
                        q=z.copy(); q["type"]=typ; q["expiration"]=e; out.append(q)
            except Exception: pass
        return pd.concat(out,ignore_index=True) if out else pd.DataFrame()
    except Exception: return pd.DataFrame()

def contracts(ch,spot,direction):
    if ch.empty:return pd.DataFrame()
    z=ch.copy(); z["expiration"]=pd.to_datetime(z.expiration); z["DTE"]=(z.expiration-pd.Timestamp.now().normalize()).dt.days
    z=z[(z.DTE>=7)&(z.DTE<=60)&(z.type==direction)].copy()
    if z.empty:return z
    z["mid"]=(z.bid.fillna(0)+z.ask.fillna(0))/2
    z["spread_pct"]=np.where(z.mid>0,(z.ask-z.bid)/z.mid*100,np.nan)
    z["dist_pct"]=(z.strike-spot).abs()/spot*100
    delta=z.delta.abs() if "delta" in z else pd.Series(np.nan,index=z.index)
    z["contract_score"]=np.log1p(z.volume.fillna(0))*7+np.log1p(z.openInterest.fillna(0))*5-z.spread_pct.clip(0,100)*.6+np.where(delta.between(.45,.70),14,0)+np.where(z.dist_pct<=7,8,0)
    cols=["contractSymbol","expiration","DTE","strike","bid","ask","mid","volume","openInterest","impliedVolatility","delta","gamma","theta","vega","spread_pct","contract_score"]
    return z.sort_values("contract_score",ascending=False)[[c for c in cols if c in z]].head(10)

# ---------------------------
# UI
# ---------------------------
syms,source=get_sp500()
st.title("📈 CALAR AI TRADER V4")
st.caption(f"Escáner completo S&P 500 · precio + PM10/20/40/100/200 + velas + volumen + opciones · universo: {len(syms)} símbolos")
with st.sidebar:
    st.markdown("### Universo")
    use_full=st.checkbox("Analizar S&P 500 completo",True)
    if use_full:
        selected=syms
        st.success(f"{len(selected)} acciones cargadas")
    else:
        txt=st.text_area("Tickers", "NVDA,CRWD,AMD,SNOW,AMZN,META,MSFT,GOOGL,TSLA,MU,AVGO,PLTR")
        selected=[s.strip().upper() for s in txt.split(",") if s.strip()]
    period=st.selectbox("Histórico diario",["6mo","1y","2y"],1)
    top_n=st.slider("Refinar intradía en top N",20,100,50,10)
    scan=st.button("🔄 ESCANEAR AHORA",type="primary",use_container_width=True)
    st.caption(f"Fuente universo: {source}")

if scan:
    bench_daily=batch_download(["QQQ"],period,"1d").get("QQQ",pd.DataFrame())
    daily=batch_download(selected,period,"1d")
    rows=[]; det={}
    bar=st.progress(0)
    for i,t in enumerate(selected):
        d=daily.get(t)
        if d is not None and not d.empty:
            try:
                s=score_symbol(d,bench_daily)
                if s:
                    rows.append([t,s["call"],s["put"],s["direction"],s["signal"],s["price"],s["rsi"],s["rvol"],", ".join(s["patterns"])])
                    det[t]=s
            except Exception: pass
        bar.progress((i+1)/max(len(selected),1))
    bar.empty()
    if not rows: st.error("No se obtuvieron datos. Revisa la conexión de datos."); st.stop()
    rank=pd.DataFrame(rows,columns=["Ticker","CALL","PUT","Dirección","Señal","Precio","RSI","RVOL","Velas"])
    rank["Score"]=rank[["CALL","PUT"]].max(axis=1)
    rank["Conflicto"]=np.where((rank.CALL>=65)&(rank.PUT>=65)&((rank.CALL-rank.PUT).abs()<12),"⚠️ Sí","No")
    rank=rank.sort_values(["Score","CALL","PUT"],ascending=False)

    # Intraday refine top N candidates and update signals.
    finalists=rank.head(min(top_n,len(rank)))["Ticker"].tolist()
    intra=get_intraday(finalists)
    for t in finalists:
        if t in det and t in intra:
            try:
                updated=score_symbol(daily[t],bench_daily,intra[t])
                det[t]=updated
                rank.loc[rank.Ticker==t,"CALL"]=updated["call"]
                rank.loc[rank.Ticker==t,"PUT"]=updated["put"]
                rank.loc[rank.Ticker==t,"Dirección"]=updated["direction"]
                rank.loc[rank.Ticker==t,"Señal"]=updated["signal"]
                rank.loc[rank.Ticker==t,"Velas"]=', '.join(updated["patterns"])
            except Exception: pass
    rank["Score"]=rank[["CALL","PUT"]].max(axis=1)
    rank["Conflicto"]=np.where((rank.CALL>=65)&(rank.PUT>=65)&((rank.CALL-rank.PUT).abs()<12),"⚠️ Sí","No")
    rank=rank.sort_values(["Score","CALL","PUT"],ascending=False)

    st.subheader("🏆 Ranking completo S&P 500")
    st.dataframe(rank.head(100),use_container_width=True,hide_index=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.subheader("🟢 TOP CALL")
        st.dataframe(rank[(rank.CALL>=75)&(rank.CALL-rank.PUT>=12)].sort_values("CALL",ascending=False).head(10)[["Ticker","CALL","PUT","Señal","Precio","Velas"]],use_container_width=True,hide_index=True)
    with c2:
        st.subheader("🔴 TOP PUT")
        st.dataframe(rank[(rank.PUT>=75)&(rank.PUT-rank.CALL>=12)].sort_values("PUT",ascending=False).head(10)[["Ticker","CALL","PUT","Señal","Precio","Velas"]],use_container_width=True,hide_index=True)
    with c3:
        st.subheader("🟡 ESPERAR / CONFLICTO")
        st.dataframe(rank[(rank.Señal.str.contains("ESPERAR|CONFLICTO"))].head(10)[["Ticker","CALL","PUT","Señal","Precio","Velas"]],use_container_width=True,hide_index=True)

    # Best actionable candidate: confirmed only. Otherwise best watchlist candidate.
    confirmed=rank[(rank.Señal.str.contains("CONFIRMADA"))].copy()
    best_t=confirmed.iloc[0].Ticker if not confirmed.empty else rank.iloc[0].Ticker
    d=det[best_t]
    st.subheader(f"🎯 Candidata principal: {best_t} — {d['signal']}")
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("CALL",f"{d['call']:.0f}/100")
    m2.metric("PUT",f"{d['put']:.0f}/100")
    m3.metric("Precio",f"${d['price']:.2f}")
    m4.metric("RSI",f"{d['rsi']:.1f}")
    m5.metric("RVOL",f"{d['rvol']:.2f}x")
    st.write("**Velas detectadas:**", ", ".join(d["patterns"]) if d["patterns"] else "Ninguna de las 8 principales")
    if d["call"]>=65 and d["put"]>=65 and abs(d["call"]-d["put"])<12:
        st.warning("⚠️ CONFLICTO: precio/indicadores no tienen suficiente separación. No se marca CALL/PUT confirmada.")
    with st.expander("🧠 Evidencia alcista",True):
        for x in d["cg"]: st.write("✅",x)
    with st.expander("🧠 Evidencia bajista",True):
        for x in d["pg"]: st.write("🔻",x)
    st.line_chart(d["data"][["Close","PM10","PM20","PM40","PM100","PM200"]].tail(180))

    direction="CALL" if d["call"]>d["put"] else "PUT" if d["put"]>d["call"] else "CALL"
    st.subheader(f"🎯 Contratos candidatos — {direction}")
    q=contracts(option_chain(best_t),d["price"],direction)
    if q.empty: st.warning("No se encontró una cadena compatible para la candidata principal.")
    else: st.dataframe(q,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Descargar ranking CSV",rank.to_csv(index=False).encode(),f"calar_v4_sp500_{datetime.now():%Y%m%d_%H%M}.csv","text/csv")
else:
    st.info("Pulsa ESCANEAR AHORA para analizar el S&P 500 completo.")
    st.markdown("### V4 — Cambios principales")
    st.markdown("- **S&P 500 completo**, no una lista fija de 12 acciones.\n- PM10/20/40/100/200.\n- Reconocimiento de 8 patrones de velas.\n- Confirmación intradía en las mejores candidatas.\n- CALL y PUT independientes.\n- Si precio y opciones se contradicen: **ESPERAR / CONFLICTO**.\n- Ranking Top CALL, Top PUT y zona de espera.\n- La cadena de opciones se consulta solo para las mejores candidatas, reduciendo carga y ruido.")
    st.warning("Los datos de Yahoo Finance pueden tener retraso y limitaciones de API. Esta herramienta es un escáner educativo, no una garantía de rentabilidad.")
