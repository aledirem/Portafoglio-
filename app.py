import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from io import StringIO
import re
from datetime import date, timedelta

# ============================================================
# 0) PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Portfolio Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1) PASSWORD GATE (Streamlit Secrets)
# ============================================================
def password_gate():
    expected = st.secrets.get("PASSWORD", None)
    if not expected:
        st.error("Secret mancante. Imposta PASSWORD in App settings → Secrets.")
        st.code('PASSWORD = "LaTuaPassword"', language="toml")
        st.stop()

    def _check():
        st.session_state["_pw_ok"] = (st.session_state.get("_pw_in", "") == expected)
        if st.session_state["_pw_ok"]:
            st.session_state["_pw_in"] = ""  # non tenere la password in memoria

    if st.session_state.get("_pw_ok", False):
        return

    st.markdown("## 🔐 Accesso riservato")
    st.text_input("Password", type="password", key="_pw_in", on_change=_check)
    if "_pw_ok" in st.session_state and not st.session_state["_pw_ok"]:
        st.error("Password errata.")
    st.stop()

password_gate()

# ============================================================
# 2) NUMERIC + CSV PARSER (Fineco robusto)
# ============================================================
def parse_eu_number(x) -> float:
    """Converte: '6 981,52' -> 6981.52, '3 055,20' -> 3055.20, gestisce NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    # rimuovi tutto tranne cifre e separatori
    s = re.sub(r"[^\d,.\-]", "", s)

    # caso EU con migliaia e decimali
    if "," in s and "." in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return np.nan


def find_header_row(lines):
    """
    Trova la riga header: cerca una riga che contenga almeno 'Titolo' e 'ISIN'.
    Funziona sia con file 'pulito' (header in prima riga) sia con righe extra sopra.
    """
    for i, ln in enumerate(lines):
        if "Titolo" in ln and "ISIN" in ln:
            return i
    return None


def parse_fineco_csv(uploaded_file) -> pd.DataFrame | None:
    try:
        raw = uploaded_file.getvalue().decode("latin1", errors="replace")
        lines = raw.splitlines()
        header_idx = find_header_row(lines)
        if header_idx is None:
            st.error("Non trovo l'intestazione (riga con 'Titolo' e 'ISIN').")
            return None

        # separatore: Fineco tipicamente ';'
        df = pd.read_csv(StringIO(raw), sep=";", skiprows=header_idx, header=0)
        df.columns = [str(c).strip() for c in df.columns]

        # colonne chiave (fuzzy match)
        def col_like(*keywords):
            for c in df.columns:
                lc = c.lower()
                if all(k in lc for k in keywords):
                    return c
            return None

        col_title = col_like("titolo") or "Titolo"
        col_isin = col_like("isin") or "ISIN"
        col_instr = col_like("strumento") or "Strumento"
        col_ccy = col_like("valuta") or "Valuta"
        col_mkt = col_like("valore", "mercato")
        col_cost = col_like("valore", "carico")
        col_pnl = col_like("var in valuta")  # Fineco spesso ha "Var in valuta"

        if col_mkt is None:
            st.error(f"Colonna 'Valore di mercato' non trovata. Colonne: {df.columns.tolist()}")
            return None

        # normalizza
        df["Titolo"] = df[col_title].astype(str)
        df["ISIN"] = df[col_isin].astype(str).str.upper()
        df["Strumento"] = df[col_instr].astype(str) if col_instr in df.columns else ""
        df["Valuta"] = df[col_ccy].astype(str).str.upper() if col_ccy in df.columns else "N/A"

        df["Valore_Mercato"] = df[col_mkt].apply(parse_eu_number)
        df["Valore_Carico"] = df[col_cost].apply(parse_eu_number) if col_cost else np.nan
        df["PnL_Export"] = df[col_pnl].apply(parse_eu_number) if col_pnl else np.nan

        # filtra righe non-asset
        df = df[df["Valore_Mercato"].notna()].copy()
        df = df[df["Valore_Mercato"] > 0].copy()

        # PnL robusto
        if df["PnL_Export"].notna().any():
            df["PnL"] = df["PnL_Export"].fillna(0.0)
        else:
            df["PnL"] = df["Valore_Mercato"] - df["Valore_Carico"].fillna(df["Valore_Mercato"])

        # return % per riga (evita div/0)
        vc = df["Valore_Carico"].fillna(0.0)
        df["Return_%"] = np.where(vc > 0, df["PnL"] / vc, np.nan)

        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Errore lettura CSV: {e}")
        return None


# ============================================================
# 3) CLASSIFICAZIONE + COUNTRY + MATURITY
# ============================================================
COUNTRY_MAP = {
    "IT": "Italia", "IE": "Irlanda", "LU": "Lussemburgo", "DE": "Germania", "FR": "Francia",
    "US": "USA", "GB": "Regno Unito", "NL": "Paesi Bassi", "ES": "Spagna", "CH": "Svizzera",
    "XS": "Internazionale (XS)",
}

MONTH_2 = {"GE":1, "FB":2, "MZ":3, "AP":4, "MG":5, "GN":6, "LG":7, "AG":8, "ST":9, "OT":10, "NV":11, "DC":12}
MONTH_3 = {"GEN":1,"FEB":2,"MAR":3,"APR":4,"MAG":5,"GIU":6,"LUG":7,"AGO":8,"SET":9,"OTT":10,"NOV":11,"DIC":12}

def isin_country(isin: str) -> str:
    if not isinstance(isin, str) or len(isin) < 2:
        return "N/A"
    cc = isin[:2].upper()
    return COUNTRY_MAP.get(cc, cc)

def infer_asset_class(row) -> str:
    instr = str(row.get("Strumento", "")).upper()
    title = str(row.get("Titolo", "")).upper()

    if "OBBLIGAZ" in instr:
        return "Bond"
    if "ETC" in instr:
        return "Commodities"
    if "ETF" in instr:
        # euristica: ETF bond
        if any(k in title for k in ["BOND", "TREASURY", "GOVT", "CORP", "HIGH YIELD", "AGGREGATE"]):
            return "Bond"
        return "Equity"
    # fallback
    return "Cash"

def extract_maturity_date(title: str):
    """Prova a estrarre una data da codici tipo BOT-31MZ26, BTP-15LG27, ecc."""
    if not isinstance(title, str):
        return pd.NaT
    t = title.upper()

    m2 = re.search(r"(\d{1,2})([A-Z]{2})(\d{2})", t)
    if m2:
        d = int(m2.group(1))
        mo = MONTH_2.get(m2.group(2), None)
        y = int("20" + m2.group(3))
        if mo:
            try:
                return pd.Timestamp(date(y, mo, d))
            except Exception:
                return pd.NaT

    m3 = re.search(r"(\d{1,2})([A-Z]{3})(\d{2})", t)
    if m3:
        d = int(m3.group(1))
        mo = MONTH_3.get(m3.group(2), None)
        y = int("20" + m3.group(3))
        if mo:
            try:
                return pd.Timestamp(date(y, mo, d))
            except Exception:
                return pd.NaT

    return pd.NaT


# ============================================================
# 4) RISK ENGINE (proxy conservative, no live)
# ============================================================
# Proxy "core ETF" (6 assets) per Markowitz: consente max peso 30% senza infeasibilità.
PROXY_ASSETS = [
    {"name": "EQ_World",   "class": "Equity",      "mu": 0.070, "vol": 0.160},
    {"name": "EQ_Europe",  "class": "Equity",      "mu": 0.060, "vol": 0.170},
    {"name": "BD_EUR_Gov", "class": "Bond",        "mu": 0.030, "vol": 0.055},
    {"name": "BD_EUR_Crp", "class": "Bond",        "mu": 0.040, "vol": 0.070},
    {"name": "CMD_Gold",   "class": "Commodities", "mu": 0.040, "vol": 0.180},
    {"name": "CASH_EUR",   "class": "Cash",        "mu": 0.020, "vol": 0.005},
]

# Correlazioni tipiche di lungo periodo (stilizzate)
# Ordine: EQ_World, EQ_Europe, BD_EUR_Gov, BD_EUR_Crp, CMD_Gold, CASH_EUR
CORR = np.array([
    [1.00, 0.90, 0.20, 0.25, 0.35, 0.00],
    [0.90, 1.00, 0.15, 0.20, 0.30, 0.00],
    [0.20, 0.15, 1.00, 0.80, 0.10, 0.10],
    [0.25, 0.20, 0.80, 1.00, 0.10, 0.10],
    [0.35, 0.30, 0.10, 0.10, 1.00, 0.05],
    [0.00, 0.00, 0.10, 0.10, 0.05, 1.00],
])

MDD_PROXY = {"Equity": 0.50, "Bond": 0.15, "Commodities": 0.35, "Cash": 0.01}

def proxy_cov_matrix():
    vols = np.array([a["vol"] for a in PROXY_ASSETS], dtype=float)
    return np.outer(vols, vols) * CORR

def portfolio_stats_from_class_weights(class_w: dict):
    """Rendimento/vol portafoglio usando proxy per classi (aggregando dai proxy asset)."""
    # converti class weights in proxy weights grezzi (split equalmente tra proxy della stessa classe)
    w = np.zeros(len(PROXY_ASSETS), dtype=float)
    for i, a in enumerate(PROXY_ASSETS):
        cls = a["class"]
        # quante proxy in quella classe
        n_cls = sum(1 for x in PROXY_ASSETS if x["class"] == cls)
        w[i] = float(class_w.get(cls, 0.0)) / max(n_cls, 1)

    # stats
    mu = np.array([a["mu"] for a in PROXY_ASSETS], dtype=float)
    cov = proxy_cov_matrix()
    pret = float(w @ mu)
    pvol = float(np.sqrt(w @ cov @ w))
    return pret, pvol

def drawdown_proxy_from_class_weights(class_w: dict):
    return float(sum(float(class_w.get(k, 0.0)) * MDD_PROXY[k] for k in MDD_PROXY))


# ============================================================
# 5) MARKOWITZ (vincolata)
# ============================================================
def solve_markowitz(
    vol_min=0.07, vol_max=0.09,
    max_equity=0.45, min_bond=0.35, max_comm=0.10, min_cash=0.05,
    max_single=0.30
):
    n = len(PROXY_ASSETS)
    mu = np.array([a["mu"] for a in PROXY_ASSETS], dtype=float)
    cov = proxy_cov_matrix()

    idx_equity = [i for i, a in enumerate(PROXY_ASSETS) if a["class"] == "Equity"]
    idx_bond   = [i for i, a in enumerate(PROXY_ASSETS) if a["class"] == "Bond"]
    idx_comm   = [i for i, a in enumerate(PROXY_ASSETS) if a["class"] == "Commodities"]
    idx_cash   = [i for i, a in enumerate(PROXY_ASSETS) if a["class"] == "Cash"]

    def port_vol(w):
        return float(np.sqrt(w @ cov @ w))

    # Obiettivo: massimizzare rendimento atteso
    def objective(w):
        return -(w @ mu)

    bounds = [(0.0, max_single) for _ in range(n)]  # no short + max peso singolo proxy

    cons = []
    cons.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    cons.append({"type": "ineq", "fun": lambda w: max_equity - np.sum(w[idx_equity])})
    cons.append({"type": "ineq", "fun": lambda w: np.sum(w[idx_bond]) - min_bond})
    cons.append({"type": "ineq", "fun": lambda w: max_comm - np.sum(w[idx_comm])})
    cons.append({"type": "ineq", "fun": lambda w: np.sum(w[idx_cash]) - min_cash})
    cons.append({"type": "ineq", "fun": lambda w: vol_max - port_vol(w)})
    cons.append({"type": "ineq", "fun": lambda w: port_vol(w) - vol_min})

    x0 = np.repeat(1.0 / n, n)

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x if res.success else x0

    ret = float(w @ mu)
    vol = port_vol(w)
    out = pd.DataFrame({
        "Proxy": [a["name"] for a in PROXY_ASSETS],
        "Class": [a["class"] for a in PROXY_ASSETS],
        "Weight": w,
        "mu": mu,
        "vol": [a["vol"] for a in PROXY_ASSETS],
    })
    return out, res.success, str(res.message), ret, vol


# ============================================================
# 6) OPTIONAL: Ratings downgrade (upload CSV)
# ============================================================
RATING_RANK = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
    "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16,
    "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC": 20, "C": 21, "D": 22,
}

def parse_ratings_csv(uploaded_file):
    try:
        raw = uploaded_file.getvalue().decode("utf-8", errors="replace")
        rdf = pd.read_csv(StringIO(raw))
        cols = {c.lower().strip(): c for c in rdf.columns}
        need = ["isin", "rating", "prev_rating"]
        if not all(k in cols for k in need):
            return None
        rdf = rdf.rename(columns={cols["isin"]: "ISIN", cols["rating"]: "Rating", cols["prev_rating"]: "Prev_Rating"})
        rdf["ISIN"] = rdf["ISIN"].astype(str).str.upper().str.strip()
        rdf["Rating"] = rdf["Rating"].astype(str).str.upper().str.strip()
        rdf["Prev_Rating"] = rdf["Prev_Rating"].astype(str).str.upper().str.strip()
        return rdf
    except Exception:
        return None


# ============================================================
# 7) SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.title("⚙️ Controlli")

    st.header("1) Upload")
    pf_file = st.file_uploader("CSV portafoglio (Fineco)", type=["csv"])
    rt_file = st.file_uploader("CSV rating (opzionale)", type=["csv"])
    st.caption("Rating CSV: colonne ISIN, Rating, Prev_Rating (es. BBB, BBB-).")

    st.divider()

    st.header("2) Alert soglie")
    max_pos_w = st.slider("Peso max posizione", 0.05, 0.60, 0.30, 0.01)
    italy_conc_th = st.slider("Soglia Italia (ISIN=IT)", 0.10, 0.95, 0.50, 0.05)
    months_maturity = st.slider("Scadenze entro (mesi)", 1, 60, 24, 1)
    vol_alert = st.slider("Volatilità alert (proxy)", 0.05, 0.20, 0.09, 0.01)
    dd_alert = st.slider("Drawdown alert (proxy)", 0.05, 0.60, 0.30, 0.01)

    st.divider()

    st.header("3) Markowitz vincoli")
    vol_min = st.slider("Vol min target", 0.03, 0.12, 0.07, 0.01)
    vol_max = st.slider("Vol max target", 0.04, 0.15, 0.09, 0.01)
    max_equity = st.slider("Max azionario", 0.0, 0.80, 0.45, 0.05)
    min_bond = st.slider("Min obbligazionario", 0.0, 0.90, 0.35, 0.05)
    max_comm = st.slider("Max commodity", 0.0, 0.30, 0.10, 0.01)
    min_cash = st.slider("Min monetario", 0.0, 0.30, 0.05, 0.01)
    max_single = st.slider("Max singolo proxy", 0.10, 0.60, 0.30, 0.01)

    st.divider()

    st.header("4) Macro monitor (manuale)")
    ecb_policy = st.selectbox("BCE stance", ["Dovish", "Neutral", "Hawkish"], index=1)
    fed_policy = st.selectbox("FED stance", ["Dovish", "Neutral", "Hawkish"], index=1)
    spread_btp = st.number_input("Spread BTP–Bund (bps)", value=145, step=5)
    eurusd = st.number_input("EUR/USD", value=1.0800, format="%.4f")
    vix = st.number_input("VIX", value=14.5, format="%.2f")

# ============================================================
# 8) MAIN
# ============================================================
st.title("📊 Dashboard Portafoglio")

if not pf_file:
    st.info("Carica il CSV Fineco dalla sidebar per iniziare.")
    st.stop()

df = parse_fineco_csv(pf_file)
if df is None or df.empty:
    st.stop()

# enrich
df["Asset_Class"] = df.apply(infer_asset_class, axis=1)
df["Paese"] = df["ISIN"].apply(isin_country)
df["Maturity_Date"] = df["Titolo"].apply(extract_maturity_date)

total_value = float(df["Valore_Mercato"].sum())
df["Peso"] = df["Valore_Mercato"] / total_value

# performance aggregate (robusta)
total_cost = float(df["Valore_Carico"].sum()) if df["Valore_Carico"].notna().any() else np.nan
total_pnl = float(df["PnL"].sum())
total_return = (total_pnl / total_cost) if (pd.notna(total_cost) and total_cost > 0) else np.nan

# per asset class
by_class = df.groupby("Asset_Class", dropna=False).agg(
    Valore=("Valore_Mercato", "sum"),
    PnL=("PnL", "sum"),
    Carico=("Valore_Carico", "sum"),
).reset_index()
by_class["Peso"] = by_class["Valore"] / total_value
by_class["Return_%"] = np.where(by_class["Carico"] > 0, by_class["PnL"] / by_class["Carico"], np.nan)

# exposures for risk proxy
class_w = (df.groupby("Asset_Class")["Peso"].sum()).to_dict()
# normalizza chiavi alle 4 classi
class_w_norm = {
    "Equity": float(class_w.get("Equity", 0.0)),
    "Bond": float(class_w.get("Bond", 0.0)),
    "Commodities": float(class_w.get("Commodities", 0.0)),
    "Cash": float(class_w.get("Cash", 0.0)),
}
pret, pvol = portfolio_stats_from_class_weights(class_w_norm)
pdd = drawdown_proxy_from_class_weights(class_w_norm)
var95 = 1.65 * pvol * total_value

# KPI row
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Valore totale", f"€ {total_value:,.2f}")
k2.metric("PnL (export)", f"€ {total_pnl:,.2f}")
k3.metric("Performance tot", f"{total_return:.2%}" if pd.notna(total_return) else "N/A")
k4.metric("Vol (proxy)", f"{pvol:.2%}")
k5.metric("Drawdown (proxy)", f"{pdd:.2%}")
k6.metric("VaR 95% (proxy)", f"€ {var95:,.0f}")

tabs = st.tabs(["📈 Dashboard", "🧯 Rischio", "🌍 Esposizioni", "⚠️ Alert", "🧠 Markowitz", "🌐 Macro", "📋 Dati"])

# --------------------
# TAB: Dashboard
# --------------------
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Allocazione per asset class")
        alloc = by_class[by_class["Peso"] > 0].copy()
        if not alloc.empty:
            fig = px.pie(alloc, values="Peso", names="Asset_Class", hole=0.45)
            fig.update_layout(uniformtext_minsize=14, uniformtext_mode="hide", legend=dict(orientation="v"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Allocazione vuota.")

    with c2:
        st.subheader("Performance per asset class")
        bb = by_class.copy()
        bb["Return_%_show"] = bb["Return_%"]
        fig = px.bar(bb, x="Asset_Class", y="Return_%_show", text_auto=".2%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top posizioni")
    top = df.sort_values("Valore_Mercato", ascending=False).head(20).copy()
    st.dataframe(
        top[["Titolo","ISIN","Asset_Class","Valuta","Paese","Valore_Mercato","Peso","PnL","Return_%","Maturity_Date"]]
        .style.format({
            "Valore_Mercato": "€ {:,.2f}",
            "Peso": "{:.2%}",
            "PnL": "€ {:,.2f}",
            "Return_%": "{:.2%}",
        }),
        hide_index=True,
        use_container_width=True
    )

# --------------------
# TAB: Rischio
# --------------------
with tabs[1]:
    st.subheader("Rischio, volatilità, drawdown (proxy)")
    st.write("Stime conservative basate su proxy ETF core e correlazioni di lungo periodo (senza prezzi live).")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rendimento atteso", f"{pret:.2%}")
    r2.metric("Volatilità attesa", f"{pvol:.2%}")
    r3.metric("Drawdown proxy", f"{pdd:.2%}")
    r4.metric("VaR 95% (proxy)", f"€ {var95:,.0f}")

    st.subheader("Stress test (semplice)")
    # shock scenario: Equity -20%, Bond -5%, Commod -10%, Cash 0
    shocks = {"Equity": -0.20, "Bond": -0.05, "Commodities": -0.10, "Cash": 0.00}
    impact = 0.0
    for cls, w in class_w_norm.items():
        impact += total_value * w * shocks.get(cls, 0.0)
    st.info(f"Scenario: Equity -20%, Bond -5%, Commod -10% ⇒ impatto stimato: € {impact:,.0f}")

# --------------------
# TAB: Esposizioni
# --------------------
with tabs[2]:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Esposizione per valuta")
        cur = df.groupby("Valuta")["Valore_Mercato"].sum().sort_values(ascending=False).reset_index()
        cur["Peso"] = cur["Valore_Mercato"] / total_value
        fig = px.bar(cur, x="Valuta", y="Peso", text_auto=".1%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Esposizione per paese (da ISIN)")
        co = df.groupby("Paese")["Valore_Mercato"].sum().sort_values(ascending=False).reset_index()
        co["Peso"] = co["Valore_Mercato"] / total_value
        fig = px.bar(co.head(20), x="Paese", y="Peso", text_auto=".1%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.subheader("Esposizione per asset class")
        ac = by_class.sort_values("Peso", ascending=False)[["Asset_Class","Peso"]].copy()
        fig = px.bar(ac, x="Asset_Class", y="Peso", text_auto=".1%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

# --------------------
# TAB: Alert
# --------------------
with tabs[3]:
    st.subheader("⚠️ Alert automatici")

    # A) overweight posizione
    ow = df[df["Peso"] > max_pos_w].copy()
    if ow.empty:
        st.success(f"✅ Nessuna posizione oltre {max_pos_w:.0%}.")
    else:
        st.error(f"🚨 Overweight: {len(ow)} posizioni oltre {max_pos_w:.0%}.")
        st.dataframe(
            ow[["Titolo","ISIN","Valore_Mercato","Peso","Asset_Class","Valuta"]]
            .sort_values("Peso", ascending=False)
            .style.format({"Valore_Mercato":"€ {:,.2f}", "Peso":"{:.2%}"}),
            hide_index=True,
            use_container_width=True
        )

    # B) volatilità portafoglio
    if pvol > vol_alert:
        st.error(f"🚨 Volatilità stimata {pvol:.2%} > soglia {vol_alert:.2%}")
    else:
        st.info(f"Volatilità stimata: {pvol:.2%}")

    # C) drawdown
    if pdd > dd_alert:
        st.error(f"🚨 Drawdown proxy {pdd:.2%} > soglia {dd_alert:.2%}")
    else:
        st.info(f"Drawdown proxy: {pdd:.2%}")

    # D) concentrazione Italia
    it_w = float(df.loc[df["ISIN"].str.startswith("IT", na=False), "Peso"].sum())
    if it_w > italy_conc_th:
        st.warning(f"⚠️ Concentrazione Italia {it_w:.2%} > soglia {italy_conc_th:.2%}")
    else:
        st.success(f"✅ Concentrazione Italia: {it_w:.2%}")

    # E) scadenze entro X mesi
    horizon = pd.Timestamp(date.today() + timedelta(days=int(months_maturity * 30.42)))
    mat = df[df["Maturity_Date"].notna() & (df["Maturity_Date"] <= horizon)].copy()
    if mat.empty:
        st.success(f"✅ Nessuna scadenza entro {months_maturity} mesi.")
    else:
        st.warning(f"📅 Scadenze entro {months_maturity} mesi: {len(mat)}")
        st.dataframe(
            mat[["Titolo","ISIN","Maturity_Date","Valore_Mercato","Asset_Class"]]
            .sort_values("Maturity_Date")
            .style.format({"Valore_Mercato":"€ {:,.2f}"}),
            hide_index=True,
            use_container_width=True
        )

    # F) rating downgrade (se caricato)
    if rt_file is None:
        st.info("Rating downgrade: carica un CSV rating (ISIN, Rating, Prev_Rating) per attivare l’alert.")
    else:
        rdf = parse_ratings_csv(rt_file)
        if rdf is None:
            st.error("Ratings CSV non valido: servono colonne ISIN, Rating, Prev_Rating.")
        else:
            mrg = df.merge(rdf, on="ISIN", how="left")
            mrg["Rank"] = mrg["Rating"].map(RATING_RANK)
            mrg["PrevRank"] = mrg["Prev_Rating"].map(RATING_RANK)

            down = mrg[mrg["Rank"].notna() & mrg["PrevRank"].notna() & (mrg["Rank"] > mrg["PrevRank"])].copy()
            if down.empty:
                st.success("✅ Nessun downgrade rilevato (sui titoli con rating caricato).")
            else:
                st.error(f"🚨 Downgrade rilevati: {len(down)}")
                st.dataframe(
                    down[["Titolo","ISIN","Prev_Rating","Rating","Valore_Mercato","Peso"]]
                    .sort_values("Peso", ascending=False)
                    .style.format({"Valore_Mercato":"€ {:,.2f}", "Peso":"{:.2%}"}),
                    hide_index=True,
                    use_container_width=True
                )

# --------------------
# TAB: Markowitz
# --------------------
with tabs[4]:
    st.subheader("Ottimizzazione Markowitz (proxy ETF core, vincolata)")
    st.write("Obiettivo: massimizzare rendimento atteso rispettando vincoli rischio/allocazione (no short, max 30% per proxy).")

    wdf, ok, msg, r_opt, v_opt = solve_markowitz(
        vol_min=vol_min, vol_max=vol_max,
        max_equity=max_equity, min_bond=min_bond, max_comm=max_comm, min_cash=min_cash,
        max_single=max_single
    )
    if not ok:
        st.warning(f"Ottimizzazione non perfetta: {msg}")

    # aggrega per classe
    w_class = wdf.groupby("Class")["Weight"].sum().reindex(["Equity","Bond","Commodities","Cash"]).fillna(0.0)
    cur_class = pd.Series(class_w_norm).reindex(["Equity","Bond","Commodities","Cash"]).fillna(0.0)

    comp = pd.DataFrame({
        "Class": w_class.index,
        "Attuale": cur_class.values,
        "Target": w_class.values
    }).melt(id_vars="Class", var_name="Tipo", value_name="Peso")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Attuale vs Target (classi)")
        fig = px.bar(comp, x="Class", y="Peso", color="Tipo", barmode="group", text_auto=".1%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Pesi proxy (max 30% per proxy)")
        show = wdf.copy()
        show["Weight_%"] = show["Weight"]
        st.dataframe(
            show[["Proxy","Class","Weight_%","mu","vol"]]
            .sort_values("Weight_%", ascending=False)
            .style.format({"Weight_%":"{:.2%}","mu":"{:.2%}","vol":"{:.2%}"}),
            hide_index=True,
            use_container_width=True
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Rendimento atteso (target)", f"{r_opt:.2%}")
    m2.metric("Volatilità attesa (target)", f"{v_opt:.2%}")
    m3.metric("Target vol range", f"{vol_min:.0%}–{vol_max:.0%}")

# --------------------
# TAB: Macro
# --------------------
with tabs[5]:
    st.subheader("Macro monitor (manuale / semi-auto)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BCE", ecb_policy)
    c2.metric("FED", fed_policy)
    c3.metric("Spread BTP–Bund", f"{spread_btp} bps", delta=("ALTO" if spread_btp >= 200 else "OK"))
    c4.metric("VIX", f"{vix:.2f}", delta=("ALTO" if vix >= 20 else "OK"))

    st.subheader("FX")
    st.metric("EUR/USD", f"{eurusd:.4f}")

# --------------------
# TAB: Dati
# --------------------
with tabs[6]:
    st.subheader("Dati importati (raw)")
    cols = ["Titolo","ISIN","Strumento","Asset_Class","Valuta","Paese","Valore_Carico","Valore_Mercato","PnL","Return_%","Maturity_Date","Peso"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(
        df[cols].style.format({
            "Valore_Carico": "€ {:,.2f}",
            "Valore_Mercato": "€ {:,.2f}",
            "PnL": "€ {:,.2f}",
            "Return_%": "{:.2%}",
            "Peso": "{:.2%}",
        }),
        hide_index=True,
        use_container_width=True
    )
