import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from io import StringIO
import re
from datetime import datetime, date, timedelta

# ----------------------------
# 0) PAGE CONFIG (deve essere la prima call Streamlit)
# ----------------------------
st.set_page_config(
    page_title="Portfolio Manager Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# 1) PASSWORD GATE
# ----------------------------
def _password_gate():
    expected = st.secrets.get("PASSWORD", None)

    if not expected:
        st.error("Secret mancante: imposta PASSWORD in App settings → Secrets (formato TOML).")
        st.code('PASSWORD = "LaTuaPasswordSegreta"', language="toml")
        st.stop()

    def _check():
        if st.session_state.get("_pw_input", "") == expected:
            st.session_state["_pw_ok"] = True
            st.session_state["_pw_input"] = ""
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok", False):
        return

    st.text_input("🔐 Password", type="password", key="_pw_input", on_change=_check)
    if "_pw_ok" in st.session_state and not st.session_state["_pw_ok"]:
        st.error("Password errata.")
    st.stop()


_password_gate()

# ----------------------------
# 2) PROXY (Markowitz a classi, senza dati live)
# ----------------------------
PROXIES = {
    "Equity": {"ret": 0.075, "vol": 0.16},
    "Bond": {"ret": 0.035, "vol": 0.06},
    "Commodities": {"ret": 0.045, "vol": 0.18},
    "Cash": {"ret": 0.025, "vol": 0.005},
}

CORRELATION = pd.DataFrame(
    [
        [1.0, 0.2, 0.4, 0.0],
        [0.2, 1.0, 0.1, 0.1],
        [0.4, 0.1, 1.0, 0.1],
        [0.0, 0.1, 0.1, 1.0],
    ],
    index=list(PROXIES.keys()),
    columns=list(PROXIES.keys()),
)

# Proxy drawdown “realistico” (ordine di grandezza, lungo periodo)
MDD_PROXY = {
    "Equity": 0.50,
    "Bond": 0.15,
    "Commodities": 0.35,
    "Cash": 0.01,
}

# ----------------------------
# 3) UTILS: parsing numeri, ISIN-country, maturity
# ----------------------------
MONTH_CODE_IT = {
    "GE": 1, "GN": 6,  # (alcuni export usano GE/GEN, qui 2-letter)
    "FB": 2,
    "MZ": 3,
    "AP": 4,
    "MG": 5,
    "GN": 6,
    "LG": 7,
    "AG": 8,
    "ST": 9,
    "OT": 10,
    "NV": 11,
    "DC": 12,
}

# Alcuni export possono usare "GEN" ecc.; gestiamo anche 3-letter
MONTH_CODE_IT_3 = {
    "GEN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAG": 5, "GIU": 6,
    "LUG": 7, "AGO": 8, "SET": 9, "OTT": 10, "NOV": 11, "DIC": 12,
}

def parse_eu_number(x) -> float:
    """Converte numeri EU tipo '6 981,52' o '6.981,52' o '6981,52' in float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    # Tieni solo cifre e separatori
    s = re.sub(r"[^\d,.\-]", "", s)

    # Se ci sono sia '.' che ',', assumiamo '.' migliaia e ',' decimali
    if "," in s and "." in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    # else: già stile US/float

    try:
        return float(s)
    except Exception:
        return np.nan


def isin_country(isin: str) -> str:
    if not isinstance(isin, str) or len(isin) < 2:
        return "N/A"
    cc = isin[:2].upper()
    # mapping minimale (puoi estenderlo)
    m = {
        "IT": "Italia",
        "IE": "Irlanda",
        "LU": "Lussemburgo",
        "DE": "Germania",
        "FR": "Francia",
        "US": "USA",
        "GB": "Regno Unito",
        "NL": "Paesi Bassi",
        "ES": "Spagna",
        "CH": "Svizzera",
        "XS": "Internazionale (XS)",
    }
    return m.get(cc, cc)


def extract_maturity_date(title: str):
    """
    Estrae una data scadenza da stringhe tipo:
    'BOT-31MZ26', 'BTP-15LG27 3,45', 'BTP-1MG31 6'
    """
    if not isinstance(title, str):
        return pd.NaT

    t = title.upper()

    # pattern 2-letter mese (MZ, LG, MG, ST, ...)
    m2 = re.search(r"(\d{1,2})([A-Z]{2})(\d{2})", t)
    if m2:
        d = int(m2.group(1))
        code = m2.group(2)
        y = int("20" + m2.group(3))
        month = MONTH_CODE_IT.get(code, None)
        if month:
            try:
                return pd.Timestamp(date(y, month, d))
            except Exception:
                return pd.NaT

    # pattern 3-letter (GEN, FEB, ...)
    m3 = re.search(r"(\d{1,2})([A-Z]{3})(\d{2})", t)
    if m3:
        d = int(m3.group(1))
        code = m3.group(2)
        y = int("20" + m3.group(3))
        month = MONTH_CODE_IT_3.get(code, None)
        if month:
            try:
                return pd.Timestamp(date(y, month, d))
            except Exception:
                return pd.NaT

    return pd.NaT


def infer_asset_class(row) -> str:
    instr = str(row.get("Strumento", "")).upper()
    title = str(row.get("Titolo", "")).upper()

    if "OBBLIGAZIONE" in instr:
        return "Bond"
    if "ETC" in instr:
        return "Commodities"
    if "ETF" in instr:
        # euristica ETF bond
        if any(k in title for k in ["BOND", "TREASURY", "CORP", "HIGH YIELD", "GOVT", "AGGREGATE", "EMB"]):
            return "Bond"
        return "Equity"

    # fallback: se non riconosciuto, cash
    return "Cash"


# ----------------------------
# 4) PARSER CSV FINECO (robusto)
# ----------------------------
def parse_fineco_csv(uploaded_file) -> pd.DataFrame | None:
    try:
        content = uploaded_file.getvalue().decode("latin1", errors="replace")
        lines = content.splitlines()

        # trova la riga header vera (quella che contiene "Titolo;ISIN;...;Valore di carico;...;Valore di mercato")
        header_idx = None
        for i, ln in enumerate(lines):
            l = ln.strip()
            if l.startswith("Titolo;") and "ISIN" in l and ("Valore di carico" in l or "Valore di mercato" in l):
                header_idx = i
                break
            # alternativa: alcuni export possono non avere ; subito dopo Titolo (spazi)
            if "Titolo" in l and "ISIN" in l and "Strumento" in l and ("Valore di mercato" in l):
                header_idx = i
                break

        if header_idx is None:
            st.error("Non riesco a trovare l'intestazione 'Titolo;ISIN;...'. Verifica che il CSV sia l'export Fineco completo.")
            return None

        df = pd.read_csv(StringIO(content), sep=";", skiprows=header_idx, header=0)

        # colonna valore mercato (può avere caratteri strani tipo "Valore di mercato ?")
        val_mkt_col = None
        for c in df.columns:
            cs = str(c).strip().lower()
            if "valore" in cs and "mercato" in cs:
                val_mkt_col = c
                break

        if val_mkt_col is None:
            st.error("Colonna 'Valore di mercato' non trovata nel CSV.")
            return None

        # colonna valore carico
        val_cost_col = None
        for c in df.columns:
            cs = str(c).strip().lower()
            if "valore" in cs and "carico" in cs:
                val_cost_col = c
                break

        # colonna PnL in valuta (se presente)
        pnl_col = None
        for c in df.columns:
            cs = str(c).strip().lower()
            if "var in valuta" in cs:
                pnl_col = c
                break

        # normalizza numeri
        df["Valore_Mercato"] = df[val_mkt_col].apply(parse_eu_number)
        if val_cost_col:
            df["Valore_Carico"] = df[val_cost_col].apply(parse_eu_number)
        else:
            df["Valore_Carico"] = np.nan

        if pnl_col:
            df["PnL"] = df[pnl_col].apply(parse_eu_number)
        else:
            df["PnL"] = df["Valore_Mercato"] - df["Valore_Carico"]

        # pulizia righe spazzatura
        df = df[df["Valore_Mercato"].notna()]
        df = df[df["Valore_Mercato"] > 0]

        # Asset class
        df["Asset_Class"] = df.apply(infer_asset_class, axis=1)

        # Valuta e paese
        if "Valuta" not in df.columns:
            df["Valuta"] = "N/A"
        if "ISIN" not in df.columns:
            df["ISIN"] = ""
        df["Paese_ISIN"] = df["ISIN"].apply(isin_country)

        # Scadenza
        df["Maturity_Date"] = df["Titolo"].apply(extract_maturity_date)

        # Ritorno % (se possibile)
        df["Return_%"] = np.where(
            df["Valore_Carico"].notna() & (df["Valore_Carico"] != 0),
            df["PnL"] / df["Valore_Carico"],
            np.nan,
        )

        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Errore lettura file: {e}")
        return None


# ----------------------------
# 5) MARKOWITZ: metriche e ottimizzazione (a classi)
# ----------------------------
def portfolio_metrics_from_weights(weights: np.ndarray):
    keys = list(PROXIES.keys())
    w = np.array(weights, dtype=float)
    ex_rets = np.array([PROXIES[k]["ret"] for k in keys], dtype=float)
    vols = np.array([PROXIES[k]["vol"] for k in keys], dtype=float)
    cov = np.outer(vols, vols) * CORRELATION.loc[keys, keys].values
    port_ret = float(np.sum(w * ex_rets))
    port_vol = float(np.sqrt(w.T @ cov @ w))
    return port_ret, port_vol


def optimize_markowitz(
    target_vol_max=0.09,
    max_equity=0.45,
    min_bond=0.35,
    max_commod=0.10,
    min_cash=0.05,
    max_single_class=0.30,  # opzionale (coerente col vincolo "peso singolo asset ≤ 30%" ma a livello classi)
):
    keys = list(PROXIES.keys())
    n = len(keys)
    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]

    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    idx = {k: i for i, k in enumerate(keys)}

    cons += [{"type": "ineq", "fun": lambda x, i=idx["Equity"]: max_equity - x[i]}]
    cons += [{"type": "ineq", "fun": lambda x, i=idx["Bond"]: x[i] - min_bond}]
    cons += [{"type": "ineq", "fun": lambda x, i=idx["Commodities"]: max_commod - x[i]}]
    cons += [{"type": "ineq", "fun": lambda x, i=idx["Cash"]: x[i] - min_cash}]
    cons += [{"type": "ineq", "fun": lambda x: target_vol_max - portfolio_metrics_from_weights(x)[1]}]

    if max_single_class is not None:
        for k in keys:
            i = idx[k]
            cons += [{"type": "ineq", "fun": lambda x, i=i: max_single_class - x[i]}]

    def objective(x):
        r, v = portfolio_metrics_from_weights(x)
        if v <= 0:
            return 0.0
        return -(r / v)

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x if res.success else x0
    return dict(zip(keys, w)), res.success, res.message


# ----------------------------
# 6) RATING (opzionale): upload file e downgrade alert
# ----------------------------
RATING_RANK = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
    "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16,
    "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC": 20, "C": 21, "D": 22,
}

def parse_ratings_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="replace")
        rdf = pd.read_csv(StringIO(content))
        # colonne richieste: ISIN, Rating, Prev_Rating
        cols = {c.lower(): c for c in rdf.columns}
        if "isin" not in cols or "rating" not in cols or "prev_rating" not in cols:
            st.warning("Ratings CSV: colonne richieste: ISIN, Rating, Prev_Rating.")
            return None
        rdf = rdf.rename(columns={
            cols["isin"]: "ISIN",
            cols["rating"]: "Rating",
            cols["prev_rating"]: "Prev_Rating"
        })
        rdf["ISIN"] = rdf["ISIN"].astype(str).str.upper()
        rdf["Rating"] = rdf["Rating"].astype(str).str.upper().str.strip()
        rdf["Prev_Rating"] = rdf["Prev_Rating"].astype(str).str.upper().str.strip()
        return rdf
    except Exception as e:
        st.warning(f"Ratings CSV non valido: {e}")
        return None


# ----------------------------
# 7) UI
# ----------------------------
with st.sidebar:
    st.title("⚙️ Controlli")

    st.header("1) Upload")
    portfolio_file = st.file_uploader("Carica CSV Fineco", type=["csv"], key="pf")
    ratings_file = st.file_uploader("Ratings CSV (opzionale)", type=["csv"], key="rf")
    st.caption("Ratings CSV atteso: colonne ISIN, Rating, Prev_Rating.")

    st.divider()

    st.header("2) Soglie Alert")
    max_pos_weight = st.slider("Peso max singola posizione", 0.05, 0.50, 0.30, 0.01)
    italy_conc_th = st.slider("Soglia concentrazione Italia (ISIN=IT)", 0.10, 0.90, 0.50, 0.05)
    maturity_months = st.slider("Scadenze entro (mesi)", 1, 60, 24, 1)
    dd_th = st.slider("Drawdown proxy max (alert)", 0.05, 0.60, 0.30, 0.01)

    st.divider()

    st.header("3) Vincoli Markowitz")
    target_vol = st.slider("Volatilità target max", 0.05, 0.15, 0.09, 0.01)
    c_max_eq = st.slider("Max azionario", 0.0, 1.0, 0.45, 0.05)
    c_min_bond = st.slider("Min obbligazionario", 0.0, 1.0, 0.35, 0.05)
    c_max_comm = st.slider("Max commodities", 0.0, 1.0, 0.10, 0.05)
    c_min_cash = st.slider("Min monetario", 0.0, 0.30, 0.05, 0.01)

    st.divider()

    st.header("4) Macro (manuale)")
    spread_btp = st.number_input("Spread BTP-Bund (bps)", value=145)
    eur_usd = st.number_input("EUR/USD", value=1.08, format="%.4f")
    vix = st.number_input("VIX", value=14.5, format="%.2f")


st.title("📊 Portfolio Intelligence Dashboard")

if not portfolio_file:
    st.info("Carica il CSV Fineco dalla sidebar per iniziare.")
    st.stop()

df = parse_fineco_csv(portfolio_file)
if df is None or df.empty:
    st.stop()

# ----------------------------
# 8) KPI & performance
# ----------------------------
total_value = float(df["Valore_Mercato"].sum())
total_cost = float(df["Valore_Carico"].sum()) if df["Valore_Carico"].notna().any() else np.nan
total_pnl = float(df["PnL"].sum()) if df["PnL"].notna().any() else np.nan
total_return = (total_value / total_cost - 1.0) if (pd.notna(total_cost) and total_cost != 0) else np.nan

df["Peso"] = df["Valore_Mercato"] / total_value

# allocazione attuale per classi
current_alloc = (
    df.groupby("Asset_Class")["Peso"].sum()
    .reindex(list(PROXIES.keys()), fill_value=0.0)
)

# metriche rischio proxy
curr_ret, curr_vol = portfolio_metrics_from_weights(current_alloc.values)

dd_proxy = float(sum(current_alloc.get(k, 0.0) * MDD_PROXY[k] for k in PROXIES.keys()))
var95_proxy_eur = 1.65 * curr_vol * total_value  # VaR parametrico (proxy)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Valore totale", f"€ {total_value:,.2f}")
if pd.notna(total_pnl):
    k2.metric("PnL (da export)", f"€ {total_pnl:,.2f}")
else:
    k2.metric("PnL (da export)", "N/A")
if pd.notna(total_return):
    k3.metric("Performance totale", f"{total_return:.2%}")
else:
    k3.metric("Performance totale", "N/A")
k4.metric("Volatilità stimata (proxy)", f"{curr_vol:.2%}")
k5.metric("VaR 95% (proxy)", f"€ {var95_proxy_eur:,.0f}")

st.divider()

tabs = st.tabs(["Dashboard", "Rischio & Drawdown", "Esposizioni", "Alert", "Markowitz", "Dati"])

# ----------------------------
# TAB: Dashboard
# ----------------------------
with tabs[0]:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Allocazione per Asset Class")
        alloc_df = (current_alloc.rename("Peso").reset_index().rename(columns={"index": "Asset Class"}))
        fig_alloc = px.pie(alloc_df, values="Peso", names="Asset Class", hole=0.45)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with c2:
        st.subheader("Performance per Asset Class (se disponibile)")
        perf_by_class = df.groupby("Asset_Class").agg(
            Valore=("Valore_Mercato", "sum"),
            PnL=("PnL", "sum"),
            Cost=("Valore_Carico", "sum"),
        ).reset_index()
        perf_by_class["Return_%"] = np.where(
            perf_by_class["Cost"].notna() & (perf_by_class["Cost"] != 0),
            perf_by_class["PnL"] / perf_by_class["Cost"],
            np.nan,
        )
        fig_perf = px.bar(
            perf_by_class,
            x="Asset_Class",
            y="Return_%",
            text_auto=".2%",
        )
        fig_perf.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Top posizioni (per valore)")
    topn = df.sort_values("Valore_Mercato", ascending=False).head(15).copy()
    topn["Peso %"] = topn["Peso"]
    st.dataframe(
        topn[["Titolo", "ISIN", "Asset_Class", "Valuta", "Paese_ISIN", "Valore_Mercato", "Peso %", "PnL", "Return_%", "Maturity_Date"]]
        .style.format({
            "Valore_Mercato": "€ {:,.2f}",
            "Peso %": "{:.2%}",
            "PnL": "€ {:,.2f}",
            "Return_%": "{:.2%}",
        }),
        hide_index=True,
        use_container_width=True,
    )

# ----------------------------
# TAB: Rischio & Drawdown
# ----------------------------
with tabs[1]:
    st.subheader("Rischio (proxy)")
    st.write(
        "Queste metriche sono stime basate su proxy di lungo periodo per classi (Equity/Bond/Commodities/Cash), "
        "utili per controllo rischio senza prezzi live."
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("Rendimento atteso (proxy)", f"{curr_ret:.2%}")
    r2.metric("Volatilità attesa (proxy)", f"{curr_vol:.2%}")
    r3.metric("Drawdown proxy (MDD)", f"{dd_proxy:.2%}")

    st.caption("Drawdown proxy = somma(peso classe × MDD tipico classe).")

    # Alert drawdown proxy
    if dd_proxy > dd_th:
        st.error(f"🚨 Alert: drawdown proxy {dd_proxy:.2%} > soglia {dd_th:.2%}")

# ----------------------------
# TAB: Esposizioni
# ----------------------------
with tabs[2]:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Per Valuta")
        cur = df.groupby("Valuta")["Valore_Mercato"].sum().sort_values(ascending=False).reset_index()
        cur["Peso"] = cur["Valore_Mercato"] / total_value
        fig = px.bar(cur, x="Valuta", y="Peso", text_auto=".1%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Per Paese (da ISIN)")
        cc = df.groupby("Paese_ISIN")["Valore_Mercato"].sum().sort_values(ascending=False).reset_index()
        cc["Peso"] = cc["Valore_Mercato"] / total_value
        fig = px.bar(cc.head(15), x="Paese_ISIN", y="Peso", text_auto=".1%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.subheader("Per Mercato/Strumento")
        if "Mercato" in df.columns:
            m = df.groupby("Mercato")["Valore_Mercato"].sum().sort_values(ascending=False).reset_index()
            m["Peso"] = m["Valore_Mercato"] / total_value
            fig = px.bar(m.head(15), x="Mercato", y="Peso", text_auto=".1%")
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonna 'Mercato' non presente nel CSV.")

# ----------------------------
# TAB: Alert
# ----------------------------
with tabs[3]:
    st.subheader("⚠️ Alert automatici")

    # 1) Overweight singola posizione
    overweight = df[df["Peso"] > max_pos_weight].copy()
    if overweight.empty:
        st.success(f"✅ Nessuna posizione oltre {max_pos_weight:.0%}.")
    else:
        st.error(f"🚨 Posizioni oltre {max_pos_weight:.0%}: {len(overweight)}")
        overweight["Peso %"] = overweight["Peso"]
        st.dataframe(
            overweight[["Titolo", "ISIN", "Valore_Mercato", "Peso %", "Asset_Class", "Valuta"]]
            .sort_values("Peso %", ascending=False)
            .style.format({"Valore_Mercato": "€ {:,.2f}", "Peso %": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )

    # 2) Concentrazione Italia (da ISIN)
    it_weight = float(df.loc[df["ISIN"].astype(str).str.upper().str.startswith("IT"), "Peso"].sum())
    if it_weight > italy_conc_th:
        st.error(f"🚨 Concentrazione Italia {it_weight:.2%} > soglia {italy_conc_th:.2%}")
    else:
        st.info(f"Concentrazione Italia: {it_weight:.2%}")

    # 3) Scadenze entro X mesi
    horizon = pd.Timestamp(date.today() + timedelta(days=int(maturity_months * 30.42)))
    maturing = df[df["Maturity_Date"].notna() & (df["Maturity_Date"] <= horizon)].copy()
    if maturing.empty:
        st.success(f"✅ Nessuna scadenza entro {maturity_months} mesi.")
    else:
        st.warning(f"📅 Scadenze entro {maturity_months} mesi: {len(maturing)}")
        st.dataframe(
            maturing[["Titolo", "ISIN", "Maturity_Date", "Valore_Mercato", "Asset_Class"]]
            .sort_values("Maturity_Date")
            .style.format({"Valore_Mercato": "€ {:,.2f}"}),
            hide_index=True,
            use_container_width=True,
        )

    # 4) Rating downgrade (da file opzionale)
    rdf = parse_ratings_file(ratings_file) if ratings_file else None
    if rdf is None:
        st.info("Rating downgrade: carica un Ratings CSV (ISIN, Rating, Prev_Rating) per abilitare l’alert.")
    else:
        merged = df.merge(rdf, on="ISIN", how="left")
        merged["RatingRank"] = merged["Rating"].map(RATING_RANK)
        merged["PrevRank"] = merged["Prev_Rating"].map(RATING_RANK)

        downgrades = merged[
            merged["RatingRank"].notna()
            & merged["PrevRank"].notna()
            & (merged["RatingRank"] > merged["PrevRank"])
        ].copy()

        if downgrades.empty:
            st.success("✅ Nessun downgrade rilevato (su titoli con rating caricato).")
        else:
            st.error(f"🚨 Downgrade rilevati: {len(downgrades)}")
            st.dataframe(
                downgrades[["Titolo", "ISIN", "Prev_Rating", "Rating", "Valore_Mercato", "Peso"]]
                .sort_values("Peso", ascending=False)
                .style.format({"Valore_Mercato": "€ {:,.2f}", "Peso": "{:.2%}"}),
                hide_index=True,
                use_container_width=True,
            )

    # 5) Macro watchlist (manuale)
    st.subheader("🌍 Macro watchlist (manuale)")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Spread BTP-Bund", f"{spread_btp} bps", delta=("ALTO" if spread_btp > 200 else "OK"))
    mc2.metric("EUR/USD", f"{eur_usd:.4f}")
    mc3.metric("VIX", f"{vix:.2f}", delta=("ALTO" if vix > 20 else "OK"))

# ----------------------------
# TAB: Markowitz
# ----------------------------
with tabs[4]:
    st.subheader("Ottimizzazione Markowitz (vincolata, proxy)")

    # Vincolo “peso singolo asset ≤ 30%” qui è applicato alle classi (opzionale) perché senza prezzi live lavoriamo a classi.
    enforce_single_class = st.checkbox("Enforce max 30% per Asset Class (proxy)", value=False)

    w_opt, ok, msg = optimize_markowitz(
        target_vol_max=target_vol,
        max_equity=c_max_eq,
        min_bond=c_min_bond,
        max_commod=c_max_comm,
        min_cash=c_min_cash,
        max_single_class=0.30 if enforce_single_class else None,
    )

    opt_vec = np.array([w_opt[k] for k in PROXIES.keys()])
    opt_ret, opt_vol = portfolio_metrics_from_weights(opt_vec)

    if not ok:
        st.warning(f"Ottimizzazione non perfetta: {msg}")

    comp = pd.DataFrame({
        "Asset Class": list(PROXIES.keys()),
        "Attuale": current_alloc.values,
        "Target (Markowitz)": opt_vec,
    })
    comp_m = comp.melt(id_vars="Asset Class", var_name="Tipo", value_name="Peso")

    fig = px.bar(comp_m, x="Asset Class", y="Peso", color="Tipo", barmode="group", text_auto=".1%")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Rendimento atteso (opt)", f"{opt_ret:.2%}")
    m2.metric("Volatilità attesa (opt)", f"{opt_vol:.2%}")
    m3.metric("Delta vol (attuale - opt)", f"{(curr_vol - opt_vol):.2%}")

# ----------------------------
# TAB: Dati
# ----------------------------
with tabs[5]:
    st.subheader("Dati importati (raw)")
    show_cols = [c for c in ["Titolo", "ISIN", "Simbolo", "Mercato", "Strumento", "Valuta",
                             "Valore_Carico", "Valore_Mercato", "PnL", "Return_%", "Asset_Class", "Paese_ISIN", "Maturity_Date", "Peso"] if c in df.columns]
    st.dataframe(
        df[show_cols].style.format({
            "Valore_Carico": "€ {:,.2f}",
            "Valore_Mercato": "€ {:,.2f}",
            "PnL": "€ {:,.2f}",
            "Return_%": "{:.2%}",
            "Peso": "{:.2%}",
        }),
        hide_index=True,
        use_container_width=True,
    )
