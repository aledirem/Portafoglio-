import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from io import StringIO
import re
from datetime import datetime, date, timedelta

# ----------------------------
# 0) CONFIGURAZIONE
# ----------------------------
st.set_page_config(
    page_title="Portfolio Manager Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# 1) PASSWORD
# ----------------------------
def _password_gate():
    expected = st.secrets.get("PASSWORD", None)
    if not expected:
        st.error("⚠️ Configurazione incompleta: Imposta PASSWORD nei Secrets.")
        st.stop()
    
    def _check():
        if st.session_state.get("_pw_input", "") == expected:
            st.session_state["_pw_ok"] = True
            st.session_state["_pw_input"] = ""
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok", False): return

    st.markdown("## 🔐 Accesso Riservato")
    st.text_input("Inserisci Password", type="password", key="_pw_input", on_change=_check)
    if "_pw_ok" in st.session_state and not st.session_state["_pw_ok"]: st.error("Password errata.")
    st.stop()

_password_gate()

# ----------------------------
# 2) PROXY DATI
# ----------------------------
PROXIES = {
    "Equity":      {"ret": 0.075, "vol": 0.16},
    "Bond":        {"ret": 0.035, "vol": 0.06},
    "Commodities": {"ret": 0.045, "vol": 0.18},
    "Cash":        {"ret": 0.025, "vol": 0.005},
}
CORRELATION = pd.DataFrame([
    [1.0, 0.2, 0.4, 0.0],
    [0.2, 1.0, 0.1, 0.1],
    [0.4, 0.1, 1.0, 0.1],
    [0.0, 0.1, 0.1, 1.0]], 
    index=PROXIES.keys(), columns=PROXIES.keys())

# ----------------------------
# 3) PARSING AVANZATO
# ----------------------------
def parse_eu_number(x) -> float:
    if pd.isna(x) or str(x).strip() == "": return 0.0
    s = re.sub(r"[^\d,.-]", "", str(x).strip())
    try:
        if "." in s and "," in s: s = s.replace(".", "").replace(",", ".")
        elif "," in s: s = s.replace(",", ".")
        return float(s)
    except: return 0.0

def infer_asset_class(row) -> str:
    instr = str(row.get("Strumento", "")).upper()
    title = str(row.get("Titolo", "")).upper()
    if "OBBLIGAZIONE" in instr or "BOND" in instr: return "Bond"
    if "ETC" in instr or "COMM" in title: return "Commodities"
    if "ETF" in instr:
        if any(x in title for x in ["BOND", "TREASURY", "CORP", "HIGH YIELD", "GOVT"]): return "Bond"
        return "Equity"
    return "Cash"

def get_maturity_year(title):
    try:
        match = re.search(r'(\d{1,2})([A-Z]{2,3})(\d{2})', str(title).upper())
        if match: return int("20" + match.group(3))
    except: pass
    return None

def parse_fineco_csv_robust(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("latin1", errors="replace")
        lines = content.splitlines()
        header_row = next((i for i, line in enumerate(lines) if "Titolo" in line and "ISIN" in line), None)
        
        if header_row is None:
            st.error("Format non riconosciuto (Manca 'Titolo'/'ISIN').")
            return None

        df = pd.read_csv(StringIO(content), sep=';', skiprows=header_row)
        df.columns = [c.strip() for c in df.columns]
        
        col_val = next((c for c in df.columns if "Valore" in c and "mercato" in c), None)
        col_cost = next((c for c in df.columns if "Valore" in c and "carico" in c), None)
        
        if not col_val: return None

        df["Valore_Mercato"] = df[col_val].apply(parse_eu_number)
        # Filtro: Valore > 1€ per evitare residui
        df = df[df["Valore_Mercato"] > 1.0].copy()

        if col_cost:
            df["Valore_Carico"] = df[col_cost].apply(parse_eu_number)
        else:
            df["Valore_Carico"] = df["Valore_Mercato"]

        df["Asset_Class"] = df.apply(infer_asset_class, axis=1)
        df["Maturity_Year"] = df["Titolo"].apply(get_maturity_year)
        
        # LOGICA PNL CORRETTA: Se è Cash o Carico è 0, PnL = 0
        df["PnL"] = df.apply(lambda x: 0.0 if (x["Asset_Class"] == "Cash" or x["Valore_Carico"] < 1) 
                             else (x["Valore_Mercato"] - x["Valore_Carico"]), axis=1)
        
        return df
    except Exception as e:
        st.error(f"Errore Parsing: {e}")
        return None

# ----------------------------
# 4) OPTIMIZER
# ----------------------------
def optimize_portfolio(constraints, current_alloc):
    keys = list(PROXIES.keys())
    n = len(keys)
    def objective(w):
        ret = np.sum(w * np.array([PROXIES[k]['ret'] for k in keys]))
        vols = np.array([PROXIES[k]['vol'] for k in keys])
        cov = np.outer(vols, vols) * CORRELATION.values
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if vol == 0: return 0
        return -(ret / vol)

    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if 'max_equity' in constraints:
        cons.append({'type': 'ineq', 'fun': lambda x: constraints['max_equity'] - x[keys.index('Equity')]})
    if 'min_bond' in constraints:
        cons.append({'type': 'ineq', 'fun': lambda x: x[keys.index('Bond')] - constraints['min_bond']})

    res = minimize(objective, np.repeat(1/n, n), bounds=tuple((0.05, 1.0) for _ in range(n)), constraints=cons)
    return dict(zip(keys, res.x)) if res.success else dict(zip(keys, np.repeat(1/n, n)))

# ----------------------------
# 5) UI PRINCIPALE
# ----------------------------
with st.sidebar:
    st.header("1. Upload")
    uploaded_file = st.file_uploader("Carica CSV Fineco", type=['csv'])
    st.divider()
    st.header("2. Obiettivi & Vincoli")
    max_eq = st.slider("Max Azionario", 0.0, 1.0, 0.50, 0.05)
    min_bond = st.slider("Min Obbligazionario", 0.0, 1.0, 0.30, 0.05)
    st.divider()
    st.header("3. Macro Input")
    spread = st.number_input("Spread BTP (bps)", 100, 500, 145)
    vix = st.number_input("VIX Index", 10.0, 50.0, 14.5)

st.title("📊 Portfolio Intelligence Dashboard")

if uploaded_file:
    df = parse_fineco_csv_robust(uploaded_file)
    if df is not None:
        
        # --- KPI ROW ---
        tot_val = df["Valore_Mercato"].sum()
        tot_pnl = df["PnL"].sum()
        df["Peso"] = df["Valore_Mercato"] / tot_val
        
        # Calcolo metriche Proxy Portafoglio
        alloc = df.groupby("Asset_Class")["Peso"].sum().reindex(PROXIES.keys(), fill_value=0)
        w = alloc.values
        # Volatilità
        v_asset = np.array([PROXIES[k]['vol'] for k in PROXIES])
        cov = np.outer(v_asset, v_asset) * CORRELATION.values
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Valore Totale", f"€ {tot_val:,.2f}")
        k2.metric("PnL (Stimato)", f"€ {tot_pnl:,.2f}", delta=f"{(tot_pnl/tot_val):.2%}")
        k3.metric("Volatilità Attesa", f"{port_vol:.2%}", help="Rischio stimato su base proxy")
        k4.metric("Asset Class", len(df["Asset_Class"].unique()))

        st.divider()

        # --- TABS COMPLETE ---
        tabs = st.tabs(["📈 Dashboard", "⚠️ Alert & Rischio", "🌍 Macro Monitor", "📋 Dati"])

        # TAB 1: DASHBOARD
        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Allocazione Attuale")
                clean_alloc = alloc[alloc > 0].reset_index(name="Peso")
                fig = px.pie(clean_alloc, values="Peso", names="Asset_Class", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.subheader("Ottimizzazione")
                opt = optimize_portfolio({'max_equity': max_eq, 'min_bond': min_bond}, alloc)
                comp = pd.DataFrame({"Asset": PROXIES.keys(), "Attuale": alloc.values, "Target": list(opt.values())})
                fig_bar = px.bar(comp.melt(id_vars="Asset"), x="Asset", y="value", color="variable", barmode="group")
                st.plotly_chart(fig_bar, use_container_width=True)

        # TAB 2: ALERT
        with tabs[1]:
            st.subheader("Centro Controllo Rischi")
            
            ac1, ac2 = st.columns(2)
            
            # Scadenze
            curr_year = datetime.now().year
            expiring = df[(df["Maturity_Year"] > 0) & (df["Maturity_Year"] <= curr_year + 2)]
            with ac1:
                if not expiring.empty:
                    st.error(f"📅 {len(expiring)} Titoli in scadenza < 24 mesi")
                    st.dataframe(expiring[["Titolo", "Maturity_Year", "Valore_Mercato"]], hide_index=True)
                else:
                    st.success("✅ Nessuna scadenza a breve termine.")

            # Concentrazione
            conc = df[df["Peso"] > 0.25]
            with ac2:
                if not conc.empty:
                    st.warning(f"🚨 {len(conc)} Posizioni concentrate (>25%)")
                    st.dataframe(conc[["Titolo", "Peso"]].style.format({"Peso": "{:.1%}"}), hide_index=True)
                else:
                    st.success("✅ Diversificazione OK (Max < 25%)")

        # TAB 3: MACRO
        with tabs[2]:
            st.subheader("Scenario Macroeconomico")
            mc1, mc2 = st.columns(2)
            mc1.metric("Spread BTP-Bund", f"{spread} bps", delta="-5 bps" if spread < 150 else "High", delta_color="inverse")
            mc2.metric("Indice Paura (VIX)", f"{vix}", delta="Stable" if vix < 20 else "Volatile", delta_color="inverse")
            
            st.info("💡 Nota: Inserisci i dati macro aggiornati nella barra laterale per calibrare gli scenari.")

        # TAB 4: DATI
        with tabs[3]:
            st.dataframe(df[["Titolo", "ISIN", "Asset_Class", "Valore_Mercato", "PnL", "Valuta"]].style.format({
                "Valore_Mercato": "€ {:,.2f}", "PnL": "€ {:,.2f}"
            }))

else:
    st.info("👋 Carica il file CSV per iniziare l'analisi.")
