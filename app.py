import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from io import StringIO
import re
from datetime import datetime, date, timedelta

# ----------------------------
# 0) CONFIGURAZIONE (Deve essere la prima istruzione)
# ----------------------------
st.set_page_config(
    page_title="Portfolio Manager Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# 1) SICUREZZA PASSWORD
# ----------------------------
def _password_gate():
    # Cerca la password nei secrets. Se non c'è, ferma tutto e avvisa.
    expected = st.secrets.get("PASSWORD", None)

    if not expected:
        st.error("⚠️ Configurazione incompleta: Imposta la chiave 'PASSWORD' nei Secrets di Streamlit Cloud.")
        st.code('PASSWORD = "LaTuaPasswordSegreta"', language="toml")
        st.stop()

    def _check():
        if st.session_state.get("_pw_input", "") == expected:
            st.session_state["_pw_ok"] = True
            st.session_state["_pw_input"] = "" # Pulisce campo input
        else:
            st.session_state["_pw_ok"] = False

    if st.session_state.get("_pw_ok", False):
        return # Accesso garantito

    # Schermata di login
    st.markdown("## 🔐 Accesso Riservato")
    st.text_input("Inserisci Password", type="password", key="_pw_input", on_change=_check)
    
    if "_pw_ok" in st.session_state and not st.session_state["_pw_ok"]:
        st.error("Password non valida.")
    
    st.stop() # Blocca esecuzione resto dello script

# Attiva il blocco password
_password_gate()

# ----------------------------
# 2) DATI PROXY (Markowitz Strategico)
# ----------------------------
PROXIES = {
    "Equity":      {"ret": 0.075, "vol": 0.16},
    "Bond":        {"ret": 0.035, "vol": 0.06},
    "Commodities": {"ret": 0.045, "vol": 0.18},
    "Cash":        {"ret": 0.025, "vol": 0.005},
}

CORRELATION = pd.DataFrame(
    [
        [1.0, 0.2, 0.4, 0.0],  # Equity
        [0.2, 1.0, 0.1, 0.1],  # Bond
        [0.4, 0.1, 1.0, 0.1],  # Comm
        [0.0, 0.1, 0.1, 1.0],  # Cash
    ],
    index=PROXIES.keys(), columns=PROXIES.keys()
)

MDD_PROXY = {"Equity": 0.50, "Bond": 0.15, "Commodities": 0.35, "Cash": 0.01}

# ----------------------------
# 3) UTILITÀ DI PARSING
# ----------------------------
def parse_eu_number(x) -> float:
    """Converte '1.234,56' -> 1234.56. Gestisce errori e NaN."""
    if pd.isna(x) or str(x).strip() == "":
        return 0.0
    
    s = str(x).strip()
    # Rimuovi valuta e spazi
    s = re.sub(r"[^\d,.-]", "", s)
    
    try:
        # Se c'è sia punto che virgola (es. 1.000,50) -> togli punto, cambia virgola in punto
        if "." in s and "," in s:
            s = s.replace(".", "").replace(",", ".")
        # Se c'è solo virgola (es. 50,5) -> cambia in punto
        elif "," in s:
            s = s.replace(",", ".")
        # Se c'è solo punto (es. 1000.50 o 1.000) -> attenzione, assumiamo standard US (float)
        
        return float(s)
    except:
        return 0.0

def infer_asset_class(row) -> str:
    instr = str(row.get("Strumento", "")).upper()
    title = str(row.get("Titolo", "")).upper()

    if "OBBLIGAZIONE" in instr or "BOND" in instr: return "Bond"
    if "ETC" in instr or "COMM" in title: return "Commodities"
    if "ETF" in instr:
        # Euristica semplice per ETF Obbligazionari
        if any(x in title for x in ["BOND", "TREASURY", "CORP", "HIGH YIELD", "GOVT", "AGGREGATE"]): 
            return "Bond"
        return "Equity"
    return "Cash"

def get_maturity_year(title):
    # Cerca pattern tipo 31MZ26, 15LG30, ecc.
    try:
        match = re.search(r'(\d{1,2})([A-Z]{2,3})(\d{2})', str(title).upper())
        if match:
            return int("20" + match.group(3))
    except:
        pass
    return None

# ----------------------------
# 4) LETTORE FILE FINECO (ROBUSTO)
# ----------------------------
def parse_fineco_csv_robust(uploaded_file):
    try:
        # Leggi tutto come testo
        content = uploaded_file.getvalue().decode("latin1", errors="replace")
        lines = content.splitlines()
        
        # Cerca la riga di intestazione corretta
        header_row = None
        for i, line in enumerate(lines):
            # La riga giusta deve avere "Titolo" e "ISIN" (o simili)
            if "Titolo" in line and "ISIN" in line:
                header_row = i
                break
        
        if header_row is None:
            st.error("❌ Formato file non riconosciuto. Assicurati che sia un export Fineco contenente 'Titolo' e 'ISIN'.")
            return None

        # Carica saltando le righe inutili prima dell'header
        df = pd.read_csv(StringIO(content), sep=';', skiprows=header_row)
        
        # Pulisci nomi colonne (rimuovi spazi, metti minuscolo per ricerca sicura)
        df.columns = [c.strip() for c in df.columns]
        
        # Trova colonne chiave
        col_valore = next((c for c in df.columns if "Valore" in c and "mercato" in c), None)
        col_carico = next((c for c in df.columns if "Valore" in c and "carico" in c), None)
        
        if not col_valore:
            st.error("❌ Colonna 'Valore di mercato' non trovata.")
            return None

        # Conversione Numerica Sicura
        df["Valore_Mercato"] = df[col_valore].apply(parse_eu_number)
        
        if col_carico:
            df["Valore_Carico"] = df[col_carico].apply(parse_eu_number)
        else:
            df["Valore_Carico"] = df["Valore_Mercato"] # Fallback per evitare div/0

        # Filtra righe vuote o totali (valore mercato deve essere > 0.01)
        df = df[df["Valore_Mercato"] > 0.01].copy()
        
        # Arricchimento Dati
        df["Asset_Class"] = df.apply(infer_asset_class, axis=1)
        df["Maturity_Year"] = df["Titolo"].apply(get_maturity_year)
        
        # PnL e Performance
        df["PnL"] = df["Valore_Mercato"] - df["Valore_Carico"]
        # Evita divisione per zero
        df["Return_Pct"] = df.apply(lambda x: (x["PnL"] / x["Valore_Carico"]) if x["Valore_Carico"] > 1 else 0.0, axis=1)
        
        # Aggiungi colonna Valuta se manca
        if "Valuta" not in df.columns:
            df["Valuta"] = "EUR" # Assumption
            
        return df

    except Exception as e:
        st.error(f"Errore tecnico nel parsing: {e}")
        return None

# ----------------------------
# 5) CALCOLATORE MARKOWITZ
# ----------------------------
def optimize_portfolio(constraints, current_alloc_weights):
    keys = list(PROXIES.keys())
    n = len(keys)
    
    # Obiettivo: Max Sharpe Ratio (semplificato)
    def objective(w):
        ret = np.sum(w * np.array([PROXIES[k]['ret'] for k in keys]))
        # Volatility
        vols = np.array([PROXIES[k]['vol'] for k in keys])
        cov = np.outer(vols, vols) * CORRELATION.values
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if vol == 0: return 0
        return -(ret / vol) # Minimize negative Sharpe

    # Vincoli
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}] # Sum = 1
    
    if 'max_equity' in constraints:
        idx = keys.index('Equity')
        cons.append({'type': 'ineq', 'fun': lambda x: constraints['max_equity'] - x[idx]})
    
    if 'min_bond' in constraints:
        idx = keys.index('Bond')
        cons.append({'type': 'ineq', 'fun': lambda x: x[idx] - constraints['min_bond']})

    bounds = tuple((0.05, 1.0) for _ in range(n)) # Minimo 5% per asset class per diversificazione
    
    init_guess = np.repeat(1/n, n)
    res = minimize(objective, init_guess, bounds=bounds, constraints=cons, method='SLSQP')
    
    return dict(zip(keys, res.x)) if res.success else dict(zip(keys, init_guess))

# ----------------------------
# 6) INTERFACCIA UTENTE (UI)
# ----------------------------
with st.sidebar:
    st.header("1. Upload Dati")
    uploaded_file = st.file_uploader("Carica CSV Fineco", type=['csv'])
    
    st.divider()
    st.header("2. Obiettivi")
    target_vol = st.slider("Volatilità Max", 0.05, 0.20, 0.09, 0.01)
    max_eq = st.slider("Max Azionario", 0.0, 1.0, 0.50, 0.05)
    min_bond = st.slider("Min Obbligazionario", 0.0, 1.0, 0.30, 0.05)

st.title("📊 Portfolio Intelligence Dashboard")

if uploaded_file:
    df = parse_fineco_csv_robust(uploaded_file)
    
    if df is not None:
        # Calcoli aggregati
        valore_totale = df["Valore_Mercato"].sum()
        df["Peso"] = df["Valore_Mercato"] / valore_totale
        
        # KPI in alto
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Valore Portafoglio", f"€ {valore_totale:,.2f}")
        
        pnl_totale = df["PnL"].sum()
        k2.metric("PnL Totale", f"€ {pnl_totale:,.2f}", delta_color="normal")
        
        # Allocazione
        alloc = df.groupby("Asset_Class")["Peso"].sum().reindex(PROXIES.keys(), fill_value=0)
        
        # Grafici
        t1, t2 = st.tabs(["Dashboard", "Dettaglio Titoli"])
        
        with t1:
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.subheader("Allocazione Attuale")
                # FIX per Plotly: Filtra valori 0 per evitare crash
                chart_data = alloc[alloc > 0].reset_index()
                chart_data.columns = ["Asset Class", "Peso"]
                
                if not chart_data.empty:
                    fig = px.pie(chart_data, values="Peso", names="Asset Class", hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Dati allocazione vuoti.")

            with c_right:
                st.subheader("Target Ottimale")
                constraints = {'max_equity': max_eq, 'min_bond': min_bond}
                opt_weights = optimize_portfolio(constraints, alloc)
                
                # Compare Bar Chart
                comp_df = pd.DataFrame({
                    "Asset": list(PROXIES.keys()),
                    "Attuale": alloc.values,
                    "Target": list(opt_weights.values())
                }).melt(id_vars="Asset", var_name="Tipo", value_name="Peso")
                
                fig_bar = px.bar(comp_df, x="Asset", y="Peso", color="Tipo", barmode="group")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Alerts
            st.divider()
            st.subheader("⚠️ Alert System")
            
            # Scadenze
            curr_year = datetime.now().year
            expiring = df[ (df["Maturity_Year"] > 0) & (df["Maturity_Year"] <= curr_year + 2) ]
            
            if not expiring.empty:
                st.warning(f"Ci sono {len(expiring)} titoli in scadenza entro il {curr_year + 2}:")
                st.dataframe(expiring[["Titolo", "Maturity_Year", "Valore_Mercato"]])
            else:
                st.success("Nessuna scadenza a breve termine rilevata.")

        with t2:
            st.dataframe(df[["Titolo", "ISIN", "Asset_Class", "Valore_Mercato", "Return_Pct", "Peso"]].style.format({
                "Valore_Mercato": "€ {:,.2f}",
                "Return_Pct": "{:.2%}",
                "Peso": "{:.2%}"
            }))

else:
    st.info("👋 Carica il tuo file CSV Fineco dalla barra laterale per iniziare.")
