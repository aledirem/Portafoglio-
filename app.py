import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from io import StringIO
import re
from datetime import datetime

# --- 1. CONFIGURAZIONE PAGINA (Deve essere la prima istruzione) ---
st.set_page_config(
    page_title="Portfolio Manager Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SISTEMA DI SICUREZZA (PASSWORD) ---
def check_password():
    """Ritorna True se l'utente ha la password corretta."""

    def password_entered():
        """Controlla se la password inserita è corretta."""
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Non conservare la password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Prima esecuzione, mostra input
        st.text_input(
            "🔐 Inserisci Password di Sicurezza", type="password", on_change=password_entered, key="password"
        )
        st.info("Nota: Imposta la password nei 'Secrets' di Streamlit Cloud.")
        return False
    elif not st.session_state["password_correct"]:
        # Password errata, riprova
        st.text_input(
            "🔐 Inserisci Password di Sicurezza", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password errata")
        return False
    else:
        # Password corretta
        return True

if not check_password():
    st.stop()  # BLOCCO APP: Niente viene caricato sotto questa linea se non loggati

# --- 3. DATI FINANZIARI (Proxy per Markowitz) ---
# Dati "Proxy" per simulazione professionale senza feed live
PROXIES = {
    'Equity':      {'ret': 0.075, 'vol': 0.16}, # Global Equity
    'Bond':        {'ret': 0.035, 'vol': 0.06}, # Aggregato EUR
    'Commodities': {'ret': 0.045, 'vol': 0.18}, # Gold/Energy
    'Cash':        {'ret': 0.025, 'vol': 0.005} # Monetario
}

# Matrice di Correlazione Stilizzata (Lungo Periodo)
CORRELATION = pd.DataFrame(
    [
        [1.0, 0.2, 0.4, 0.0],  # Equity
        [0.2, 1.0, 0.1, 0.1],  # Bond
        [0.4, 0.1, 1.0, 0.1],  # Comm
        [0.0, 0.1, 0.1, 1.0]   # Cash
    ],
    index=PROXIES.keys(), columns=PROXIES.keys()
)

# --- 4. FUNZIONI DI CALCOLO ---

def get_portfolio_metrics(weights):
    """Calcola Rendimento e Volatilità attesi del portafoglio dato un array di pesi"""
    weights = np.array(weights)
    ex_rets = np.array([PROXIES[c]['ret'] for c in PROXIES])
    vols = np.array([PROXIES[c]['vol'] for c in PROXIES])
    
    # Rendimento
    port_ret = np.sum(weights * ex_rets)
    
    # Volatilità (w^T * Cov * w)
    cov_matrix = np.outer(vols, vols) * CORRELATION.values
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return port_ret, port_vol

def optimize_portfolio(constraints_dict):
    """
    Ottimizza usando Markowitz vincolato.
    Obiettivo: Massimizzare Sharpe Ratio (Rendimento / Volatilità)
    """
    n_assets = len(PROXIES)
    init_guess = np.repeat(1/n_assets, n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Vincolo: Somma pesi = 1
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    keys = list(PROXIES.keys()) # ['Equity', 'Bond', 'Commodities', 'Cash']
    
    # Applicazione vincoli utente dinamici
    if 'max_equity' in constraints_dict:
        idx = keys.index('Equity')
        cons.append({'type': 'ineq', 'fun': lambda x: constraints_dict['max_equity'] - x[idx]})
    
    if 'min_bond' in constraints_dict:
        idx = keys.index('Bond')
        cons.append({'type': 'ineq', 'fun': lambda x: x[idx] - constraints_dict['min_bond']})
        
    if 'max_comm' in constraints_dict:
        idx = keys.index('Commodities')
        cons.append({'type': 'ineq', 'fun': lambda x: constraints_dict['max_comm'] - x[idx]})
        
    if 'min_cash' in constraints_dict:
        idx = keys.index('Cash')
        cons.append({'type': 'ineq', 'fun': lambda x: x[idx] - constraints_dict['min_cash']})
        
    if 'target_vol_max' in constraints_dict:
        cons.append({'type': 'ineq', 'fun': lambda x: constraints_dict['target_vol_max'] - get_portfolio_metrics(x)[1]})

    def objective(weights):
        ret, vol = get_portfolio_metrics(weights)
        if vol == 0: return 0
        return -(ret / vol) # Maximize Sharpe

    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    return dict(zip(keys, result.x))

def parse_fineco_csv(uploaded_file):
    try:
        # Lettura CSV (salta prime 2 righe, separatore ;)
        stringio = StringIO(uploaded_file.getvalue().decode("latin1"))
        df = pd.read_csv(stringio, sep=';', skiprows=2)
        
        # Identifica colonna valore (gestisce eventuali caratteri strani)
        val_col = [c for c in df.columns if str(c).startswith('Valore di mercato')][0]
        
        # Pulizia Dati Numerici
        def clean_num(x):
            if isinstance(x, str):
                x = x.replace(' ', '').replace(',', '.')
            return float(x)
        
        df['Valore_Mercato'] = df[val_col].apply(clean_num)
        
        # Mappatura Asset Class (Euristica)
        def get_asset_class(row):
            instr = str(row['Strumento']).upper()
            title = str(row['Titolo']).upper()
            
            if 'ETC' in instr or 'COMM' in title: return 'Commodities'
            if 'OBBLIGAZIONE' in instr: return 'Bond'
            if 'ETF' in instr:
                if any(x in title for x in ['BOND', 'TREASURY', 'CORP', 'HIGH YIELD', 'GOVT']): return 'Bond'
                return 'Equity'
            return 'Cash' # Default fallback

        df['Asset_Class'] = df.apply(get_asset_class, axis=1)
        
        # Estrazione Scadenza (da Titolo es: BTP-15LG27)
        def get_maturity(title):
            match = re.search(r'(\d{1,2})([A-Z]{2})(\d{2})', str(title))
            if match:
                try:
                    yr = int("20" + match.group(3))
                    return yr
                except:
                    return None
            return None

        df['Maturity_Year'] = df['Titolo'].apply(get_maturity)
        
        return df
    except Exception as e:
        st.error(f"Errore lettura file: {e}")
        return None

# --- 5. INTERFACCIA UTENTE (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Controlli")
    
    st.header("1. Upload")
    uploaded_file = st.file_uploader("Carica CSV Fineco", type=['csv'])
    
    st.divider()
    
    st.header("2. Vincoli Ottimizzazione")
    target_vol = st.slider("Volatilità Target Max", 0.05, 0.15, 0.09, 0.01)
    c_max_eq = st.slider("Max Azionario", 0.0, 1.0, 0.45, 0.05)
    c_min_bond = st.slider("Min Obbligazionario", 0.0, 1.0, 0.35, 0.05)
    c_max_comm = st.slider("Max Commodities", 0.0, 1.0, 0.10, 0.05)
    
    st.divider()
    
    st.header("3. Monitor Macro (Input)")
    spread_btp = st.number_input("Spread BTP-Bund (bps)", value=145)
    eur_usd = st.number_input("EUR/USD", value=1.08)
    vix = st.number_input("VIX Volatility", value=14.5)

# --- 6. INTERFACCIA UTENTE (MAIN) ---
st.title("📊 Portfolio Intelligence Dashboard")

if uploaded_file is not None:
    df = parse_fineco_csv(uploaded_file)
    
    if df is not None:
        total_value = df['Valore_Mercato'].sum()
        df['Peso %'] = df['Valore_Mercato'] / total_value
        
        # 1. ANALISI ATTUALE
        current_alloc = df.groupby('Asset_Class')['Peso %'].sum().reindex(PROXIES.keys(), fill_value=0)
        
        # Calcolo metriche attuali
        curr_ret, curr_vol = get_portfolio_metrics(current_alloc.values)
        curr_var_95 = 1.65 * curr_vol * total_value # VaR 95%
        
        # 2. OTTIMIZZAZIONE MARKOWITZ
        constraints = {
            'target_vol_max': target_vol,
            'max_equity': c_max_eq,
            'min_bond': c_min_bond,
            'max_comm': c_max_comm,
            'min_cash': 0.05
        }
        opt_weights = optimize_portfolio(constraints)
        opt_ret, opt_vol = get_portfolio_metrics(list(opt_weights.values()))
        
        # --- TOP KPI ROW ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valore Totale", f"€ {total_value:,.2f}")
        col2.metric("Volatilità Annuale", f"{curr_vol:.1%}", delta=f"{(curr_vol-opt_vol):.1%} vs Ottimale", delta_color="inverse")
        col3.metric("Rendimento Atteso", f"{curr_ret:.1%}", delta=f"{(curr_ret-opt_ret):.1%} vs Ottimale")
        col4.metric("VaR (Rischio 95%)", f"€ {curr_var_95:,.0f}", help="Perdita massima stimata a 1 anno con confidenza 95%")
        
        st.divider()

        # --- GRAFICI PRINCIPALI ---
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("Allocazione: Attuale vs Target")
            comp_df = pd.DataFrame({
                'Asset Class': list(PROXIES.keys()),
                'Attuale': current_alloc.values,
                'Target (Markowitz)': list(opt_weights.values())
            })
            # Trasforma per grafico grouped
            comp_df_melted = comp_df.melt(id_vars='Asset Class', var_name='Tipo', value_name='Peso')
            
            fig_bar = px.bar(comp_df_melted, x='Asset Class', y='Peso', color='Tipo',
                             barmode='group', height=400, text_auto='.1%')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("Composizione Dettagliata")
            fig_sun = px.sunburst(df, path=['Asset_Class', 'Titolo'], values='Valore_Mercato',
                                  height=400)
            st.plotly_chart(fig_sun, use_container_width=True)

        # --- SEZIONE ALERTS ---
        st.subheader("⚠️ Centro Notifiche & Rischi")
        
        alert_cols = st.columns(3)
        
        # Alert 1: Scadenze
        curr_year = datetime.now().year
        expiring = df[ (df['Maturity_Year'] > 0) & (df['Maturity_Year'] < curr_year + 2) ]
        with alert_cols[0]:
            st.info(f"📅 Scadenze < 24 Mesi: {len(expiring)}")
            if not expiring.empty:
                st.dataframe(expiring[['Titolo', 'Maturity_Year', 'Valore_Mercato']].style.format({'Valore_Mercato': "{:.2f}"}), hide_index=True)
        
        # Alert 2: Concentrazione
        concentrated = df[df['Peso %'] > 0.30]
        with alert_cols[1]:
            if not concentrated.empty:
                st.error(f"🚨 Concentrazione >30%: {len(concentrated)}")
                st.dataframe(concentrated[['Titolo', 'Peso %']].style.format({'Peso %': "{:.1%}"}), hide_index=True)
            else:
                st.success("✅ Diversificazione OK (Max < 30%)")

        # Alert 3: Macro Watchlist
        with alert_cols[2]:
            st.warning("🌍 Macro Watchlist")
            cols_macro = st.columns(2)
            cols_macro[0].metric("Spread BTP", f"{spread_btp}", delta="-5" if spread_btp < 200 else "High", delta_color="inverse")
            cols_macro[1].metric("VIX", f"{vix}", delta="Stable" if vix < 20 else "High", delta_color="inverse")

        # --- DATI GREZZI ---
        with st.expander("📂 Vedi Dati Importati"):
            st.dataframe(df.style.format({'Valore_Mercato': '€ {:.2f}', 'Peso %': '{:.2%}', 'Maturity_Year': '{:.0f}'}))

else:
    st.info("👋 Benvenuto. Per iniziare, carica il file CSV dalla barra laterale (Menu a sinistra).")
    st.markdown("""
    ---
    **Istruzioni:**
    1. Scarica il portafoglio da Fineco (Excel/CSV).
    2. Caricalo qui a sinistra.
    3. Imposta i vincoli di rischio.
    
    *I dati sono elaborati solo in memoria e non vengono salvati.*
    """)
