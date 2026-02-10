import streamlit as st
import pandas as pd
import yaml

# -------------------------
# CONFIG STREAMLIT
# -------------------------
st.set_page_config(
    page_title="Portfolio Monitor PRO",
    layout="wide"
)

# -------------------------
# LOGIN
# -------------------------
def load_users():
    with open("users.yaml", "r") as f:
        return yaml.safe_load(f)["users"]

users = load_users()

with st.sidebar:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if username and password:
        if username in users and users[username]["password"] == password:
            st.success("Login OK")
            authenticated = True
        else:
            st.error("Credenziali errate")
            authenticated = False
    else:
        authenticated = False

if not authenticated:
    st.stop()

# -------------------------
# HEADER
# -------------------------
st.title("📊 Portfolio Monitor PRO – Profilo Bilanciato EUR")
st.caption("Dashboard avanzata – ottimizzata per iPad")

# -------------------------
# UPLOAD CSV
# -------------------------
uploaded_file = st.sidebar.file_uploader(
    "📁 Carica CSV bancario",
    type=["csv"]
)

if uploaded_file is None:
    st.info("⬅️ Carica un file CSV per iniziare")
    st.stop()

# -------------------------
# LETTURA CSV
# -------------------------
try:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
except Exception:
    st.error("Errore nella lettura del CSV")
    st.stop()

df.columns = [c.strip() for c in df.columns]

# -------------------------
# RICONOSCIMENTO COLONNA VALORE
# -------------------------
possible_value_cols = [
    "Valore",
    "Controvalore",
    "Valore di mercato",
    "Valore attuale",
    "Importo",
    "Totale",
    "Valore (€)"
]

value_col = None
for col in possible_value_cols:
    if col in df.columns:
        value_col = col
        break

if value_col is None:
    st.error("❌ Colonna valore non trovata nel CSV")
    st.write("Colonne disponibili:", list(df.columns))
    st.stop()

df["Valore"] = (
    df[value_col]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# -------------------------
# RICONOSCIMENTO STRUMENTO
# -------------------------
possible_name_cols = [
    "Strumento",
    "Descrizione",
    "Titolo",
    "Nome",
    "ISIN"
]

name_col = None
for col in possible_name_cols:
    if col in df.columns:
        name_col = col
        break

if name_col is None:
    df["Strumento"] = "Non specificato"
else:
    df["Strumento"] = df[name_col]

# -------------------------
# METRICHE PRINCIPALI
# -------------------------
totale = df["Valore"].sum()
num_strumenti = df.shape[0]

col1, col2 = st.columns(2)
col1.metric("💰 Valore totale portafoglio", f"{totale:,.2f} €")
col2.metric("📦 Numero strumenti", num_strumenti)

# -------------------------
# ALLOCAZIONE
# -------------------------
alloc = (
    df.groupby("Strumento")["Valore"]
    .sum()
    .sort_values(ascending=False)
)

st.subheader("📌 Allocazione per strumento")
st.bar_chart(alloc)

# -------------------------
# TABELLA DETTAGLIO
# -------------------------
st.subheader("📋 Dettaglio portafoglio")
st.dataframe(
    df[[ "Strumento", "Valore" ]]
    .sort_values("Valore", ascending=False),
    use_container_width=True
)

# -------------------------
# ALERT BASE
# -------------------------
st.subheader("⚠️ Alert automatici")

max_weight = alloc.max() / totale

if max_weight > 0.25:
    st.error("🚨 Concentrazione elevata: uno strumento >25% del portafoglio (5★ URGENTE)")
elif max_weight > 0.15:
    st.warning("⚠️ Concentrazione significativa >15% (3★ IMPORTANTE)")
else:
    st.success("✅ Nessuna concentrazione critica")

# -------------------------
# FOOTER
# -------------------------
st.caption("Portfolio Monitor PRO – versione cloud")
