import streamlit as st
import pandas as pd
import yaml

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Portfolio Monitor PRO", layout="wide")

# -------------------------
# LOGIN
# -------------------------
def load_users():
    with open("users.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["users"]

users = load_users()

with st.sidebar:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    authenticated = False
    if username and password:
        if username in users and users[username]["password"] == password:
            st.success("Login OK")
            authenticated = True
        else:
            st.error("Credenziali errate")

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
# LETTURA CSV ROBUSTA
# -------------------------
df = None
read_errors = []

for encoding in ["utf-8", "latin1", "cp1252"]:
    for sep in [";", ","]:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
            if df.shape[1] > 1:
                break
        except Exception as e:
            read_errors.append(str(e))
    if df is not None and df.shape[1] > 1:
        break

if df is None or df.shape[1] <= 1:
    st.error("❌ Impossibile leggere il CSV bancario")
    st.write("Suggerimento: assicurati che sia un vero CSV (non Excel).")
    st.stop()

df.columns = [c.strip() for c in df.columns]

# -------------------------
# COLONNA VALORE
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

value_col = next((c for c in possible_value_cols if c in df.columns), None)

if value_col is None:
    st.error("❌ Colonna valore non trovata")
    st.write("Colonne disponibili:", list(df.columns))
    st.stop()

df["Valore"] = (
    df[value_col]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.replace("€", "", regex=False)
    .astype(float)
)

# -------------------------
# COLONNA STRUMENTO
# -------------------------
possible_name_cols = [
    "Strumento",
    "Descrizione",
    "Titolo",
    "Nome",
    "ISIN"
]

name_col = next((c for c in possible_name_cols if c in df.columns), None)

df["Strumento"] = df[name_col] if name_col else "Non specificato"

# -------------------------
# METRICHE
# -------------------------
totale = df["Valore"].sum()

c1, c2 = st.columns(2)
c1.metric("💰 Valore totale", f"{totale:,.2f} €")
c2.metric("📦 Numero strumenti", len(df))

# -------------------------
# ALLOCAZIONE
# -------------------------
alloc = df.groupby("Strumento")["Valore"].sum().sort_values(ascending=False)

st.subheader("📊 Allocazione per strumento")
st.bar_chart(alloc)

# -------------------------
# TABELLA
# -------------------------
st.subheader("📋 Dettaglio")
st.dataframe(
    df[["Strumento", "Valore"]].sort_values("Valore", ascending=False),
    use_container_width=True
)

# -------------------------
# ALERT
# -------------------------
st.subheader("⚠️ Alert")

peso_max = alloc.max() / totale

if peso_max > 0.25:
    st.error("🚨 Concentrazione >25% (5★ URGENTE)")
elif peso_max > 0.15:
    st.warning("⚠️ Concentrazione >15% (3★ IMPORTANTE)")
else:
    st.success("✅ Allocazione equilibrata")

st.caption("Portfolio Monitor PRO – cloud edition")
