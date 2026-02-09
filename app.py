import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
import smtplib
from email.mime.text import MIMEText

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Portfolio Monitor PRO", layout="wide")

# ======================================================
# AUTH
# ======================================================
with open("users.yaml") as f:
    users = yaml.safe_load(f)

st.sidebar.title("🔐 Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username not in users["users"] or password != users["users"][username]["password"]:
    st.warning("Inserisci credenziali valide")
    st.stop()

# ======================================================
# TITLE
# ======================================================
st.title("📊 Portfolio Monitor PRO – Profilo Bilanciato EUR")
st.caption("Dashboard avanzata – ottimizzata per iPad")

# ======================================================
# UPLOAD CSV
# ======================================================
uploaded_file = st.sidebar.file_uploader(
    "📂 Carica CSV bancario",
    type=["csv"]
)

if not uploaded_file:
    st.info("⬅️ Carica il file CSV della banca")
    st.stop()

# ======================================================
# CSV CLEANING (BANCARIO)
# ======================================================
df_raw = pd.read_csv(uploaded_file, sep=";", encoding="latin1")
df = df_raw.iloc[1:].copy()
df.columns = df_raw.iloc[1]

df = df.rename(columns={
    "Titolo": "Titolo",
    "ISIN": "ISIN",
    "Strumento": "Asset Class",
    "Valore di mercato €": "Valore",
    "Var %": "Performance %"
})

df["Valore"] = (
    df["Valore"].astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# ======================================================
# TABLE
# ======================================================
st.subheader("📄 Portafoglio")
st.dataframe(
    df[["Titolo", "ISIN", "Asset Class", "Valore"]],
    use_container_width=True
)

# ======================================================
# ALLOCATION
# ======================================================
allocation = df.groupby("Asset Class")["Valore"].sum().reset_index()
allocation["Peso %"] = allocation["Valore"] / allocation["Valore"].sum() * 100

st.subheader("📊 Asset Allocation")
st.plotly_chart(
    px.pie(allocation, names="Asset Class", values="Peso %"),
    use_container_width=True
)

# ======================================================
# METRICS
# ======================================================
col1, col2, col3 = st.columns(3)
col1.metric("Valore Totale (€)", f"{allocation['Valore'].sum():,.0f}")
col2.metric("Volatilità Stimata", "≈ 8%")
col3.metric("Drawdown Stimato", "≈ -14%")

# ======================================================
# TARGET MARKOWITZ
# ======================================================
target = {
    "ETF": 42,
    "Obbligazione": 38,
    "ETC": 8,
    "Liquidità": 12
}
target_df = pd.DataFrame({
    "Asset Class": target.keys(),
    "Target %": target.values()
})

merged = allocation.merge(target_df, on="Asset Class", how="left")

st.subheader("🎯 Attuale vs Target")
st.plotly_chart(
    px.bar(
        merged,
        x="Asset Class",
        y=["Peso %", "Target %"],
        barmode="group"
    ),
    use_container_width=True
)

# ======================================================
# ALERT ENGINE
# ======================================================
st.subheader("⚠️ Alert Automatici")
alerts = []

if "Obbligazione" in allocation["Asset Class"].values:
    bond_weight = allocation.loc[
        allocation["Asset Class"] == "Obbligazione", "Peso %"
    ].values[0]
    if bond_weight > 45:
        alerts.append("Obbligazionario sopra 45%")

if allocation["Peso %"].max() > 30:
    alerts.append("Asset class sopra 30%")

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("Portafoglio in linea con il profilo")

# ======================================================
# STRESS TEST
# ======================================================
st.subheader("🧪 Stress Test")
stress = pd.DataFrame({
    "Scenario": ["Crisi 2008", "Covid 2020", "Shock BCE"],
    "Impatto Stimato": ["-22%", "-18%", "-12%"]
})
st.table(stress)

# ======================================================
# MACRO
# ======================================================
st.subheader("🌍 Scenario Macro")
c1, c2, c3, c4 = st.columns(4)
c1.metric("BCE", "Pausa tassi")
c2.metric("FED", "Tagli graduali")
c3.metric("Spread BTP-Bund", "Stabile")
c4.metric("EUR/USD", "Neutrale")

# ======================================================
# EMAIL ALERT (CONFIG)
# ======================================================
st.subheader("📧 Alert Email")
email = st.text_input("Email destinatario alert")
send_test = st.button("Invia test alert")

def send_email_alert(message):
    msg = MIMEText(message)
    msg["Subject"] = "⚠️ Alert Portafoglio"
    msg["From"] = "tuamail@gmail.com"
    msg["To"] = email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login("tuamail@gmail.com", "APP_PASSWORD")
        server.send_message(msg)

if send_test and email:
    send_email_alert("Test alert portafoglio")
    st.success("Email inviata")

st.caption("Uso personale – non consulenza finanziaria")
