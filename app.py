import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="CAC x LTV (Semestres)", layout="wide")

# Caminho local ou URL no GitHub
CSV_PATH = st.secrets.get("CSV_PATH", "ies_cac_ltv.csv")
GITHUB_RAW_URL = st.secrets.get("GITHUB_RAW_URL", "")  
# Exemplo: 'https://raw.githubusercontent.com/<user>/<repo>/main/ies_cac_ltv.csv'

@st.cache_data
def load_data(csv_path: str, raw_url: str):
    if raw_url:
        return pd.read_csv(raw_url)
    return pd.read_csv(csv_path)

df = load_data(CSV_PATH, GITHUB_RAW_URL)

st.title("CAC × LTV — Fluxo acumulado por semestre (S0…S12)")
ies_list = df["IES"].tolist()
ies = st.selectbox("Selecione a IES", ies_list, index=0)

row = df[df["IES"] == ies].iloc[0]
cac = float(row["CAC"])
receita_cols = [c for c in df.columns if c.startswith("Proj_Receita_")]
receitas = row[receita_cols].values.astype(float)
acc = np.concatenate(([cac], cac + np.cumsum(receitas)))
sem_labels = ["S0"] + [f"S{i}" for i in range(1, len(receitas)+1)]
pb_idx = next((i for i, v in enumerate(acc) if v >= 0), None)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sem_labels, acc, marker="o", linewidth=2, color="purple")
ax.axhline(0, linestyle="--", linewidth=1, color="gray")

# CAC
ax.annotate(f"CAC (S0): R$ {cac:,.0f}",
            xy=(sem_labels[0], acc[0]),
            xytext=(0.8, acc[0]*0.6),
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8))

# Payback
if pb_idx is not None:
    ax.axvline(pb_idx, linestyle="--", linewidth=1, color="gray")
    ax.annotate(f"Payback: {sem_labels[pb_idx]}",
                xy=(sem_labels[pb_idx], 0),
                xytext=(pb_idx+0.2, max(acc)*0.08),
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8))

# Rótulos
for x, y in zip(sem_labels, acc):
    ax.text(x, y, f"{y:,.0f}", ha="center", va="bottom", fontsize=8)

ax.set_title(f"{ies} — CAC vs LTV (Acumulado por semestre) — Linha Roxa")
ax.set_xlabel("Semestres")
ax.set_ylabel("R$ acumulado")
plt.xticks(rotation=45)
st.pyplot(fig)

st.caption("💡 Suba o arquivo 'ies_cac_ltv.csv' no seu repositório GitHub e aponte "
           "'GITHUB_RAW_URL' nos *Secrets* do Streamlit para atualizar os dados.")
