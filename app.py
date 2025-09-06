
import io
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="CAC × LTV (Semestres)", layout="wide")

# ==============================
# Leitura de dados (flexível)
# ==============================
CSV_PATH = st.secrets.get("CSV_PATH", "ies_cac_ltv.csv")
GITHUB_RAW_URL = st.secrets.get("GITHUB_RAW_URL", "")  # ex: https://raw.githubusercontent.com/<user>/<repo>/main/ies_cac_ltv.csv

uploaded = st.file_uploader("Opcional: envie um CSV (colunas: IES, CAC, Proj_Receita_S1..S12)", type=["csv"])
raw_url_input = st.text_input("Ou cole uma URL raw do GitHub (opcional)", value=GITHUB_RAW_URL)

def _try_clean_currency(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        s = series.astype(str)
        # remove R$, pontos de milhar, espaços não quebráveis; troca vírgula por ponto
        s = s.str.replace("R$", "", regex=False).str.replace(".", "", regex=False).str.replace("\u00a0","", regex=False).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")

@st.cache_data
def load_data(uploaded_file, csv_path: str, raw_url: str) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif raw_url:
        df = pd.read_csv(raw_url)
    else:
        df = pd.read_csv(csv_path)
    # Normaliza numéricos (CAC e Proj_Receita_*)
    num_cols = [c for c in df.columns if c == "CAC" or c.startswith("Proj_Receita_")]
    for c in num_cols:
        df[c] = _try_clean_currency(df[c])
    return df

df = load_data(uploaded, CSV_PATH, raw_url_input)

if "IES" not in df.columns:
    st.error("CSV inválido: falta a coluna 'IES'.")
    st.stop()

receita_cols = sorted([c for c in df.columns if c.startswith("Proj_Receita_")],
                      key=lambda c: int(c.split('_S')[1]))

if not receita_cols:
    st.error("CSV inválido: não encontrei colunas 'Proj_Receita_Sn'.")
    st.stop()

# ==============================
# Funções de cálculo e plot
# ==============================
def payback_index(acc: np.ndarray):
    for i, v in enumerate(acc):
        if v >= 0:
            return i
    return None

def compute_curve(row: pd.Series):
    cac = float(row["CAC"])
    receitas = row[receita_cols].to_numpy(dtype=float)
    acc = np.concatenate(([cac], cac + np.cumsum(receitas)))
    sem_labels = ["S0"] + [f"S{i}" for i in range(1, len(receitas)+1)]
    pb_idx = payback_index(acc)
    return sem_labels, acc, pb_idx

def make_figure(sem_labels, acc, title, point_labels=True, line_color="purple"):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(sem_labels, acc, marker="o", linewidth=2, color=line_color)
    ax.axhline(0, linestyle="--", linewidth=1, color="gray")  # equilíbrio
    # CAC
    ax.annotate(f"CAC (S0): R$ {acc[0]:,.0f}",
                xy=(sem_labels[0], acc[0]),
                xytext=(0.8, acc[0]*0.6 if acc[0] != 0 else -1),
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8))
    # Payback
    pb_idx = payback_index(acc)
    if pb_idx is not None:
        ax.axvline(pb_idx, linestyle="--", linewidth=1, color="gray")
        ax.annotate(f"Payback: {sem_labels[pb_idx]}",
                    xy=(sem_labels[pb_idx], 0),
                    xytext=(pb_idx+0.2, max(acc)*0.08 if np.nanmax(acc) > 0 else 1),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8))
    # Rótulos
    if point_labels:
        for x, y in zip(sem_labels, acc):
            ax.text(x, y, f"{y:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Semestres (S0…S12)")
    ax.set_ylabel("R$ acumulado por aluno")
    plt.xticks(rotation=45)
    return fig

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ==============================
# UI
# ==============================
st.title("CAC × LTV — Fluxo acumulado por semestre")

mode = st.radio("Modo de visualização:", ["1 IES", "Todos (grade)", "Baixar pacote (19 gráficos)"], index=0, horizontal=True)
show_labels = st.checkbox("Exibir rótulos nos pontos", value=True)

if mode == "1 IES":
    ies = st.selectbox("Selecione a IES", df["IES"].tolist(), index=0)
    row = df[df["IES"] == ies].iloc[0]
    sem_labels, acc, _ = compute_curve(row)
    fig = make_figure(sem_labels, acc, f"{ies} — CAC vs LTV (Acumulado por semestre) — Linha Roxa", point_labels=show_labels)
    st.pyplot(fig)
    # download PNG
    png_bytes = fig_to_png_bytes(fig)
    st.download_button("Baixar PNG", data=png_bytes, file_name=f"{ies}.png", mime="image/png")

elif mode == "Todos (grade)":
    cols = st.slider("Colunas na grade", min_value=2, max_value=4, value=3)
    ies_list = df["IES"].tolist()
    rows = (len(ies_list) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        cols_container = st.columns(cols)
        for c in range(cols):
            if idx >= len(ies_list):
                break
            ies = ies_list[idx]
            row = df[df["IES"] == ies].iloc[0]
            sem_labels, acc, _ = compute_curve(row)
            fig = make_figure(sem_labels, acc, f"{ies} — CAC vs LTV", point_labels=False)
            with cols_container[c]:
                st.pyplot(fig, clear_figure=True)
            idx += 1

elif mode == "Baixar pacote (19 gráficos)":
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, row in df.iterrows():
            ies = row["IES"]
            sem_labels, acc, _ = compute_curve(row)
            fig = make_figure(sem_labels, acc, f"{ies} — CAC vs LTV", point_labels=show_labels)
            png = fig_to_png_bytes(fig)
            zf.writestr(f"{ies}.png", png)
    zip_buf.seek(0)
    st.download_button("Baixar ZIP com todos os gráficos", data=zip_buf, file_name="graficos_cac_ltv.zip", mime="application/zip")

st.caption("Dica: configure 'GITHUB_RAW_URL' nos *Secrets* do Streamlit para ler direto do GitHub, ou faça upload do CSV acima.")
