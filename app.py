# -*- coding: utf-8 -*-
"""
SPK Rekomendasi Laptop untuk Pemrograman — AHP + TOPSIS (Streamlit)

Dikonversi dari notebook Google Colab SPK_Laptop_AHP_TOPSIS.
Logika ekstraksi, AHP, dan TOPSIS sama dengan notebook.

Jalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import re
import io
import html
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rekomendasi Laptop Pemrograman",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {padding-top: 2.4rem; max-width: 1180px;}
h1 {font-weight: 700; letter-spacing: -0.01em;}
.rec-card {
    border: 1px solid rgba(128,128,128,.28);
    border-radius: 12px;
    padding: 18px 18px 14px 18px;
    background: var(--secondary-background-color, #f6f8fa);
    min-height: 300px;
}
.rec-card.top {border: 1.5px solid var(--primary-color, #0F62FE);}
.rec-tag {
    font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
    opacity: .7; margin-bottom: 6px;
}
.rec-model {
    font-size: 1.02rem; font-weight: 600; line-height: 1.35;
    min-height: 2.8em; margin-bottom: 6px;
}
.rec-price {font-size: 1.35rem; font-weight: 700; margin-bottom: 14px;}
.rec-specs {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px 12px; margin-bottom: 14px;
}
.rec-specs div {font-size: .92rem; font-weight: 500;}
.rec-specs span {
    display: block; font-size: .68rem; text-transform: uppercase;
    letter-spacing: .05em; opacity: .6; font-weight: 400;
}
.rec-score {
    border-top: 1px solid rgba(128,128,128,.25); padding-top: 10px;
    font-size: .85rem; display: flex; justify-content: space-between;
}
.section-note {opacity: .75; font-size: .92rem; margin-bottom: .6rem;}
</style>
""",
    unsafe_allow_html=True,
)

DATASET_HANDLE = "sumanbera19/laptop-price-dataset"
DATASET_URL = "https://www.kaggle.com/datasets/sumanbera19/laptop-price-dataset"

# ---------------------------------------------------------------------------
# Definisi kriteria (sesuai kesepakatan tim)
# ---------------------------------------------------------------------------
criteria = ["price_idr", "cpu_cores", "ram_gb", "storage_gb", "resolution_px"]
criteria_names = {
    "price_idr": "Harga",
    "cpu_cores": "Jumlah Core CPU",
    "ram_gb": "RAM",
    "storage_gb": "Storage",
    "resolution_px": "Resolusi Layar",
}
criteria_types = {
    "price_idr": "cost",
    "cpu_cores": "benefit",
    "ram_gb": "benefit",
    "storage_gb": "benefit",
    "resolution_px": "benefit",
}

# Matriks perbandingan berpasangan sementara
DEFAULT_PAIRWISE = np.array([
    [1,   2,   2,   3,   4],
    [1/2, 1,   1,   2,   3],
    [1/2, 1,   1,   2,   3],
    [1/3, 1/2, 1/2, 1,   2],
    [1/4, 1/3, 1/3, 1/2, 1],
], dtype=float)

RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ---------------------------------------------------------------------------
# Fungsi ekstraksi kriteria
# ---------------------------------------------------------------------------
def parse_price_inr(value):
    digits = re.sub(r"[^0-9]", "", str(value))
    return int(digits) if digits else np.nan


def parse_cpu_cores(value):
    text = str(value)
    numeric = re.search(r"\b(\d+)\s*Cores?\b", text, flags=re.I)
    if numeric:
        return int(numeric.group(1))

    words = {
        "single": 1, "dual": 2, "triple": 3, "quad": 4,
        "penta": 5, "hexa": 6, "octa": 8, "deca": 10,
    }
    named = re.search(
        r"\b(Single|Dual|Triple|Quad|Penta|Hexa|Octa|Deca)\s+Core\b",
        text, flags=re.I,
    )
    return words[named.group(1).lower()] if named else np.nan


def parse_ram_gb(value):
    text = str(value)
    # Harus ada penanda RAM/Memory/DDR agar storage/graphics tidak salah dibaca sebagai RAM.
    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*GB\b(?=.*(?:RAM|Memory|DDR|LPDDR))",
        text, flags=re.I,
    )
    return float(match.group(1)) if match else np.nan


def parse_storage_gb(value):
    text = str(value)
    if not re.search(r"\b(SSD|HDD|Hard Disk|eMMC|NVMe)\b", text, flags=re.I):
        return np.nan
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(TB|GB)\b", text, flags=re.I)
    if not match:
        return np.nan
    amount = float(match.group(1))
    return amount * 1024 if match.group(2).upper() == "TB" else amount


def parse_resolution_px(value):
    match = re.search(r"(\d{3,5})\s*x\s*(\d{3,5})", str(value), flags=re.I)
    return int(match.group(1)) * int(match.group(2)) if match else np.nan


# ---------------------------------------------------------------------------
# Pemuatan data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Mengunduh dataset dari Kaggle...")
def load_from_kaggle():
    import kagglehub

    # Token opsional (untuk dataset publik biasanya tidak diperlukan).
    try:
        token = st.secrets.get("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
    except Exception:
        pass

    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)

    candidates = [
        p for p in Path(dataset_path).rglob("*")
        if p.is_file() and p.name.lower() == "laptop.csv"
    ]
    if not candidates:
        available = [str(p) for p in Path(dataset_path).rglob("*") if p.is_file()]
        raise FileNotFoundError(
            "File laptop.csv tidak ditemukan di hasil unduhan. File yang tersedia:\n"
            + "\n".join(available)
        )

    raw_file = candidates[0]
    sha = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    return pd.read_csv(raw_file), sha, raw_file.name


@st.cache_data
def load_from_upload(file_bytes: bytes, name: str):
    sha = hashlib.sha256(file_bytes).hexdigest()
    return pd.read_csv(io.BytesIO(file_bytes)), sha, name


@st.cache_data
def extract_criteria(df_raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "model": df_raw["Model"].astype(str).str.strip(),
        "price_inr": df_raw["Price"].apply(parse_price_inr),
        "cpu_cores": df_raw["Core"].apply(parse_cpu_cores),
        "ram_gb": df_raw["Ram"].apply(parse_ram_gb),
        "storage_gb": df_raw["SSD"].apply(parse_storage_gb),
        "resolution_px": df_raw["Display"].apply(parse_resolution_px),
    })


# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------
def calculate_ahp(matrix: np.ndarray):
    n = matrix.shape[0]
    geometric_means = np.prod(matrix, axis=1) ** (1 / n)
    weights = geometric_means / geometric_means.sum()

    weighted_sum = matrix @ weights
    lambda_max = (weighted_sum / weights).mean()
    ci = (lambda_max - n) / (n - 1)
    ri = RI_TABLE[n]
    cr = ci / ri if ri else 0.0
    return weights, lambda_max, ci, ri, cr


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------
def calculate_topsis(data, criteria, weights, criteria_types):
    matrix = data[criteria].astype(float).to_numpy()

    denominator = np.sqrt((matrix ** 2).sum(axis=0))
    if np.any(denominator == 0):
        raise ValueError("Terdapat kolom kriteria dengan penyebut normalisasi nol.")
    normalized = matrix / denominator
    weighted = normalized * weights

    ideal_positive = np.zeros(len(criteria))
    ideal_negative = np.zeros(len(criteria))
    for j, criterion in enumerate(criteria):
        if criteria_types[criterion] == "benefit":
            ideal_positive[j] = weighted[:, j].max()
            ideal_negative[j] = weighted[:, j].min()
        else:
            ideal_positive[j] = weighted[:, j].min()
            ideal_negative[j] = weighted[:, j].max()

    distance_positive = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))

    denominator_pref = distance_positive + distance_negative
    preference = np.divide(
        distance_negative,
        denominator_pref,
        out=np.zeros_like(distance_negative),
        where=denominator_pref != 0,
    )

    result = data.copy()
    result["distance_positive"] = distance_positive
    result["distance_negative"] = distance_negative
    result["preference_score"] = preference
    result["rank"] = result["preference_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    return result.sort_values(
        ["preference_score", "model"], ascending=[False, True]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utilitas tampilan
# ---------------------------------------------------------------------------
def format_rupiah(value):
    return "Rp." + f"{int(round(value)):,}".replace(",", ".")


def format_storage(gb):
    if gb >= 1024:
        return f"{gb / 1024:g} TB"
    return f"{gb:g} GB"


def parse_budget(text):
    """Membaca anggaran dari teks bebas.

    Contoh yang diterima: 10000000, 10.000.000, Rp 10.000.000, 10jt,
    10 juta, 7,5 juta, 8500rb.
    Mengembalikan integer (IDR) atau None jika tidak dapat dibaca.
    """
    t = re.sub(r"\s+", "", str(text).lower())
    t = re.sub(r"^rp\.?", "", t)
    if not t:
        return None

    multiplier = 1
    for suffix, factor in (("juta", 1_000_000), ("jt", 1_000_000),
                           ("ribu", 1_000), ("rb", 1_000)):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            multiplier = factor
            break

    try:
        if multiplier != 1:
            value = float(t.replace(",", ".")) * multiplier
        else:
            digits = re.sub(r"[.,]", "", t)
            if not digits.isdigit():
                return None
            value = float(digits)
    except ValueError:
        return None

    return int(round(value)) if value > 0 else None


def recommendation_cards(rows):
    cards = []
    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        tag = "Rekomendasi utama" if i == 1 else f"Peringkat {i}"
        top_class = " top" if i == 1 else ""
        cards.append(
            f'<div class="rec-card{top_class}">'
            f'<div class="rec-tag">{tag}</div>'
            f'<div class="rec-model">{html.escape(str(r["model"]))}</div>'
            f'<div class="rec-price">{format_rupiah(r["price_idr"])}</div>'
            f'<div class="rec-specs">'
            f'<div><span>CPU</span>{int(r["cpu_cores"])} core</div>'
            f'<div><span>RAM</span>{r["ram_gb"]:g} GB</div>'
            f'<div><span>Storage</span>{format_storage(r["storage_gb"])}</div>'
            f'<div><span>Resolusi</span>{r["resolution_px"] / 1e6:.2f} MP</div>'
            f'</div>'
            f'<div class="rec-score"><span>Skor TOPSIS</span>'
            f'<strong>{r["preference_score"]:.4f}</strong></div>'
            f'</div>'
        )
    return cards


def ranking_table(data, top_n):
    view = data.head(top_n).copy()
    return pd.DataFrame({
        "No": np.arange(1, len(view) + 1),
        "Model": view["model"].values,
        "Harga": [format_rupiah(v) for v in view["price_idr"]],
        "Core CPU": view["cpu_cores"].astype(int).values,
        "RAM (GB)": view["ram_gb"].values,
        "Storage (GB)": view["storage_gb"].values,
        "Resolusi (MP)": (view["resolution_px"] / 1e6).round(2).values,
        "Skor TOPSIS": view["preference_score"].round(4).values,
        "Peringkat Keseluruhan": view["rank"].values,
    })


def show_table(table):
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "No": st.column_config.NumberColumn(width="small"),
            "RAM (GB)": st.column_config.NumberColumn(format="%d"),
            "Storage (GB)": st.column_config.NumberColumn(format="%d"),
            "Resolusi (MP)": st.column_config.NumberColumn(format="%.2f"),
            "Skor TOPSIS": st.column_config.ProgressColumn(
                "Skor TOPSIS", min_value=0.0, max_value=1.0, format="%.4f"
            ),
        },
    )


def score_chart(data, top_n):
    top = data.head(top_n).copy()
    top["harga"] = top["price_idr"].apply(format_rupiah)
    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("preference_score:Q", title="Skor TOPSIS"),
            y=alt.Y("model:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=340)),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("harga:N", title="Harga"),
                alt.Tooltip("preference_score:Q", title="Skor", format=".4f"),
            ],
        )
        .properties(height=max(120, 30 * len(top) + 40))
    )
    st.altair_chart(chart, use_container_width=True)


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Sidebar: pengaturan lanjutan (sumber data dan kurs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pengaturan Lanjutan")

    st.subheader("Sumber data")
    source_mode = st.radio(
        "Pilih sumber dataset",
        ["Unduh dari Kaggle", "Unggah laptop.csv"],
        label_visibility="collapsed",
    )
    uploaded = None
    if source_mode == "Unggah laptop.csv":
        uploaded = st.file_uploader("File laptop.csv", type=["csv"])

    st.subheader("Kurs INR ke IDR")
    INR_TO_IDR_RATE = st.number_input(
        "Kurs (1 INR = ... IDR)", min_value=0.0, value=196.1, step=0.1, format="%.4f"
    )
    EXCHANGE_RATE_SOURCE = st.text_input(
        "Sumber kurs",
        value="Wise - https://wise.com/in/currency-converter/inr-to-idr-rate/history",
    )
    EXCHANGE_RATE_DATE = st.text_input("Tanggal kurs", value="2026-11-24")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Rekomendasi Laptop untuk Pemrograman")
st.markdown(
    '<div class="section-note">Masukkan anggaran Anda. Sistem akan memilih dan '
    "mengurutkan laptop yang sesuai menggunakan metode AHP (pembobotan kriteria) "
    "dan TOPSIS (perangkingan).</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Muat data
# ---------------------------------------------------------------------------
if source_mode == "Unduh dari Kaggle":
    try:
        df_raw, source_sha256, source_name = load_from_kaggle()
    except Exception as exc:
        st.error(
            "Gagal mengunduh dataset dari Kaggle. Buka sidebar (panah di kiri atas) "
            "lalu pilih opsi Unggah laptop.csv."
        )
        st.exception(exc)
        st.stop()
else:
    if uploaded is None:
        st.info("Buka sidebar (panah di kiri atas) dan unggah file laptop.csv untuk memulai.")
        st.stop()
    df_raw, source_sha256, source_name = load_from_upload(uploaded.getvalue(), uploaded.name)

required_columns = ["Model", "Price", "Core", "Ram", "SSD", "Display"]
missing_columns = [c for c in required_columns if c not in df_raw.columns]
if missing_columns:
    st.error(f"Kolom wajib tidak ditemukan: {missing_columns}")
    st.stop()

if INR_TO_IDR_RATE <= 0:
    st.error("Kurs INR ke IDR harus berupa angka positif.")
    st.stop()
if not EXCHANGE_RATE_SOURCE.strip() or not EXCHANGE_RATE_DATE.strip():
    st.error("Sumber dan tanggal kurs wajib diisi.")
    st.stop()

# ---------------------------------------------------------------------------
# Ekstraksi, pembersihan, konversi
# ---------------------------------------------------------------------------
df_extracted = extract_criteria(df_raw)

criteria_columns = ["price_inr", "cpu_cores", "ram_gb", "storage_gb", "resolution_px"]
positive_mask = (df_extracted[criteria_columns] > 0).all(axis=1)
complete_mask = df_extracted[criteria_columns].notna().all(axis=1)
valid_mask = positive_mask & complete_mask

df_excluded = df_extracted.loc[~valid_mask].copy()
df_clean_inr = df_extracted.loc[valid_mask].copy().reset_index(drop=True)

if len(df_clean_inr) < 45:
    st.error(
        f"Alternatif valid hanya {len(df_clean_inr)} (kurang dari 45). "
        "Periksa aturan ekstraksi sebelum melanjutkan."
    )
    st.stop()

df_clean = df_clean_inr.copy()
df_clean["price_idr"] = (df_clean["price_inr"] * INR_TO_IDR_RATE).round().astype("int64")

# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------
tab_rec, tab_rank, tab_ahp, tab_data, tab_valid, tab_export = st.tabs([
    "Rekomendasi", "Seluruh Ranking", "Bobot AHP",
    "Data dan Audit", "Validasi", "Unduh",
])

# ======================= TAB AHP (dihitung lebih dulu karena bobot dipakai tab lain) =======================
labels = [criteria_names[c] for c in criteria]
n_crit = len(criteria)

with tab_ahp:
    st.subheader("Matriks perbandingan berpasangan")
    st.markdown(
        '<div class="section-note">Matriks ini bersifat sementara dan perlu disetujui tim. '
        "Isi hanya sel di atas diagonal (skala 1/9 sampai 9). Sel di bawah diagonal "
        "dihitung otomatis sebagai kebalikannya.</div>",
        unsafe_allow_html=True,
    )

    upper = np.full((n_crit, n_crit), np.nan)
    for i in range(n_crit):
        for j in range(i + 1, n_crit):
            upper[i, j] = DEFAULT_PAIRWISE[i, j]

    edited = st.data_editor(
        pd.DataFrame(upper, index=labels, columns=labels),
        column_config={
            label: st.column_config.NumberColumn(
                label, min_value=1 / 9, max_value=9.0, format="%.3f"
            )
            for label in labels
        },
        use_container_width=True,
        key="pairwise_editor",
    )

    edited_values = edited.to_numpy(dtype=float)
    pairwise_matrix = np.ones((n_crit, n_crit))
    valid_matrix = True
    for i in range(n_crit):
        for j in range(i + 1, n_crit):
            v = edited_values[i, j]
            if np.isnan(v) or v <= 0:
                valid_matrix = False
            else:
                pairwise_matrix[i, j] = v
                pairwise_matrix[j, i] = 1.0 / v

    if not valid_matrix:
        st.error(
            "Semua sel di atas diagonal harus berisi angka positif. "
            "Sementara ini perhitungan memakai matriks bawaan."
        )
        pairwise_matrix = DEFAULT_PAIRWISE.copy()

    st.markdown("**Matriks lengkap (setelah kebalikan dihitung)**")
    st.dataframe(
        pd.DataFrame(pairwise_matrix, index=labels, columns=labels).style.format("{:.4f}"),
        use_container_width=True,
    )

    weights, lambda_max, CI, RI, CR = calculate_ahp(pairwise_matrix)

    weight_table = pd.DataFrame({
        "Kriteria": labels,
        "Jenis": [criteria_types[c].title() for c in criteria],
        "Bobot": weights,
    })

    st.subheader("Bobot kriteria")
    w1, w2 = st.columns(2)
    with w1:
        st.dataframe(
            weight_table.style.format({"Bobot": "{:.4f}"}),
            hide_index=True,
            use_container_width=True,
        )
    with w2:
        weight_chart = (
            alt.Chart(weight_table)
            .mark_bar()
            .encode(
                x=alt.X("Bobot:Q", title="Bobot AHP"),
                y=alt.Y("Kriteria:N", sort="-x", title=None),
                tooltip=["Kriteria", alt.Tooltip("Bobot:Q", format=".4f")],
            )
            .properties(height=220)
        )
        st.altair_chart(weight_chart, use_container_width=True)

    st.subheader("Uji konsistensi")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lambda max", f"{lambda_max:.6f}")
    k2.metric("CI", f"{CI:.6f}")
    k3.metric("RI", f"{RI:.2f}")
    k4.metric("CR", f"{CR:.6f}")
    if CR < 0.10:
        st.success("Konsisten (CR < 0,10)")
    else:
        st.error("Belum konsisten (CR >= 0,10). Perbaiki penilaian matriks.")

# ======================= TOPSIS =======================
ranking = calculate_topsis(df_clean, criteria, weights, criteria_types)

# ======================= TAB REKOMENDASI =======================
with tab_rec:
    st.subheader("Cari laptop sesuai anggaran")

    with st.form("budget_form"):
        f1, f2 = st.columns([3, 1])
        budget_text = f1.text_input(
            "Anggaran maksimal (Rupiah)",
            value="10.000.000",
            placeholder="Contoh: 10.000.000 atau 10 juta",
            help="Anda dapat menulis 10000000, 10.000.000, Rp 10.000.000, 10jt, "
                 "10 juta, atau 7,5 juta.",
        )
        top_n = f2.selectbox("Jumlah rekomendasi", [5, 10, 15, 20, 30], index=1)
        submitted = st.form_submit_button("Tampilkan Rekomendasi", type="primary")

    budget = parse_budget(budget_text)

    if budget is None:
        st.error(
            "Anggaran tidak dapat dibaca. Gunakan angka, misalnya 10.000.000, "
            "10jt, atau 7,5 juta."
        )
    else:
        budget_result = ranking[ranking["price_idr"] <= budget].copy().reset_index(drop=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Anggaran", format_rupiah(budget))
        m2.metric("Laptop sesuai anggaran", f"{len(budget_result)} dari {len(ranking)}")
        if budget_result.empty:
            m3.metric("Skor tertinggi", "-")
        else:
            m3.metric("Skor tertinggi", f"{budget_result.loc[0, 'preference_score']:.4f}")

        if budget_result.empty:
            cheapest = int(ranking["price_idr"].min())
            st.warning(
                "Tidak ada laptop yang memenuhi anggaran ini. "
                f"Harga laptop termurah pada data adalah {format_rupiah(cheapest)}."
            )
        else:
            st.markdown("### Rekomendasi terbaik")
            cards = recommendation_cards(budget_result.head(3))
            card_cols = st.columns(3)
            for col, card in zip(card_cols, cards):
                with col:
                    st.markdown(card, unsafe_allow_html=True)

            shown = min(top_n, len(budget_result))
            st.markdown(f"### {shown} laptop teratas dalam anggaran")
            table = ranking_table(budget_result, shown)
            show_table(table)

            st.markdown("### Perbandingan skor")
            score_chart(budget_result, shown)

            st.download_button(
                "Unduh hasil sesuai anggaran (CSV)",
                data=to_csv_bytes(budget_result),
                file_name="hasil_rekomendasi_sesuai_anggaran.csv",
                mime="text/csv",
            )

        st.markdown(
            '<div class="section-note">Skor TOPSIS menunjukkan kedekatan relatif '
            "terhadap solusi ideal berdasarkan kriteria dan bobot yang digunakan. "
            "Skor bukan persentase kualitas laptop dan bukan jaminan bahwa laptop "
            "tersebut paling cocok untuk semua kebutuhan pemrograman. Skor dihitung "
            "dari seluruh data, lalu difilter berdasarkan anggaran.</div>",
            unsafe_allow_html=True,
        )

# ======================= TAB SELURUH RANKING =======================
with tab_rank:
    st.subheader("Peringkat TOPSIS seluruh alternatif")
    st.caption(f"Jumlah alternatif yang dirangking: {len(ranking)}")

    rank_top_n = st.slider("Jumlah laptop yang ditampilkan", 5, len(ranking), min(20, len(ranking)))
    show_table(ranking_table(ranking, rank_top_n))
    score_chart(ranking, min(rank_top_n, 30))

# ======================= TAB DATA =======================
with tab_data:
    with st.expander("Catatan akademik"):
        st.warning(
            "Instruksi Final Project menyebutkan alternatif minimal 45 data dummy. "
            "Dataset ini adalah data sekunder Kaggle, bukan data dummy. Penggunaannya "
            "perlu dikonfirmasi kepada dosen. Jangan menyebut dataset ini sebagai "
            "data dummy sebelum ada persetujuan atau arahan dosen."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data awal", len(df_raw))
    c2.metric("Data valid", len(df_clean))
    c3.metric("Dikeluarkan", len(df_excluded))
    c4.metric("Kolom sumber", df_raw.shape[1])

    st.caption(
        f"Sumber: [{DATASET_HANDLE}]({DATASET_URL}) | File: `{source_name}` | "
        f"SHA-256: `{source_sha256}`"
    )

    st.subheader("Cuplikan data mentah")
    st.dataframe(df_raw.head(), use_container_width=True)

    st.subheader("Audit awal")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Nilai kosong per kolom (data mentah)**")
        st.dataframe(
            df_raw.isna().sum().to_frame("jumlah_kosong"), use_container_width=True
        )
    with a2:
        st.markdown("**Nilai kosong setelah ekstraksi**")
        st.dataframe(
            df_extracted.isna().sum().to_frame("jumlah_kosong"), use_container_width=True
        )
    st.write("Jumlah baris duplikat identik:", int(df_raw.duplicated().sum()))

    st.subheader("Data bersih dan konversi IDR")
    st.caption(
        f"Kurs {INR_TO_IDR_RATE} | {EXCHANGE_RATE_SOURCE} | {EXCHANGE_RATE_DATE}"
    )
    st.dataframe(df_clean.head(20), use_container_width=True)

    if not df_excluded.empty:
        with st.expander(f"Baris yang tidak lolos ({len(df_excluded)})"):
            st.dataframe(df_excluded, use_container_width=True)

# ======================= TAB VALIDASI =======================
with tab_valid:
    st.subheader("Validasi hasil perhitungan")
    checks = [
        ("Jumlah bobot AHP sama dengan 1", bool(np.isclose(weights.sum(), 1.0))),
        ("Skor TOPSIS berada pada rentang 0 sampai 1",
         bool(ranking["preference_score"].between(0, 1).all())),
        ("Peringkat pertama ditemukan", bool(ranking["rank"].min() == 1)),
        ("Jumlah alternatif minimal 45", bool(len(ranking) >= 45)),
        ("Tidak ada nama model kosong", bool(ranking["model"].notna().all())),
    ]
    st.dataframe(
        pd.DataFrame({
            "Pemeriksaan": [c[0] for c in checks],
            "Hasil": ["Lolos" if c[1] else "Gagal" for c in checks],
        }),
        hide_index=True,
        use_container_width=True,
    )

    v1, v2, v3 = st.columns(3)
    v1.metric("Jumlah alternatif", len(ranking))
    v2.metric("Jumlah bobot", round(float(weights.sum()), 6))
    v3.metric("CR AHP", round(float(CR), 6))

# ======================= TAB UNDUH =======================
with tab_export:
    st.subheader("Unduh hasil")

    metadata = pd.DataFrame([{
        "dataset_source": DATASET_URL,
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "exchange_rate_inr_to_idr": INR_TO_IDR_RATE,
        "exchange_rate_source": EXCHANGE_RATE_SOURCE,
        "exchange_rate_date": EXCHANGE_RATE_DATE,
        "initial_rows": len(df_raw),
        "valid_rows": len(df_clean),
        "excluded_rows": len(df_excluded),
        "ahp_cr": CR,
    }])

    downloads = [
        ("hasil_ranking_topsis.csv", ranking),
        ("bobot_ahp.csv", weight_table),
        ("laptop_clean_idr.csv", df_clean),
        ("laptop_excluded_for_review.csv", df_excluded),
        ("laptop_extracted_all.csv", df_extracted),
        ("metadata_proses.csv", metadata),
    ]
    cols = st.columns(3)
    for idx, (fname, frame) in enumerate(downloads):
        with cols[idx % 3]:
            st.download_button(
                fname,
                data=to_csv_bytes(frame),
                file_name=fname,
                mime="text/csv",
                use_container_width=True,
            )
