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

# ---------------------------------------------------------------------------
# Tema terang / gelap
# ---------------------------------------------------------------------------
PALETTES = {
    "light": {
        "bg": "#F7F9FE", "glow1": "rgba(139,92,246,.10)", "glow2": "rgba(15,98,254,.10)",
        "surface": "#FFFFFF", "surface2": "#F4F7FF", "input_bg": "#F4F7FF",
        "border": "#E1E7F3", "text": "#1B1F27", "muted": "#5B6475", "faint": "#7A8396",
        "accent": "#0F62FE", "accent2": "#8B5CF6", "soft": "#EEF3FF",
        "hover_border": "#B9D0FF", "shadow": "rgba(20,40,90,.45)",
        "shadow_hover": "rgba(15,98,254,.55)", "track": "#E7ECF8",
        "rank": "rgba(15,98,254,.07)", "grid_filter": "none",
        "scheme": "light",
        # warna grafik
        "chart_label": "#44506A", "chart_grid": "#EDF1FA", "chart_domain": "#D5DCEB",
        "chart_range": ["#93C5FD", "#0F62FE"], "chart_solid": "#0F62FE",
    },
    "dark": {
        "bg": "#0B1020", "glow1": "rgba(139,92,246,.20)", "glow2": "rgba(59,130,246,.16)",
        "surface": "#141B2E", "surface2": "#1A2440", "input_bg": "#1A2440",
        "border": "#27325A", "text": "#E8ECF6", "muted": "#A5B1CB", "faint": "#7F8CA8",
        "accent": "#5B9BFF", "accent2": "#A78BFA", "soft": "rgba(91,155,255,.16)",
        "hover_border": "#3C5BA8", "shadow": "rgba(0,0,0,.70)",
        "shadow_hover": "rgba(91,155,255,.35)", "track": "#222C4A",
        "rank": "rgba(91,155,255,.10)", "grid_filter": "invert(.92) hue-rotate(180deg)",
        "scheme": "dark",
        "chart_label": "#A5B1CB", "chart_grid": "#222C4A", "chart_domain": "#33406A",
        "chart_range": ["#3B5FA8", "#7FB0FF"], "chart_solid": "#5B9BFF",
    },
}


def _initial_theme():
    """Tema awal: ikut parameter ?tema=gelap pada alamat, selain itu terang."""
    try:
        value = st.query_params.get("tema", "")
        if isinstance(value, list):
            value = value[0] if value else ""
        return "dark" if str(value).lower() in ("gelap", "dark") else "light"
    except Exception:
        return "light"


if "theme_dark" not in st.session_state:
    st.session_state["theme_dark"] = _initial_theme() == "dark"


def on_theme_change():
    """Menyimpan pilihan tema pada alamat halaman agar bertahan saat dimuat ulang."""
    try:
        st.query_params["tema"] = "gelap" if st.session_state["theme_dark"] else "terang"
    except Exception:
        pass


IS_DARK = bool(st.session_state["theme_dark"])
P = PALETTES["dark" if IS_DARK else "light"]


def _is_mobile():
    """Deteksi ponsel dari User-Agent, dipakai untuk menyesuaikan grafik."""
    try:
        agent = st.context.headers.get("User-Agent", "") or ""
        return bool(re.search(r"Mobi|iPhone|iPod", agent, re.I))
    except Exception:
        return False


IS_MOBILE = _is_mobile()

# ---------------------------------------------------------------------------
# Gaya visual dan animasi
# ---------------------------------------------------------------------------
THEME_VARS = f"""
<style>
:root {{
    --bg: {P['bg']}; --glow1: {P['glow1']}; --glow2: {P['glow2']};
    --surface: {P['surface']}; --surface2: {P['surface2']}; --input-bg: {P['input_bg']};
    --border: {P['border']}; --text: {P['text']}; --muted: {P['muted']}; --faint: {P['faint']};
    --accent: {P['accent']}; --accent2: {P['accent2']}; --soft: {P['soft']};
    --hover-border: {P['hover_border']}; --shadow: {P['shadow']};
    --shadow-hover: {P['shadow_hover']}; --track: {P['track']}; --rank: {P['rank']};
    --grid-filter: {P['grid_filter']};
    color-scheme: {P['scheme']};
}}
</style>
"""

STATIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ---------- Font ---------- */
html, body, .stApp, .stApp div, .stApp p, .stApp label, .stApp li, .stApp button,
.stApp input, .stApp textarea, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp span:not([data-testid="stIconMaterial"]) {
    font-family: 'Plus Jakarta Sans', 'Source Sans', sans-serif;
}

/* ---------- Latar dan warna teks ---------- */
.stApp {
    background:
        radial-gradient(900px 380px at 100% -5%, var(--glow1), transparent 60%),
        radial-gradient(800px 360px at -5% 0%, var(--glow2), transparent 60%),
        var(--bg);
    color: var(--text);
    transition: background-color .4s ease, color .4s ease;
}
.block-container {padding-top: 4.4rem; max-width: 1200px;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {color: var(--text);}
.stApp h2, .stApp h3 {font-weight: 800;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: var(--muted) !important;}
.stApp a {color: var(--accent);}
.stApp code {background: var(--surface2) !important; color: var(--accent) !important; border-radius: 6px;}
section[data-testid="stSidebar"] {
    background: var(--surface2) !important; border-right: 1px solid var(--border);
}
[data-testid="stExpandSidebarButton"] span, [data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpandSidebarButton"] svg, [data-testid="stSidebarCollapseButton"] svg {
    color: var(--muted) !important;
}

/* ---------- Animasi ---------- */
@keyframes fadeUp {
    from {opacity: 0; transform: translateY(18px);}
    to   {opacity: 1; transform: none;}
}
@keyframes gradientShift {
    0%   {background-position: 0% 50%;}
    50%  {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
@keyframes floaty {
    0%, 100% {transform: translate(0, 0);}
    50%      {transform: translate(-18px, 22px);}
}
@keyframes grow {
    from {width: 0;}
    to   {width: var(--w);}
}
@keyframes pulseGlow {
    0%, 100% {box-shadow: 0 0 0 0 rgba(91,155,255,.38);}
    50%      {box-shadow: 0 0 0 8px rgba(91,155,255,0);}
}

/* ---------- Bilah atas dan tombol tema ---------- */
.brand {
    display: flex; align-items: center; gap: 10px; padding: 10px 0;
    font-size: .95rem; font-weight: 700; letter-spacing: .02em; color: var(--text);
}
.brand-dot {
    width: 12px; height: 12px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    animation: pulseGlow 2.6s ease-in-out infinite;
}
/* Saklar tema Light / Dark: bentuk oval, bulatan putih bergeser, label berubah sesuai mode */
.st-key-theme_dark {display: flex; justify-content: flex-end;}
.st-key-theme_dark label {
    padding: 0 !important; background: none !important; border: none !important;
    box-shadow: none !important; min-height: 0 !important; cursor: pointer;
}
.st-key-theme_dark label [data-testid="stWidgetLabel"] {display: none !important;}
.st-key-theme_dark label > div:first-of-type {
    position: relative !important; box-sizing: border-box !important;
    width: 164px !important; height: 44px !important; margin: 0 !important; padding: 0 !important;
    border-radius: 999px !important; overflow: hidden;
    background: linear-gradient(90deg, #F0658F, #FFB04D) !important;
    box-shadow: 0 12px 22px -12px rgba(240,101,143,.8) !important;
    transition: background .45s ease, box-shadow .45s ease, transform .2s ease !important;
}
.st-key-theme_dark label[data-selected="true"] > div:first-of-type {
    background: linear-gradient(90deg, #1B2F75, #2B4CB8) !important;
    box-shadow: 0 12px 22px -12px rgba(43,76,184,.85) !important;
}
.st-key-theme_dark label:hover > div:first-of-type {transform: translateY(-2px) !important;}
.st-key-theme_dark label:has(input:focus-visible) > div:first-of-type {
    outline: 2px solid var(--accent); outline-offset: 3px;
}
/* Tulisan mode lawan pada latar saklar (bulan saat terang, matahari saat gelap) */
.st-key-theme_dark label > div:first-of-type::before {
    content: "Dark"; position: absolute; top: 50%; right: 20px; transform: translateY(-50%);
    padding-left: 21px; line-height: 16px; pointer-events: none;
    font-size: .74rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #fff;
    background: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/%3E%3C/svg%3E") no-repeat left center / 15px 15px;
}
.st-key-theme_dark label[data-selected="true"] > div:first-of-type::before {
    content: "Light"; right: auto; left: 15px;
    background: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='4' fill='white'/%3E%3Cpath d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4'/%3E%3C/svg%3E") no-repeat left center / 15px 15px;
}
/* Bulatan putih bertuliskan mode aktif */
.st-key-theme_dark label > div:first-of-type > div {
    position: absolute !important; top: 4px !important; left: 4px !important; right: auto !important;
    width: 72px !important; height: 36px !important; margin: 0 !important;
    border-radius: 999px !important; background: #fff !important; transform: none !important;
    box-shadow: 0 6px 14px -6px rgba(0,0,0,.45) !important;
    display: flex !important; align-items: center; justify-content: center;
    transition: left .4s cubic-bezier(.3,.9,.3,1) !important;
}
.st-key-theme_dark label[data-selected="true"] > div:first-of-type > div {left: calc(100% - 76px) !important;}
.st-key-theme_dark label > div:first-of-type > div::after {
    content: "Light"; font-size: .76rem; font-weight: 800; letter-spacing: .1em;
    text-transform: uppercase; color: #E5487E;
}
.st-key-theme_dark label[data-selected="true"] > div:first-of-type > div::after {
    content: "Dark"; color: #1B2F75;
}

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden;
    border-radius: 28px; padding: 40px 44px; margin-bottom: 26px;
    color: #fff;
    background: linear-gradient(120deg, #0B1F5C, #0F62FE, #6C3BFF, #0F62FE);
    background-size: 300% 300%;
    animation: fadeUp .7s ease both, gradientShift 14s ease infinite;
    box-shadow: 0 22px 44px -22px var(--shadow-hover);
}
.hero::before, .hero::after {
    content: ""; position: absolute; border-radius: 50%;
    filter: blur(42px); opacity: .5; animation: floaty 9s ease-in-out infinite;
}
.hero::before {width: 280px; height: 280px; background: #7DD3FC; top: -100px; right: -60px;}
.hero::after  {width: 230px; height: 230px; background: #C084FC; bottom: -110px; left: 35%; animation-delay: -4s;}
.hero > * {position: relative; z-index: 1;}
.hero .hero-eyebrow {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3); color: #fff;
}
.hero .hero-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.15;
    margin: 16px 0 12px; max-width: 760px; color: #fff;
}
.hero .hero-sub {max-width: 640px; font-size: 1.02rem; line-height: 1.65; opacity: .92; color: #fff;}
.hero .hero-chips {display: flex; gap: 10px; flex-wrap: wrap; margin-top: 22px;}
.hero .chip {
    padding: 7px 15px; border-radius: 999px; font-size: .82rem; font-weight: 600; color: #fff;
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.3);
    transition: transform .2s ease, background .2s ease;
}
.hero .chip:hover {transform: translateY(-2px); background: rgba(255,255,255,.26);}

/* ---------- Judul seksi ---------- */
.sec-title {
    display: flex; align-items: center; gap: 12px;
    font-size: 1.2rem; font-weight: 800; color: var(--text);
    margin: 28px 0 14px; animation: fadeUp .5s ease both;
}
.sec-title::before {
    content: ""; width: 5px; height: 24px; border-radius: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
}
.section-note {color: var(--muted); font-size: .9rem; line-height: 1.6; margin: 8px 0 4px;}

/* ---------- Tab oval ---------- */
[data-testid="stTabs"] [role="tablist"] {
    gap: 10px !important; border-bottom: none !important; box-shadow: none !important;
    padding: 6px 4px 12px 4px !important; flex-wrap: wrap;
}
[data-testid="stTabs"] [role="tablist"]::after,
[data-testid="stTabs"] [role="tablist"]::before {display: none !important;}
[data-testid="stTab"] {
    height: auto !important; margin: 0 !important;
    padding: 10px 24px !important; border-radius: 999px !important;
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease !important;
}
[data-testid="stTab"] p {color: var(--muted) !important; font-weight: 600; font-size: .9rem;}
[data-testid="stTab"]:hover {
    transform: translateY(-2px); border-color: var(--accent) !important;
    box-shadow: 0 10px 18px -12px var(--shadow-hover);
}
[data-testid="stTab"][aria-selected="true"] {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-color: transparent !important;
    box-shadow: 0 12px 22px -10px var(--shadow-hover);
}
[data-testid="stTab"][aria-selected="true"] p {color: #fff !important;}
.react-aria-SelectionIndicator {display: none !important;}
[data-testid="stTabs"] [role="tabpanel"] {animation: fadeUp .45s ease both; padding-top: .6rem;}

/* ---------- Form anggaran ---------- */
div[data-testid="stForm"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 24px;
    padding: 24px 24px 14px 24px;
    box-shadow: 0 14px 30px -20px var(--shadow);
    animation: fadeUp .6s ease both;
}
/* Kotak input teks, angka, dan pilihan: dukung struktur lama (baseweb) dan baru (react-aria) */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div,
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] [role="group"], [data-testid="stTextAreaRootElement"] {
    background: var(--input-bg) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 14px !important;
    transition: border-color .2s ease, box-shadow .2s ease, background .3s ease;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="base-input"]:focus-within,
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stSelectbox"] [role="group"]:focus-within, [data-testid="stTextAreaRootElement"]:focus-within {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--soft);
}
.stApp input, .stApp textarea {
    color: var(--text) !important; -webkit-text-fill-color: var(--text);
    background: transparent !important;
}
.stApp input::placeholder, .stApp textarea::placeholder {
    color: var(--faint) !important; -webkit-text-fill-color: var(--faint);
}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"],
[data-testid="stNumberInput"] button {
    background: var(--surface) !important; color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-testid="stNumberInputStepDown"] svg, [data-testid="stNumberInputStepUp"] svg {
    color: var(--text) !important; fill: var(--text) !important;
}
[data-testid="stSelectbox"] svg, div[data-baseweb="select"] svg {
    color: var(--muted) !important; fill: var(--muted) !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div {color: var(--text);}
/* Daftar pilihan (dropdown) */
div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul, div[data-baseweb="menu"],
[data-testid="stSelectboxVirtualDropdown"], [role="listbox"] {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border); border-radius: 14px !important;
}
[role="option"], [role="option"] div, li[role="option"] {color: var(--text) !important;}
[role="option"] {border-radius: 10px !important;}
[role="option"]:hover, [role="option"][aria-selected="true"], [role="option"][data-focused="true"] {
    background: var(--soft) !important;
}
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent)) !important;
    background-size: 200% 100% !important;
    color: #fff !important; border: none !important; border-radius: 999px !important;
    padding: .6rem 1.8rem !important; font-weight: 700 !important;
    transition: transform .2s ease, box-shadow .2s ease, background-position .4s ease !important;
    animation: pulseGlow 2.6s ease-in-out infinite;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px);
    background-position: 100% 0 !important;
    box-shadow: 0 14px 24px -10px var(--shadow-hover);
}
div[data-testid="stFormSubmitButton"] button p {color: #fff !important; font-weight: 700;}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {color: var(--text) !important;}
[data-testid="stFileUploaderDropzone"] {
    background: var(--input-bg) !important; border: 1px dashed var(--border) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small {color: var(--muted) !important;}
[data-testid="stFileUploaderDropzone"] button {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 999px !important;
}

/* ---------- Kartu metrik ---------- */
div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 20px;
    padding: 16px 20px; animation: fadeUp .55s ease both;
    box-shadow: 0 10px 22px -16px var(--shadow);
    transition: transform .25s ease, box-shadow .25s ease, background .4s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 28px -16px var(--shadow-hover);
}
[data-testid="stMetricLabel"] p {
    font-size: .74rem !important; letter-spacing: .06em; text-transform: uppercase;
    color: var(--faint) !important; font-weight: 700 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
    font-weight: 800 !important; font-size: clamp(1.3rem, 2.1vw, 2rem) !important;
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}

/* ---------- Kartu rekomendasi ---------- */
.rec-card {
    position: relative; overflow: hidden; min-height: 380px;
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border); border-radius: 26px;
    padding: 24px 22px 20px 22px;
    box-shadow: 0 12px 26px -18px var(--shadow);
    animation: fadeUp .65s ease both;
    transition: transform .28s ease, box-shadow .28s ease, border-color .28s ease;
}
.rec-card:hover {
    transform: translateY(-8px);
    border-color: var(--hover-border);
    box-shadow: 0 28px 40px -22px var(--shadow-hover);
}
.rec-card.top {
    border: 1.6px solid transparent;
    background:
        linear-gradient(var(--surface), var(--surface)) padding-box,
        linear-gradient(135deg, var(--accent), var(--accent2)) border-box;
}
.rec-rank {
    position: absolute; top: -8px; right: 16px;
    font-size: 5.6rem; font-weight: 800; line-height: 1;
    color: var(--rank); pointer-events: none;
}
.rec-tag {
    display: inline-block; margin-bottom: 14px; padding: 5px 12px; border-radius: 999px;
    font-size: .66rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
    background: var(--soft); color: var(--accent);
}
.rec-card.top .rec-tag {background: linear-gradient(90deg, var(--accent), var(--accent2)); color: #fff;}
.rec-model {
    font-size: 1rem; font-weight: 700; line-height: 1.4;
    min-height: 2.8em; margin-bottom: 10px; color: var(--text);
}
.rec-price {
    font-size: 1.6rem; font-weight: 800; margin-bottom: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
    width: fit-content;
}
.rec-save {font-size: .78rem; color: var(--muted); margin-bottom: 18px;}
.rec-specs {display: grid; grid-template-columns: 1fr 1fr; gap: 10px;}
.spec {
    display: flex; align-items: center; gap: 10px;
    background: var(--surface2); border-radius: 16px; padding: 10px 12px;
    transition: background .2s ease, transform .2s ease;
}
.spec:hover {background: var(--soft); transform: scale(1.03);}
.spec svg {width: 20px; height: 20px; flex: none; color: var(--accent);}
.spec small {
    display: block; font-size: .6rem; letter-spacing: .07em;
    text-transform: uppercase; color: var(--faint); font-weight: 600;
}
.spec b {font-size: .88rem; color: var(--text); font-weight: 700;}
.score-row {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: .8rem; color: var(--muted); margin: 18px 0 7px;
}
.score-row b {color: var(--text); font-size: .95rem; font-weight: 800;}
.bar {height: 9px; background: var(--track); border-radius: 999px; overflow: hidden;}
.bar > i {
    display: block; height: 100%; width: var(--w); border-radius: 999px;
    background: linear-gradient(90deg, #60A5FA, var(--accent), var(--accent2));
    animation: grow 1.2s cubic-bezier(.2,.8,.2,1) both; animation-delay: .35s;
}

/* ---------- Pemberitahuan ---------- */
.notice {
    display: flex; gap: 14px; align-items: stretch;
    padding: 14px 18px; margin: 10px 0; border-radius: 18px;
    background: var(--n-bg); border: 1px solid var(--n-border);
    color: var(--text); line-height: 1.55; font-size: .95rem;
    animation: fadeUp .5s ease both;
}
.notice::before {
    content: ""; flex: none; width: 5px; border-radius: 4px; background: var(--n-color);
}
.notice div {color: var(--text);}
.notice.success {--n-color: #16A34A; --n-bg: rgba(22,163,74,.10); --n-border: rgba(22,163,74,.32);}
.notice.error   {--n-color: #DC2626; --n-bg: rgba(220,38,38,.10); --n-border: rgba(220,38,38,.32);}
.notice.warning {--n-color: #D97706; --n-bg: rgba(217,119,6,.11); --n-border: rgba(217,119,6,.34);}
.notice.info    {--n-color: var(--accent); --n-bg: var(--soft); --n-border: var(--hover-border);}

/* ---------- Keadaan kosong ---------- */
.empty {
    text-align: center; padding: 44px 24px; margin-top: 12px;
    border: 1.6px dashed var(--hover-border); border-radius: 24px; background: var(--surface);
    color: var(--muted); line-height: 1.6; animation: fadeUp .5s ease both;
}
.empty b {display: block; font-size: 1.15rem; margin-bottom: 6px; color: var(--text);}

/* ---------- Komponen lain ---------- */
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
    border: 1px solid var(--border); border-radius: 20px; overflow: hidden;
    box-shadow: 0 10px 22px -18px var(--shadow);
    filter: var(--grid-filter);
}
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important; border-radius: 20px !important;
    background: var(--surface) !important;
}
div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary {background: transparent !important;}
div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary span {color: var(--text) !important;}
.stDownloadButton button {
    border-radius: 999px; border: 1px solid var(--border); background: var(--surface);
    font-weight: 600;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.stDownloadButton button p {color: var(--text) !important;}
.stDownloadButton button:hover {
    transform: translateY(-2px); border-color: var(--accent);
    box-shadow: 0 12px 20px -12px var(--shadow-hover);
}
.footer-note {
    margin-top: 34px; padding-top: 16px; border-top: 1px solid var(--border);
    font-size: .8rem; color: var(--faint); text-align: center;
}

/* ---------- Responsif: tablet ---------- */
@media (max-width: 900px) {
    .block-container {padding-left: 1.4rem !important; padding-right: 1.4rem !important;}
    .hero {padding: 32px 30px;}
    .hero .hero-title {font-size: 2rem;}
}

/* ---------- Responsif: ponsel ---------- */
@media (max-width: 640px) {
    .block-container {padding: 4.2rem .9rem 2.5rem .9rem !important;}

    /* bilah atas tetap satu baris: nama di kiri, saklar di kanan */
    .st-key-topbar [data-testid="stHorizontalBlock"] {
        flex-direction: row !important; flex-wrap: nowrap !important;
        align-items: center !important; gap: .5rem !important;
    }
    .st-key-topbar [data-testid="stColumn"], .st-key-topbar [data-testid="column"] {
        min-width: 0 !important; width: auto !important;
    }
    .st-key-topbar [data-testid="stColumn"]:first-child,
    .st-key-topbar [data-testid="column"]:first-child {flex: 1 1 auto !important;}
    .st-key-topbar [data-testid="stColumn"]:last-child,
    .st-key-topbar [data-testid="column"]:last-child {flex: 0 0 auto !important;}
    .brand {font-size: .85rem; padding: 6px 0;}

    .hero {padding: 26px 22px; border-radius: 22px; margin-bottom: 18px;}
    .hero .hero-title {font-size: 1.55rem; line-height: 1.22; margin: 12px 0 10px;}
    .hero .hero-sub {font-size: .92rem; line-height: 1.55;}
    .hero .hero-chips {gap: 8px; margin-top: 16px;}
    .hero .chip {padding: 5px 12px; font-size: .74rem;}
    .hero .hero-eyebrow {font-size: .64rem; padding: 5px 12px;}

    /* tab dalam satu baris yang bisa digeser */
    [data-testid="stTabs"] [role="tablist"] {
        flex-wrap: nowrap !important; overflow-x: auto !important;
        scrollbar-width: none; -webkit-overflow-scrolling: touch;
    }
    [data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {display: none;}
    [data-testid="stTab"] {flex: none !important; white-space: nowrap; padding: 8px 16px !important;}
    [data-testid="stTab"] p {font-size: .82rem;}

    .sec-title {font-size: 1.05rem; margin: 22px 0 12px;}
    div[data-testid="stForm"] {padding: 18px 16px 10px 16px; border-radius: 20px;}
    div[data-testid="stFormSubmitButton"], div[data-testid="stFormSubmitButton"] button {width: 100% !important;}
    div[data-testid="stMetric"] {padding: 12px 16px; border-radius: 16px;}

    .rec-card {min-height: 0; padding: 20px 18px 16px 18px; border-radius: 22px;}
    .rec-card:hover {transform: none;}
    .rec-model {min-height: 0;}
    .rec-price {font-size: 1.45rem;}
    .rec-rank {font-size: 4.4rem;}
    .empty {padding: 30px 18px;}
    .footer-note {font-size: .72rem;}
}

@media (prefers-reduced-motion: reduce) {
    * {animation: none !important; transition: none !important;}
    .bar > i {width: var(--w);}
}
</style>
"""
st.markdown(THEME_VARS + STATIC_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bilah atas (nama aplikasi dan tombol tema)
# ---------------------------------------------------------------------------
try:
    topbar = st.container(key="topbar")
except TypeError:
    topbar = st.container()

with topbar:
    try:
        top_left, top_right = st.columns([4, 1], vertical_alignment="center")
    except TypeError:
        top_left, top_right = st.columns([4, 1])

    with top_left:
        st.markdown(
            '<div class="brand"><span class="brand-dot"></span>SPK Laptop Pemrograman</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        try:
            st.toggle("Mode gelap", key="theme_dark", on_change=on_theme_change)
        except AttributeError:
            st.checkbox("Mode gelap", key="theme_dark", on_change=on_theme_change)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """<div class="hero">
<div class="hero-eyebrow">Sistem Pendukung Keputusan</div>
<div class="hero-title">Temukan laptop pemrograman terbaik sesuai anggaran Anda</div>
<div class="hero-sub">Masukkan batas anggaran, lalu sistem menyaring dan mengurutkan laptop menggunakan pembobotan AHP dan perangkingan TOPSIS.</div>
<div class="hero-chips"><span class="chip">Metode AHP</span><span class="chip">Metode TOPSIS</span><span class="chip">5 kriteria</span><span class="chip">Data Kaggle</span></div>
</div>""",
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

# Ikon SVG untuk kartu spesifikasi
SVG_OPEN = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round">')
ICON_CPU = (SVG_OPEN + '<rect x="6" y="6" width="12" height="12" rx="2"/>'
            '<rect x="9.5" y="9.5" width="5" height="5"/>'
            '<path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>')
ICON_RAM = (SVG_OPEN + '<rect x="2" y="7" width="20" height="9" rx="1.5"/>'
            '<path d="M6 16v3M10 16v3M14 16v3M18 16v3M6 10.5h2M11 10.5h2M16 10.5h2"/></svg>')
ICON_DISK = (SVG_OPEN + '<rect x="3" y="6" width="18" height="12" rx="2.5"/>'
             '<circle cx="17" cy="12" r="1"/><path d="M7 12h6"/></svg>')
ICON_SCREEN = (SVG_OPEN + '<rect x="3" y="4" width="18" height="12" rx="2"/>'
               '<path d="M8 20h8M12 16v4"/></svg>')


# ---------------------------------------------------------------------------
# Pembantu tampilan
# ---------------------------------------------------------------------------
def stretch(func, *args, **kwargs):
    """Memanggil komponen Streamlit dengan lebar penuh.

    Versi baru memakai width="stretch", versi lama memakai use_container_width.
    """
    try:
        return func(*args, width="stretch", **kwargs)
    except Exception:
        return func(*args, use_container_width=True, **kwargs)


def notice(kind, text):
    """Kotak pemberitahuan yang ikut tema. kind: success, error, warning, info."""
    st.markdown(
        f'<div class="notice {kind}"><div>{html.escape(str(text))}</div></div>',
        unsafe_allow_html=True,
    )


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


CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


@st.cache_data
def load_from_upload(file_bytes: bytes, name: str):
    """Membaca CSV unggahan. Mencoba beberapa encoding karena banyak CSV Kaggle
    tidak berformat UTF-8 (misalnya Latin-1 atau Windows-1252)."""
    sha = hashlib.sha256(file_bytes).hexdigest()
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding), sha, name
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


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
# Utilitas tampilan data
# ---------------------------------------------------------------------------
def format_rupiah(value):
    return "Rp." + f"{int(round(value)):,}".replace(",", ".")


def format_storage(gb):
    if gb >= 1024:
        return f"{gb / 1024:g} TB"
    return f"{gb:g} GB"


def section_title(text):
    st.markdown(f'<div class="sec-title">{html.escape(text)}</div>', unsafe_allow_html=True)


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


def recommendation_cards(rows, budget):
    cards = []
    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        tag = "Rekomendasi utama" if i == 1 else f"Peringkat {i}"
        top_class = " top" if i == 1 else ""
        delay = (i - 1) * 0.14
        score_pct = max(0.0, min(1.0, float(r["preference_score"]))) * 100
        remaining = budget - r["price_idr"]
        if remaining <= 0:
            save_text = "Tepat sesuai anggaran"
        else:
            save_text = f"Hemat {format_rupiah(remaining)} dari anggaran"

        cards.append(
            f'<div class="rec-card{top_class}" style="animation-delay:{delay:.2f}s">'
            f'<div class="rec-rank">{i}</div>'
            f'<div class="rec-tag">{tag}</div>'
            f'<div class="rec-model">{html.escape(str(r["model"]))}</div>'
            f'<div class="rec-price">{format_rupiah(r["price_idr"])}</div>'
            f'<div class="rec-save">{save_text}</div>'
            f'<div class="rec-specs">'
            f'<div class="spec">{ICON_CPU}<div><small>CPU</small><b>{int(r["cpu_cores"])} core</b></div></div>'
            f'<div class="spec">{ICON_RAM}<div><small>RAM</small><b>{r["ram_gb"]:g} GB</b></div></div>'
            f'<div class="spec">{ICON_DISK}<div><small>Storage</small><b>{format_storage(r["storage_gb"])}</b></div></div>'
            f'<div class="spec">{ICON_SCREEN}<div><small>Resolusi</small><b>{r["resolution_px"] / 1e6:.2f} MP</b></div></div>'
            f'</div>'
            f'<div class="score-row"><span>Skor TOPSIS</span><b>{r["preference_score"]:.4f}</b></div>'
            f'<div class="bar"><i style="--w:{score_pct:.1f}%"></i></div>'
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
    stretch(
        st.dataframe,
        table,
        hide_index=True,
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


def axis_style(**extra):
    """Gaya sumbu grafik yang mengikuti tema (dipasang langsung pada tiap sumbu)."""
    options = dict(
        gridColor=P["chart_grid"], domainColor=P["chart_domain"],
        tickColor=P["chart_domain"], labelColor=P["chart_label"],
        titleColor=P["chart_label"], labelFontSize=12,
    )
    options.update(extra)
    return alt.Axis(**options)


def style_chart(chart):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            gridColor=P["chart_grid"], domainColor=P["chart_domain"],
            tickColor=P["chart_domain"], labelColor=P["chart_label"],
            titleColor=P["chart_label"], labelFontSize=12,
        )
        .configure(background="transparent")
    )


def score_chart(data, top_n):
    top = data.head(top_n).copy()
    top["harga"] = top["price_idr"].apply(format_rupiah)
    chart = (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=7, size=18)
        .encode(
            x=alt.X("preference_score:Q", title="Skor TOPSIS",
                    axis=axis_style(tickCount=4 if IS_MOBILE else 8, format=".2f")),
            y=alt.Y("model:N", sort="-x", title=None,
                    axis=axis_style(labelLimit=110 if IS_MOBILE else 340,
                                    labelFontSize=11 if IS_MOBILE else 12)),
            color=alt.Color(
                "preference_score:Q",
                scale=alt.Scale(range=P["chart_range"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("harga:N", title="Harga"),
                alt.Tooltip("preference_score:Q", title="Skor", format=".4f"),
            ],
        )
        .properties(height=max(120, (28 if IS_MOBILE else 32) * len(top) + 40))
    )
    stretch(st.altair_chart, style_chart(chart))


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
# Muat data
# ---------------------------------------------------------------------------
if source_mode == "Unduh dari Kaggle":
    try:
        df_raw, source_sha256, source_name = load_from_kaggle()
    except Exception as exc:
        notice(
            "error",
            "Gagal mengunduh dataset dari Kaggle. Buka sidebar (panah di kiri atas) "
            "lalu pilih opsi Unggah laptop.csv.",
        )
        st.exception(exc)
        st.stop()
else:
    if uploaded is None:
        notice("info", "Buka sidebar (panah di kiri atas) dan unggah file laptop.csv untuk memulai.")
        st.stop()
    df_raw, source_sha256, source_name = load_from_upload(uploaded.getvalue(), uploaded.name)

required_columns = ["Model", "Price", "Core", "Ram", "SSD", "Display"]
missing_columns = [c for c in required_columns if c not in df_raw.columns]
if missing_columns:
    notice(
        "error",
        f"Kolom wajib tidak ditemukan: {missing_columns}. "
        f"Kolom pada file Anda: {list(df_raw.columns)}. "
        "Aplikasi ini membaca format dataset Kaggle sumanbera19/laptop-price-dataset "
        "(kolom Model, Price, Core, Ram, SSD, dan Display).",
    )
    st.stop()

if INR_TO_IDR_RATE <= 0:
    notice("error", "Kurs INR ke IDR harus berupa angka positif.")
    st.stop()
if not EXCHANGE_RATE_SOURCE.strip() or not EXCHANGE_RATE_DATE.strip():
    notice("error", "Sumber dan tanggal kurs wajib diisi.")
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
    notice(
        "error",
        f"Alternatif valid hanya {len(df_clean_inr)} (kurang dari 45). "
        "Periksa aturan ekstraksi sebelum melanjutkan.",
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
    section_title("Matriks perbandingan berpasangan")
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

    edited = stretch(
        st.data_editor,
        pd.DataFrame(upper, index=labels, columns=labels),
        column_config={
            label: st.column_config.NumberColumn(
                label, min_value=1 / 9, max_value=9.0, format="%.3f"
            )
            for label in labels
        },
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
        notice(
            "error",
            "Semua sel di atas diagonal harus berisi angka positif. "
            "Sementara ini perhitungan memakai matriks bawaan.",
        )
        pairwise_matrix = DEFAULT_PAIRWISE.copy()

    st.markdown("**Matriks lengkap (setelah kebalikan dihitung)**")
    stretch(
        st.dataframe,
        pd.DataFrame(pairwise_matrix, index=labels, columns=labels).style.format("{:.4f}"),
    )

    weights, lambda_max, CI, RI, CR = calculate_ahp(pairwise_matrix)

    weight_table = pd.DataFrame({
        "Kriteria": labels,
        "Jenis": [criteria_types[c].title() for c in criteria],
        "Bobot": weights,
    })

    section_title("Bobot kriteria")
    w1, w2 = st.columns(2)
    with w1:
        stretch(
            st.dataframe,
            weight_table.style.format({"Bobot": "{:.4f}"}),
            hide_index=True,
        )
    with w2:
        weight_chart = (
            alt.Chart(weight_table)
            .mark_bar(cornerRadiusEnd=7, size=20, color=P["chart_solid"])
            .encode(
                x=alt.X("Bobot:Q", title="Bobot AHP", axis=axis_style()),
                y=alt.Y("Kriteria:N", sort="-x", title=None, axis=axis_style()),
                tooltip=["Kriteria", alt.Tooltip("Bobot:Q", format=".4f")],
            )
            .properties(height=220)
        )
        stretch(st.altair_chart, style_chart(weight_chart))

    section_title("Uji konsistensi")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lambda max", f"{lambda_max:.6f}")
    k2.metric("CI", f"{CI:.6f}")
    k3.metric("RI", f"{RI:.2f}")
    k4.metric("CR", f"{CR:.6f}")
    if CR < 0.10:
        notice("success", "Konsisten (CR < 0,10)")
    else:
        notice("error", "Belum konsisten (CR >= 0,10). Perbaiki penilaian matriks.")

# ======================= TOPSIS =======================
ranking = calculate_topsis(df_clean, criteria, weights, criteria_types)

# ======================= TAB REKOMENDASI =======================
with tab_rec:
    section_title("Cari laptop sesuai anggaran")

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
        st.form_submit_button("Tampilkan Rekomendasi", type="primary")

    budget = parse_budget(budget_text)

    if budget is None:
        notice(
            "error",
            "Anggaran tidak dapat dibaca. Gunakan angka, misalnya 10.000.000, "
            "10jt, atau 7,5 juta.",
        )
    else:
        budget_result = ranking[ranking["price_idr"] <= budget].copy().reset_index(drop=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Anggaran", format_rupiah(budget))
        m2.metric("Laptop sesuai anggaran", f"{len(budget_result)} dari {len(ranking)}")
        if budget_result.empty:
            m3.metric("Skor tertinggi", "-")
        else:
            m3.metric("Skor tertinggi", f"{budget_result.loc[0, 'preference_score']:.4f}")

        if budget_result.empty:
            cheapest = int(ranking["price_idr"].min())
            st.markdown(
                f'<div class="empty"><b>Belum ada laptop yang sesuai anggaran</b>'
                f"Harga laptop termurah pada data adalah {format_rupiah(cheapest)}. "
                f"Naikkan anggaran untuk melihat rekomendasi.</div>",
                unsafe_allow_html=True,
            )
        else:
            section_title("Rekomendasi terbaik")
            cards = recommendation_cards(budget_result.head(3), budget)
            card_cols = st.columns(3)
            for col, card in zip(card_cols, cards):
                with col:
                    st.markdown(card, unsafe_allow_html=True)

            shown = min(top_n, len(budget_result))
            section_title(f"{shown} laptop teratas dalam anggaran")
            show_table(ranking_table(budget_result, shown))

            section_title("Perbandingan skor")
            score_chart(budget_result, shown)

            stretch(
                st.download_button,
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
    section_title("Peringkat TOPSIS seluruh alternatif")
    st.caption(f"Jumlah alternatif yang dirangking: {len(ranking)}")

    rank_top_n = st.slider("Jumlah laptop yang ditampilkan", 5, len(ranking), min(20, len(ranking)))
    show_table(ranking_table(ranking, rank_top_n))
    score_chart(ranking, min(rank_top_n, 30))

# ======================= TAB DATA =======================
with tab_data:
    with st.expander("Catatan akademik"):
        notice(
            "warning",
            "Instruksi Final Project menyebutkan alternatif minimal 45 data dummy. "
            "Dataset ini adalah data sekunder Kaggle, bukan data dummy. Penggunaannya "
            "perlu dikonfirmasi kepada dosen. Jangan menyebut dataset ini sebagai "
            "data dummy sebelum ada persetujuan atau arahan dosen.",
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data awal", len(df_raw))
    c2.metric("Data valid", len(df_clean))
    c3.metric("Dikeluarkan", len(df_excluded))
    c4.metric("Kolom sumber", df_raw.shape[1])

    st.caption(
        f"Sumber: [{DATASET_HANDLE}]({DATASET_URL}) | File: `{source_name}` | "
        f"SHA-256: `{source_sha256}`"
    )

    section_title("Cuplikan data mentah")
    stretch(st.dataframe, df_raw.head())

    section_title("Audit awal")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Nilai kosong per kolom (data mentah)**")
        stretch(st.dataframe, df_raw.isna().sum().to_frame("jumlah_kosong"))
    with a2:
        st.markdown("**Nilai kosong setelah ekstraksi**")
        stretch(st.dataframe, df_extracted.isna().sum().to_frame("jumlah_kosong"))
    st.write("Jumlah baris duplikat identik:", int(df_raw.duplicated().sum()))

    section_title("Data bersih dan konversi IDR")
    st.caption(
        f"Kurs {INR_TO_IDR_RATE} | {EXCHANGE_RATE_SOURCE} | {EXCHANGE_RATE_DATE}"
    )
    stretch(st.dataframe, df_clean.head(20))

    if not df_excluded.empty:
        with st.expander(f"Baris yang tidak lolos ({len(df_excluded)})"):
            stretch(st.dataframe, df_excluded)

# ======================= TAB VALIDASI =======================
with tab_valid:
    section_title("Validasi hasil perhitungan")
    checks = [
        ("Jumlah bobot AHP sama dengan 1", bool(np.isclose(weights.sum(), 1.0))),
        ("Skor TOPSIS berada pada rentang 0 sampai 1",
         bool(ranking["preference_score"].between(0, 1).all())),
        ("Peringkat pertama ditemukan", bool(ranking["rank"].min() == 1)),
        ("Jumlah alternatif minimal 45", bool(len(ranking) >= 45)),
        ("Tidak ada nama model kosong", bool(ranking["model"].notna().all())),
    ]
    stretch(
        st.dataframe,
        pd.DataFrame({
            "Pemeriksaan": [c[0] for c in checks],
            "Hasil": ["Lolos" if c[1] else "Gagal" for c in checks],
        }),
        hide_index=True,
    )

    v1, v2, v3 = st.columns(3)
    v1.metric("Jumlah alternatif", len(ranking))
    v2.metric("Jumlah bobot", round(float(weights.sum()), 6))
    v3.metric("CR AHP", round(float(CR), 6))

# ======================= TAB UNDUH =======================
with tab_export:
    section_title("Unduh hasil")

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
            stretch(
                st.download_button,
                fname,
                data=to_csv_bytes(frame),
                file_name=fname,
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Catatan kaki
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="footer-note">Data: Kaggle ({html.escape(DATASET_HANDLE)}) | '
    f"Kurs 1 INR = {INR_TO_IDR_RATE:g} IDR ({html.escape(EXCHANGE_RATE_DATE)}) | "
    f"Metode: AHP dan TOPSIS</div>",
    unsafe_allow_html=True,
)
