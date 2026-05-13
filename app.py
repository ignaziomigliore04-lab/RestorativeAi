from pathlib import Path
import sys
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.engine import add_material_decision_metadata, assess_material_case_feasibility, compute_case_indices, interpret_indices, load_database, rank_materials, summarize_case
from utils.validation import run_material_validation

st.set_page_config(page_title="Restorative AI", layout="wide")

st.markdown(
    """
<style>
:root {
    --bg-main: #f7f9ff;
    --bg-soft: #eef4ff;
    --bg-soft-2: #f6f3ff;
    --card: #ffffff;
    --line: #d9e4ff;
    --line-soft: #e7ecf8;
    --text-main: #111827;
    --text-soft: #52607a;
    --blue: #3b82f6;
    --blue-deep: #2563eb;
    --blue-soft: #e9f1ff;
    --violet: #8b5cf6;
    --violet-deep: #7c3aed;
    --violet-soft: #f1ebff;
    --shadow: 0 12px 30px rgba(76, 104, 180, 0.10);
}
html, body, [data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #fbfcff 0%, #f3f7ff 58%, #f7f4ff 100%);}
[data-testid="stHeader"] {background: rgba(255,255,255,0.72);}
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1380px;}
.ra-hero {background: radial-gradient(circle at top left, rgba(139,92,246,0.18), transparent 28%), linear-gradient(135deg, #f7f1ff 0%, #efe7ff 36%, #ecf2ff 72%, #ffffff 100%); color: var(--text-main); border: 1px solid #d9ccff; border-radius: 34px; padding: 1.8rem 1.8rem; margin-bottom: 1rem; box-shadow: 0 18px 40px rgba(124, 58, 237, 0.15);}
.ra-card {background: var(--card); border: 1px solid var(--line-soft); border-radius: 24px; padding: 1rem 1.1rem; margin-bottom: 0.9rem; box-shadow: var(--shadow);}
.feature-card {background: linear-gradient(180deg,#ffffff 0%,#faf7ff 48%,#f5f9ff 100%); border: 1px solid #e6ddff; border-radius: 28px; padding: 1.2rem 1.2rem; min-height: 190px; box-shadow: var(--shadow); position:relative; overflow:hidden;}
.feature-card:before {content:''; position:absolute; inset:0 auto auto 0; width:100%; height:4px; background:linear-gradient(90deg,var(--violet),var(--blue));}
.feature-icon {width:54px; height:54px; border-radius:18px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#f0e9ff,#eaf0ff); border:1px solid #ddd2ff; font-size:1.55rem; margin-bottom:0.8rem;}
.feature-title {font-size: 1.12rem; font-weight: 800; margin-bottom: 0.45rem; color: var(--text-main);}
.feature-copy {color: var(--text-soft); line-height: 1.6; font-size: 0.98rem;}
.index-panel {background:linear-gradient(180deg,#ffffff,#f7faff 55%,#f7f3ff 100%); border:1px solid var(--line); border-radius:24px; padding:1.05rem 1.1rem; box-shadow:var(--shadow);}
.index-row {padding:0.7rem 0; border-bottom:1px solid #edf1fa;}
.index-row:last-child {border-bottom:none;}
.index-name {font-weight:800; color:var(--text-main);}
.index-copy {color:var(--text-soft); font-size:0.93rem; line-height:1.45;}
.cta-button-wrap {text-align:center; margin-top:1.15rem;}
.output-mini {background:linear-gradient(180deg,#ffffff,#f8fbff); border:1px solid var(--line-soft); border-radius:16px; padding:0.8rem 0.9rem; margin-bottom:0.65rem;}
.result-top {background: linear-gradient(180deg, #ffffff 0%, #f7fbff 55%, #f8f4ff 100%); border: 1px solid var(--line); border-radius: 24px; padding: 1rem 1.1rem; box-shadow: var(--shadow);}
.reason-box {background: #fafcff; border-left: 5px solid var(--violet); border-radius: 12px; padding: 0.8rem 0.9rem; margin-bottom: 0.7rem;}
.sem-card {border-radius: 18px; padding: 0.95rem 1rem; margin-bottom: 0.7rem; border: 1px solid transparent;}
.sem-green {background: #f0fdf4; border-color: #86efac;}
.sem-yellow {background: #fffbeb; border-color: #fde68a;}
.sem-red {background: #fef2f2; border-color: #fca5a5;}
.small-muted {color: var(--text-soft); font-size: 0.93rem; line-height: 1.55;}
.radar-note {background:linear-gradient(180deg,#ffffff,#f8fbff); border:1px solid var(--line-soft); border-radius:16px; padding:0.9rem 1rem; margin-bottom:0.75rem;}
.idx-chip {display:inline-block; padding:0.3rem 0.65rem; border-radius:999px; background:linear-gradient(135deg,#eef4ff,#f2ecff); border:1px solid var(--line); margin-right:0.35rem; margin-bottom:0.35rem; font-size:0.85rem;}
.section-title {font-size: 2.55rem; text-align: center; font-weight: 800; color: var(--text-main); margin-top: 0.8rem;}
.section-subtitle {font-size: 1.07rem; text-align: center; color: var(--text-soft); max-width: 760px; margin: 0 auto 1.3rem auto; line-height: 1.55;}
.cta-wrap {text-align:center; margin: 1.1rem 0 0.5rem 0;}
.app-tag {display:inline-block; padding:0.30rem 0.68rem; border-radius:999px; background:linear-gradient(135deg,#efe7ff,#f5efff); color:var(--violet-deep); border:1px solid #ddd2ff; font-size:0.83rem; font-weight:700; margin-bottom:0.35rem;}
.result-shell {background: linear-gradient(180deg,#ffffff 0%,#f8fbff 55%,#f8f4ff 100%); border:1px solid var(--line); border-radius:26px; padding:1.15rem 1.2rem; box-shadow:var(--shadow);}
.result-image-card {background:white; border:1px solid var(--line-soft); border-radius:26px; padding:0.85rem; box-shadow:var(--shadow);} 
.result-summary-card {background: radial-gradient(circle at top right, rgba(139,92,246,0.12), transparent 25%), linear-gradient(180deg,#ffffff,#f8fbff 55%,#f7f1ff 100%); border:1px solid #ddd2ff; border-radius:28px; padding:1.05rem 1.1rem; min-height:100%; box-shadow:var(--shadow);} 
.result-kicker {font-size:0.82rem; color:var(--violet-deep); font-weight:800; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.35rem;}
.result-main-title {font-size:2rem; color:var(--text-main); font-weight:800; margin:0 0 0.35rem 0; line-height:1.15;} 
.result-subtitle {color:var(--text-soft); font-size:0.98rem; line-height:1.55; margin-bottom:0.7rem;} 
.premium-metric {background:linear-gradient(180deg,#ffffff,#fbfcff 55%,#faf6ff 100%); border:1px solid #e7defe; border-radius:22px; padding:0.95rem 1rem; box-shadow:var(--shadow); min-height:98px;} 
.premium-metric-label {font-size:0.86rem; color:var(--text-soft); margin-bottom:0.4rem;} 
.premium-metric-value {font-size:1.72rem; color:var(--text-main); font-weight:800; line-height:1.1;} 
.premium-metric-note {font-size:0.84rem; color:var(--text-soft); margin-top:0.35rem;} 
.premium-panel {background:linear-gradient(180deg,#ffffff,#fcfcff 55%,#faf6ff 100%); border:1px solid #e7defe; border-radius:24px; padding:1rem 1.05rem; box-shadow:var(--shadow); height:100%;}
.premium-panel-title {font-size:1.02rem; font-weight:800; color:var(--text-main); margin-bottom:0.65rem;} 
.premium-list {margin:0; padding-left:1.1rem; color:#334155;} .premium-list li {margin-bottom:0.48rem; line-height:1.45;} 
.chart-shell {background:linear-gradient(180deg,#ffffff,#fcfcff 48%,#faf6ff 100%); border:1px solid #e7defe; border-radius:26px; padding:1rem 1rem 0.7rem 1rem; box-shadow:var(--shadow);} 
.section-kicker {font-size:0.84rem; color:var(--violet-deep); font-weight:800; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.2rem;} 
.expander-label {font-weight:700; color:var(--text-main);} 
.material-chip {display:inline-block; padding:0.28rem 0.55rem; background:linear-gradient(135deg,#eef4ff,#f2ecff); border:1px solid var(--line); border-radius:999px; font-size:0.82rem; color:#334155; margin-right:0.35rem; margin-bottom:0.35rem;} 
.home-grid-shell {background:linear-gradient(180deg,#ffffff,#f8f4ff 40%,#f5f8ff 100%); border:1px solid #e5dcff; border-radius:28px; padding:1rem 1rem 0.4rem 1rem; box-shadow:var(--shadow);} 
.home-side-shell {background:linear-gradient(180deg,#ffffff,#f7f3ff 45%,#eef4ff 100%); border:1px solid #ddd2ff; border-radius:28px; padding:1.1rem 1.15rem; box-shadow:var(--shadow);} 
.home-stat-card {background:white; border:1px solid var(--line-soft); border-radius:18px; padding:0.8rem 0.9rem; box-shadow:0 8px 20px rgba(76, 104, 180, 0.06); margin-bottom:0.7rem;} 
.takeaway-shell {background: linear-gradient(135deg, #f2eaff 0%, #eef4ff 58%, #ffffff 100%); color: var(--text-main); border:1px solid #ddd2ff; border-radius:24px; padding:1.05rem 1.15rem; box-shadow:var(--shadow);} 
.takeaway-title {font-size:1.04rem; font-weight:800; margin-bottom:0.45rem;} 
.takeaway-copy {font-size:0.97rem; line-height:1.6; color: var(--text-soft);} 
.stButton > button, div[data-testid="stDownloadButton"] > button {background: linear-gradient(135deg, var(--blue) 0%, var(--violet) 100%); color: white; border: none; border-radius: 999px; padding: 0.62rem 1.15rem; font-weight: 700; box-shadow: 0 10px 22px rgba(97, 96, 255, 0.20);} 
.stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {background: linear-gradient(135deg, var(--blue-deep) 0%, var(--violet-deep) 100%); color:white;}
.stRadio > div {background: rgba(255,255,255,0.72); border:1px solid var(--line-soft); border-radius: 999px; padding: 0.25rem 0.45rem;}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {border-radius: 18px; overflow:hidden;}
.hero-purple-note {color:#6d28d9; font-weight:700;}

.hero-grid {display:grid; grid-template-columns: 1.2fr 0.8fr; gap:1rem; align-items:center;}
.hero-badge {display:inline-block; padding:0.34rem 0.72rem; border-radius:999px; background:linear-gradient(135deg,#f2eaff,#ecefff); border:1px solid #ddd2ff; color:#6d28d9; font-weight:700; font-size:0.82rem; margin-bottom:0.55rem;}
.hero-title {font-size:3rem; line-height:1.02; font-weight:800; color:var(--text-main); margin:0.15rem 0 0.5rem 0;}
.hero-title span {background:linear-gradient(135deg,#6d28d9,#2563eb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;}
.hero-copy {font-size:1.03rem; line-height:1.65; color:var(--text-soft); max-width:720px;}
.hero-chip-row {margin-top:0.9rem;}
.hero-chip {display:inline-block; padding:0.34rem 0.7rem; border-radius:999px; background:rgba(255,255,255,0.82); border:1px solid #ddd2ff; color:#4b5563; font-size:0.84rem; margin-right:0.4rem; margin-bottom:0.45rem;}
.hero-side-card {background:linear-gradient(180deg,rgba(255,255,255,0.92),rgba(247,241,255,0.94)); border:1px solid #ddd2ff; border-radius:26px; padding:1rem 1.05rem; box-shadow:0 10px 24px rgba(124,58,237,0.10);} 
.hero-mini {background:white; border:1px solid #e8e0ff; border-radius:18px; padding:0.7rem 0.8rem; margin-bottom:0.55rem;}
.hero-mini-label {font-size:0.8rem; color:var(--text-soft);}
.hero-mini-value {font-size:1.15rem; font-weight:800; color:#6d28d9;}
.hero-mini-copy {font-size:0.84rem; color:var(--text-soft);}
.result-main-title {font-size:2.08rem; color:var(--text-main); font-weight:800; margin:0 0 0.35rem 0; line-height:1.1;} 
.result-main-title span {background:linear-gradient(135deg,#6d28d9,#2563eb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;}
.section-title {font-size: 2.65rem; text-align: center; font-weight: 800; color: var(--text-main); margin-top: 0.8rem;}
@media (max-width: 900px) {.hero-grid {grid-template-columns:1fr;}}


.confidence-badge {display:inline-block; padding:0.42rem 0.72rem; border-radius:999px; font-weight:800; font-size:0.86rem; margin-bottom:0.55rem; border:1px solid transparent;}
.confidence-high {background:#ecfdf5; border-color:#86efac; color:#166534;}
.confidence-medium {background:#fffbeb; border-color:#fde68a; color:#92400e;}
.confidence-borderline {background:#fff1f2; border-color:#fecdd3; color:#9f1239;}
.near-alt-box {background:linear-gradient(135deg,#fff7ed,#f8f4ff); border:1px solid #fed7aa; border-radius:20px; padding:0.85rem 0.95rem; margin-top:0.75rem;}
.validation-kpi {background:linear-gradient(180deg,#ffffff,#fbfcff 58%,#f8f4ff 100%); border:1px solid #e7defe; border-radius:22px; padding:0.9rem 1rem; box-shadow:var(--shadow); min-height:92px;}
.validation-kpi-label {font-size:0.82rem; color:var(--text-soft); margin-bottom:0.32rem;}
.validation-kpi-value {font-size:1.55rem; font-weight:800; color:var(--text-main);}


.keyword-strip {display:flex; flex-wrap:wrap; gap:0.45rem; margin:0.7rem 0 0.75rem 0;}
.keyword-pill {display:inline-flex; align-items:center; gap:0.35rem; padding:0.42rem 0.68rem; border-radius:999px; background:linear-gradient(135deg,#eef4ff,#f4efff); border:1px solid #ddd2ff; color:#334155; font-size:0.86rem; font-weight:700;}
.keyword-pill strong {color:#111827;}
.key-message {background:linear-gradient(135deg,#f8fbff,#f7f2ff); border:1px solid #ded6ff; border-radius:20px; padding:0.85rem 0.95rem; margin:0.72rem 0; box-shadow:0 8px 22px rgba(76,104,180,0.07);}
.key-message-title {font-size:0.82rem; font-weight:800; color:#6d28d9; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.25rem;}
.key-message-copy {font-size:1rem; color:#1f2937; line-height:1.45;}
.clinical-rec-card {background:radial-gradient(circle at top right, rgba(139,92,246,0.14), transparent 28%), linear-gradient(135deg,#ffffff 0%,#f7fbff 48%,#f5efff 100%); border:1px solid #d9ccff; border-radius:24px; padding:1rem 1.05rem; margin:0.75rem 0 0.7rem 0; box-shadow:0 12px 26px rgba(76,104,180,0.10);}
.clinical-rec-eyebrow {font-size:0.78rem; color:#6d28d9; font-weight:850; text-transform:uppercase; letter-spacing:0.055em; margin-bottom:0.35rem;}
.clinical-rec-main {font-size:1.25rem; line-height:1.28; color:#111827; font-weight:750; margin-bottom:0.48rem;}
.clinical-rec-copy {font-size:0.98rem; line-height:1.55; color:#334155;}
.clinical-disclaimer {background:#ffffffcc; border:1px solid #e7defe; border-radius:18px; padding:0.7rem 0.8rem; margin-top:0.65rem; color:#4b5563; font-size:0.92rem; line-height:1.45;}
.recommendation-strip {display:grid; grid-template-columns:repeat(3,1fr); gap:0.55rem; margin:0.75rem 0 0.45rem 0;}
.recommendation-tile {background:#ffffff; border:1px solid #e7defe; border-radius:18px; padding:0.72rem 0.75rem; min-height:76px;}
.recommendation-tile-label {font-size:0.75rem; color:#6d28d9; font-weight:800; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.25rem;}
.recommendation-tile-value {font-size:0.95rem; color:#111827; font-weight:750; line-height:1.25;}
.clinician-note {background:linear-gradient(135deg,#eef4ff,#f7f2ff); border:1px solid #d9ccff; border-radius:20px; padding:0.85rem 0.95rem; margin:0.75rem 0; color:#334155; font-size:0.95rem; line-height:1.5;}
.clinician-note strong {color:#111827;}
.why-optimal-list {margin:0.35rem 0 0 0; padding:0; display:grid; grid-template-columns:1fr; gap:0.38rem;}
.why-optimal-item {list-style:none; background:rgba(255,255,255,0.72); border:1px solid #e7defe; border-radius:14px; padding:0.48rem 0.58rem; color:#1f2937; font-size:0.92rem; line-height:1.35;}
.why-optimal-item strong {color:#111827;}
.alert-chip-row {display:flex; flex-wrap:wrap; gap:0.45rem; margin-top:0.45rem;}
.alert-chip {display:inline-flex; align-items:center; padding:0.42rem 0.68rem; border-radius:999px; background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12; font-size:0.86rem; font-weight:800;}
.weight-grid {display:grid; grid-template-columns:1fr 1fr; gap:0.48rem; margin-top:0.4rem;}
.weight-tile {background:white; border:1px solid #e7defe; border-radius:16px; padding:0.58rem 0.66rem;}
.weight-value {font-size:1.05rem; font-weight:900; color:#6d28d9; line-height:1.1;}
.weight-label {font-size:0.78rem; color:#52607a; font-weight:700;}
.gate-card {background:linear-gradient(135deg,#ffffff,#f7f2ff 55%,#eef4ff); border:1px solid #ddd2ff; border-radius:28px; padding:1.25rem 1.35rem; box-shadow:var(--shadow); margin:1rem 0;}
.no-material-card {background:linear-gradient(135deg,#fff7ed,#fff1f2 55%,#ffffff); border:1px solid #fed7aa; border-radius:26px; padding:1rem 1.1rem; box-shadow:var(--shadow); margin:1rem 0;}
.no-material-title {font-size:1.35rem; font-weight:900; color:#9a3412; margin-bottom:0.35rem;}
.gate-title {font-size:1.45rem; font-weight:900; color:#111827; margin-bottom:0.35rem;}
.gate-copy {color:#52607a; line-height:1.55; font-size:0.98rem;}

</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def _load():
    return load_database()

materials_df, props_df, sources_df = _load()




def material_aligned_restoration_context(rest_df: pd.DataFrame, top_material: pd.Series) -> pd.Series:
    """Keep the visible envelope aligned with the material-first decision.

    The restoration envelope is secondary. If the best material class is direct and
    the direct envelope is clinically acceptable, the UI should not visually pull the
    user back toward an indirect/prosthetic reading of the case.
    """
    if str(top_material.get("direct_or_indirect", "")) == "Direct":
        direct_rows = rest_df[rest_df["restoration"] == "Restauro diretto in composito"]
        if not direct_rows.empty and float(direct_rows.iloc[0].get("score", 0)) >= 55:
            return direct_rows.iloc[0]
    return rest_df.iloc[0]

def semaforo(score: float):
    if score >= 82:
        return "🟢", "sem-green", "Raccomandato"
    if score >= 65:
        return "🟡", "sem-yellow", "Possibile con limiti"
    return "🔴", "sem-red", "Non indicato"



OPTION_LABELS = {
    "cavity_size": {
        "Piccola": "Piccola (<3 mm / <1⁄3 intercuspidale)",
        "Media": "Media (3–5 mm / 1⁄3–1⁄2)",
        "Ampia": "Ampia (>5 mm / >1⁄2)",
    },
    "wall_thickness": {
        "Adeguato": "Adeguato (≥2,0 mm)",
        "Sottile": "Sottile (1,0–1,9 mm)",
        "Molto sottile": "Molto sottile (<1,0 mm)",
    },
    "ferrule": {
        "Presente": "Presente (≥2,0 mm)",
        "Parziale": "Parziale (1,0–1,9 mm)",
        "Assente": "Assente (<1,0 mm)",
    },
    "pulp_proximity": {
        "Bassa": "Bassa (>2,0 mm dentina residua)",
        "Media": "Media (0,5–2,0 mm)",
        "Alta": "Alta (<0,5 mm / rischio esposizione)",
    },
    "caries_risk": {
        "Basso": "Basso (0 nuove lesioni/12 mesi)",
        "Medio": "Medio (1–2 lesioni/12 mesi)",
        "Alto": "Alto (≥3 lesioni/12 mesi o carie attiva)",
    },
    "plaque_control": {
        "Buono": "Buono (<20%)",
        "Medio": "Medio (20–40%)",
        "Scarso": "Scarso (>40%)",
    },
    "margin": {
        "Sovragengivale": "Sovragengivale (>1 mm)",
        "Juxtagengivale": "Juxtagengivale (0–1 mm)",
        "Subgengivale": "Subgengivale (<0 mm)",
    },
    "compliance": {
        "Alta": "Alta (richiami regolari)",
        "Media": "Media (richiami irregolari)",
        "Bassa": "Bassa (scarsa adesione)",
    },
    "esthetic_demand": {
        "Bassa": "Bassa (priorità 0–3/10)",
        "Media": "Media (priorità 4–7/10)",
        "Alta": "Alta (priorità 8–10/10 / zona critica)",
    },
    "parafunction_severity": {
        "Assente": "Assente",
        "Lieve": "Lieve (<1 mm)",
        "Moderata": "Moderata (1–2 mm)",
        "Severa": "Severa (>2 mm / chipping)",
    },
    "occlusal_load": {
        "Basso": "Basso (contatti fisiologici)",
        "Medio": "Medio (carico aumentato controllabile)",
        "Alto": "Alto (bruxismo/fratture/restauri ripetuti)",
    },
    "tooth_wear": {
        "Assente": "Assente (BEWE 0)",
        "Lieve": "Lieve (BEWE 1)",
        "Moderata": "Moderata (BEWE 2, <50%)",
        "Severa": "Severa (BEWE 3, ≥50%)",
    },
    "budget_level": {
        "Basso": "Basso (forte vincolo)",
        "Medio": "Medio (indiretto selettivo)",
        "Alto": "Alto (workflow premium possibile)",
    },
}

OPTION_DETAILS = {
    "cavity_size": {
        "Piccola": "<3 mm oppure <1/3 della distanza intercuspidale: di solito compatibile con approccio conservativo se gli altri fattori sono favorevoli.",
        "Media": "3–5 mm oppure 1/3–1/2 della distanza intercuspidale: zona intermedia, da leggere insieme a pareti residue, cuspidi e carico.",
        "Ampia": ">5 mm oppure >1/2 della distanza intercuspidale: aumenta il peso di resistenza, supporto e copertura della classe materiale.",
    },
    "wall_thickness": {
        "Adeguato": "≥2,0 mm nel punto critico della parete/cuspide residua: rischio strutturale relativamente più basso.",
        "Sottile": "1,0–1,9 mm: parete più soggetta a flessione/frattura, aumenta il peso meccanico.",
        "Molto sottile": "<1,0 mm: condizione strutturalmente critica, spesso richiede una classe/material design più protettivi.",
    },
    "ferrule": {
        "Presente": "Circa ≥2,0 mm continui di ferrule sfruttabile.",
        "Parziale": "Circa 1,0–1,9 mm o ferrule discontinuo.",
        "Assente": "<1,0 mm o ferrule clinicamente non sfruttabile.",
    },
    "pulp_proximity": {
        "Bassa": "Dentina residua stimata >2,0 mm.",
        "Media": "Dentina residua stimata 0,5–2,0 mm.",
        "Alta": "Dentina residua <0,5 mm o rischio di esposizione pulpare.",
    },
    "caries_risk": {
        "Basso": "0 nuove lesioni o recidive negli ultimi 12 mesi.",
        "Medio": "1–2 nuove lesioni o recidive negli ultimi 12 mesi.",
        "Alto": "≥3 nuove lesioni/recidive, carie attiva o fattori salivari importanti.",
    },
    "plaque_control": {
        "Buono": "Indice di placca indicativo <20%.",
        "Medio": "Indice di placca indicativo 20–40%.",
        "Scarso": "Indice di placca indicativo >40%.",
    },
    "margin": {
        "Sovragengivale": "Margine visibile e isolabile, indicativamente >1 mm sopra il margine gengivale.",
        "Juxtagengivale": "Margine entro circa 0–1 mm dal margine gengivale.",
        "Subgengivale": "Margine oltre il margine gengivale: aumenta criticità adesiva/operativa.",
    },
    "compliance": {
        "Alta": "Richiami regolari e buona adesione alle istruzioni.",
        "Media": "Richiami irregolari o igiene variabile.",
        "Bassa": "Frequenti mancati richiami o scarsa adesione terapeutica.",
    },
    "esthetic_demand": {
        "Bassa": "Priorità estetica 0–3/10 o area poco visibile.",
        "Media": "Priorità estetica 4–7/10 o sorriso parzialmente coinvolto.",
        "Alta": "Priorità estetica 8–10/10 o zona estetica critica.",
    },
    "parafunction_severity": {
        "Assente": "Nessun segno clinico rilevante.",
        "Lieve": "Faccette isolate <1 mm, anamnesi non dominante.",
        "Moderata": "Faccette 1–2 mm o più segni funzionali associati.",
        "Severa": "Faccette >2 mm, dentina esposta, fratture/chipping ricorrenti.",
    },
    "occlusal_load": {
        "Basso": "Contatti fisiologici, nessun segno evidente di sovraccarico.",
        "Medio": "Faccette ≤1–2 mm, contatti eccentrici o carico aumentato ma controllabile.",
        "Alto": "Bruxismo/parafunzione, faccette >2 mm, fratture o restauri ripetuti.",
    },
    "tooth_wear": {
        "Assente": "BEWE 0: nessuna usura/erosione clinicamente rilevante.",
        "Lieve": "BEWE 1: perdita iniziale della texture superficiale.",
        "Moderata": "BEWE 2: difetto distinto, perdita di tessuto duro <50% della superficie.",
        "Severa": "BEWE 3: perdita di tessuto duro ≥50% della superficie.",
    },
    "budget_level": {
        "Basso": "Forte vincolo economico: preferenza per soluzioni dirette quando clinicamente accettabili.",
        "Medio": "Accetta indiretti selettivi se il beneficio clinico è chiaro.",
        "Alto": "Accetta workflow premium, CAD/CAM o laboratorio quando indicati.",
    },
}

def option_label(group: str, value: object) -> str:
    return OPTION_LABELS.get(group, {}).get(str(value), str(value))


def option_detail(group: str, value: object) -> str:
    return OPTION_DETAILS.get(group, {}).get(str(value), option_label(group, value))


def option_guide_df(groups) -> pd.DataFrame:
    rows = []
    for group, title in groups:
        for level, detail in OPTION_DETAILS.get(group, {}).items():
            rows.append({"Parametro": title, "Livello": level, "Riferimento clinico": detail})
    return pd.DataFrame(rows)


def restoration_type_label(restoration: str) -> str:
    if restoration == "Restauro diretto in composito":
        return "Restauro diretto"
    return "Restauro indiretto"


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def restoration_image_path(restoration: str) -> Path:
    mapping = {
        "Corona completa": "crown_complete.png",
        "Restauro diretto in composito": "direct_restoration.png",
        "Inlay": "inlay.png",
        "Onlay": "onlay.png",
        "Overlay": "overlay.png",
    }
    return ROOT / "assets" / mapping.get(restoration, "direct_restoration.png")


def restoration_brief_description(restoration: str) -> str:
    descriptions = {
        "Restauro diretto in composito": "Approccio conservativo indicato quando la struttura residua consente una ricostruzione diretta affidabile.",
        "Inlay": "Restauro indiretto intracoronale indicato quando la perdita di tessuto è moderata e non serve una copertura cuspidale estesa.",
        "Onlay": "Restauro indiretto con copertura cuspidale selettiva, utile quando una o più cuspidi necessitano rinforzo.",
        "Overlay": "Restauro indiretto con copertura cuspidale ampia, indicato nei casi con compromissione strutturale importante.",
        "Corona completa": "Restauro indiretto totale indicato quando il dente richiede protezione circonferenziale e massima stabilità strutturale.",
    }
    return descriptions.get(restoration, "Restauro selezionato in base all'equilibrio tra struttura, biologia, funzione e workflow.")


def _cell_list(value):
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v) for v in value if str(v).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    return [text] if text else []


def row_examples(row, n: int = 3):
    examples = _cell_list(row.get("example_names", []))
    if not examples:
        examples = _cell_list(row.get("display_name", ""))
    return list(dict.fromkeys(examples))[:n]


def class_examples(ranked: pd.DataFrame, class_name: str, n: int = 3):
    subset = ranked[ranked["primary_class_name"] == class_name]
    if subset.empty:
        return []
    examples = row_examples(subset.iloc[0], n=n)
    if examples:
        return examples[:n]
    fallback = subset["display_name"].dropna().astype(str).drop_duplicates().tolist()
    return fallback[:n]


def class_why(case, indices, top_row):
    reasons = []
    mode = "diretta" if top_row["direct_or_indirect"] == "Direct" else "indiretta"
    reasons.append(f"Classe materiale {mode}: scelta per proprietà meccaniche, adesive, estetiche e di workflow, non perché il restauro sia il protagonista.")
    if top_row["structural_fit"] >= 70:
        reasons.append("Profilo meccanico adeguato alla struttura residua e alla richiesta di supporto/copertura.")
    if top_row["bio_fit"] >= 65:
        reasons.append("Compatibilità adeguata con isolamento, margine e contesto adesivo del caso.")
    if top_row["esthetic_fit"] >= 70:
        reasons.append("Prestazione estetica coerente con la domanda clinica senza sacrificare il driver biomeccanico.")
    if top_row["workflow_fit"] >= 65:
        reasons.append("Workflow compatibile con sedute, budget, CAD/CAM e accettazione del piano.")
    if top_row["evidence_score"] >= 75:
        reasons.append("Base documentale più solida rispetto alle alternative vicine nel database.")
    if len(reasons) == 1:
        reasons.append("Miglior equilibrio materiale complessivo tra struttura, biologia, funzione, estetica e workflow.")
    return reasons[:4]


def material_weight_breakdown(top_row) -> pd.DataFrame:
    axes = [
        ("Meccanica / struttura", "weight_mechanical", "structural_fit", "mechanical_points", "Quanto il materiale risponde a carico, tessuto residuo, cuspidi e resistenza richiesta."),
        ("Biologia / adesione", "weight_biology", "bio_fit", "biology_points", "Quanto tollera isolamento, margini, rischio carie e qualità adesiva del caso."),
        ("Estetica", "weight_esthetic", "esthetic_fit", "esthetic_points", "Quanto il profilo ottico è coerente con settore anteriore/posteriore e richiesta estetica."),
        ("Workflow", "weight_workflow", "workflow_fit", "workflow_points", "Quanto è compatibile con sedute, budget, CAD/CAM, laboratorio e preferenza chairside."),
        ("Evidenza database", "weight_evidence", "evidence_score", "evidence_points", "Quanto la classe è supportata da verifica, completezza quantitativa e fonti nel database."),
        ("Tipo materiale", "weight_material_path", "scenario_match", "material_path_points", "Compatibilità diretto/indiretto come caratteristica della classe, non come scelta restaurativa principale."),
    ]
    rows = []
    for label, weight_col, fit_col, points_col, meaning in axes:
        rows.append({
            "Asse materiale": label,
            "Peso applicato": f"{float(top_row.get(weight_col, 0)):.1f}%",
            "Fit classe": f"{float(top_row.get(fit_col, 0)):.1f}%",
            "Punti score": f"{float(top_row.get(points_col, 0)):.1f}",
            "Logica clinica": meaning,
        })
    rows.append({
        "Asse materiale": "Bonus / penalità",
        "Peso applicato": "regole",
        "Fit classe": "—",
        "Punti score": f"+{float(top_row.get('bonus_points', 0)):.1f} / -{float(top_row.get('penalty_points', 0)):.1f}",
        "Logica clinica": "Correzioni cliniche esplicite: per esempio flowable sotto carico, zirconia molto opaca in estetica alta, indiretto non accettato, ecc.",
    })
    return pd.DataFrame(rows)


def database_weight_breakdown(top_row) -> pd.DataFrame:
    axes = [
        ("Meccanica", "case_weight_mechanical", "db_weight_mechanical", "weight_mechanical"),
        ("Biologia/adesione", "case_weight_biology", "db_weight_biology", "weight_biology"),
        ("Estetica", "case_weight_esthetic", "db_weight_esthetic", "weight_esthetic"),
        ("Workflow", "case_weight_workflow", "db_weight_workflow", "weight_workflow"),
        ("Evidenza", "case_weight_evidence", "db_weight_evidence", "weight_evidence"),
        ("Tipo materiale", "case_weight_material_path", "db_weight_material_path", "weight_material_path"),
    ]
    return pd.DataFrame([
        {
            "Asse": label,
            "Peso dal caso": f"{float(top_row.get(case_col, 0)):.1f}%",
            "Peso seed database": f"{float(top_row.get(db_col, 0)):.1f}%",
            "Peso finale usato": f"{float(top_row.get(final_col, 0)):.1f}%",
        }
        for label, case_col, db_col, final_col in axes
    ])


def database_reliability_table(top_row) -> pd.DataFrame:
    """Compact transparency table: what is directly database-backed vs interpreted.

    This avoids overclaiming. The material profile shown in the UI is partly
    class-level clinical interpretation, while score evidence uses the workbook
    fields: verification status, quantitative completeness and source records.
    """
    completeness = float(top_row.get('quantitative_completeness_pct', 0.0))
    evidence = float(top_row.get('evidence_score', 0.0))
    verification = safe_text(top_row.get('verification_status', 'N/D'))
    product_count = int(float(top_row.get('class_product_count', 1) or 1))
    source_count = int(float(top_row.get('class_source_count', len(top_row.get('source_urls', []) or [])) or 0))
    quant_obs = int(float(top_row.get('class_quantitative_observation_count', 0) or 0))
    status = 'Alta' if evidence >= 78 and completeness >= 45 else 'Intermedia' if evidence >= 62 else 'Da consolidare'
    return pd.DataFrame([
        {'Parametro': 'Affidabilità database', 'Valore': status, 'Lettura': 'Sintesi di verifica, completezza quantitativa e fonti.'},
        {'Parametro': 'Materiali nella classe', 'Valore': product_count, 'Lettura': 'Numero di esempi/database record usati per aggregare la classe.'},
        {'Parametro': 'Fonti tracciate', 'Valore': source_count, 'Lettura': 'Record fonte collegati ai materiali della classe.'},
        {'Parametro': 'Osservazioni quantitative', 'Valore': quant_obs, 'Lettura': 'Valori numerici disponibili nel database.'},
        {'Parametro': 'Completezza quantitativa', 'Valore': f'{completeness:.1f}%', 'Lettura': 'Quanto la classe è coperta da dati numerici nel workbook.'},
        {'Parametro': 'Verification status', 'Valore': verification, 'Lettura': 'Stato di verifica importato dal database.'},
    ])


def confidence_short_message(label: str) -> str:
    label = str(label or '')
    if label == 'Alta':
        return 'Vantaggio netto sulla seconda classe.'
    if label == 'Borderline':
        return 'Prime due classi quasi equivalenti.'
    return 'Alternativa materiale ancora vicina.'


def compact_material_reason(case, idx, top_row) -> str:
    driver = material_driver_sentence(top_row)
    cls = safe_text(top_row.get('primary_class_name', 'classe materiale'))
    if driver in ['meccanica / struttura', 'tipo materiale diretto/indiretto']:
        reason = 'priorità strutturale/funzionale'
    elif driver == 'biologia / adesione':
        reason = 'adesione e controllo del rischio'
    elif driver == 'estetica':
        reason = 'resa ottica e integrazione estetica'
    elif driver == 'workflow':
        reason = 'fattibilità clinica e operativa'
    else:
        reason = 'supporto del database'
    return f"**{cls}** · {reason} · driver: **{driver}**"


def _dominant_axis_values(top_row):
    axis = str(top_row.get("dominant_material_axis", "mechanical"))
    mapping = {
        "mechanical": ("meccanica/struttura", "weight_mechanical", "structural_fit"),
        "biology": ("biologia/adesione", "weight_biology", "bio_fit"),
        "esthetic": ("estetica", "weight_esthetic", "esthetic_fit"),
        "workflow": ("workflow", "weight_workflow", "workflow_fit"),
        "evidence": ("evidenza database", "weight_evidence", "evidence_score"),
        "material_path": ("tipo materiale", "weight_material_path", "scenario_match"),
    }
    label, weight_col, fit_col = mapping.get(axis, mapping["mechanical"])
    return label, float(top_row.get(weight_col, 0.0)), float(top_row.get(fit_col, 0.0))


def optimal_material_bullets(case, idx, top_row, ranked=None) -> list[str]:
    """Very short visible explanation: enough to understand why the material wins
    without turning the result page into a long clinical paragraph.
    """
    label, weight, fit = _dominant_axis_values(top_row)
    mode = "diretto" if top_row.get("direct_or_indirect") == "Direct" else "indiretto"
    sector = safe_text(case.get("clinical_sector", "N/D"))
    score = float(top_row.get("final_score", 0.0))
    pss = float(top_row.get("pss", 0.0))

    bullets = [
        f"**Driver dominante:** {label} — peso {weight:.0f}%, fit {fit:.0f}%.",
        f"**Coerenza clinica:** settore {sector.lower()}, strategia {mode}, score {score:.1f}.",
    ]
    if ranked is not None and getattr(ranked, "shape", (0,))[0] > 1:
        second = ranked.iloc[1]
        gap = score - float(second.get("final_score", 0.0))
        if gap < 2.0:
            bullets.append(f"**Nota:** alternativa vicina ({safe_text(second.get('primary_class_name'))}, gap {gap:.1f}).")
        else:
            bullets.append(f"**Vantaggio:** migliore equilibrio globale vs seconda classe, gap {gap:.1f}.")
    else:
        bullets.append(f"**Predicibilità:** PSS {pss:.1f}% sul profilo inserito.")
    return bullets[:3]


def optimal_material_html(case, idx, top_row, ranked=None) -> str:
    html_items = "".join([f"<li class='why-optimal-item'>{_inline_bold_html(x)}</li>" for x in optimal_material_bullets(case, idx, top_row, ranked)])
    return f"<ul class='why-optimal-list'>{html_items}</ul>"


def compact_weight_breakdown(top_row) -> pd.DataFrame:
    df = material_weight_breakdown(top_row).copy()
    keep = [c for c in ['Asse materiale', 'Peso applicato', 'Fit classe', 'Punti score'] if c in df.columns]
    return df[keep]


def compact_database_reliability(top_row) -> pd.DataFrame:
    full = database_reliability_table(top_row)
    wanted = ['Affidabilità database', 'Materiali nella classe', 'Fonti tracciate', 'Completezza quantitativa']
    return full[full['Parametro'].isin(wanted)][['Parametro', 'Valore']]


def compact_class_table(ranked: pd.DataFrame) -> pd.DataFrame:
    table = class_comparison_table(ranked).copy()
    if table.empty:
        return table
    return table[['Semaforo', 'Classe', 'Tipo', 'Score', 'PSS']].head(5)


def concise_material_summary(case, idx, top_row) -> str:
    cls = safe_text(top_row.get('primary_class_name', 'classe materiale'))
    driver = material_driver_sentence(top_row)
    sector = case.get('clinical_sector', 'settore clinico')
    return f"{cls}: scelta guidata soprattutto da {driver}, con settore {sector.lower()} e vincoli clinici inseriti."


def compact_alert_items(alert_items: list[str], n: int = 2) -> list[str]:
    return alert_items[:n] if alert_items else []


def compact_alert_keyword(alert: str) -> str:
    a = safe_text(alert).lower()
    if "diretta" in a and "carico" in a:
        return "Carico elevato"
    if "isolamento" in a:
        return "Adesione critica"
    if "estetica" in a or "traslucenza" in a:
        return "Check estetico"
    if "budget" in a:
        return "Budget"
    if "borderline" in a or "seconda classe" in a:
        return "Alternativa vicina"
    if "nessun alert" in a:
        return "Nessun alert dominante"
    return safe_text(alert).split(":")[0][:36]


def material_keyword_chips(case, idx, top_row, ranked=None) -> list[str]:
    sector = case.get("clinical_sector", "N/D")
    mode = "Diretto" if top_row.get("direct_or_indirect") == "Direct" else "Indiretto"
    chips = [
        f"**{mode}**",
        f"Driver: **{material_driver_sentence(top_row)}**",
        f"Settore: **{sector}**",
    ]
    if idx.fsi >= 0.65:
        chips.append("Funzione: **alta**")
    if idx.ssi >= 0.65:
        chips.append("Struttura: **critica**")
    if idx.edi >= 0.65:
        chips.append("Estetica: **alta**")
    if idx.bri >= 0.65:
        chips.append("Adesione: **da controllare**")
    if ranked is not None and getattr(ranked, 'shape', (0,))[0] > 1:
        gap = float(top_row.get("final_score", 0)) - float(ranked.iloc[1].get("final_score", 0))
        if gap < 2:
            chips.append("Gap: **borderline**")
    evidence = float(top_row.get("evidence_score", 0))
    chips.append(f"DB: **{evidence:.0f}%**")
    return list(dict.fromkeys(chips))[:8]


def _inline_bold_html(text: str) -> str:
    # Convert compact **keyword** markers into HTML bold while escaping text.
    parts = safe_text(text).split("**")
    rendered = []
    for i, part in enumerate(parts):
        part = escape(part)
        if i % 2 == 1:
            rendered.append(f"<strong>{part}</strong>")
        else:
            rendered.append(part)
    return "".join(rendered)


def chips_html(chips: list[str]) -> str:
    return "<div class='keyword-strip'>" + "".join([f"<span class='keyword-pill'>{_inline_bold_html(c)}</span>" for c in chips]) + "</div>"


def compact_class_verdict(case, idx, top_row) -> str:
    cls = safe_text(top_row.get("primary_class_name", "classe materiale"))
    driver = material_driver_sentence(top_row)
    mode = "diretta" if top_row.get("direct_or_indirect") == "Direct" else "indiretta"
    return f"**{cls}** è la scelta migliore perché massimizza il fit di **{driver}** in una strategia **{mode}**."


def _clinical_driver_phrase(case, idx, top_row) -> str:
    driver = material_driver_sentence(top_row)
    if driver == "meccanica/struttura":
        if idx.fsi >= 0.60:
            return "serve una classe con buon controllo di carico, usura e stabilità strutturale"
        return "la struttura residua richiede un materiale affidabile ma ancora conservativo"
    if driver == "biologia/adesione":
        return "isolamento, margine e rischio biologico condizionano la predicibilità adesiva"
    if driver == "estetica":
        return "la richiesta estetica e il settore rendono prioritari resa ottica e lucidabilità"
    if driver == "workflow":
        return "tempi, budget e accettazione del piano orientano verso la soluzione più praticabile"
    if driver == "evidenza database":
        return "la classe ha una base dati più solida nel database rispetto alle alternative"
    return "il percorso diretto/indiretto della classe è coerente con i vincoli inseriti"


def assisted_recommendation_html(case, idx, top_row, ranked=None) -> str:
    cls = escape(safe_text(top_row.get("primary_class_name", "classe materiale")))
    mode = "diretta" if top_row.get("direct_or_indirect") == "Direct" else "indiretta"
    driver = escape(material_driver_sentence(top_row))
    sector = escape(safe_text(case.get("clinical_sector", "settore clinico"))).lower()
    score = float(top_row.get("final_score", 0.0))
    phrase = escape(_clinical_driver_phrase(case, idx, top_row))
    advantage = ""
    if ranked is not None and getattr(ranked, "shape", (0,))[0] > 1:
        second = ranked.iloc[1]
        gap = score - float(second.get("final_score", 0.0))
        if gap < 2.0:
            advantage = f" La seconda classe, <strong>{escape(safe_text(second.get('primary_class_name')))}</strong>, è molto vicina: va considerata come alternativa reale."
        else:
            advantage = f" Il margine sulla seconda classe è <strong>{gap:.1f} punti</strong>."
    return (
        "<div class='clinical-rec-card'>"
        "<div class='clinical-rec-eyebrow'>Raccomandazione assistita</div>"
        f"<div class='clinical-rec-main'>La classe più coerente è <strong>{cls}</strong>.</div>"
        f"<div class='clinical-rec-copy'>È proposta perché, in questo caso, <strong>{driver}</strong> è il driver principale: {phrase}. "
        f"Il profilo è <strong>{mode}</strong>, in settore <strong>{sector}</strong>, con score <strong>{score:.1f}</strong>.{advantage}</div>"
        "<div class='clinical-disclaimer'><strong>Ruolo del sistema:</strong> supporta la scelta del materiale, ma non sostituisce diagnosi, esperienza clinica, controllo intraoperatorio e consenso del paziente.</div>"
        "</div>"
    )


def recommendation_tiles_html(case, idx, top_row, ranked=None) -> str:
    mode = "Diretto" if top_row.get("direct_or_indirect") == "Direct" else "Indiretto"
    driver = material_driver_sentence(top_row)
    confidence = safe_text(top_row.get("confidence_label", "Intermedia"))
    alt_text = "Non dominante"
    if ranked is not None and getattr(ranked, "shape", (0,))[0] > 1:
        second = ranked.iloc[1]
        gap = float(top_row.get("final_score", 0)) - float(second.get("final_score", 0))
        alt_text = "Quasi equivalente" if gap < 2.0 else f"Gap {gap:.1f}"
    tiles = [
        ("Strategia", mode),
        ("Driver", driver),
        ("Decisione", f"{confidence} · {alt_text}"),
    ]
    html = "<div class='recommendation-strip'>"
    for label, value in tiles:
        html += f"<div class='recommendation-tile'><div class='recommendation-tile-label'>{escape(label)}</div><div class='recommendation-tile-value'>{escape(safe_text(value))}</div></div>"
    html += "</div>"
    return html


def clinician_support_note_html() -> str:
    return (
        "<div class='clinician-note'><strong>Interpretazione:</strong> questa è una raccomandazione di classe materiale basata sui dati inseriti. "
        "La scelta definitiva resta clinica e va confermata su campo con isolamento, spessori, occlusione, substrato e preferenze del paziente.</div>"
    )


def public_restricted_page():
    st.markdown("<div class='gate-card'><div class='gate-title'>Accesso limitato</div><div class='gate-copy'>Questo decision support è destinato a odontoiatri, studenti di odontoiatria e professionisti del settore dentale. Non è uno strumento per autodiagnosi o scelta autonoma di materiali/restauri.</div></div>", unsafe_allow_html=True)
    st.info("Per dubbi clinici o restaurativi, consulta un odontoiatra. Puoi tornare alla Home, ma il Material Decision Assistant resta filtrato per uso professionale.")


def material_driver_sentence(top_row) -> str:
    axis_labels = {
        "mechanical": "meccanica/struttura",
        "biology": "biologia/adesione",
        "esthetic": "estetica",
        "workflow": "workflow",
        "evidence": "evidenza database",
        "material_path": "tipo materiale diretto/indiretto",
    }
    axis = str(top_row.get("dominant_material_axis", "mechanical"))
    return axis_labels.get(axis, axis)





def confidence_css(label: str) -> str:
    label = str(label or '').lower()
    if 'alta' in label:
        return 'confidence-high'
    if 'borderline' in label:
        return 'confidence-borderline'
    return 'confidence-medium'


def near_equivalent_material(ranked: pd.DataFrame, threshold: float = 2.0):
    if ranked is None or ranked.shape[0] < 2:
        return None
    top = ranked.iloc[0]
    second = ranked.iloc[1]
    gap = float(top.get('final_score', 0.0)) - float(second.get('final_score', 0.0))
    if gap < threshold:
        return second, round(gap, 1)
    return None


def material_vs_alternatives(top_row, ranked):
    alts = ranked.iloc[1:3].copy()
    criteria = [
        ("Resistenza biomeccanica", "structural_fit"),
        ("Compatibilità biologico-adesiva", "bio_fit"),
        ("Compatibilità di workflow", "workflow_fit"),
        ("Domanda estetica", "esthetic_fit"),
        ("Affidabilità documentale", "evidence_score"),
    ]
    rows = []
    for label, col in criteria:
        entry = {"Criterio": label}
        for i, (_, row) in enumerate(pd.concat([pd.DataFrame([top_row]), alts]).iterrows(), start=1):
            icon, _, status = semaforo(float(row[col]))
            entry[f"Opzione {i}"] = f"{icon} {int(round(float(row[col])))}"
        rows.append(entry)
    names = [top_row["display_name"]] + alts["display_name"].tolist()
    return pd.DataFrame(rows), names


def top_material_limitations(case, top_row, ranked):
    limits = []
    cls = safe_text(top_row["primary_class_name"]).lower()
    if case["isolation"] == "Impossibile" and top_row["direct_or_indirect"] == "Indirect":
        limits.append("La qualità dell'isolamento resta un punto critico da controllare nel workflow indiretto adesivo.")
    if case["esthetic_demand"] == "Alta" and "3y" in cls:
        limits.append("La classe top massimizza la robustezza, ma può richiedere attenzione sul compromesso estetico rispetto ad alternative più traslucide.")
    if case["budget_level"] == "Basso" and top_row["direct_or_indirect"] == "Indirect":
        limits.append("Il vincolo economico potrebbe favorire una seconda scelta più accessibile se il paziente non accetta il piano ideale.")
    if top_row["quantitative_completeness_pct"] < 45:
        limits.append("La completezza quantitativa del materiale non è massima: la raccomandazione resta valida, ma con base dati non totalmente piena.")
    if ranked.shape[0] > 1 and (float(top_row["final_score"]) - float(ranked.iloc[1]["final_score"])) < 5:
        limits.append("Il vantaggio sulla seconda opzione è contenuto: le prime alternative restano clinicamente discutibili.")
    if not limits:
        limits.append("Non emergono criticità dominanti: il materiale proposto ha un vantaggio clinico abbastanza netto sulle alternative principali.")
    return limits


def _fit_level(value: float) -> str:
    value = float(value)
    if value >= 80:
        return "Alto"
    if value >= 65:
        return "Buono"
    if value >= 50:
        return "Intermedio"
    return "Critico"


def _class_family_text(class_name: str) -> dict:
    c = safe_text(class_name).lower()
    if 'flowable' in c and 'bulk-fill' not in c:
        return {
            "family": "Composito flowable",
            "adhesion": "Adesione obbligatoria; indicazione principale come liner/base o piccole aree a basso stress.",
            "thickness": "Strati sottili; non ideale come massa portante occlusale.",
            "repair": "Riparabile e lucidabile, ma meno resistente come materiale principale.",
            "indications": "Piccole cavità, liner, base elastica, aree non portanti.",
            "limits": "Da evitare come protagonista in carichi elevati, cuspidi coinvolte o ampie superfici occlusali.",
        }
    if 'bulk-fill' in c:
        return {
            "family": "Composito bulk-fill",
            "adhesion": "Adesione obbligatoria; controllo dell'isolamento determinante.",
            "thickness": "Incrementi più profondi secondo indicazione del produttore; rifinitura occlusale da controllare.",
            "repair": "Buona riparabilità intraorale; lucidabilità variabile in base alla formulazione.",
            "indications": "Posteriori diretti, cavità moderate/profonde, workflow rapido.",
            "limits": "Meno ideale se la richiesta estetica è molto alta o se serve massima stratificazione ottica.",
        }
    if 'nanoibrido' in c or 'nanofilled' in c or 'nanoriempito' in c or 'microibrido' in c or 'universale' in c:
        return {
            "family": "Composito restaurativo diretto",
            "adhesion": "Adesione obbligatoria; molto dipendente da isolamento, margine e protocollo adesivo.",
            "thickness": "Incrementale/stratificato; spessori e profondità secondo tecnica e materiale usato.",
            "repair": "Alta riparabilità intraorale; lucidabilità generalmente favorevole, soprattutto nei nanofilled/nanoibridi.",
            "indications": "Restauri diretti conservativi, anteriori estetici e posteriori selezionati con buon controllo operativo.",
            "limits": "Più sensibile a tecnica, contrazione, carico elevato e isolamento sfavorevole rispetto ad alcune opzioni indirette.",
        }
    if 'feldsp' in c or 'leucite' in c:
        return {
            "family": "Vetroceramica estetica",
            "adhesion": "Cementazione adesiva raccomandata/necessaria per massimizzare supporto e predicibilità.",
            "thickness": "Richiede spessore e supporto adeguati; molto sensibile a preparazione e bonding.",
            "repair": "Riparazione possibile ma meno semplice rispetto ai compositi; grande valore estetico.",
            "indications": "Settore anteriore, faccette/restauri estetici, casi con carico controllato.",
            "limits": "Meno indicata con parafunzione severa, posteriori ad alto carico o richiesta meccanica dominante.",
        }
    if 'disilicato' in c or 'lithium disilicate' in c or 'zls' in c:
        return {
            "family": "Vetroceramica rinforzata",
            "adhesion": "Cementazione adesiva spesso preferibile, soprattutto nei restauri parziali e spessori ridotti.",
            "thickness": "Spessore clinico da verificare in base a indicazione, preparazione e produttore.",
            "repair": "Riparabilità intermedia; buona estetica con resistenza superiore alle ceramiche più estetiche tradizionali.",
            "indications": "Inlay/onlay/overlay estetici, anteriori estesi, posteriori selezionati con carico gestibile.",
            "limits": "Richiede controllo di adesione, spessore e carico; attenzione in bruxismo severo o margini molto sfavorevoli.",
        }
    if 'zirconia 3y' in c or 'alta resistenza' in c:
        return {
            "family": "Zirconia alta resistenza",
            "adhesion": "Può consentire cementazione meno adesiva in alcune indicazioni; bonding specifico se necessario.",
            "thickness": "Buona efficienza meccanica anche a spessori ridotti, da confermare secondo sistema/produttore.",
            "repair": "Riparabilità intraorale più complessa; grande vantaggio meccanico.",
            "indications": "Posteriori ad alto carico, parafunzione, antagonista protesico, corone/pilastri e casi strutturali dove domina la resistenza.",
            "limits": "Non è la scelta automatica negli indiretti adesivi conservativi; attenzione a estetica, lucidatura, antagonista e protocollo di bonding.",
        }
    if 'alta traslucenza' in c or '4y' in c or '5y' in c:
        return {
            "family": "Zirconia alta traslucenza",
            "adhesion": "Cementazione/bonding da scegliere secondo preparazione, ritenzione e protocollo del sistema.",
            "thickness": "Richiede verifica dello spessore minimo perché la maggiore traslucenza può ridurre margine meccanico rispetto a 3Y.",
            "repair": "Riparabilità più complessa dei compositi; buon compromesso tra estetica e robustezza.",
            "indications": "Posteriori/anteriori selezionati quando serve equilibrio tra resistenza ed estetica, non massima robustezza assoluta.",
            "limits": "Da evitare come default nei parziali conservativi se compositi, ibridi o vetroceramiche adesive sono più coerenti.",
        }
    if 'cad/cam composite' in c or 'resin nanoceramic' in c or 'picn' in c:
        return {
            "family": "Ibrido CAD/CAM / composito indiretto",
            "adhesion": "Workflow adesivo generalmente importante; buon controllo della cementazione richiesto.",
            "thickness": "Spessore da verificare secondo blocchetto/sistema; comportamento più elastico rispetto alle ceramiche pure.",
            "repair": "Buona riparabilità e minore fragilità percepita; estetica buona ma non sempre pari alle vetroceramiche più estetiche.",
            "indications": "Restauri indiretti conservativi, workflow digitale, casi in cui riparabilità e modulo più favorevole sono utili.",
            "limits": "Può essere meno indicato se serve massima resistenza o massima stabilità ottica a lungo termine.",
        }
    return {
        "family": "Classe materiale restaurativa",
        "adhesion": "Protocollo adesivo/cementazione da modulare secondo classe, margine e isolamento.",
        "thickness": "Spessore minimo da verificare secondo indicazione clinica e produttore.",
        "repair": "Riparabilità e lucidabilità da valutare rispetto alla famiglia materiale.",
        "indications": "Indicazione da definire in base al bilanciamento tra struttura, biologia, funzione, estetica e workflow.",
        "limits": "Limiti clinici dipendenti da carico, adesione, spessore, margine e richiesta estetica.",
    }


def material_profile_table(top_row) -> pd.DataFrame:
    fam = _class_family_text(top_row.get("primary_class_name", ""))
    return pd.DataFrame([
        {"Parametro": "Famiglia", "Valutazione": fam["family"], "Keyword clinica": "indicazione"},
        {"Parametro": "Resistenza", "Valutazione": _fit_level(top_row.get("structural_fit", 0)), "Keyword clinica": f"fit {float(top_row.get('structural_fit', 0)):.0f}%"},
        {"Parametro": "Estetica", "Valutazione": _fit_level(top_row.get("esthetic_fit", 0)), "Keyword clinica": f"fit {float(top_row.get('esthetic_fit', 0)):.0f}%"},
        {"Parametro": "Adesione", "Valutazione": _fit_level(top_row.get("bio_fit", 0)), "Keyword clinica": "isolamento / margini"},
        {"Parametro": "Workflow", "Valutazione": _fit_level(top_row.get("workflow_fit", 0)), "Keyword clinica": "sedute / budget"},
        {"Parametro": "Spessore", "Valutazione": "Verificare", "Keyword clinica": "secondo produttore"},
        {"Parametro": "Riparabilità", "Valutazione": "Profilo di classe", "Keyword clinica": "follow-up"},
    ])


def material_profile_bullets(top_row) -> tuple[list[str], list[str]]:
    fam = _class_family_text(top_row.get("primary_class_name", ""))
    indications = [fam["indications"]]
    limits = [fam["limits"]]
    if float(top_row.get("evidence_score", 0)) >= 75:
        indications.append("Base dati favorevole nel database: evidenza/verifica della classe sopra soglia.")
    if float(top_row.get("quantitative_completeness_pct", 0)) < 45:
        limits.append("Completezza quantitativa non massima: utile confermare valori specifici del prodotto scelto.")
    return indications[:3], limits[:3]


def material_decision_explanation(case, idx, top_row) -> str:
    cls = top_row["primary_class_name"]
    parts = [f"La classe {cls} emerge perché il motore assegna più peso a {material_driver_sentence(top_row)} nel profilo inserito."]
    if idx.fsi >= 0.65:
        parts.append("Lo stress funzionale elevato aumenta il peso di resistenza, comportamento all’usura e stabilità sotto carico.")
    if idx.ssi >= 0.65:
        parts.append("La compromissione strutturale spinge verso una classe capace di sostenere protezione e supporto del tessuto residuo.")
    if idx.bri >= 0.65:
        parts.append("Il contesto biologico-adesivo rende più importante la tolleranza a isolamento, margine e rischio carie.")
    if idx.edi >= 0.65:
        parts.append("La richiesta estetica e il settore clinico aumentano il peso del profilo ottico e della lucidabilità.")
    if idx.wci >= 0.60:
        parts.append("I vincoli di workflow riducono il vantaggio teorico di opzioni più complesse se non coerenti con sedute, budget o accettazione.")
    if len(parts) == 1:
        parts.append("Non c’è un singolo vincolo dominante: la scelta deriva dall’equilibrio tra meccanica, adesione, estetica, workflow e dati del database.")
    return " ".join(parts[:4])


def alternative_material_guidance(case, idx, ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked is None or ranked.shape[0] < 2:
        return pd.DataFrame()
    top = ranked.iloc[0]
    alt = ranked.iloc[1]
    rows = []
    rows.append({
        "Domanda clinica": "Perché la prima classe",
        "Lettura": f"{top['primary_class_name']} ha lo score globale migliore ({float(top['final_score']):.1f}) e il miglior equilibrio tra i pesi del caso.",
    })
    rows.append({
        "Domanda clinica": "Quando considerare la seconda",
        "Lettura": f"{alt['primary_class_name']} è reale se il clinico vuole privilegiare il suo punto forte: meccanica {float(alt.get('structural_fit', 0)):.0f}%, estetica {float(alt.get('esthetic_fit', 0)):.0f}%, biologia {float(alt.get('bio_fit', 0)):.0f}%, workflow {float(alt.get('workflow_fit', 0)):.0f}%.",
    })
    if top["direct_or_indirect"] != alt["direct_or_indirect"]:
        rows.append({
            "Domanda clinica": "Differenza pratica",
            "Lettura": "Le due classi appartengono a percorsi diversi diretto/indiretto: la scelta va rifinita con accettazione del paziente, controllo adesivo e quantità di struttura residua.",
        })
    elif float(top.get("esthetic_fit", 0)) - float(alt.get("esthetic_fit", 0)) > 8:
        rows.append({"Domanda clinica": "Se domina l’estetica", "Lettura": f"La prima classe mantiene un vantaggio estetico più netto rispetto a {alt['primary_class_name']} in questo caso."})
    elif float(top.get("structural_fit", 0)) - float(alt.get("structural_fit", 0)) > 8:
        rows.append({"Domanda clinica": "Se domina il carico", "Lettura": f"La prima classe mantiene un vantaggio biomeccanico più netto rispetto a {alt['primary_class_name']} in questo caso."})
    else:
        rows.append({"Domanda clinica": "Se il gap è basso", "Lettura": "La differenza è clinicamente sottile: preferenza operativa, disponibilità del materiale e manualità possono orientare la scelta finale."})
    return pd.DataFrame(rows)


def material_risk_alerts(case, idx, top_row, ranked) -> list[str]:
    alerts = []
    cls = safe_text(top_row.get("primary_class_name", "")).lower()
    if top_row["direct_or_indirect"] == "Direct" and (idx.fsi >= 0.70 or case.get("occlusal_load") == "Alto"):
        alerts.append("Classe diretta sotto carico elevato: controllare anatomia, contatti occlusali, stratificazione e follow-up.")
    if top_row["direct_or_indirect"] == "Direct" and case.get("isolation") in ["Difficile", "Impossibile"]:
        alerts.append("Materiale adesivo diretto con isolamento non ideale: il rischio tecnico può pesare più del punteggio teorico.")
    if ('feldsp' in cls or 'leucite' in cls) and idx.fsi >= 0.60:
        alerts.append("Classe molto estetica ma più delicata: attenzione se il carico funzionale è moderato/alto.")
    if ('zirconia 3y' in cls or 'alta resistenza' in cls) and case.get("clinical_sector") == "Anteriore" and case.get("esthetic_demand") == "Alta":
        alerts.append("Zirconia ad alta resistenza in area estetica: verificare traslucenza, spessore e mascheramento cromatico.")
    if ('zirconia 3y' in cls or 'alta resistenza' in cls or 'alta traslucenza' in cls or '4y' in cls or '5y' in cls) and case.get("clinical_sector") == "Posteriore" and int(case.get("involved_cusps", 0)) <= 1 and case.get("endo_treated") == "No" and case.get("occlusal_load") != "Alto":
        alerts.append("Zirconia in caso conservativo: confermare che serva davvero alta resistenza rispetto a diretto, ibridi o vetroceramiche adesive.")
    if top_row["direct_or_indirect"] == "Indirect" and case.get("budget_level") == "Basso":
        alerts.append("Classe indiretta con budget basso: discutere costo/beneficio e seconda classe quasi equivalente se presente.")
    if ranked is not None and ranked.shape[0] > 1 and (float(top_row["final_score"]) - float(ranked.iloc[1]["final_score"])) < 2:
        alerts.append("Decisione borderline: la seconda classe è clinicamente quasi equivalente e va mostrata al clinico come alternativa reale.")
    if not alerts:
        alerts.append("Nessun alert dominante: la classe è coerente con i principali vincoli meccanici, biologici, estetici e di workflow inseriti.")
    return alerts[:4]


def class_comparison_table(ranked: pd.DataFrame) -> pd.DataFrame:
    cols=[]
    dedup = ranked.drop_duplicates(subset=["primary_class_name"], keep="first").head(6)
    for _, row in dedup.iterrows():
        icon, _, status = semaforo(float(row["final_score"]))
        cols.append({
            "Semaforo": icon,
            "Esito": status,
            "Classe": row["primary_class_name"],
            "Esempi": ", ".join(row_examples(row, n=2)),
            "Score": round(float(row["final_score"]),1),
            "PSS": round(float(row["pss"]),1),
            "Tipo": row["direct_or_indirect"],
        })
    return pd.DataFrame(cols)


def material_class_reason(case, idx, top_row, row) -> str:
    if row["primary_class_name"] == top_row["primary_class_name"]:
        return "Classe scelta: miglior equilibrio materiale per struttura residua, rischio biologico-adesivo, funzione, estetica e workflow del caso."

    score = float(row["final_score"])
    prefix = "Possibile con limiti: " if score >= 65 else "Non indicata: "
    reasons = []

    # coerenza diretto/indiretto della classe
    if row["direct_or_indirect"] != top_row["direct_or_indirect"]:
        if top_row["direct_or_indirect"] == "Indirect":
            reasons.append("la classe diretta è meno coerente con la necessità di protezione strutturale/cuspidale")
        else:
            reasons.append("la classe indiretta è più invasiva o meno necessaria rispetto alla scelta conservativa")

    # structural and functional deficits
    if float(row.get("structural_fit", 0)) + 8 < float(top_row.get("structural_fit", 0)):
        reasons.append("fit biomeccanico inferiore rispetto alla classe prescelta")
    if idx.fsi >= 0.65 and float(row.get("structural_fit", 0)) < 70:
        reasons.append("meno indicata con carico funzionale/parafunzione elevati")

    # biological / adhesive context
    if float(row.get("bio_fit", 0)) + 8 < float(top_row.get("bio_fit", 0)):
        reasons.append("compatibilità biologico-adesiva più debole")
    if idx.bri >= 0.65 and float(row.get("bio_fit", 0)) < 65:
        reasons.append("penalizzata dal contesto biologico o dall'isolamento")

    # esthetic profile
    if idx.edi >= 0.65 and float(row.get("esthetic_fit", 0)) + 8 < float(top_row.get("esthetic_fit", 0)):
        reasons.append("resa estetica meno coerente con la richiesta del caso")

    # workflow
    if float(row.get("workflow_fit", 0)) + 8 < float(top_row.get("workflow_fit", 0)):
        reasons.append("workflow meno compatibile con sedute, budget o accettazione del piano")
    if case["indirect_acceptance"] == "No" and row["direct_or_indirect"] == "Indirect":
        reasons.append("limitata dall'accettazione sfavorevole del restauro indiretto")

    # evidence/data quality
    if float(row.get("evidence_score", 0)) + 10 < float(top_row.get("evidence_score", 0)):
        reasons.append("supporto documentale meno forte nel database")

    if not reasons:
        score_gap = float(top_row["final_score"]) - score
        if score_gap < 6:
            reasons.append("resta una seconda scelta discutibile, ma con vantaggio minore rispetto alla classe top")
        else:
            reasons.append("score globale inferiore per equilibrio meno favorevole tra i principali criteri clinici")

    return prefix + "; ".join(reasons[:2]) + "."


def class_comparison_table_with_reasons(ranked: pd.DataFrame, case, idx, top_row) -> pd.DataFrame:
    cols=[]
    dedup = ranked.drop_duplicates(subset=["primary_class_name"], keep="first").head(6)
    for _, row in dedup.iterrows():
        icon, _, status = semaforo(float(row["final_score"]))
        cols.append({
            "Semaforo": icon,
            "Esito": status,
            "Classe": row["primary_class_name"],
            "Esempi": ", ".join(row_examples(row, n=2)),
            "Score": round(float(row["final_score"]),1),
            "PSS": round(float(row["pss"]),1),
            "Tipo": row["direct_or_indirect"],
            "Motivo clinico": material_class_reason(case, idx, top_row, row),
        })
    return pd.DataFrame(cols)


def direct_output_points(case, top_rest, top_material, ranked):
    points=[]
    points.append(f"Restauro indicato: {top_rest['restoration']}.")
    points.append(f"Classe più coerente: {top_material['primary_class_name']}.")
    if case['cusp_loss'] == 'Sì' or int(case['involved_cusps']) >= 2:
        points.append("Serve protezione cuspidale o rinforzo strutturale.")
    if case['bruxism'] != 'Assente' or case['occlusal_load'] == 'Alto':
        points.append("Il carico funzionale pesa nella scelta della classe materiale.")
    if case['esthetic_demand'] == 'Alta':
        points.append("L'estetica resta un criterio rilevante, ma non dominante sulla sicurezza biomeccanica.")
    if ranked.shape[0] > 1 and float(ranked.iloc[1]['final_score']) >= 55:
        points.append(f"Seconda opzione discutibile: {ranked.iloc[1]['primary_class_name']}.")
    return points[:5]


def restoration_reasoning(case, indices, top_rest_row):
    reasons = []
    restoration = top_rest_row["restoration"]
    if restoration == "Restauro diretto in composito":
        reasons.append("La struttura residua consente ancora un approccio conservativo.")
    else:
        reasons.append("La perdita strutturale rende più indicata una soluzione indiretta.")
    if case["cusp_loss"] == "Sì" or int(case["involved_cusps"]) >= 2:
        reasons.append("Il coinvolgimento cuspidale richiede maggiore protezione.")
    if case["endo_treated"] == "Sì":
        reasons.append("Il trattamento endodontico aumenta il bisogno di supporto strutturale.")
    if indices.fsi >= 0.65:
        reasons.append("Il carico funzionale favorisce opzioni biomeccanicamente più affidabili.")
    if case["indirect_acceptance"] == "No":
        reasons.append("Le preferenze del paziente limitano il ventaglio delle opzioni indirette.")
    return list(dict.fromkeys(reasons))[:4]


def restoration_alternative_reason(case, idx, selected_restoration: str, option_restoration: str, option_score: float) -> str:
    if option_restoration == selected_restoration:
        return "È la soluzione con il miglior equilibrio complessivo per questo caso."

    structural_severe = idx.ssi >= 0.65
    functional_high = idx.fsi >= 0.65
    biologic_high = idx.bri >= 0.65
    cuspal_need = case["cusp_loss"] == "Sì" or int(case["involved_cusps"]) >= 2
    conservative_case = idx.ssi < 0.45 and int(case["residual_walls"]) >= 3 and case["endo_treated"] == "No"

    if option_score >= 65:
        prefix = "Possibile con limiti: "
    else:
        prefix = "Non indicato: "

    if option_restoration == "Restauro diretto in composito":
        if structural_severe or cuspal_need:
            return prefix + "non offre una protezione cuspidale e strutturale sufficiente rispetto alla soluzione prescelta."
        if functional_high:
            return prefix + "il carico funzionale previsto rende la soluzione diretta meno affidabile della scelta top."
        if selected_restoration != "Restauro diretto in composito":
            return prefix + "risulta meno protettivo della soluzione indiretta selezionata."

    if option_restoration == "Inlay":
        if cuspal_need or structural_severe:
            return prefix + "la perdita cuspidale o la compromissione strutturale richiedono più copertura di quella offerta da un inlay."
        if selected_restoration in ["Onlay", "Overlay", "Corona completa"]:
            return prefix + "è più conservativo ma meno protettivo rispetto al restauro selezionato."
        if case["indirect_acceptance"] == "No":
            return prefix + "il paziente non accetta facilmente una soluzione indiretta."

    if option_restoration == "Onlay":
        if selected_restoration == "Overlay" and (int(case["involved_cusps"]) >= 3 or structural_severe):
            return prefix + "fornisce una copertura più limitata rispetto all'overlay scelto."
        if selected_restoration == "Corona completa" and (case["endo_treated"] == "Sì" or case["coronal_tissue"] in ["25-50%", "<25%"]):
            return prefix + "può risultare meno protettivo di una corona completa nel livello di compromissione attuale."
        if case["indirect_acceptance"] == "No":
            return prefix + "resta condizionato dall'accettazione dell'indiretto."

    if option_restoration == "Overlay":
        if conservative_case and selected_restoration in ["Restauro diretto in composito", "Inlay"]:
            return prefix + "offrirebbe una copertura eccessiva rispetto alla struttura residua presente."
        if selected_restoration == "Corona completa" and case["endo_treated"] == "Sì" and case["coronal_tissue"] == "<25%":
            return prefix + "può risultare meno stabile di una corona completa nel caso specifico."
        if case["indirect_acceptance"] == "No":
            return prefix + "è frenato dai vincoli operativi e di accettazione del piano."

    if option_restoration == "Corona completa":
        if conservative_case or (case["vitality"] == "Vitale" and int(case["residual_walls"]) >= 3 and idx.ssi < 0.55):
            return prefix + "sarebbe più invasiva del necessario rispetto alla soluzione prescelta."
        if selected_restoration in ["Onlay", "Overlay"]:
            return prefix + "può essere presa in considerazione, ma comporta un sacrificio tissutale maggiore."

    if biologic_high and option_restoration != selected_restoration:
        return prefix + "il contesto biologico/adesivo penalizza questa opzione rispetto alla scelta top."
    if functional_high and option_restoration != selected_restoration:
        return prefix + "il carico funzionale riduce l'affidabilità relativa di questa alternativa."
    return prefix + "ha uno score inferiore perché meno coerente con il profilo strutturale, biologico e operativo del caso."



def source_block(urls):
    if not urls:
        st.write("Nessuna fonte tracciata disponibile per la classe/materiale top del database.")
        return
    for url in urls[:3]:
        st.markdown(f"- {url}")



def patient_context_note(case) -> str:
    sex = case.get("patient_sex", "Non specificato")
    sector = case.get("clinical_sector", "Posteriore" if case.get("tooth_group") in ["Premolare", "Molare"] else "Anteriore")
    notes = [f"Settore {sector.lower()}: {'priorità estetico-ottica più alta' if sector == 'Anteriore' else 'priorità meccanico-funzionale più alta'}."]
    if sex == "Femmina":
        notes.append("Sesso registrato come micro-fattore biologico: influenza solo in presenza di rischio carie/xerostomia già indicati.")
    elif sex == "Maschio":
        notes.append("Sesso registrato come micro-fattore biologico: influenza solo se associato a placca, compliance o supporto parodontale sfavorevoli.")
    else:
        notes.append("Sesso non usato come driver del materiale; nessun micro-aggiustamento demografico applicato.")
    return " ".join(notes)



def build_report(case, idx, top_rest, top_material, ranked, interpretation) -> str:
    example_list = class_examples(ranked, top_material["primary_class_name"], n=3)
    alt_classes = ranked[["primary_class_name", "final_score"]].drop_duplicates(subset=["primary_class_name"]).head(5)
    lines = []
    lines.append("# Restorative AI - Scheda decisionale")
    lines.append("")
    lines.append("## Caso")
    lines.append(f"- Dente: {case['tooth_number']} ({case['tooth_group']})")
    lines.append(f"- Settore clinico: {case.get('clinical_sector', 'N/D')}")
    lines.append(f"- Sesso paziente: {case.get('patient_sex', 'Non specificato')}")
    lines.append(f"- Vitalità: {case['vitality']}")
    lines.append(f"- Endodonzia: {case['endo_treated']}")
    lines.append(f"- Pareti residue: {case['residual_walls']}")
    lines.append(f"- Dimensione cavità: {option_detail('cavity_size', case['cavity_size'])}")
    lines.append(f"- Spessore pareti residue: {option_detail('wall_thickness', case['wall_thickness'])}")
    lines.append(f"- Cusp loss: {case['cusp_loss']}")
    lines.append(f"- Carico occlusale: {option_detail('occlusal_load', case['occlusal_load'])}")
    lines.append(f"- Usura dentale: {option_detail('tooth_wear', case.get('tooth_wear', 'Assente'))}")
    lines.append("")
    lines.append("## Decisione materiale")
    lines.append(f"- Classe materiale consigliata: {top_material['primary_class_name']}")
    lines.append(f"- Tipo materiale: {'Diretto' if top_material['direct_or_indirect'] == 'Direct' else 'Indiretto'}")
    lines.append(f"- Score classe: {top_material['final_score']}%")
    lines.append(f"- Confidence materiale: {top_material.get('confidence_label', 'N/D')} — {top_material.get('confidence_message', '')}")
    if ranked.shape[0] > 1 and (float(top_material['final_score']) - float(ranked.iloc[1]['final_score'])) < 2:
        lines.append(f"- Alternativa quasi equivalente: {ranked.iloc[1]['primary_class_name']} ({ranked.iloc[1]['final_score']}%, gap {float(top_material['final_score']) - float(ranked.iloc[1]['final_score']):.1f})")
    lines.append(f"- PSS: {top_material['pss']}%")
    if example_list:
        lines.append(f"- Esempi commerciali coerenti: {', '.join(example_list)}")
    lines.append(f"- Driver dominante dello score: {material_driver_sentence(top_material)}")
    lines.append(f"- Restauro compatibile secondario: {top_rest['restoration']} ({restoration_type_label(top_rest['restoration'])})")
    lines.append("")
    lines.append("## Perché è il materiale ottimale")
    for item in optimal_material_bullets(case, idx, top_material, ranked):
        lines.append(f"- {item.replace('**', '')}")
    lines.append("")
    lines.append("## Pesi materiali usati nello score")
    for _, r in material_weight_breakdown(top_material).iterrows():
        lines.append(f"- {r['Asse materiale']}: peso {r['Peso applicato']}, fit {r['Fit classe']}, punti {r['Punti score']}")
    lines.append("")
    lines.append("## Perché questa classe")
    lines.append(material_decision_explanation(case, idx, top_material))
    for item in class_why(case, idx, top_material)[:3]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Profilo della classe materiale")
    for _, r in material_profile_table(top_material).iterrows():
        lines.append(f"- {r['Parametro']}: {r['Valutazione']} — {r.get('Keyword clinica', '')}")
    lines.append("")
    lines.append("## Alert materiale")
    for item in material_risk_alerts(case, idx, top_material, ranked)[:3]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Restauro compatibile secondario")
    lines.append("Il restauro non determina lo score materiale: viene indicato solo come envelope operativo compatibile con la classe scelta.")
    for item in restoration_reasoning(case, idx, top_rest)[:2]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Index")
    lines.append(f"- SSI: {round(idx.ssi*100,1)}")
    lines.append(f"- BRI: {round(idx.bri*100,1)}")
    lines.append(f"- FSI: {round(idx.fsi*100,1)}")
    lines.append(f"- EDI: {round(idx.edi*100,1)}")
    lines.append(f"- WCI: {round(idx.wci*100,1)}")
    lines.append(f"- Asse dominante: {interpretation['dominant_axis']}")
    lines.append("")
    lines.append("## Alternative di classe")
    for _, row in alt_classes.iterrows():
        lines.append(f"- {row['primary_class_name']}: {row['final_score']}")
    lines.append("")
    lines.append("## Cautela")
    for item in top_material_limitations(case, top_material, ranked)[:2]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Clinical take-home message")
    lines.append(clinical_take_home(case, idx, top_rest, top_material, interpretation))
    return "\n".join(lines)


def clinical_take_home(case, idx, top_rest, top_material, interpretation):
    lines = []
    lines.append(f"La decisione principale riguarda la classe materiale: {top_material['primary_class_name']}.")
    lines.append(f"Lo score è guidato soprattutto da {material_driver_sentence(top_material)}, integrando pesi del caso e seed del database.")
    lines.append(f"Il restauro {top_rest['restoration'].lower()} è solo un envelope operativo compatibile, non il centro della raccomandazione.")
    lines.append(patient_context_note(case))
    if idx.fsi >= 0.65:
        lines.append("Il carico funzionale impone di privilegiare affidabilità meccanica e protezione rispetto al solo risultato estetico.")
    elif idx.ssi >= 0.65:
        lines.append("La compromissione strutturale richiede una classe capace di sostenere copertura e rinforzo." )
    elif idx.bri >= 0.65:
        lines.append("Il contesto biologico/adesivo rende decisive tolleranza clinica, isolamento e posizione del margine." )
    else:
        lines.append("Il profilo consente una scelta materiale relativamente conservativa e stabile, senza un singolo vincolo dominante critico.")
    return " ".join(lines[:5])


def home_stats(materials_df, sources_df):
    published = materials_df[materials_df["include_in_v1_database"] == "yes"]
    return [
        ("Classi materiali", int(published["primary_class_name"].nunique())),
        ("Esempi commerciali", int(len(published))),
        ("Fonti tracciate", int(len(sources_df))),
        ("Index del motore", 5),
    ]


def professional_access_gate():
    """First-load professional filter.

    The assistant is intended for dental professionals. If the user is not in
    the dental field, the decision engine is not shown.
    """
    if st.session_state.get("professional_access") in {"professional", "public"}:
        return

    def _gate_body():
        st.markdown("**Restorative AI è un supporto decisionale sui materiali dentari destinato a uso professionale.**")
        st.markdown("Non sostituisce diagnosi, visita clinica, responsabilità professionale o consenso informato.")
        c_yes, c_no = st.columns(2)
        with c_yes:
            if st.button("Sì, sono del settore odontoiatrico", use_container_width=True):
                st.session_state["professional_access"] = "professional"
                st.rerun()
        with c_no:
            if st.button("No, non faccio parte del settore odontoiatrico", use_container_width=True):
                st.session_state["professional_access"] = "public"
                st.rerun()

    if hasattr(st, "dialog"):
        @st.dialog("Accesso professionale")
        def _professional_dialog():
            _gate_body()
        _professional_dialog()
        st.stop()

    st.markdown("<div class='gate-card'><div class='gate-title'>Accesso professionale</div><div class='gate-copy'>Seleziona il tuo profilo per continuare.</div></div>", unsafe_allow_html=True)
    _gate_body()
    st.stop()


def render_home():
    st.markdown(
        """
<div class='ra-hero'>
  <div class='hero-grid'>
    <div>
      <div class='hero-badge'>AI-powered restorative decision support</div>
      <div class='hero-title'>Restorative <span>AI</span></div>
      <div class='hero-copy'>Sistema di supporto decisionale sui materiali dentari in odontoiatria restaurativa. Integra struttura, biologia, funzione, estetica e workflow per proporre prima la classe di materiale più coerente; il tipo di restauro resta come cornice clinica di supporto.</div>
      <div class='hero-chip-row'>
        <span class='hero-chip'>5 clinical indices</span>
        <span class='hero-chip'>Database-driven engine</span>
        <span class='hero-chip'>Interpretable output</span>
        <span class='hero-chip'>Validation mode</span>
      </div>
    </div>
    <div class='hero-side-card'>
      <div class='hero-mini'>
        <div class='hero-mini-label'>Output principale</div>
        <div class='hero-mini-value'>Classe materiale</div>
        <div class='hero-mini-copy'>La raccomandazione parte dalla classe, non dal brand commerciale.</div>
      </div>
      <div class='hero-mini'>
        <div class='hero-mini-label'>Decision engine</div>
        <div class='hero-mini-value'>SSI • BRI • FSI • EDI • WCI</div>
        <div class='hero-mini-copy'>La scelta nasce dall'integrazione di struttura, biologia, funzione, estetica e workflow.</div>
      </div>
      <div class='hero-mini'>
        <div class='hero-mini-label'>Orientamento clinico</div>
        <div class='hero-mini-value'>Material class first</div>
        <div class='hero-mini-copy'>Restauro, radar chart e alternative restano leggibili ma secondari.</div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-title'>Analisi clinica completa</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Una piattaforma pensata per trasformare i dati clinici del caso in una scelta di materiale più chiara, tracciabile e difendibile.</div>",
        unsafe_allow_html=True,
    )

    main, side = st.columns([1.68, 0.88])
    with main:
        st.markdown("<div class='home-grid-shell'>", unsafe_allow_html=True)
        features = [
            ("🧠", "Analisi materiale-centrica", "Legge il caso con indici clinici e regole decisionali orientate alla scelta della classe materiale."),
            ("🧱", "Classe materiale al centro", "La classe è la decisione principale; i prodotti sono esempi commerciali coerenti e secondari."),
            ("📊", "Confronto semaforico", "Mostra subito ciò che è consigliato, borderline o sfavorevole nel caso specifico."),
            ("🛡️", "Valutazione del rischio", "Integra rischio biologico, meccanico e funzionale nella scelta finale."),
            ("🩺", "Lettura del caso completo", "Considera struttura residua, cuspidi, carico, adesione, estetica e workflow."),
            ("✅", "Output clinico diretto", "Restituisce una raccomandazione materiale sintetica, utile e difendibile al riunito."),
        ]
        for row in [features[:3], features[3:]]:
            cols = st.columns(3)
            for col, (icon, title, copy) in zip(cols, row):
                with col:
                    st.markdown(
                        f"""
<div class='feature-card'>
  <div class='feature-icon'>{icon}</div>
  <div class='feature-title'>{title}</div>
  <div class='feature-copy'>{copy}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        if st.button("Valuta il tuo caso", type="primary", use_container_width=True):
            st.query_params["page"] = "evaluate"
            st.rerun()
        for label, value in home_stats(materials_df, sources_df):
            st.markdown(f"<div class='home-stat-card'><div class='premium-metric-label'>{label}</div><div class='premium-metric-value' style='font-size:1.32rem'>{value}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with side:
        st.markdown("<div class='home-side-shell'>", unsafe_allow_html=True)
        st.markdown("<div class='index-panel'>", unsafe_allow_html=True)
        st.subheader("Gli index del motore")
        index_rows = [
            ("SSI", "Structural Severity Index", "Quanto il dente è compromesso", "Più è alto, più servono protezione cuspidale e restauri indiretti."),
            ("BRI", "Biological Risk Index", "Quanto è delicato il contesto adesivo e biologico", "Sale con isolamento difficile, margini sfavorevoli e rischio carie alto."),
            ("FSI", "Functional Stress Index", "Quanto il caso è sotto carico", "Sale con bruxismo, parafunzione, usura e carico occlusale elevato."),
            ("EDI", "Esthetic Demand Index", "Quanto conta l'estetica", "Aiuta a spingere verso classi più estetiche quando il caso lo richiede."),
            ("WCI", "Workflow Constraint Index", "Quanto pesano i vincoli pratici", "Sedute, budget, CAD/CAM e accettazione del piano influenzano la scelta."),
        ]
        for short, name, copy, mini in index_rows:
            st.markdown(f"<div class='index-row'><div class='index-name'>{short} — {name}</div><div class='index-copy'>{copy}</div><div class='index-mini'>{mini}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_clinical_page():
    with st.expander("Metodo del motore decisionale", expanded=False):
        st.write(
            "Il motore usa cinque indici clinici — SSI, BRI, FSI, EDI e WCI — per scegliere prima la classe/tipo di materiale. Lo score materiale è composto da assi espliciti: meccanica, biologia/adesione, estetica, workflow, evidenza database e compatibilità diretto/indiretto. I pesi derivano per il 78% dal caso clinico e per il 22% dal foglio Class_Scoring_Seed del database. Il restauro è solo un envelope operativo secondario."
        )

    col_form, col_help = st.columns([1.55, 0.75])

    with col_help:
        st.markdown("<div class='ra-card'>", unsafe_allow_html=True)
        st.subheader("Material Decision Assistant")
        st.markdown("<div class='small-muted'>Inserisci i dati del caso. Il motore produrrà prima la classe/tipo di materiale, poi esempi commerciali coerenti e indicazione restaurativa di contesto.</div>", unsafe_allow_html=True)
        st.divider()
        st.subheader("Quick guide agli index")
        st.markdown("<div class='small-muted'><strong>SSI</strong> struttura • <strong>BRI</strong> biologia • <strong>FSI</strong> funzione • <strong>EDI</strong> estetica • <strong>WCI</strong> workflow</div>", unsafe_allow_html=True)
        st.divider()
        st.metric("Classi materiali", int(materials_df[materials_df["include_in_v1_database"] == "yes"]["primary_class_name"].nunique()))
        st.metric("Fonti tracciate", len(sources_df))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_form:
        st.markdown("<div class='ra-card'>", unsafe_allow_html=True)
        with st.form("clinical_case"):
            st.subheader("1. Paziente e Dente")
            c1, c2, c3 = st.columns(3)
            with c1:
                patient_age = st.number_input("Età paziente", min_value=10, max_value=100, value=25, step=1)
                patient_sex = st.selectbox("Sesso paziente", ["Non specificato", "Femmina", "Maschio"], help="Usato solo come micro-fattore biologico contestuale; non prescrive mai direttamente una classe materiale.")
                tooth_number = st.selectbox(
                    "Numero dente",
                    [
                        "11", "12", "13", "14", "15", "16", "17", "18",
                        "21", "22", "23", "24", "25", "26", "27", "28",
                        "31", "32", "33", "34", "35", "36", "37", "38",
                        "41", "42", "43", "44", "45", "46", "47", "48",
                    ],
                )
            with c2:
                sector_choice = st.selectbox("Settore clinico", ["Automatico dal dente", "Anteriore", "Posteriore"], help="Il settore pesa sulla priorità estetica nei casi anteriori e sulla richiesta meccanico-funzionale nei posteriori.")
                black_class = st.selectbox("Classe di Black", ["I", "II", "III", "IV", "V", "VI"])
                vitality = st.selectbox("Vitalità", ["Vitale", "Non vitale"])
            with c3:
                endo_treated = st.selectbox("Trattamento endodontico", ["No", "Sì"])
                clinical_priority = st.selectbox("Priorità clinica", ["Conservatività", "Durata", "Estetica", "Rapidità"])

            second_digit = tooth_number[1]
            tooth_group = "Anteriore" if second_digit in ["1", "2", "3"] else "Premolare" if second_digit in ["4", "5"] else "Molare"
            inferred_sector = "Anteriore" if tooth_group == "Anteriore" else "Posteriore"
            clinical_sector = inferred_sector if sector_choice == "Automatico dal dente" else sector_choice
            arch = "Mascellare" if tooth_number.startswith(("1", "2")) else "Mandibolare"
            st.caption(f"Settore usato dal motore: {clinical_sector} · gruppo dentale: {tooth_group}")

            st.subheader("2. Fattori Strutturali")
            st.caption("Range numerici compatti nei menu. La guida completa resta disponibile senza appesantire la compilazione.")
            with st.expander("Guida rapida ai range strutturali", expanded=False):
                st.dataframe(option_guide_df([
                    ("cavity_size", "Dimensione cavità"),
                    ("wall_thickness", "Spessore pareti"),
                    ("ferrule", "Ferrule"),
                    ("pulp_proximity", "Prossimità pulpare"),
                ]), use_container_width=True, hide_index=True)
            s1, s2, s3 = st.columns(3)
            with s1:
                residual_walls = st.slider("Numero di pareti residue", 0, 4, 2)
                marginal_ridges = st.selectbox("Creste marginali residue", [0, 1, 2], index=1)
                cusp_loss = st.selectbox("Perdita di cuspide", ["No", "Sì"])
            with s2:
                involved_cusps = st.selectbox("Numero di cuspidi coinvolte", [0, 1, 2, 3, 4], index=1)
                ferrule = st.selectbox("Ferrule", ["Presente", "Parziale", "Assente"], format_func=lambda v: option_label("ferrule", v))
                cavity_size = st.selectbox("Dimensione cavità", ["Piccola", "Media", "Ampia"], format_func=lambda v: option_label("cavity_size", v), help="Range orientativi: adattare alla morfologia del dente e alla visibilità clinica.")
            with s3:
                coronal_tissue = st.selectbox("% tessuto coronale residuo", [">75%", "50-75%", "25-50%", "<25%"])
                wall_thickness = st.selectbox("Spessore pareti residue", ["Adeguato", "Sottile", "Molto sottile"], format_func=lambda v: option_label("wall_thickness", v), help="Valore stimato clinicamente/radiograficamente nel punto più critico della parete o cuspide residua.")
                crack = st.selectbox("Crack sospetta/presente", ["No", "Sì"])
                pulp_proximity = st.selectbox("Prossimità pulpare", ["Bassa", "Media", "Alta"], format_func=lambda v: option_label("pulp_proximity", v))

            st.subheader("3. Fattori Biologici e Contestuali")
            with st.expander("Guida rapida ai range biologico-adesivi", expanded=False):
                st.dataframe(option_guide_df([
                    ("caries_risk", "Rischio carie"),
                    ("plaque_control", "Controllo placca"),
                    ("margin", "Margine"),
                    ("compliance", "Compliance"),
                    ("esthetic_demand", "Richiesta estetica"),
                ]), use_container_width=True, hide_index=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                caries_risk = st.selectbox("Rischio carie", ["Basso", "Medio", "Alto"], format_func=lambda v: option_label("caries_risk", v))
                plaque_control = st.selectbox("Controllo di placca", ["Buono", "Medio", "Scarso"], format_func=lambda v: option_label("plaque_control", v))
                xerostomia = st.selectbox("Xerostomia", ["No", "Sì"])
            with b2:
                isolation = st.selectbox("Isolamento con diga", ["Facile", "Difficile", "Impossibile"])
                margin = st.selectbox("Posizione del margine", ["Sovragengivale", "Juxtagengivale", "Subgengivale"], format_func=lambda v: option_label("margin", v))
                periodontal_support = st.selectbox("Supporto parodontale", ["Buono", "Ridotto"])
            with b3:
                compliance = st.selectbox("Compliance del paziente", ["Alta", "Media", "Bassa"], format_func=lambda v: option_label("compliance", v))
                adhesive_context = st.selectbox("Contesto adesivo", ["Favorevole", "Intermedio", "Sfavorevole"])
                esthetic_demand = st.selectbox("Richiesta estetica", ["Bassa", "Media", "Alta"], format_func=lambda v: option_label("esthetic_demand", v))

            st.subheader("4. Fattori Funzionali e Occlusali")
            st.caption("Usura, parafunzione e carico modulano il peso di resistenza, tenacità e comportamento all’usura della classe materiale.")
            with st.expander("Guida rapida ai range funzionali", expanded=False):
                st.dataframe(option_guide_df([
                    ("parafunction_severity", "Severità parafunzione"),
                    ("occlusal_load", "Carico occlusale"),
                    ("tooth_wear", "Usura dentale"),
                ]), use_container_width=True, hide_index=True)
            f1, f2, f3 = st.columns(3)
            with f1:
                bruxism = st.selectbox("Parafunzione", ["Assente", "Sospetta", "Confermata"])
                parafunction_severity = st.selectbox("Severità parafunzione", ["Assente", "Lieve", "Moderata", "Severa"], format_func=lambda v: option_label("parafunction_severity", v))
                occlusal_load = st.selectbox("Carico occlusale", ["Basso", "Medio", "Alto"], format_func=lambda v: option_label("occlusal_load", v))
            with f2:
                eccentric_contacts = st.selectbox("Contatti eccentrici", ["Assenti", "Presenti"])
                antagonist = st.selectbox("Antagonista", ["Naturale", "Restauro", "Protesico"])
            with f3:
                tooth_wear = st.selectbox("Livello di usura dentale (BEWE)", ["Assente", "Lieve", "Moderata", "Severa"], format_func=lambda v: option_label("tooth_wear", v), help="Screening orientativo ispirato al BEWE: serve a modulare lo stress funzionale/materiale, non a fare diagnosi definitiva.")

            st.subheader("5. Fattori Operativi e di Workflow")
            with st.expander("Guida rapida ai vincoli di workflow", expanded=False):
                st.dataframe(option_guide_df([
                    ("budget_level", "Budget"),
                ]), use_container_width=True, hide_index=True)
            w1, w2, w3 = st.columns(3)
            with w1:
                cadcam_available = st.selectbox("CAD/CAM disponibile", ["No", "Sì"])
                indirect_acceptance = st.selectbox("Accettazione workflow indiretto", ["Sì", "No", "Incerta"])
            with w2:
                max_sessions = st.selectbox("Numero massimo sedute accettabili", ["1", "2", "3+"])
                budget_level = st.selectbox("Vincolo economico", ["Basso", "Medio", "Alto"], format_func=lambda v: option_label("budget_level", v))
            with w3:
                workflow_preference = st.selectbox("Workflow preferito", ["Chairside", "Laboratorio", "Indifferente"])

            submitted = st.form_submit_button("Ottieni raccomandazione")
        st.markdown("</div>", unsafe_allow_html=True)

    case = {
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "tooth_number": tooth_number,
        "tooth_group": tooth_group,
        "clinical_sector": clinical_sector,
        "arch": arch,
        "black_class": black_class,
        "vitality": vitality,
        "endo_treated": endo_treated,
        "clinical_priority": clinical_priority,
        "residual_walls": residual_walls,
        "marginal_ridges": marginal_ridges,
        "cusp_loss": cusp_loss,
        "involved_cusps": involved_cusps,
        "ferrule": ferrule,
        "cavity_size": cavity_size,
        "coronal_tissue": coronal_tissue,
        "wall_thickness": wall_thickness,
        "crack": crack,
        "pulp_proximity": pulp_proximity,
        "caries_risk": caries_risk,
        "plaque_control": plaque_control,
        "xerostomia": xerostomia,
        "isolation": isolation,
        "margin": margin,
        "periodontal_support": periodontal_support,
        "compliance": compliance,
        "adhesive_context": adhesive_context,
        "esthetic_demand": esthetic_demand,
        "bruxism": bruxism,
        "parafunction_severity": parafunction_severity,
        "occlusal_load": occlusal_load,
        "eccentric_contacts": eccentric_contacts,
        "antagonist": antagonist,
        "tooth_wear": tooth_wear,
        "cadcam_available": cadcam_available,
        "indirect_acceptance": indirect_acceptance,
        "max_sessions": max_sessions,
        "budget_level": budget_level,
        "workflow_preference": workflow_preference,
    }

    if submitted:
        st.session_state["last_material_case"] = case
    elif "last_material_case" in st.session_state:
        case = st.session_state["last_material_case"]
    else:
        return

    idx = compute_case_indices(case)
    interpretation = interpret_indices(idx)
    feasibility = assess_material_case_feasibility(case, idx)
    if not feasibility['is_actionable']:
        st.markdown("<div class='app-tag'>Material-first decision support</div>", unsafe_allow_html=True)
        st.markdown("## Nessuna classe materiale raccomandata")
        st.markdown(
            f"<div class='no-material-card'><div class='no-material-title'>{feasibility['title']}</div>"
            f"<div class='gate-copy'>{feasibility['message']}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("### Motivi di esclusione")
        for reason in feasibility['reasons']:
            st.markdown(f"- **{reason}**")
        st.info("In questo scenario il sito non forza una scelta di materiale: prima serve rendere il caso restaurativamente/operativamente gestibile.")
        with st.expander("Index calcolati sul caso", expanded=False):
            st.dataframe(pd.DataFrame([
                {"Index": "SSI", "Valore": round(idx.ssi * 100, 1), "Ruolo": "struttura"},
                {"Index": "BRI", "Valore": round(idx.bri * 100, 1), "Ruolo": "biologia/adesione"},
                {"Index": "FSI", "Valore": round(idx.fsi * 100, 1), "Ruolo": "funzione"},
                {"Index": "EDI", "Valore": round(idx.edi * 100, 1), "Ruolo": "estetica"},
                {"Index": "WCI", "Valore": round(idx.wci * 100, 1), "Ruolo": "workflow"},
            ]), use_container_width=True, hide_index=True)
        return
    rest_df, ranked = rank_materials(case, idx, materials_df, sources_df)
    ranked = add_material_decision_metadata(ranked)
    top_material = ranked.iloc[0]
    top_rest = material_aligned_restoration_context(rest_df, top_material)

    rest_icon, rest_css, rest_status = semaforo(float(top_rest["score"]))
    mat_icon, mat_css, mat_status = semaforo(float(top_material["final_score"]))

    st.markdown("<div class='app-tag'>Material-first decision support</div>", unsafe_allow_html=True)
    st.markdown("## Classe materiale consigliata")
    class_examples_list = class_examples(ranked, top_material["primary_class_name"], n=3)
    class_note = safe_text(top_material.get("class_note", ""))
    material_mode = "Diretto" if top_material["direct_or_indirect"] == "Direct" else "Indiretto"
    confidence_label = str(top_material.get("confidence_label", "Intermedia"))
    confidence_message = safe_text(top_material.get("confidence_message", ""))
    near_alt = near_equivalent_material(ranked, threshold=2.0)
    alert_items = material_risk_alerts(case, idx, top_material, ranked)
    compact_alerts = compact_alert_items(alert_items, n=3)

    hero_left, hero_right = st.columns([1.18, 0.82])
    with hero_left:
        st.markdown("<div class='result-summary-card'>", unsafe_allow_html=True)
        st.markdown("<div class='result-kicker'>Decisione principale: materiale</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-main-title'><span>{top_material['primary_class_name']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<span class='confidence-badge {confidence_css(confidence_label)}'>Confidence: {confidence_label}</span>", unsafe_allow_html=True)
        st.markdown(chips_html(material_keyword_chips(case, idx, top_material, ranked)), unsafe_allow_html=True)
        st.markdown(assisted_recommendation_html(case, idx, top_material, ranked), unsafe_allow_html=True)
        st.markdown(recommendation_tiles_html(case, idx, top_material, ranked), unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0.55rem'><strong style='font-size:0.9rem;'>Esempi commerciali secondari</strong><br/>" + ''.join([f"<span class='material-chip'>{x}</span>" for x in class_examples_list]) + "</div>", unsafe_allow_html=True)
        if near_alt is not None:
            alt_row, alt_gap = near_alt
            st.markdown(
                f"<div class='near-alt-box'><strong>Alternativa vicina:</strong> {alt_row['primary_class_name']} "
                f"· gap {alt_gap:.1f}</div>",
                unsafe_allow_html=True,
            )
        with st.expander("Dettagli clinici della decisione", expanded=False):
            st.markdown(f"**Contesto:** {patient_context_note(case)}")
            if class_note:
                st.markdown(f"**Nota di classe:** {class_note}")
            if confidence_message:
                st.markdown(f"**Confidence:** {confidence_message}")
        st.markdown("</div>", unsafe_allow_html=True)
    with hero_right:
        st.markdown("<div class='premium-panel' style='min-height:100%'>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel-title'>Prima di confermare</div>", unsafe_allow_html=True)
        st.markdown("<div class='alert-chip-row'>" + ''.join([f"<span class='alert-chip'>{compact_alert_keyword(x)}</span>" for x in compact_alerts]) + "</div>", unsafe_allow_html=True)
        st.markdown(clinician_support_note_html(), unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'><strong>Envelope secondario:</strong> {top_rest['restoration']}</div>", unsafe_allow_html=True)
        with st.expander("Metodo e base dati", expanded=False):
            st.markdown("<strong>Pesi principali</strong>", unsafe_allow_html=True)
            st.markdown(
                "<div class='weight-grid'>"
                f"<div class='weight-tile'><div class='weight-value'>{float(top_material.get('weight_mechanical', 0)):.0f}%</div><div class='weight-label'>Meccanica</div></div>"
                f"<div class='weight-tile'><div class='weight-value'>{float(top_material.get('weight_biology', 0)):.0f}%</div><div class='weight-label'>Adesione</div></div>"
                f"<div class='weight-tile'><div class='weight-value'>{float(top_material.get('weight_esthetic', 0)):.0f}%</div><div class='weight-label'>Estetica</div></div>"
                f"<div class='weight-tile'><div class='weight-value'>{float(top_material.get('weight_workflow', 0)):.0f}%</div><div class='weight-label'>Workflow</div></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Base dati:** {int(top_material.get('class_product_count', 1))} materiali · evidenza {float(top_material.get('evidence_score', 0)):.0f}% · completezza {float(top_material.get('quantitative_completeness_pct', 0)):.0f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='premium-metric'><div class='premium-metric-label'>Coerenza</div><div class='premium-metric-value'>{top_material['final_score']:.1f}</div><div class='premium-metric-note'>Fit globale della classe</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='premium-metric'><div class='premium-metric-label'>Predicibilità</div><div class='premium-metric-value'>{top_material['pss']:.1f}%</div><div class='premium-metric-note'>Sul profilo inserito</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='premium-metric'><div class='premium-metric-label'>Tipo</div><div class='premium-metric-value' style='font-size:1.38rem'>{material_mode}</div><div class='premium-metric-note'>Classe materiale</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='premium-metric'><div class='premium-metric-label'>Confidence</div><div class='premium-metric-value' style='font-size:1.18rem'>{confidence_label}</div><div class='premium-metric-note'>{confidence_short_message(confidence_label)}</div></div>", unsafe_allow_html=True)

    report_text = build_report(case, idx, top_rest, top_material, ranked, interpretation)
    st.download_button(
        "Esporta scheda decisionale",
        data=report_text,
        file_name=f"restorative_ai_report_{case['tooth_number']}.md",
        mime="text/markdown",
        use_container_width=False,
    )

    st.markdown("### Profilo materiale")
    st.dataframe(material_profile_table(top_material), use_container_width=True, hide_index=True)
    with st.expander("Indicazioni, limiti e spiegazione completa", expanded=False):
        indications, limits = material_profile_bullets(top_material)
        left_i, right_i = st.columns(2)
        with left_i:
            st.markdown("**Indicazioni**")
            st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in indications]) + "</ul>", unsafe_allow_html=True)
        with right_i:
            st.markdown("**Limiti**")
            st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in limits]) + "</ul>", unsafe_allow_html=True)
        st.markdown("**Razionale sintetico**")
        st.markdown(concise_material_summary(case, idx, top_material))
        st.markdown(material_decision_explanation(case, idx, top_material))

    with st.expander("Metodo, pesi e affidabilità del database", expanded=False):
        st.markdown("**Pesi principali usati nello score**")
        st.dataframe(compact_weight_breakdown(top_material), use_container_width=True, hide_index=True)
        st.markdown("**Dettaglio tecnico**")
        st.dataframe(material_weight_breakdown(top_material), use_container_width=True, hide_index=True)
        st.dataframe(database_weight_breakdown(top_material), use_container_width=True, hide_index=True)
        st.markdown("**Affidabilità database**")
        st.dataframe(compact_database_reliability(top_material), use_container_width=True, hide_index=True)
        st.dataframe(database_reliability_table(top_material), use_container_width=True, hide_index=True)
        st.caption("La raccomandazione resta material-first: i dati del caso guidano lo score, il database ne calibra la classe e il restauro resta secondario.")

    with st.expander("Alert clinici completi", expanded=False):
        st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in alert_items]) + "</ul>", unsafe_allow_html=True)

    alt_guidance = alternative_material_guidance(case, idx, ranked)
    if not alt_guidance.empty:
        with st.expander("Prima classe vs alternativa materiale", expanded=False):
            st.dataframe(alt_guidance, use_container_width=True, hide_index=True)

    radar_col, insight_col = st.columns([1.05, 0.95])
    with radar_col:
        st.markdown("<div class='section-kicker'>Profilo del caso</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-shell'>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel-title'>Radar chart clinico</div>", unsafe_allow_html=True)
        radar_values = {
            "SSI": idx.ssi * 100,
            "BRI": idx.bri * 100,
            "FSI": idx.fsi * 100,
            "EDI": idx.edi * 100,
            "WCI": idx.wci * 100,
        }
        labels = list(radar_values.keys()) + [list(radar_values.keys())[0]]
        values = list(radar_values.values()) + [list(radar_values.values())[0]]
        threshold = [60, 60, 60, 60, 60, 60]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=threshold, theta=labels, mode="lines", name="Soglia", line=dict(dash="dash")))
        fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill="toself", name="Caso clinico"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=410, margin=dict(l=30,r=30,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div class='radar-note'><strong>{interpretation['headline']}.</strong></div>", unsafe_allow_html=True)
        with st.expander("Leggi nota radar completa", expanded=False):
            st.markdown(interpretation['radar_note'])
        st.markdown("</div>", unsafe_allow_html=True)
    with insight_col:
        st.markdown("<div class='section-kicker'>Sintesi operativa</div>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel-title'>Parole chiave</div>", unsafe_allow_html=True)
        st.markdown(chips_html(material_keyword_chips(case, idx, top_material, ranked)[:6]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("Sintesi operativa completa", expanded=False):
            st.markdown("**Perché questa classe materiale**")
            st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in class_why(case, idx, top_material)[:3]]) + "</ul>", unsafe_allow_html=True)
            st.markdown("**Envelope restaurativo compatibile**")
            st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in restoration_reasoning(case, idx, top_rest)[:2]]) + "</ul>", unsafe_allow_html=True)
            st.markdown("**Cautela principale**")
            st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in top_material_limitations(case, top_material, ranked)[:2]]) + "</ul>", unsafe_allow_html=True)

    with st.expander("Clinical take-home message", expanded=False):
        st.markdown(clinical_take_home(case, idx, top_rest, top_material, interpretation))

    with st.expander("Influenza degli index", expanded=False):
        idx_rows = pd.DataFrame([
            {"Index": "SSI", "Valore": round(idx.ssi * 100, 1), "Ruolo": "Struttura", "Impatto clinico": "Alto" if idx.ssi >= 0.65 else "Medio" if idx.ssi >= 0.40 else "Basso"},
            {"Index": "BRI", "Valore": round(idx.bri * 100, 1), "Ruolo": "Biologia/adesione", "Impatto clinico": "Alto" if idx.bri >= 0.65 else "Medio" if idx.bri >= 0.40 else "Basso"},
            {"Index": "FSI", "Valore": round(idx.fsi * 100, 1), "Ruolo": "Funzione", "Impatto clinico": "Alto" if idx.fsi >= 0.65 else "Medio" if idx.fsi >= 0.40 else "Basso"},
            {"Index": "EDI", "Valore": round(idx.edi * 100, 1), "Ruolo": "Estetica", "Impatto clinico": "Alto" if idx.edi >= 0.65 else "Medio" if idx.edi >= 0.40 else "Basso"},
            {"Index": "WCI", "Valore": round(idx.wci * 100, 1), "Ruolo": "Workflow", "Impatto clinico": "Alto" if idx.wci >= 0.65 else "Medio" if idx.wci >= 0.40 else "Basso"},
            {"Index": "Settore", "Valore": case.get("clinical_sector", "N/D"), "Ruolo": "Contesto", "Impatto clinico": "Estetico" if case.get("clinical_sector") == "Anteriore" else "Funzionale"},
            {"Index": "Sesso", "Valore": case.get("patient_sex", "Non specificato"), "Ruolo": "Micro-contesto", "Impatto clinico": "Debole/condizionato"},
        ])
        st.dataframe(idx_rows, use_container_width=True, hide_index=True)
        for bullet in interpretation["bullets"]:
            st.markdown(f"- {bullet}")

    st.markdown("### Ranking materiali")
    st.dataframe(compact_class_table(ranked), use_container_width=True, hide_index=True)
    with st.expander("Confronto completo con motivi clinici", expanded=False):
        class_table = class_comparison_table_with_reasons(ranked, case, idx, top_material)
        st.dataframe(class_table, use_container_width=True, hide_index=True)
        st.caption("Verde = top · Giallo = possibile · Rosso = non indicato. Motivo nella tabella.")

    with st.expander("Envelope restaurativi compatibili (secondari)", expanded=False):
        view = rest_df.head(5).copy()
        view["Semaforo"] = view["score"].apply(lambda x: semaforo(float(x))[0])
        view["Esito"] = view["score"].apply(lambda x: semaforo(float(x))[2])
        view["Motivo clinico"] = view.apply(lambda r: restoration_alternative_reason(case, idx, top_rest["restoration"], r["restoration"], float(r["score"])), axis=1)
        st.dataframe(view[["Semaforo","Esito","restoration","score","Motivo clinico"]].rename(columns={"restoration":"Envelope restaurativo","score":"Compatibilità"}), use_container_width=True, hide_index=True)

        s_cols = st.columns(min(5, len(rest_df)))
        for i, (_, row) in enumerate(view.iterrows()):
            icon, css, status = semaforo(float(row["score"]))
            reason = row["Motivo clinico"]
            with s_cols[i]:
                st.markdown(f"<div class='{css} sem-card'><strong>{icon} {row['restoration']}</strong><br/>{status}<br/><span class='small-muted'>{reason}</span><br/><strong>Compatibilità:</strong> {row['score']:.1f}</div>", unsafe_allow_html=True)

    with st.expander("Classi materiali ed esempi commerciali", expanded=False):
        table = ranked.head(6).copy()
        table["Semaforo"] = table["final_score"].apply(lambda x: semaforo(float(x))[0])
        table["Esito"] = table["final_score"].apply(lambda x: semaforo(float(x))[2])
        table["Esempi commerciali"] = table.apply(lambda r: ", ".join(row_examples(r, n=3)), axis=1)
        table["Nota di classe"] = table["class_note"].fillna("") if "class_note" in table.columns else ""
        table["Motivo clinico"] = table.apply(lambda r: material_class_reason(case, idx, top_material, r), axis=1)
        table = table[["Semaforo", "Esito", "primary_class_name", "Esempi commerciali", "direct_or_indirect", "final_score", "pss", "confidence_label", "score_gap_to_next", "Nota di classe", "Motivo clinico"]]
        table.columns = ["Semaforo", "Esito", "Classe", "Esempi commerciali", "Tipo", "Score", "PSS", "Confidence", "Gap vs successiva", "Nota di classe", "Motivo clinico"]
        st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Fonti della classe/materiale top", expanded=False):
        source_block(top_material["source_urls"])


def render_validation_page():
    st.markdown("<div class='app-tag'>Material-first validation mode</div>", unsafe_allow_html=True)
    st.markdown("## Validazione sintetica del motore materiale")
    st.markdown("200 casi sintetici plausibili · focus su classi materiali · restauro solo come envelope secondario.")
    with st.spinner("Esecuzione validazione materiale su 200 casi..."):
        df, summary = run_material_validation(materials_df, sources_df, n=200, seed=42)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Casi actionable", summary['actionable_cases']),
        ("Red flag", summary['red_flags']),
        ("Yellow flag", summary['yellow_flags']),
        ("Gap <2", summary['gap_under_2']),
        ("Copertura input", f"{summary['coverage_covered_levels']}/{summary['coverage_total_levels']}"),
    ]
    for col, (label, value) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(f"<div class='validation-kpi'><div class='validation-kpi-label'>{label}</div><div class='validation-kpi-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("### Lettura rapida")
    st.write(
        f"Score medio top: **{summary['mean_top_score']}** · Gap medio: **{summary['mean_gap']}** · "
        f"Diretti: **{summary['direct_pct']}%** · Indiretti: **{summary['indirect_pct']}%**."
    )
    if summary['red_flags'] == 0:
        st.success("Nessuna red flag clinica grave rilevata nei 200 casi sintetici.")
    else:
        st.warning("Sono presenti red flag: controllare la tabella casi per capire dove il motore forza una classe poco coerente.")
    if summary['gap_under_2'] > 0:
        st.info("I casi con gap <2 non sono automaticamente errori: indicano zone di equivalenza tra classi materiali. Il sito ora le comunica come incertezza controllata.")
    if summary.get('coverage_missing_levels', 0) == 0:
        st.success("Copertura input completa: tutti i livelli categoriali previsti sono comparsi almeno una volta nei 200 casi actionable.")
    else:
        st.warning(f"Copertura input non completa: {summary['coverage_missing_levels']} livelli non coperti.")
    with st.expander("Copertura degli input testati", expanded=False):
        st.dataframe(summary['coverage_table'], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Distribuzione classi top")
        st.dataframe(df['top_class'].value_counts().rename_axis('Classe').reset_index(name='Casi'), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### Confidence")
        st.dataframe(df['confidence'].value_counts().rename_axis('Confidence').reset_index(name='Casi'), use_container_width=True, hide_index=True)

    st.markdown("### Casi validati")
    display_cols = [
        'case_id', 'archetype', 'sector', 'tooth_group', 'actionable', 'top_class', 'top_type', 'top_score',
        'second_class', 'second_score', 'gap', 'confidence', 'SSI', 'BRI', 'FSI', 'EDI', 'WCI',
        'red_flags', 'yellow_flags'
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Scarica CSV validazione 200 casi",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='restorative_ai_material_validation_200_cases.csv',
        mime='text/csv',
    )


professional_access_gate()

page_param = st.query_params.get("page", "home")
if page_param == "evaluate":
    current_nav = "Material Decision Assistant"
elif page_param == "validation":
    current_nav = "Validation Mode"
else:
    current_nav = "Home"

nav_options = ["Home", "Material Decision Assistant", "Validation Mode"]
selected_nav = st.radio(
    "Navigazione",
    nav_options,
    index=nav_options.index(current_nav),
    horizontal=True,
    label_visibility="collapsed",
)
if selected_nav != current_nav:
    st.query_params["page"] = "evaluate" if selected_nav == "Material Decision Assistant" else "validation" if selected_nav == "Validation Mode" else "home"
    st.rerun()

if selected_nav == "Home":
    render_home()
elif st.session_state.get("professional_access") == "public":
    public_restricted_page()
elif selected_nav == "Validation Mode":
    render_validation_page()
else:
    render_clinical_page()
