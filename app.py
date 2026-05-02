from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.engine import compute_case_indices, interpret_indices, load_database, rank_materials, summarize_case

st.set_page_config(page_title="Restorative AI", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1380px;}
.ra-hero {background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0f766e 100%); color: white; border-radius: 24px; padding: 1.4rem 1.45rem; margin-bottom: 1rem; box-shadow: 0 12px 28px rgba(15,23,42,0.16);}
.ra-card {background: white; border: 1px solid #e5e7eb; border-radius: 20px; padding: 1rem 1.1rem; margin-bottom: 0.9rem; box-shadow: 0 6px 16px rgba(15,23,42,0.06);}
.feature-card {background: linear-gradient(180deg,#ffffff 0%,#f8fbff 100%); border: 1px solid #e2e8f0; border-radius: 24px; padding: 1.2rem 1.2rem; min-height: 180px; box-shadow: 0 10px 24px rgba(15,23,42,0.07); position:relative; overflow:hidden;}
.feature-card:before {content:''; position:absolute; inset:0 auto auto 0; width:100%; height:4px; background:linear-gradient(90deg,#2563eb,#0f766e);}
.feature-icon {width:52px; height:52px; border-radius:16px; display:flex; align-items:center; justify-content:center; background:#eff6ff; border:1px solid #bfdbfe; font-size:1.55rem; margin-bottom:0.8rem;}
.feature-title {font-size: 1.12rem; font-weight: 700; margin-bottom: 0.45rem; color: #0f172a;}
.feature-copy {color: #475569; line-height: 1.6; font-size: 0.98rem;}
.index-panel {background:linear-gradient(180deg,#f8fbff,#ffffff); border:1px solid #dbeafe; border-radius:22px; padding:1.05rem 1.1rem; box-shadow:0 8px 20px rgba(15,23,42,0.06);}
.index-row {padding:0.7rem 0; border-bottom:1px solid #e5e7eb;}
.index-row:last-child {border-bottom:none;}
.index-name {font-weight:800; color:#0f172a;}
.index-copy {color:#475569; font-size:0.93rem; line-height:1.45;}
.cta-button-wrap {text-align:center; margin-top:1.15rem;}
.output-mini {background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:0.8rem 0.9rem; margin-bottom:0.65rem;}
.result-top {background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%); border: 1px solid #dbeafe; border-radius: 22px; padding: 1rem 1.1rem; box-shadow: 0 6px 16px rgba(15,23,42,0.06);}
.reason-box {background: #f8fafc; border-left: 5px solid #1d4ed8; border-radius: 12px; padding: 0.8rem 0.9rem; margin-bottom: 0.7rem;}
.sem-card {border-radius: 18px; padding: 0.95rem 1rem; margin-bottom: 0.7rem; border: 1px solid transparent;}
.sem-green {background: #f0fdf4; border-color: #86efac;}
.sem-yellow {background: #fffbeb; border-color: #fde68a;}
.sem-red {background: #fef2f2; border-color: #fca5a5;}
.small-muted {color: #475569; font-size: 0.93rem; line-height: 1.5;}
.radar-note {background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:0.9rem 1rem; margin-bottom:0.75rem;}
.idx-chip {display:inline-block; padding:0.3rem 0.65rem; border-radius:999px; background:#eff6ff; border:1px solid #bfdbfe; margin-right:0.35rem; margin-bottom:0.35rem; font-size:0.85rem;}
.section-title {font-size: 2.55rem; text-align: center; font-weight: 800; color: #0f172a; margin-top: 0.8rem;}
.section-subtitle {font-size: 1.07rem; text-align: center; color: #64748b; max-width: 760px; margin: 0 auto 1.3rem auto; line-height: 1.55;}
.cta-wrap {text-align:center; margin: 1.1rem 0 0.5rem 0;}

.app-tag {display:inline-block; padding:0.28rem 0.62rem; border-radius:999px; background:#e0ecff; color:#1d4ed8; border:1px solid #bfdbfe; font-size:0.83rem; font-weight:700; margin-bottom:0.35rem;}
.result-shell {background: linear-gradient(180deg,#ffffff 0%,#f8fbff 100%); border:1px solid #dbeafe; border-radius:26px; padding:1.15rem 1.2rem; box-shadow:0 14px 30px rgba(15,23,42,0.08);}
.result-image-card {background:white; border:1px solid #e5e7eb; border-radius:24px; padding:0.8rem; box-shadow:0 10px 24px rgba(15,23,42,0.07);} 
.result-summary-card {background:linear-gradient(180deg,#fcfdff,#f8fbff); border:1px solid #dbeafe; border-radius:24px; padding:1.05rem 1.1rem; min-height:100%; box-shadow:0 10px 24px rgba(15,23,42,0.07);} 
.result-kicker {font-size:0.82rem; color:#2563eb; font-weight:800; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.35rem;}
.result-main-title {font-size:2rem; color:#0f172a; font-weight:800; margin:0 0 0.35rem 0; line-height:1.15;} 
.result-subtitle {color:#475569; font-size:0.98rem; line-height:1.55; margin-bottom:0.7rem;} 
.premium-metric {background:white; border:1px solid #e5e7eb; border-radius:20px; padding:0.95rem 1rem; box-shadow:0 8px 18px rgba(15,23,42,0.06); min-height:98px;} 
.premium-metric-label {font-size:0.86rem; color:#64748b; margin-bottom:0.4rem;} 
.premium-metric-value {font-size:1.72rem; color:#0f172a; font-weight:800; line-height:1.1;} 
.premium-metric-note {font-size:0.84rem; color:#475569; margin-top:0.35rem;} 
.premium-panel {background:white; border:1px solid #e5e7eb; border-radius:22px; padding:1rem 1.05rem; box-shadow:0 10px 24px rgba(15,23,42,0.06); height:100%;}
.premium-panel-title {font-size:1.02rem; font-weight:800; color:#0f172a; margin-bottom:0.65rem;} 
.premium-list {margin:0; padding-left:1.1rem; color:#334155;} .premium-list li {margin-bottom:0.48rem; line-height:1.45;} 
.chart-shell {background:white; border:1px solid #e5e7eb; border-radius:24px; padding:1rem 1rem 0.7rem 1rem; box-shadow:0 10px 24px rgba(15,23,42,0.06);} 
.section-kicker {font-size:0.84rem; color:#2563eb; font-weight:800; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.2rem;} 
.expander-label {font-weight:700; color:#0f172a;} 
.material-chip {display:inline-block; padding:0.28rem 0.55rem; background:#f8fafc; border:1px solid #e2e8f0; border-radius:999px; font-size:0.82rem; color:#334155; margin-right:0.35rem; margin-bottom:0.35rem;} 
.home-grid-shell {background:linear-gradient(180deg,#ffffff,#f8fbff); border:1px solid #e2e8f0; border-radius:28px; padding:1rem 1rem 0.4rem 1rem; box-shadow:0 12px 28px rgba(15,23,42,0.07);} 
.home-side-shell {background:linear-gradient(180deg,#f8fbff,#ffffff); border:1px solid #dbeafe; border-radius:28px; padding:1.1rem 1.15rem; box-shadow:0 12px 28px rgba(15,23,42,0.07);} 
.home-stat-card {background:white; border:1px solid #e2e8f0; border-radius:18px; padding:0.8rem 0.9rem; box-shadow:0 8px 20px rgba(15,23,42,0.05); margin-bottom:0.7rem;} 
.takeaway-shell {background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 70%,#0f766e 100%); color:white; border-radius:24px; padding:1.05rem 1.15rem; box-shadow:0 16px 34px rgba(15,23,42,0.18);} 
.takeaway-title {font-size:1.04rem; font-weight:800; margin-bottom:0.45rem;} 
.takeaway-copy {font-size:0.97rem; line-height:1.6; opacity:0.97;} 

</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def _load():
    return load_database()

materials_df, props_df, sources_df = _load()


def semaforo(score: float):
    if score >= 82:
        return "🟢", "sem-green", "Raccomandato"
    if score >= 65:
        return "🟡", "sem-yellow", "Possibile con limiti"
    return "🔴", "sem-red", "Non indicato"


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


def class_examples(ranked: pd.DataFrame, class_name: str, n: int = 3):
    subset = ranked[ranked["primary_class_name"] == class_name]["display_name"].dropna().astype(str).drop_duplicates().tolist()
    return subset[:n]


def class_why(case, indices, top_row):
    reasons = []
    if top_row["structural_fit"] >= 70:
        reasons.append("Fit strutturale superiore per quantità di tessuto residuo e richiesta di protezione cuspidale.")
    if top_row["bio_fit"] >= 65:
        reasons.append("Compatibilità adeguata con isolamento, margine e contesto adesivo del caso.")
    if top_row["esthetic_fit"] >= 70:
        reasons.append("Prestazione estetica coerente con la domanda clinica.")
    if top_row["workflow_fit"] >= 65:
        reasons.append("Workflow compatibile con sedute, accettazione dell'indiretto e vincoli operativi.")
    if top_row["evidence_score"] >= 75:
        reasons.append("Base documentale più solida rispetto alle alternative vicine.")
    if not reasons:
        reasons.append("Miglior equilibrio complessivo tra struttura, biologia, funzione e workflow.")
    return reasons[:4]



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
        limits.append("L'opzione top massimizza la robustezza, ma può richiedere attenzione sul compromesso estetico rispetto ad alternative più traslucide.")
    if case["budget_level"] == "Basso" and top_row["direct_or_indirect"] == "Indirect":
        limits.append("Il vincolo economico potrebbe favorire una seconda scelta più accessibile se il paziente non accetta il piano ideale.")
    if top_row["quantitative_completeness_pct"] < 45:
        limits.append("La completezza quantitativa del materiale non è massima: la raccomandazione resta valida, ma con base dati non totalmente piena.")
    if ranked.shape[0] > 1 and (float(top_row["final_score"]) - float(ranked.iloc[1]["final_score"])) < 5:
        limits.append("Il vantaggio sulla seconda opzione è contenuto: le prime alternative restano clinicamente discutibili.")
    if not limits:
        limits.append("Non emergono criticità dominanti: il materiale proposto ha un vantaggio clinico abbastanza netto sulle alternative principali.")
    return limits


def class_comparison_table(ranked: pd.DataFrame) -> pd.DataFrame:
    cols=[]
    dedup = ranked.drop_duplicates(subset=["primary_class_name"], keep="first").head(6)
    for _, row in dedup.iterrows():
        icon, _, status = semaforo(float(row["final_score"]))
        cols.append({
            "Semaforo": icon,
            "Esito": status,
            "Classe": row["primary_class_name"],
            "Esempio": row["display_name"],
            "Score": round(float(row["final_score"]),1),
            "PSS": round(float(row["pss"]),1),
            "Tipo": row["direct_or_indirect"],
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



def source_block(urls):
    if not urls:
        st.write("Nessuna fonte tracciata disponibile per la riga top del database.")
        return
    for url in urls[:3]:
        st.markdown(f"- {url}")


def build_report(case, idx, top_rest, top_material, ranked, interpretation) -> str:
    example_list = class_examples(ranked, top_material["primary_class_name"], n=3)
    alt_classes = ranked[["primary_class_name", "final_score"]].drop_duplicates(subset=["primary_class_name"]).head(5)
    lines = []
    lines.append("# Restorative AI - Scheda decisionale")
    lines.append("")
    lines.append("## Caso")
    lines.append(f"- Dente: {case['tooth_number']} ({case['tooth_group']})")
    lines.append(f"- Vitalità: {case['vitality']}")
    lines.append(f"- Endodonzia: {case['endo_treated']}")
    lines.append(f"- Pareti residue: {case['residual_walls']}")
    lines.append(f"- Cusp loss: {case['cusp_loss']}")
    lines.append(f"- Carico occlusale: {case['occlusal_load']}")
    lines.append("")
    lines.append("## Decisione")
    lines.append(f"- Tipo di restauro: {restoration_type_label(top_rest['restoration'])}")
    lines.append(f"- Restauro consigliato: {top_rest['restoration']}")
    lines.append(f"- Classe materiale consigliata: {top_material['primary_class_name']}")
    lines.append(f"- PSS: {top_material['pss']}%")
    if example_list:
        lines.append(f"- Possibili esempi: {', '.join(example_list)}")
    lines.append("")
    lines.append("## Perché questa scelta")
    for item in restoration_reasoning(case, idx, top_rest)[:3]:
        lines.append(f"- {item}")
    for item in class_why(case, idx, top_material)[:3]:
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
    lines.append(f"Caso guidato principalmente da {interpretation['dominant_axis'].lower()}.")
    lines.append(f"Il restauro più coerente è {top_rest['restoration'].lower()}.")
    lines.append(f"La classe materiale con miglior equilibrio è {top_material['primary_class_name']}.")
    if idx.fsi >= 0.65:
        lines.append("Il carico funzionale impone una scelta più protettiva e meccanicamente affidabile.")
    elif idx.ssi >= 0.65:
        lines.append("La compromissione strutturale sposta la scelta verso copertura e rinforzo.")
    elif idx.bri >= 0.65:
        lines.append("Il contesto biologico/adesivo condiziona fortemente l'esecuzione clinica.")
    else:
        lines.append("Il caso consente una decisione relativamente stabile senza un singolo vincolo dominante critico.")
    return " ".join(lines[:4])


def home_stats(materials_df, sources_df):
    return [
        ("Classi/materiali gestiti", int((materials_df["include_in_v1_database"] == "yes").sum())),
        ("Fonti tracciate", int(len(sources_df))),
        ("Index del motore", 5),
    ]


def render_home():
    st.markdown(
        """
<div class='ra-hero'>
  <div style='font-size:0.84rem; opacity:0.9;'>Clinical decision support • restorative dentistry • database-driven</div>
  <h1 style='margin:0.15rem 0 0.35rem 0;'>🦷 Restorative AI</h1>
  <div style='font-size:1rem; opacity:0.95;'>Sistema di supporto decisionale clinico per l'odontoiatria restaurativa. Integra dati strutturali, biologici, funzionali, estetici e operativi per suggerire il tipo di restauro e la classe di materiale più appropriati.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-title'>Analisi clinica completa</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Una piattaforma pensata per trasformare i dati clinici del caso in una scelta restaurativa più chiara, tracciabile e difendibile.</div>",
        unsafe_allow_html=True,
    )

    main, side = st.columns([1.68, 0.88])
    with main:
        st.markdown("<div class='home-grid-shell'>", unsafe_allow_html=True)
        features = [
            ("🧠", "Analisi basata su AI", "Legge il caso con indici clinici e regole decisionali orientate alla pratica restaurativa."),
            ("🧱", "Selezione della classe materiale", "Propone prima la classe più adatta e solo dopo gli esempi commerciali coerenti."),
            ("📊", "Confronto semaforico", "Mostra subito ciò che è consigliato, borderline o sfavorevole nel caso specifico."),
            ("🛡️", "Valutazione del rischio", "Integra rischio biologico, meccanico e funzionale nella scelta finale."),
            ("🩺", "Lettura del caso completo", "Considera struttura residua, cuspidi, carico, adesione, estetica e workflow."),
            ("✅", "Output clinico diretto", "Restituisce una decisione sintetica, utile e difendibile al riunito."),
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
            "Il motore usa cinque indici clinici — SSI, BRI, FSI, EDI e WCI — per stimare la soluzione restaurativa più coerente e poi classifica i materiali del database in base a fit strutturale, biologico, estetico, workflow e affidabilità documentale."
        )

    col_form, col_help = st.columns([1.55, 0.75])

    with col_help:
        st.markdown("<div class='ra-card'>", unsafe_allow_html=True)
        st.subheader("Do your better Restoration")
        st.markdown("<div class='small-muted'>Inserisci i dati del caso. Il motore produrrà: tipo di restauro, classe di materiale, top materiali e confronto con alternative.</div>", unsafe_allow_html=True)
        st.divider()
        st.subheader("Quick guide agli index")
        st.markdown("<div class='small-muted'><strong>SSI</strong> struttura • <strong>BRI</strong> biologia • <strong>FSI</strong> funzione • <strong>EDI</strong> estetica • <strong>WCI</strong> workflow</div>", unsafe_allow_html=True)
        st.divider()
        st.metric("Materiali pubblicabili", int((materials_df["include_in_v1_database"] == "yes").sum()))
        st.metric("Fonti tracciate", len(sources_df))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_form:
        st.markdown("<div class='ra-card'>", unsafe_allow_html=True)
        with st.form("clinical_case"):
            st.subheader("1. Paziente e Dente")
            c1, c2, c3 = st.columns(3)
            with c1:
                patient_age = st.number_input("Età paziente", min_value=10, max_value=100, value=25, step=1)
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
                black_class = st.selectbox("Classe di Black", ["I", "II", "III", "IV", "V", "VI"])
                vitality = st.selectbox("Vitalità", ["Vitale", "Non vitale"])
            with c3:
                endo_treated = st.selectbox("Trattamento endodontico", ["No", "Sì"])
                clinical_priority = st.selectbox("Priorità clinica", ["Conservatività", "Durata", "Estetica", "Rapidità"])

            second_digit = tooth_number[1]
            tooth_group = "Anteriore" if second_digit in ["1", "2", "3"] else "Premolare" if second_digit in ["4", "5"] else "Molare"
            arch = "Mascellare" if tooth_number.startswith(("1", "2")) else "Mandibolare"

            st.subheader("2. Fattori Strutturali")
            s1, s2, s3 = st.columns(3)
            with s1:
                residual_walls = st.slider("Numero di pareti residue", 0, 4, 2)
                marginal_ridges = st.selectbox("Creste marginali residue", [0, 1, 2], index=1)
                cusp_loss = st.selectbox("Perdita di cuspide", ["No", "Sì"])
            with s2:
                involved_cusps = st.selectbox("Numero di cuspidi coinvolte", [0, 1, 2, 3, 4], index=1)
                ferrule = st.selectbox("Ferrule", ["Presente", "Parziale", "Assente"])
                cavity_size = st.selectbox("Dimensione cavità", ["Piccola", "Media", "Ampia"])
            with s3:
                coronal_tissue = st.selectbox("% tessuto coronale residuo", [">75%", "50-75%", "25-50%", "<25%"])
                wall_thickness = st.selectbox("Spessore pareti residue", ["Adeguato", "Sottile", "Molto sottile"])
                crack = st.selectbox("Crack sospetta/presente", ["No", "Sì"])
                pulp_proximity = st.selectbox("Prossimità pulpare", ["Bassa", "Media", "Alta"])

            st.subheader("3. Fattori Biologici e Contestuali")
            b1, b2, b3 = st.columns(3)
            with b1:
                caries_risk = st.selectbox("Rischio carie", ["Basso", "Medio", "Alto"])
                plaque_control = st.selectbox("Controllo di placca", ["Buono", "Medio", "Scarso"])
                xerostomia = st.selectbox("Xerostomia", ["No", "Sì"])
            with b2:
                isolation = st.selectbox("Isolamento con diga", ["Facile", "Difficile", "Impossibile"])
                margin = st.selectbox("Posizione del margine", ["Sovragengivale", "Juxtagengivale", "Subgengivale"])
                periodontal_support = st.selectbox("Supporto parodontale", ["Buono", "Ridotto"])
            with b3:
                compliance = st.selectbox("Compliance del paziente", ["Alta", "Media", "Bassa"])
                adhesive_context = st.selectbox("Contesto adesivo", ["Favorevole", "Intermedio", "Sfavorevole"])
                esthetic_demand = st.selectbox("Richiesta estetica", ["Bassa", "Media", "Alta"])

            st.subheader("4. Fattori Funzionali e Occlusali")
            f1, f2, f3 = st.columns(3)
            with f1:
                bruxism = st.selectbox("Parafunzione", ["Assente", "Sospetta", "Confermata"])
                parafunction_severity = st.selectbox("Severità parafunzione", ["Assente", "Lieve", "Moderata", "Severa"])
                occlusal_load = st.selectbox("Carico occlusale", ["Basso", "Medio", "Alto"])
            with f2:
                eccentric_contacts = st.selectbox("Contatti eccentrici", ["Assenti", "Presenti"])
                antagonist = st.selectbox("Antagonista", ["Naturale", "Restauro", "Protesico"])
            with f3:
                tooth_wear = st.selectbox("Usura dentale generale", ["Assente", "Lieve", "Moderata", "Severa"])

            st.subheader("5. Fattori Operativi e di Workflow")
            w1, w2, w3 = st.columns(3)
            with w1:
                cadcam_available = st.selectbox("CAD/CAM disponibile", ["No", "Sì"])
                indirect_acceptance = st.selectbox("Accettazione restauro indiretto", ["Sì", "No", "Incerta"])
            with w2:
                max_sessions = st.selectbox("Numero massimo sedute accettabili", ["1", "2", "3+"])
                budget_level = st.selectbox("Vincolo economico", ["Basso", "Medio", "Alto"])
            with w3:
                workflow_preference = st.selectbox("Workflow preferito", ["Chairside", "Laboratorio", "Indifferente"])

            submitted = st.form_submit_button("Ottieni raccomandazione")
        st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return

    case = {
        "patient_age": patient_age,
        "tooth_number": tooth_number,
        "tooth_group": tooth_group,
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

    idx = compute_case_indices(case)
    interpretation = interpret_indices(idx)
    rest_df, ranked = rank_materials(case, idx, materials_df, sources_df)
    top_rest = rest_df.iloc[0]
    top_material = ranked.iloc[0]

    rest_icon, rest_css, rest_status = semaforo(float(top_rest["score"]))
    mat_icon, mat_css, mat_status = semaforo(float(top_material["final_score"]))

    st.markdown("<div class='app-tag'>Report decisionale</div>", unsafe_allow_html=True)
    st.markdown("## Risultato clinico")
    class_examples_list = class_examples(ranked, top_material["primary_class_name"], n=3)

    hero_left, hero_right = st.columns([0.92, 1.08])
    with hero_left:
        st.markdown("<div class='result-image-card'>", unsafe_allow_html=True)
        st.image(str(restoration_image_path(top_rest["restoration"])), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with hero_right:
        st.markdown("<div class='result-summary-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-kicker'>{restoration_type_label(top_rest['restoration'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-main-title'>{top_rest['restoration']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-subtitle'><strong>Classe materiale consigliata:</strong> {top_material['primary_class_name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='{rest_css} sem-card'><strong>{rest_icon} {rest_status}</strong><br/>Decisione restaurativa principale per il caso clinico inserito.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='{mat_css} sem-card'><strong>{mat_icon} {mat_status}</strong><br/>Classe materiale con miglior equilibrio tra struttura, biologia, funzione, estetica e workflow.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'><strong>Perché questo restauro:</strong> {restoration_brief_description(top_rest['restoration'])}</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0.65rem'><strong style='font-size:0.9rem;'>Possibili esempi nel database</strong><br/>" + ''.join([f"<span class='material-chip'>{x}</span>" for x in class_examples_list]) + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='premium-metric'><div class='premium-metric-label'>Score restauro</div><div class='premium-metric-value'>{top_rest['score']:.1f}</div><div class='premium-metric-note'>Adeguatezza del tipo di restauro</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='premium-metric'><div class='premium-metric-label'>Score classe</div><div class='premium-metric-value'>{top_material['final_score']:.1f}</div><div class='premium-metric-note'>Coerenza globale della classe</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='premium-metric'><div class='premium-metric-label'>PSS</div><div class='premium-metric-value'>{top_material['pss']:.1f}%</div><div class='premium-metric-note'>Predicted Success Score</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='premium-metric'><div class='premium-metric-label'>Index dominante</div><div class='premium-metric-value' style='font-size:1.35rem'>{interpretation['dominant_axis']}</div><div class='premium-metric-note'>Driver principale della decisione</div></div>", unsafe_allow_html=True)

    report_text = build_report(case, idx, top_rest, top_material, ranked, interpretation)
    st.download_button(
        "Esporta scheda decisionale",
        data=report_text,
        file_name=f"restorative_ai_report_{case['tooth_number']}.md",
        mime="text/markdown",
        use_container_width=False,
    )

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
        st.markdown(f"<div class='radar-note'><strong>{interpretation['headline']}.</strong> {interpretation['radar_note']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with insight_col:
        st.markdown("<div class='section-kicker'>Sintesi operativa</div>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel-title'>Scelta del restauro</div>", unsafe_allow_html=True)
        st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in restoration_reasoning(case, idx, top_rest)[:3]]) + "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel-title'>Scelta della classe</div>", unsafe_allow_html=True)
        st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in class_why(case, idx, top_material)[:3]]) + "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='premium-panel-title'>Cautela principale</div>", unsafe_allow_html=True)
        st.markdown("<ul class='premium-list'>" + ''.join([f"<li>{x}</li>" for x in top_material_limitations(case, top_material, ranked)[:2]]) + "</ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='takeaway-shell'><div class='takeaway-title'>Clinical take-home message</div><div class='takeaway-copy'>" + clinical_take_home(case, idx, top_rest, top_material, interpretation) + "</div></div>", unsafe_allow_html=True)

    with st.expander("Influenza degli index", expanded=False):
        idx_rows = pd.DataFrame([
            {"Index": "SSI", "Valore": round(idx.ssi * 100, 1), "Ruolo": "Struttura", "Impatto clinico": "Alto" if idx.ssi >= 0.65 else "Medio" if idx.ssi >= 0.40 else "Basso"},
            {"Index": "BRI", "Valore": round(idx.bri * 100, 1), "Ruolo": "Biologia/adesione", "Impatto clinico": "Alto" if idx.bri >= 0.65 else "Medio" if idx.bri >= 0.40 else "Basso"},
            {"Index": "FSI", "Valore": round(idx.fsi * 100, 1), "Ruolo": "Funzione", "Impatto clinico": "Alto" if idx.fsi >= 0.65 else "Medio" if idx.fsi >= 0.40 else "Basso"},
            {"Index": "EDI", "Valore": round(idx.edi * 100, 1), "Ruolo": "Estetica", "Impatto clinico": "Alto" if idx.edi >= 0.65 else "Medio" if idx.edi >= 0.40 else "Basso"},
            {"Index": "WCI", "Valore": round(idx.wci * 100, 1), "Ruolo": "Workflow", "Impatto clinico": "Alto" if idx.wci >= 0.65 else "Medio" if idx.wci >= 0.40 else "Basso"},
        ])
        st.dataframe(idx_rows, use_container_width=True, hide_index=True)
        for bullet in interpretation["bullets"]:
            st.markdown(f"- {bullet}")

    with st.expander("Confronto classi materiali", expanded=True):
        class_table = class_comparison_table(ranked)
        st.dataframe(class_table, use_container_width=True, hide_index=True)
        st.caption("Verde = scelta raccomandata. Giallo = utilizzabile con limiti. Rosso = poco sensata nel caso inserito.")

    with st.expander("Confronto tipi di restauro", expanded=False):
        s_cols = st.columns(min(5, len(rest_df)))
        for i, (_, row) in enumerate(rest_df.head(5).iterrows()):
            icon, css, status = semaforo(float(row["score"]))
            with s_cols[i]:
                st.markdown(f"<div class='{css} sem-card'><strong>{icon} {row['restoration']}</strong><br/>{status}<br/><span class='small-muted'>{row['rationale']}</span><br/><strong>Score:</strong> {row['score']:.1f}</div>", unsafe_allow_html=True)

    with st.expander("Dettaglio materiali (solo come esempi)", expanded=False):
        table = ranked.head(5).copy()
        table["Semaforo"] = table["final_score"].apply(lambda x: semaforo(float(x))[0])
        table["Esito"] = table["final_score"].apply(lambda x: semaforo(float(x))[2])
        table = table[["Semaforo", "Esito", "primary_class_name", "display_name", "direct_or_indirect", "final_score", "pss"]]
        table.columns = ["Semaforo", "Esito", "Classe", "Esempio", "Tipo", "Score", "PSS"]
        st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Fonti del materiale top", expanded=False):
        source_block(top_material["source_urls"])


page_param = st.query_params.get("page", "home")
current_nav = "Do your better Restoration" if page_param == "evaluate" else "Home"

nav_options = ["Home", "Do your better Restoration"]
selected_nav = st.radio(
    "Navigazione",
    nav_options,
    index=nav_options.index(current_nav),
    horizontal=True,
    label_visibility="collapsed",
)
if selected_nav != current_nav:
    st.query_params["page"] = "evaluate" if selected_nav == "Do your better Restoration" else "home"
    st.rerun()

if selected_nav == "Home":
    render_home()
else:
    render_clinical_page()