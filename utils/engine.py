from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[1] / 'data' / 'materialmatch_site_optimized_database_v1.xlsx'


@dataclass
class CaseIndices:
    ssi: float
    bri: float
    fsi: float
    edi: float
    wci: float

    def as_dict(self) -> Dict[str, float]:
        return {
            'SSI': round(self.ssi, 3),
            'BRI': round(self.bri, 3),
            'FSI': round(self.fsi, 3),
            'EDI': round(self.edi, 3),
            'WCI': round(self.wci, 3),
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def load_database() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    materials = pd.read_excel(DATA_FILE, sheet_name='Site_Materials')
    props = pd.read_excel(DATA_FILE, sheet_name='Property_Observations')
    sources = pd.read_excel(DATA_FILE, sheet_name='Material_Sources')

    props_num = props.dropna(subset=['value_numeric']).copy()
    pivot = props_num.pivot_table(index='material_id', columns='property_name', values='value_numeric', aggfunc='mean').reset_index()
    merged = materials.merge(pivot, on='material_id', how='left')

    merged['verification_score'] = merged['verification_status'].map({
        'verified': 1.0,
        'verified_variant': 0.9,
        'partial_verified': 0.65,
    }).fillna(0.5)
    merged['quantitative_completeness_pct'] = merged['quantitative_completeness_pct'].fillna(0.0)
    merged['evidence_score'] = 0.6 * merged['verification_score'] + 0.4 * (merged['quantitative_completeness_pct'] / 100.0)
    merged['site_readiness_tier'] = merged['site_readiness_tier'].fillna('catalog_only')
    return merged, props, sources


def _yes_no(value: str, yes='Sì') -> float:
    return 1.0 if value == yes else 0.0


def compute_case_indices(case: Dict[str, object]) -> CaseIndices:
    # Structural Severity Index
    walls_component = {4: 0.0, 3: 0.25, 2: 0.55, 1: 0.8, 0: 1.0}[int(case['residual_walls'])]
    ridges_component = {2: 0.0, 1: 0.5, 0: 1.0}[int(case['marginal_ridges'])]
    cusp_component = _yes_no(case['cusp_loss'])
    cusps_involved_component = min(int(case['involved_cusps']) / 4.0, 1.0)
    ferrule_component = {'Presente': 0.0, 'Parziale': 0.5, 'Assente': 1.0}[case['ferrule']]
    cavity_component = {'Piccola': 0.15, 'Media': 0.5, 'Ampia': 0.9}[case['cavity_size']]
    tissue_component = {'>75%': 0.1, '50-75%': 0.35, '25-50%': 0.7, '<25%': 1.0}[case['coronal_tissue']]
    wall_thickness_component = {'Adeguato': 0.15, 'Sottile': 0.6, 'Molto sottile': 0.95}[case['wall_thickness']]
    vitality_component = 0.35 if case['vitality'] == 'Non vitale' else 0.0
    endo_component = 0.8 if case['endo_treated'] == 'Sì' else 0.0

    ssi = _clamp(
        0.18 * endo_component +
        0.16 * walls_component +
        0.10 * ridges_component +
        0.13 * cusp_component +
        0.08 * cusps_involved_component +
        0.10 * ferrule_component +
        0.08 * cavity_component +
        0.10 * tissue_component +
        0.07 * wall_thickness_component +
        0.06 * vitality_component
    )

    # Biological Risk Index
    bri = _clamp(
        0.18 * {'Basso': 0.0, 'Medio': 0.55, 'Alto': 1.0}[case['caries_risk']] +
        0.10 * {'Buono': 0.0, 'Medio': 0.55, 'Scarso': 1.0}[case['plaque_control']] +
        0.06 * _yes_no(case['xerostomia']) +
        0.12 * _yes_no(case['crack']) +
        0.08 * {'Bassa': 0.0, 'Media': 0.5, 'Alta': 1.0}[case['pulp_proximity']] +
        0.18 * {'Facile': 0.0, 'Difficile': 0.55, 'Impossibile': 1.0}[case['isolation']] +
        0.16 * {'Sovragengivale': 0.0, 'Juxtagengivale': 0.45, 'Subgengivale': 1.0}[case['margin']] +
        0.05 * {'Buono': 0.0, 'Ridotto': 1.0}[case['periodontal_support']] +
        0.07 * {'Alta': 0.0, 'Media': 0.5, 'Bassa': 1.0}[case['compliance']] +
        0.10 * {'Favorevole': 0.0, 'Intermedio': 0.55, 'Sfavorevole': 1.0}[case['adhesive_context']]
    )

    # Functional Stress Index
    fsi = _clamp(
        0.18 * {'Assente': 0.0, 'Sospetta': 0.55, 'Confermata': 1.0}[case['bruxism']] +
        0.14 * {'Assente': 0.0, 'Lieve': 0.3, 'Moderata': 0.7, 'Severa': 1.0}[case['parafunction_severity']] +
        0.24 * {'Basso': 0.0, 'Medio': 0.5, 'Alto': 1.0}[case['occlusal_load']] +
        0.10 * {'Assenti': 0.0, 'Presenti': 1.0}[case['eccentric_contacts']] +
        0.10 * {'Naturale': 0.15, 'Restauro': 0.45, 'Protesico': 0.8}[case['antagonist']] +
        0.14 * {'Assente': 0.0, 'Lieve': 0.3, 'Moderata': 0.7, 'Severa': 1.0}[case['tooth_wear']] +
        0.10 * (1.0 if case['tooth_group'] == 'Molare' else 0.6 if case['tooth_group'] == 'Premolare' else 0.25)
    )

    # Esthetic Demand Index
    esthetic_zone = 1.0 if case['tooth_group'] == 'Anteriore' else 0.55 if case['tooth_group'] == 'Premolare' else 0.2
    priority_esthetic = 1.0 if case['clinical_priority'] == 'Estetica' else 0.25
    edi = _clamp(
        0.45 * esthetic_zone +
        0.35 * {'Bassa': 0.0, 'Media': 0.5, 'Alta': 1.0}[case['esthetic_demand']] +
        0.20 * priority_esthetic
    )

    # Workflow Constraint Index
    direct_preference = 1.0 if case['workflow_preference'] == 'Chairside' else 0.0
    wci = _clamp(
        0.25 * {'No': 1.0, 'Sì': 0.0}[case['cadcam_available']] +
        0.25 * {'Sì': 0.0, 'No': 1.0, 'Incerta': 0.5}[case['indirect_acceptance']] +
        0.20 * {'1': 1.0, '2': 0.4, '3+': 0.0}[case['max_sessions']] +
        0.20 * {'Basso': 1.0, 'Medio': 0.45, 'Alto': 0.0}[case['budget_level']] +
        0.10 * direct_preference
    )

    return CaseIndices(ssi=ssi, bri=bri, fsi=fsi, edi=edi, wci=wci)


def recommend_restoration(case: Dict[str, object], idx: CaseIndices) -> pd.DataFrame:
    direct = 79 - 34 * idx.ssi - 18 * idx.bri - 16 * idx.fsi - 12 * idx.wci + 6 * idx.edi
    inlay = 63 + 10 * idx.edi + 9 * (1 - idx.wci) + 6 * (1 - idx.bri) + 6 * (0.75 - abs(idx.ssi - 0.45))
    onlay = 60 + 15 * idx.ssi + 8 * idx.fsi + 5 * idx.edi + 8 * (1 - idx.wci)
    overlay = 56 + 23 * idx.ssi + 11 * idx.fsi + 4 * idx.edi + 8 * (1 - idx.wci)
    crown = 48 + 28 * idx.ssi + 15 * idx.fsi + 5 * (1 - idx.wci) - 6 * idx.edi

    if case['tooth_group'] == 'Anteriore':
        direct += 7
        inlay += 2
        onlay -= 3
        crown -= 4
    elif case['tooth_group'] == 'Premolare':
        inlay += 2
        onlay += 3
    elif case['tooth_group'] == 'Molare':
        direct -= 6
        inlay -= 2
        onlay += 3
        overlay += 5
        crown += 3

    if int(case['involved_cusps']) >= 1 or case['cusp_loss'] == 'Sì':
        direct -= 6
        inlay -= 2
        onlay += 5
    if int(case['involved_cusps']) >= 2:
        onlay += 4
        overlay += 4
        direct -= 4
    if int(case['involved_cusps']) >= 3:
        inlay -= 5
        overlay += 5
        crown += 3
    if int(case['residual_walls']) <= 2:
        onlay += 3
        overlay += 4
        crown += 2
        direct -= 5
    if int(case['residual_walls']) <= 1:
        inlay -= 6
        direct -= 6
        crown += 4
    if case['coronal_tissue'] in ['25-50%', '<25%']:
        overlay += 4
        crown += 4
        direct -= 5
    if case['wall_thickness'] in ['Sottile', 'Molto sottile']:
        onlay += 2
        overlay += 3
        crown += 2
    if case['vitality'] == 'Vitale' and int(case['residual_walls']) >= 3:
        crown -= 6
    if case['endo_treated'] == 'Sì':
        direct -= 6
        inlay -= 2
        onlay += 3
        overlay += 5
        crown += 6
    if case['crack'] == 'Sì':
        direct -= 4
        inlay -= 2
        onlay += 2
        overlay += 4
        crown += 3
    if idx.fsi >= 0.65:
        direct -= 4
        inlay -= 3
        onlay += 2
        overlay += 3
        crown += 3
    if case['indirect_acceptance'] == 'No':
        inlay -= 20
        onlay -= 20
        overlay -= 20
        crown -= 20
    if case['cadcam_available'] == 'No':
        inlay -= 4
        onlay -= 4
        overlay -= 5
        crown -= 3

    rows = [
        {'restoration': 'Restauro diretto in composito', 'score': round(direct, 1), 'rationale': 'Approccio conservativo con massimo risparmio di tessuto.'},
        {'restoration': 'Inlay', 'score': round(inlay, 1), 'rationale': 'Soluzione intracoronale indiretta per perdita moderata senza ampia copertura cuspidale.'},
        {'restoration': 'Onlay', 'score': round(onlay, 1), 'rationale': 'Restauro indiretto con copertura cuspidale selettiva quando una o più cuspidi richiedono rinforzo.'},
        {'restoration': 'Overlay', 'score': round(overlay, 1), 'rationale': 'Restauro indiretto con copertura cuspidale ampia per compromissione strutturale marcata.'},
        {'restoration': 'Corona completa', 'score': round(crown, 1), 'rationale': 'Scelta più invasiva ma più protettiva nei casi molto compromessi o ad alto carico.'},
    ]
    df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    return df


def _class_profile(class_name: str) -> Dict[str, float]:
    c = (class_name or '').lower()
    profile = {
        'esthetic': 0.55,
        'strength': 0.55,
        'workflow': 0.55,
        'adhesive_tolerance': 0.5,
        'directness': 1.0,
    }
    if 'microibrido' in c or 'nanoibrido' in c or 'universale' in c:
        profile.update({'esthetic': 0.72, 'strength': 0.65, 'workflow': 0.85, 'adhesive_tolerance': 0.55, 'directness': 1.0})
    if 'nanofilled' in c or 'nanoriempito' in c:
        profile.update({'esthetic': 0.88, 'strength': 0.62, 'workflow': 0.82, 'adhesive_tolerance': 0.55, 'directness': 1.0})
    if 'flowable' in c and 'bulk-fill' not in c:
        profile.update({'esthetic': 0.62, 'strength': 0.38, 'workflow': 0.92, 'adhesive_tolerance': 0.5, 'directness': 1.0})
    if 'bulk-fill flowable' in c:
        profile.update({'esthetic': 0.55, 'strength': 0.48, 'workflow': 0.96, 'adhesive_tolerance': 0.5, 'directness': 1.0})
    if 'bulk-fill sculptable' in c or 'packable' in c:
        profile.update({'esthetic': 0.58, 'strength': 0.72, 'workflow': 0.9, 'adhesive_tolerance': 0.55, 'directness': 1.0})
    if 'feldsp' in c or 'layering' in c:
        profile.update({'esthetic': 0.95, 'strength': 0.3, 'workflow': 0.25, 'adhesive_tolerance': 0.35, 'directness': 0.0})
    if 'leucite' in c:
        profile.update({'esthetic': 0.9, 'strength': 0.4, 'workflow': 0.35, 'adhesive_tolerance': 0.45, 'directness': 0.0})
    if 'disilicato' in c or 'lithium disilicate' in c:
        profile.update({'esthetic': 0.9, 'strength': 0.82, 'workflow': 0.45, 'adhesive_tolerance': 0.5, 'directness': 0.0})
    if 'zls' in c or 'zirconia-reinforced lithium silicate' in c:
        profile.update({'esthetic': 0.88, 'strength': 0.76, 'workflow': 0.45, 'adhesive_tolerance': 0.5, 'directness': 0.0})
    if 'zirconia 3y' in c or 'alta resistenza' in c:
        profile.update({'esthetic': 0.45, 'strength': 0.98, 'workflow': 0.42, 'adhesive_tolerance': 0.8, 'directness': 0.0})
    if 'alta traslucenza' in c or '4y' in c or '5y' in c:
        profile.update({'esthetic': 0.78, 'strength': 0.78, 'workflow': 0.42, 'adhesive_tolerance': 0.75, 'directness': 0.0})
    if 'allumina' in c:
        profile.update({'esthetic': 0.55, 'strength': 0.7, 'workflow': 0.25, 'adhesive_tolerance': 0.65, 'directness': 0.0})
    if 'indiretto da laboratorio' in c:
        profile.update({'esthetic': 0.72, 'strength': 0.52, 'workflow': 0.28, 'adhesive_tolerance': 0.55, 'directness': 0.0})
    if 'cad/cam composite' in c or 'resin nanoceramic' in c or 'picn' in c:
        profile.update({'esthetic': 0.8, 'strength': 0.68, 'workflow': 0.5, 'adhesive_tolerance': 0.58, 'directness': 0.0})
    return profile


def _mechanical_from_numeric(row: pd.Series, fallback_strength: float) -> float:
    values = []
    if pd.notna(row.get('flexural_strength_mpa')):
        values.append(min(float(row['flexural_strength_mpa']) / 1200.0, 1.0))
    if pd.notna(row.get('compressive_strength_mpa')):
        values.append(min(float(row['compressive_strength_mpa']) / 500.0, 1.0))
    if pd.notna(row.get('elastic_modulus_gpa')):
        values.append(min(float(row['elastic_modulus_gpa']) / 220.0, 1.0))
    return sum(values) / len(values) if values else fallback_strength


def rank_materials(case: Dict[str, object], idx: CaseIndices, materials: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    rest_df = recommend_restoration(case, idx)
    best_rest = rest_df.iloc[0]['restoration']
    restoration_type = 'Direct' if best_rest == 'Restauro diretto in composito' else 'Indirect'
    rest_score_norm = max(0.0, min(rest_df.iloc[0]['score'] / 100.0, 1.0))

    rows: List[Dict[str, object]] = []
    for _, row in materials.iterrows():
        profile = _class_profile(str(row.get('primary_class_name', '')))
        material_type = str(row.get('direct_or_indirect', ''))
        direct_match = 1.0 if material_type.lower() == restoration_type.lower() else 0.0
        if direct_match == 0.0 and best_rest in ['Onlay', 'Overlay', 'Corona completa'] and material_type == 'Indirect':
            direct_match = 1.0

        mech = _mechanical_from_numeric(row, profile['strength'])
        esthetic = profile['esthetic']
        workflow = profile['workflow']
        adhesive_tolerance = profile['adhesive_tolerance']
        evidence = float(row.get('evidence_score', 0.5))
        completeness = float(row.get('quantitative_completeness_pct', 0.0)) / 100.0

        structural_fit = 1 - abs(mech - max(idx.ssi, idx.fsi * 0.9))
        structural_fit = _clamp(structural_fit)
        esthetic_fit = 1 - abs(esthetic - idx.edi)
        esthetic_fit = _clamp(esthetic_fit)
        workflow_need = 1 - idx.wci
        workflow_fit = 1 - abs(workflow - workflow_need)
        workflow_fit = _clamp(workflow_fit)
        bio_fit = 1 - abs(adhesive_tolerance - (1 - idx.bri))
        bio_fit = _clamp(bio_fit)

        penalty = 0.0
        class_name = str(row.get('primary_class_name', ''))
        lc = class_name.lower()
        if case['tooth_group'] == 'Molare' and idx.fsi > 0.65 and ('feldsp' in lc or 'leucite' in lc or 'allumina' in lc):
            penalty += 0.18
        if case['isolation'] == 'Impossibile' and material_type == 'Indirect' and idx.bri > 0.65:
            penalty += 0.08
        if best_rest == 'Restauro diretto in composito' and material_type == 'Indirect':
            penalty += 0.25
        if best_rest in ['Onlay', 'Overlay', 'Corona completa'] and material_type == 'Direct':
            penalty += 0.18
        if case['indirect_acceptance'] == 'No' and material_type == 'Indirect':
            penalty += 0.35
        if idx.edi > 0.75 and 'zirconia 3y' in lc:
            penalty += 0.08
        if 'allumina' in lc and best_rest != 'Corona':
            penalty += 0.22
        if ('feldsp' in lc or 'leucite' in lc) and best_rest in ['Onlay', 'Overlay', 'Corona completa']:
            penalty += 0.16
        if row.get('site_readiness_tier') == 'catalog_only':
            penalty += 0.05

        final = (
            0.24 * direct_match +
            0.24 * structural_fit +
            0.16 * esthetic_fit +
            0.14 * bio_fit +
            0.12 * workflow_fit +
            0.10 * evidence +
            0.05 * completeness +
            0.05 * rest_score_norm - penalty
        )
        final = _clamp(final)
        pss = _clamp(
            1 / (1 + pow(2.71828, -(1.8 * structural_fit + 1.2 * bio_fit + 0.9 * esthetic_fit + 0.7 * workflow_fit - 1.1 * idx.bri - 1.1 * idx.fsi - 1.0 * idx.ssi)))
        )

        material_sources = sources[sources['material_id'] == row['material_id']].head(3)
        source_urls = [u for u in material_sources['source_url'].dropna().tolist()[:3]]

        drivers = []
        if structural_fit > 0.7:
            drivers.append('buon fit biomeccanico')
        if esthetic_fit > 0.7:
            drivers.append('coerenza con domanda estetica')
        if workflow_fit > 0.7:
            drivers.append('workflow compatibile')
        if evidence > 0.75:
            drivers.append('buona affidabilità documentale')
        if not drivers:
            drivers.append('bilanciamento complessivo accettabile')

        rows.append({
            'material_id': row['material_id'],
            'display_name': row['display_name'],
            'manufacturer_name': row['manufacturer_name'],
            'primary_class_name': row['primary_class_name'],
            'direct_or_indirect': material_type,
            'verification_status': row['verification_status'],
            'quantitative_completeness_pct': round(float(row.get('quantitative_completeness_pct', 0.0)), 1),
            'final_score': round(final * 100, 1),
            'pss': round(pss * 100, 1),
            'structural_fit': round(structural_fit * 100, 1),
            'esthetic_fit': round(esthetic_fit * 100, 1),
            'workflow_fit': round(workflow_fit * 100, 1),
            'bio_fit': round(bio_fit * 100, 1),
            'evidence_score': round(evidence * 100, 1),
            'scenario_match': round(direct_match * 100, 1),
            'top_drivers': '; '.join(drivers),
            'source_urls': source_urls,
            'official_manufacturer_wording': row.get('official_manufacturer_wording', ''),
        })

    ranked = pd.DataFrame(rows).sort_values(['final_score', 'pss', 'evidence_score'], ascending=False).reset_index(drop=True)
    return rest_df, ranked


def summarize_case(case: Dict[str, object], idx: CaseIndices) -> List[str]:
    items = []
    if idx.ssi >= 0.7:
        items.append('compromissione strutturale elevata')
    elif idx.ssi >= 0.45:
        items.append('compromissione strutturale moderata')
    if idx.bri >= 0.65:
        items.append('contesto biologico/adesivo sfavorevole')
    if idx.fsi >= 0.65:
        items.append('stress funzionale elevato')
    if idx.edi >= 0.7:
        items.append('domanda estetica elevata')
    if idx.wci >= 0.6:
        items.append('vincoli di workflow importanti')
    if not items:
        items.append('profilo clinico abbastanza equilibrato')
    return items


def _level_from_index(value: float) -> str:
    if value >= 0.75:
        return "elevato"
    if value >= 0.5:
        return "moderato"
    return "basso"


def interpret_indices(idx: CaseIndices) -> Dict[str, object]:
    values = {"SSI": idx.ssi, "BRI": idx.bri, "FSI": idx.fsi, "EDI": idx.edi, "WCI": idx.wci}
    labels = {
        "SSI": "Severità strutturale",
        "BRI": "Rischio biologico/adesivo",
        "FSI": "Stress funzionale",
        "EDI": "Domanda estetica",
        "WCI": "Vincoli di workflow",
    }

    sorted_axes = sorted(values.items(), key=lambda x: x[1], reverse=True)
    dominant_key, dominant_value = sorted_axes[0]
    secondary_key, secondary_value = sorted_axes[1]

    headline = []
    if idx.ssi >= 0.75:
        headline.append("caso strutturalmente critico")
    elif idx.ssi >= 0.5:
        headline.append("caso con compromissione strutturale intermedia")
    else:
        headline.append("caso conservativo dal punto di vista strutturale")

    if idx.fsi >= 0.7:
        headline.append("alto carico funzionale")
    if idx.bri >= 0.65:
        headline.append("contesto biologico/adesivo delicato")
    if idx.edi >= 0.7:
        headline.append("forte esigenza estetica")
    if idx.wci >= 0.65:
        headline.append("workflow condizionato da vincoli pratici")

    bullet_map = {
        "SSI": (
            "La perdita di struttura residua spinge verso restauri più protettivi e con maggiore copertura cuspidale.",
            "La struttura residua è ancora favorevole a soluzioni più conservative."
        ),
        "BRI": (
            "Il contesto biologico rende più importanti isolamento, posizione del margine e affidabilità adesiva.",
            "Il contesto biologico è relativamente favorevole e consente maggiore flessibilità nella scelta del materiale."
        ),
        "FSI": (
            "Carico occlusale e/o parafunzione aumentano il peso della resistenza meccanica del restauro.",
            "Il carico funzionale non appare il driver dominante del caso."
        ),
        "EDI": (
            "L'estetica ha un peso rilevante nella scelta finale del materiale e del tipo di restauro.",
            "L'estetica non è il driver principale rispetto alla protezione strutturale."
        ),
        "WCI": (
            "Sedute, budget e accettazione dell'indiretto influenzano in modo concreto la decisione clinica.",
            "I vincoli di workflow non sembrano limitare fortemente le opzioni terapeutiche."
        ),
    }

    bullets = []
    for key, value in sorted_axes[:3]:
        high, low = bullet_map[key]
        bullets.append(f"{labels[key]} {_level_from_index(value)}: {high if value >= 0.5 else low}")

    radar_note = (
        f"Asse dominante: {labels[dominant_key]} ({round(dominant_value * 100)}%). "
        f"Secondo asse: {labels[secondary_key]} ({round(secondary_value * 100)}%)."
    )

    return {
        "headline": "; ".join(headline),
        "bullets": bullets,
        "radar_note": radar_note,
        "dominant_axis": labels[dominant_key],
        "dominant_value": round(dominant_value * 100, 1),
    }
