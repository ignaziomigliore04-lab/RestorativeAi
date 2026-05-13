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

    # The workbook already contains a class-level scoring seed.  We merge it into
    # the material table so the engine can use database-maintained class priors
    # instead of hard-coding every weighting decision in Python.
    try:
        class_seed = pd.read_excel(DATA_FILE, sheet_name='Class_Scoring_Seed')
        class_seed = class_seed.rename(columns={
            'class_name': 'primary_class_name',
            'esthetic_weight': 'seed_esthetic_weight',
            'mechanical_weight': 'seed_mechanical_weight',
            'handling_weight': 'seed_handling_weight',
            'evidence_weight': 'seed_evidence_weight',
            'cost_weight': 'seed_cost_weight',
        })
        seed_cols = [
            'primary_class_name', 'seed_esthetic_weight', 'seed_mechanical_weight',
            'seed_handling_weight', 'seed_evidence_weight', 'seed_cost_weight'
        ]
        materials = materials.merge(class_seed[seed_cols], on='primary_class_name', how='left')
    except Exception:
        for col, value in {
            'seed_esthetic_weight': 0.20,
            'seed_mechanical_weight': 0.35,
            'seed_handling_weight': 0.20,
            'seed_evidence_weight': 0.20,
            'seed_cost_weight': 0.05,
        }.items():
            materials[col] = value

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

    for col, value in {
        'seed_esthetic_weight': 0.20,
        'seed_mechanical_weight': 0.35,
        'seed_handling_weight': 0.20,
        'seed_evidence_weight': 0.20,
        'seed_cost_weight': 0.05,
    }.items():
        merged[col] = pd.to_numeric(merged.get(col, value), errors='coerce').fillna(value)

    return merged, props, sources

def _yes_no(value: str, yes='Sì') -> float:
    return 1.0 if value == yes else 0.0


def _clinical_sector(case: Dict[str, object]) -> str:
    """Return the clinical sector used by the material engine.

    The tooth number still drives premolar/molar nuance, but the explicit sector
    lets the clinician force anterior/posterior context when the case demands it.
    """
    explicit = str(case.get('clinical_sector', case.get('sector', 'Automatico')))
    if explicit in ['Anteriore', 'Posteriore']:
        return explicit
    return 'Anteriore' if case.get('tooth_group') == 'Anteriore' else 'Posteriore'


def _is_anterior(case: Dict[str, object]) -> bool:
    return _clinical_sector(case) == 'Anteriore'


def _is_posterior(case: Dict[str, object]) -> bool:
    return _clinical_sector(case) == 'Posteriore'


def _sex_biologic_context_modifier(case: Dict[str, object]) -> float:
    """Very small, transparent population-level calibration.

    Sex is not used as a direct prescription for a material. It can only add a
    tiny biological-risk modifier when it aligns with already present clinical
    risk signals, so it never overrides structure, function, isolation or sector.
    """
    sex = str(case.get('patient_sex', 'Non specificato'))
    modifier = 0.0
    if sex == 'Femmina' and (case.get('caries_risk') in ['Medio', 'Alto'] or case.get('xerostomia') == 'Sì'):
        modifier += 0.015
    if sex == 'Maschio' and (
        case.get('plaque_control') == 'Scarso' or
        case.get('periodontal_support') == 'Ridotto' or
        case.get('compliance') == 'Bassa'
    ):
        modifier += 0.015
    return modifier



def _posterior_direct_cuspal_viability(case: Dict[str, object], idx: CaseIndices) -> Dict[str, object]:
    """Evaluate whether a posterior one-cusp defect is still a direct-material case.

    This is a material-centred guardrail. A single missing cusp in a vital molar
    must not automatically push the engine toward ceramic/indirect classes. Modern
    adhesive direct composites remain realistic when structure, load and operative
    field are favourable. The function returns a graded signal rather than a hard
    prescription.
    """
    posterior = _is_posterior(case)
    single_cusp = case.get('cusp_loss') == 'Sì' and int(case.get('involved_cusps', 0)) == 1
    not_complex = (
        case.get('endo_treated') == 'No' and
        case.get('vitality') == 'Vitale' and
        case.get('crack') == 'No' and
        int(case.get('residual_walls', 0)) >= 2 and
        case.get('coronal_tissue') in ['>75%', '50-75%', '25-50%'] and
        case.get('ferrule') != 'Assente'
    )
    adhesive_ok = (
        case.get('isolation') in ['Facile', 'Difficile'] and
        case.get('margin') in ['Sovragengivale', 'Juxtagengivale'] and
        case.get('adhesive_context') in ['Favorevole', 'Intermedio']
    )
    load_ok = (
        idx.fsi < 0.62 and
        case.get('occlusal_load') != 'Alto' and
        case.get('bruxism') != 'Confermata' and
        case.get('parafunction_severity') not in ['Severa'] and
        case.get('tooth_wear') not in ['Severa'] and
        case.get('antagonist') != 'Protesico'
    )
    size_ok = (
        case.get('cavity_size') in ['Media', 'Ampia'] and
        idx.ssi <= 0.55 and
        int(case.get('involved_cusps', 0)) <= 1
    )
    direct_context = (
        case.get('clinical_priority') in ['Conservatività', 'Rapidità'] or
        case.get('workflow_preference') == 'Chairside' or
        case.get('max_sessions') == '1' or
        case.get('budget_level') == 'Basso' or
        case.get('indirect_acceptance') in ['No', 'Incerta']
    )
    viable = bool(posterior and single_cusp and not_complex and adhesive_ok and load_ok and size_ok)
    # Stronger signal when the clinician/patient constraints also favour a direct path.
    strength = 0.0
    if viable:
        strength = 0.62 + (0.20 if direct_context else 0.0)
        if int(case.get('residual_walls', 0)) >= 3:
            strength += 0.08
        if case.get('wall_thickness') == 'Adeguato':
            strength += 0.05
        if case.get('cavity_size') == 'Media':
            strength += 0.04
        if case.get('indirect_acceptance') == 'Sì' and case.get('max_sessions') in ['2', '3+'] and case.get('budget_level') in ['Medio', 'Alto']:
            # Indirect acceptance should not erase direct viability; it only softens the push.
            strength -= 0.08
    return {
        'viable': viable,
        'strength': _clamp(strength),
        'reason': 'posteriore vitale con singola cuspide mancante, carico non severo e campo adesivo gestibile',
    }



def assess_material_case_feasibility(case: Dict[str, object], idx: CaseIndices | None = None) -> Dict[str, object]:
    """Screen whether a material recommendation should be generated.

    This is intentionally a *material-selection* gate, not a prosthetic planning
    module. When the entered case is biologically/structurally not actionable,
    the engine does not pretend that one restorative material can solve it.
    """
    if idx is None:
        idx = compute_case_indices(case)

    residual_walls = int(case.get('residual_walls', 0))
    involved_cusps = int(case.get('involved_cusps', 0))
    reasons: List[str] = []
    cautions: List[str] = []

    non_restorable_structure = (
        residual_walls == 0 and
        case.get('coronal_tissue') == '<25%' and
        case.get('ferrule') == 'Assente'
    )
    endo_no_ferrule_extreme = (
        case.get('endo_treated') == 'Sì' and
        case.get('coronal_tissue') == '<25%' and
        case.get('ferrule') == 'Assente' and
        residual_walls <= 1
    )
    biologic_not_actionable = (
        case.get('isolation') == 'Impossibile' and
        case.get('margin') == 'Subgengivale' and
        case.get('adhesive_context') == 'Sfavorevole'
    )
    caries_control_not_actionable = (
        case.get('caries_risk') == 'Alto' and
        case.get('plaque_control') == 'Scarso' and
        case.get('compliance') == 'Bassa' and
        case.get('xerostomia') == 'Sì'
    )
    functional_not_controlled = (
        case.get('bruxism') == 'Confermata' and
        case.get('parafunction_severity') == 'Severa' and
        case.get('occlusal_load') == 'Alto' and
        case.get('tooth_wear') == 'Severa' and
        int(case.get('involved_cusps', 0)) >= 3 and
        case.get('indirect_acceptance') == 'No'
    )

    if non_restorable_structure:
        reasons.append('Struttura residua non sufficiente: 0 pareti, tessuto coronale <25% e ferrule assente.')
    if endo_no_ferrule_extreme:
        reasons.append('Dente endodonticamente trattato con ferrule assente e struttura coronale estremamente ridotta.')
    if biologic_not_actionable:
        reasons.append('Condizioni adesive non controllabili: isolamento impossibile, margine subgengivale e contesto sfavorevole.')
    if caries_control_not_actionable:
        reasons.append('Rischio biologico non controllato: carie alta, placca scarsa, bassa compliance e xerostomia.')
    if functional_not_controlled:
        reasons.append('Stress funzionale severo non controllato associato a mancata accettazione del percorso indiretto/protettivo.')

    if not reasons:
        if idx.ssi >= 0.85:
            cautions.append('Compromissione strutturale molto alta: confermare restaurabilità prima della scelta materiale definitiva.')
        if idx.bri >= 0.82:
            cautions.append('Rischio biologico/adesivo alto: controllare margini, isolamento e gestione del rischio prima della finalizzazione.')
        if idx.fsi >= 0.85:
            cautions.append('Stress funzionale molto alto: prevedere controllo occlusale/parafunzionale e follow-up.')

    return {
        'is_actionable': len(reasons) == 0,
        'status': 'actionable' if len(reasons) == 0 else 'not_actionable',
        'title': 'Caso idoneo alla selezione del materiale' if len(reasons) == 0 else 'Material recommendation non generata',
        'reasons': reasons,
        'cautions': cautions,
        'message': (
            'Il caso è compatibile con una raccomandazione material-first.' if len(reasons) == 0 else
            'Il caso inserito non è un contesto realistico/sicuro per scegliere una classe di materiale. Prima serve risolvere restaurabilità, controllo biologico o condizioni operative.'
        ),
    }

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
    # Patient sex is deliberately a weak contextual modifier only. The material
    # decision remains driven by explicit clinical variables: caries risk, plaque,
    # isolation, margin, pulp proximity and adhesive context.
    bri_base = (
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
    bri = _clamp(bri_base + _sex_biologic_context_modifier(case))

    # Functional Stress Index
    fsi = _clamp(
        0.18 * {'Assente': 0.0, 'Sospetta': 0.55, 'Confermata': 1.0}[case['bruxism']] +
        0.14 * {'Assente': 0.0, 'Lieve': 0.3, 'Moderata': 0.7, 'Severa': 1.0}[case['parafunction_severity']] +
        0.22 * {'Basso': 0.0, 'Medio': 0.5, 'Alto': 1.0}[case['occlusal_load']] +
        0.10 * {'Assenti': 0.0, 'Presenti': 1.0}[case['eccentric_contacts']] +
        0.09 * {'Naturale': 0.15, 'Restauro': 0.45, 'Protesico': 0.8}[case['antagonist']] +
        0.17 * {'Assente': 0.0, 'Lieve': 0.3, 'Moderata': 0.7, 'Severa': 1.0}[case['tooth_wear']] +
        0.10 * (1.0 if case['tooth_group'] == 'Molare' else 0.68 if case['tooth_group'] == 'Premolare' else 0.35 if _is_posterior(case) else 0.22)
    )

    # Esthetic Demand Index
    if _is_anterior(case):
        esthetic_zone = 1.0
    elif case['tooth_group'] == 'Premolare':
        esthetic_zone = 0.55
    else:
        esthetic_zone = 0.22
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


def _restoration_score_status(score: float) -> float:
    """Display score on a clinical 0-100 scale without hiding high-score differences."""
    display = score
    if score > 80:
        display = 80 + (score - 80) * 0.42
    elif score < 20:
        display = score * 0.85
    return round(_clamp(display, 0.0, 99.5), 1)


def recommend_restoration(case: Dict[str, object], idx: CaseIndices) -> pd.DataFrame:
    """
    Secondary restoration-envelope recommendation.

    The restoration type is kept as clinical context for material selection. Scores are
    intentionally conservative: direct restorations are favoured in small/low-risk
    cases, while indirect cuspal-coverage options emerge only when structure,
    endodontic status or functional load justify a more protective design.
    """
    residual_walls = int(case['residual_walls'])
    involved_cusps = int(case['involved_cusps'])
    posterior = _is_posterior(case)
    anterior = _is_anterior(case)
    cusp_loss = case['cusp_loss'] == 'Sì'
    endo = case['endo_treated'] == 'Sì'
    low_tissue = case['coronal_tissue'] in ['25-50%', '<25%']
    very_low_tissue = case['coronal_tissue'] == '<25%'
    thin_walls = case['wall_thickness'] in ['Sottile', 'Molto sottile']

    conservative_case = (
        idx.ssi < 0.43 and
        residual_walls >= 3 and
        not cusp_loss and
        involved_cusps == 0 and
        not endo and
        case['cavity_size'] in ['Piccola', 'Media']
    )
    moderate_intracoronal_case = (
        posterior and
        0.28 <= idx.ssi <= 0.58 and
        residual_walls >= 3 and
        not cusp_loss and
        involved_cusps <= 1 and
        case['cavity_size'] in ['Media', 'Ampia'] and
        not very_low_tissue
    )
    direct_cuspal_signal = _posterior_direct_cuspal_viability(case, idx)
    direct_cuspal_viable = bool(direct_cuspal_signal['viable'])
    cuspal_protection_need = (cusp_loss or involved_cusps >= 1) and not direct_cuspal_viable
    broad_coverage_need = involved_cusps >= 2 or low_tissue or endo or idx.ssi >= 0.62
    circumferential_need = (
        residual_walls <= 1 or
        very_low_tissue or
        case['ferrule'] == 'Assente' or
        (endo and idx.fsi >= 0.7 and residual_walls <= 2)
    )

    # Base envelope. WCI is a constraint index: higher values make simple/chairside
    # workflows more attractive and make complex indirect workflows less attractive.
    direct = 82 - 43 * idx.ssi - 15 * idx.bri - 18 * idx.fsi + 12 * idx.wci + 4 * idx.edi
    inlay = 55 + 8 * idx.edi + 8 * (1 - idx.wci) + 6 * (1 - idx.bri) + 10 * (0.70 - abs(idx.ssi - 0.42))
    onlay = 54 + 17 * idx.ssi + 10 * idx.fsi + 4 * idx.edi + 7 * (1 - idx.wci)
    overlay = 49 + 25 * idx.ssi + 13 * idx.fsi + 3 * idx.edi + 7 * (1 - idx.wci)
    crown = 40 + 34 * idx.ssi + 17 * idx.fsi + 5 * (1 - idx.wci) - 7 * idx.edi

    if conservative_case:
        direct += 18
        inlay -= 8
        onlay -= 16
        overlay -= 22
        crown -= 28

    if moderate_intracoronal_case:
        inlay += 10
        direct += 2
        onlay -= 3
        overlay -= 8
        crown -= 14

    if anterior:
        direct += 9
        inlay -= 18
        onlay -= 14
        overlay -= 10
        crown -= 7
        if case.get('black_class') in ['III', 'IV', 'V']:
            inlay -= 10
            onlay -= 8
            overlay -= 6
    elif case['tooth_group'] == 'Premolare':
        inlay += 2
        onlay += 4
        overlay += 1
    elif case['tooth_group'] == 'Molare':
        direct -= 4
        inlay -= 2
        onlay += 4
        overlay += 6
        crown += 3

    if cuspal_protection_need:
        direct -= 10
        inlay -= 5
        onlay += 9
        overlay += 4
    elif direct_cuspal_viable:
        # A vital posterior tooth with one cusp missing is often still a direct-material case
        # when the adhesive field and load are controlled. Do not let the envelope force
        # an indirect onlay solely because one cusp is involved.
        direct += 13 * float(direct_cuspal_signal['strength'])
        inlay -= 4
        onlay -= 9 * float(direct_cuspal_signal['strength'])
        overlay -= 10
        crown -= 14
    if involved_cusps >= 2:
        direct -= 8
        inlay -= 8
        onlay += 5
        overlay += 8
        crown += 2
    if involved_cusps >= 3:
        onlay -= 2
        overlay += 8
        crown += 5
    if residual_walls <= 2:
        direct -= 8
        inlay -= 5
        onlay += 4
        overlay += 7
        crown += 5
    if residual_walls <= 1:
        direct -= 10
        inlay -= 10
        onlay -= 2
        overlay += 4
        crown += 10
    if low_tissue:
        direct -= 7
        inlay -= 6
        onlay += 3
        overlay += 7
        crown += 7
    if very_low_tissue:
        onlay -= 3
        overlay += 4
        crown += 10
    if thin_walls:
        direct -= 4
        inlay -= 2
        onlay += 3
        overlay += 4
        crown += 3
    if endo:
        direct -= 8
        inlay -= 5
        onlay += 5
        overlay += 8
        crown += 8
    if case['vitality'] == 'Vitale' and residual_walls >= 3 and idx.ssi < 0.6:
        crown -= 10
    if case['crack'] == 'Sì':
        direct -= 5
        inlay -= 4
        onlay += 3
        overlay += 5
        crown += 4
    if idx.fsi >= 0.65:
        direct -= 5
        inlay -= 5
        onlay += 3
        overlay += 5
        crown += 4
    if idx.fsi >= 0.8 and broad_coverage_need:
        overlay += 4
        crown += 5
    if circumferential_need:
        direct -= 10
        inlay -= 10
        onlay -= 3
        overlay += 4
        crown += 14
    if case['indirect_acceptance'] == 'No':
        inlay -= 18
        onlay -= 18
        overlay -= 20
        crown -= 22
        direct += 6
    elif case['indirect_acceptance'] == 'Incerta':
        inlay -= 5
        onlay -= 5
        overlay -= 6
        crown -= 7
    if case['cadcam_available'] == 'No' and case['workflow_preference'] == 'Chairside':
        inlay -= 4
        onlay -= 5
        overlay -= 6
        crown -= 5
        direct += 3

    rows = [
        {'restoration': 'Restauro diretto in composito', 'score': _restoration_score_status(direct), 'raw_score': round(direct, 1), 'rationale': 'Approccio conservativo con massimo risparmio di tessuto.'},
        {'restoration': 'Inlay', 'score': _restoration_score_status(inlay), 'raw_score': round(inlay, 1), 'rationale': 'Soluzione intracoronale indiretta per perdita moderata senza ampia copertura cuspidale.'},
        {'restoration': 'Onlay', 'score': _restoration_score_status(onlay), 'raw_score': round(onlay, 1), 'rationale': 'Restauro indiretto con copertura cuspidale selettiva quando una o più cuspidi richiedono rinforzo.'},
        {'restoration': 'Overlay', 'score': _restoration_score_status(overlay), 'raw_score': round(overlay, 1), 'rationale': 'Restauro indiretto con copertura cuspidale ampia per compromissione strutturale marcata.'},
        {'restoration': 'Corona completa', 'score': _restoration_score_status(crown), 'raw_score': round(crown, 1), 'rationale': 'Scelta più invasiva ma più protettiva nei casi molto compromessi o ad alto carico.'},
    ]
    df = pd.DataFrame(rows).sort_values('raw_score', ascending=False).reset_index(drop=True)
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


def _material_modality_fits(case: Dict[str, object], idx: CaseIndices) -> Tuple[float, float]:
    """Return direct and indirect material-path compatibility, independent of a single restoration label."""
    residual_walls = int(case['residual_walls'])
    involved_cusps = int(case['involved_cusps'])
    cusp_factor = 1.0 if case['cusp_loss'] == 'Sì' else 0.0
    endo_factor = 1.0 if case['endo_treated'] == 'Sì' else 0.0
    low_tissue_factor = 1.0 if case['coronal_tissue'] in ['25-50%', '<25%'] else 0.0
    very_low_tissue_factor = 1.0 if case['coronal_tissue'] == '<25%' else 0.0
    wall_loss_factor = {4: 0.0, 3: 0.25, 2: 0.55, 1: 0.85, 0: 1.0}[residual_walls]
    cuspal_involvement = min(involved_cusps / 4.0, 1.0)

    protection_need = _clamp(
        0.36 * idx.ssi +
        0.22 * idx.fsi +
        0.14 * cusp_factor +
        0.10 * cuspal_involvement +
        0.08 * endo_factor +
        0.06 * low_tissue_factor +
        0.04 * wall_loss_factor
    )

    simple_workflow_pull = _clamp(0.55 * idx.wci + 0.20 * (1.0 if case['workflow_preference'] == 'Chairside' else 0.0))
    indirect_feasibility = {
        'Sì': 1.0,
        'Incerta': 0.62,
        'No': 0.12,
    }[case['indirect_acceptance']]
    cadcam_lab_feasibility = 1.0 if case['cadcam_available'] == 'Sì' or case['workflow_preference'] == 'Laboratorio' else 0.72

    direct_fit = _clamp(
        0.88 -
        0.72 * protection_need -
        0.12 * idx.bri +
        0.22 * simple_workflow_pull +
        0.10 * (1.0 if case['clinical_priority'] in ['Conservatività', 'Rapidità'] else 0.0)
    )
    indirect_fit = _clamp(
        0.34 +
        0.78 * protection_need +
        0.12 * idx.edi +
        0.10 * (1 - idx.wci) +
        0.16 * indirect_feasibility +
        0.06 * cadcam_lab_feasibility -
        0.22 * (1 - indirect_feasibility) -
        0.12 * very_low_tissue_factor * (1.0 if case['margin'] == 'Subgengivale' else 0.0)
    )
    direct_cuspal_signal = _posterior_direct_cuspal_viability(case, idx)
    if direct_cuspal_signal['viable']:
        # A single missing cusp in a vital, controlled posterior case is not
        # automatically an indirect-material indication. Keep direct classes
        # clinically competitive even when indirect workflow is acceptable.
        sig = float(direct_cuspal_signal['strength'])
        direct_fit = _clamp(direct_fit + 0.18 * sig)
        indirect_fit = _clamp(indirect_fit - 0.10 * sig)
    return direct_fit, indirect_fit


def _target_material_strength(case: Dict[str, object], idx: CaseIndices) -> float:
    cuspal = 1.0 if case['cusp_loss'] == 'Sì' else min(int(case['involved_cusps']) / 4.0, 1.0)
    endo = 1.0 if case['endo_treated'] == 'Sì' else 0.0
    posterior = 1.0 if case['tooth_group'] == 'Molare' else 0.7 if case['tooth_group'] == 'Premolare' else 0.35 if _is_posterior(case) else 0.22
    low_tissue = 1.0 if case['coronal_tissue'] in ['25-50%', '<25%'] else 0.0
    return _clamp(0.28 + 0.38 * idx.ssi + 0.26 * idx.fsi + 0.08 * cuspal + 0.05 * endo + 0.04 * posterior + 0.04 * low_tissue)


def _target_esthetic_level(case: Dict[str, object], idx: CaseIndices) -> float:
    # In high-risk posterior cases aesthetics should matter, but should not overrule safety.
    safety_pull = 0.16 * max(idx.ssi, idx.fsi)
    sector_pull = 0.06 if _is_anterior(case) else -0.04
    return _clamp(idx.edi - safety_pull + sector_pull + 0.06 * (1.0 if case['clinical_priority'] == 'Estetica' else 0.0))


def _target_adhesive_tolerance(case: Dict[str, object], idx: CaseIndices) -> float:
    isolation_penalty = {'Facile': 0.0, 'Difficile': 0.16, 'Impossibile': 0.28}[case['isolation']]
    margin_penalty = {'Sovragengivale': 0.0, 'Juxtagengivale': 0.10, 'Subgengivale': 0.20}[case['margin']]
    return _clamp(0.36 + 0.48 * idx.bri + isolation_penalty + margin_penalty)


def _target_workflow_simplicity(case: Dict[str, object], idx: CaseIndices) -> float:
    chairside = 0.12 if case['workflow_preference'] == 'Chairside' else -0.08 if case['workflow_preference'] == 'Laboratorio' else 0.0
    return _clamp(0.45 + 0.45 * idx.wci + chairside)


def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(cleaned.values()) or 1.0
    return {k: v / total for k, v in cleaned.items()}


def _case_material_axis_weights(case: Dict[str, object], idx: CaseIndices, direct_fit: float, indirect_fit: float) -> Dict[str, float]:
    """Dynamic material-first weights generated from the clinical case.

    These weights decide how much the material engine should care about each
    material axis. They do not use the selected restoration type. Structural
    loss, load and posterior sector push mechanical weight up; anterior sector
    and esthetic demand push optical weight up; isolation/margins/caries risk
    push biologic-adhesive tolerance up; time/budget/preferences push workflow.
    """
    posterior = 1.0 if _is_posterior(case) else 0.0
    anterior = 1.0 if _is_anterior(case) else 0.0
    cuspal_need = 1.0 if case['cusp_loss'] == 'Sì' or int(case['involved_cusps']) >= 1 else 0.0
    endo = 1.0 if case['endo_treated'] == 'Sì' else 0.0
    poor_adhesion = {
        'Facile': 0.0,
        'Difficile': 0.55,
        'Impossibile': 1.0,
    }[case['isolation']]
    deep_margin = {
        'Sovragengivale': 0.0,
        'Juxtagengivale': 0.45,
        'Subgengivale': 1.0,
    }[case['margin']]
    one_session = 1.0 if case['max_sessions'] == '1' else 0.0
    low_budget = 1.0 if case['budget_level'] == 'Basso' else 0.0
    mode_decision_pressure = abs(float(direct_fit) - float(indirect_fit))

    raw = {
        'mechanical': 0.25 + 0.23 * idx.ssi + 0.18 * idx.fsi + 0.06 * posterior + 0.05 * cuspal_need + 0.03 * endo,
        'biology': 0.18 + 0.24 * idx.bri + 0.07 * poor_adhesion + 0.05 * deep_margin,
        'esthetic': 0.15 + 0.28 * idx.edi + 0.07 * anterior,
        'workflow': 0.13 + 0.22 * idx.wci + 0.05 * one_session + 0.04 * low_budget,
        'evidence': 0.11,
        'material_path': 0.08 + 0.06 * mode_decision_pressure,
    }
    return _normalise_weights(raw)


def _database_axis_weights(row: pd.Series) -> Dict[str, float]:
    """Class-level priors stored in the Excel database.

    Class_Scoring_Seed defines which properties are usually important for a
    material class. The clinical case remains dominant, but these priors keep
    the score aligned with the database rather than with a generic formula.
    """
    raw = {
        'mechanical': float(row.get('seed_mechanical_weight', 0.35) or 0.35),
        'esthetic': float(row.get('seed_esthetic_weight', 0.20) or 0.20),
        'workflow': float(row.get('seed_handling_weight', 0.20) or 0.20) + 0.35 * float(row.get('seed_cost_weight', 0.05) or 0.05),
        'evidence': float(row.get('seed_evidence_weight', 0.20) or 0.20),
        'biology': 0.16,
        'material_path': 0.08,
    }
    return _normalise_weights(raw)


def _blend_axis_weights(case_weights: Dict[str, float], database_weights: Dict[str, float]) -> Dict[str, float]:
    axes = ['mechanical', 'biology', 'esthetic', 'workflow', 'evidence', 'material_path']
    blended = {axis: 0.78 * case_weights.get(axis, 0.0) + 0.22 * database_weights.get(axis, 0.0) for axis in axes}
    return _normalise_weights(blended)


def _class_clinical_notes(class_name: str, material_type: str) -> str:
    c = (class_name or '').lower()
    if 'flowable' in c and 'bulk-fill' not in c:
        return 'utile soprattutto come liner/base o in cavità a basso stress; non dovrebbe essere protagonista nei carichi elevati.'
    if 'bulk-fill flowable' in c:
        return 'favorisce workflow rapido in cavità profonde, ma richiede copertura/gestione dello strato occlusale se il carico è rilevante.'
    if 'bulk-fill sculptable' in c or 'packable' in c:
        return 'classe diretta orientata a posteriori e workflow efficiente, con attenzione a contatti occlusali e stratificazione finale.'
    if 'nanofilled' in c or 'nanoibrido' in c or 'microibrido' in c or 'universale' in c:
        return 'classe diretta versatile; indicata quando conservatività, estetica e controllo adesivo sono compatibili.'
    if 'feldsp' in c or 'leucite' in c:
        return 'classe molto estetica ma meno indicata quando dominano carico, parafunzione o richiesta di copertura ampia.'
    if 'disilicato' in c or 'lithium disilicate' in c or 'zls' in c:
        return 'classe indiretta adesiva con buon equilibrio estetico-meccanico, utile in restauri parziali e coperture quando l’adesione è gestibile.'
    if 'zirconia 3y' in c or 'alta resistenza' in c:
        return 'classe indiretta ad alta resistenza, più orientata a protezione meccanica che a massima resa estetica.'
    if 'alta traslucenza' in c or '4y' in c or '5y' in c:
        return 'classe indiretta che bilancia robustezza e resa estetica, con indicazione da modulare in base a spessore e carico.'
    if 'cad/cam composite' in c or 'resin nanoceramic' in c or 'picn' in c:
        return 'classe indiretta a modulo più favorevole e workflow digitale, utile quando si cerca un compromesso tra riparabilità e supporto.'
    if 'indiretto da laboratorio' in c:
        return 'opzione indiretta polimerica/ibrida, utile come alternativa meno fragile ma meno ceramica nella resa ottica.'
    if 'allumina' in c:
        return 'classe storica/strutturale oggi meno centrale rispetto a zirconie e vetroceramiche moderne.'
    return f"classe {'diretta' if material_type == 'Direct' else 'indiretta'} da valutare in base al profilo clinico complessivo."


def _build_class_ranking(product_rows: List[Dict[str, object]]) -> pd.DataFrame:
    products_ranked = pd.DataFrame(product_rows).sort_values(['final_score', 'pss', 'evidence_score'], ascending=False).reset_index(drop=True)
    class_rows: List[Dict[str, object]] = []
    for (class_name, modality), group in products_ranked.groupby(['primary_class_name', 'direct_or_indirect'], sort=False):
        group = group.sort_values(['final_score', 'pss', 'evidence_score'], ascending=False).reset_index(drop=True)
        best = group.iloc[0].to_dict()
        top3 = group.head(3)
        top_score_mean = float(top3['final_score'].mean())
        class_score = 0.72 * float(best['final_score']) + 0.18 * top_score_mean + 0.10 * float(top3['evidence_score'].mean())
        class_pss = 0.70 * float(best['pss']) + 0.30 * float(top3['pss'].mean())
        examples = top3['display_name'].dropna().astype(str).drop_duplicates().tolist()
        manufacturers = top3['manufacturer_name'].dropna().astype(str).drop_duplicates().tolist()
        urls = []
        for source_list in top3['source_urls'].tolist():
            for url in source_list:
                if url and url not in urls:
                    urls.append(url)

        mean_cols = [
            'weight_mechanical', 'weight_biology', 'weight_esthetic', 'weight_workflow', 'weight_evidence', 'weight_material_path',
            'case_weight_mechanical', 'case_weight_biology', 'case_weight_esthetic', 'case_weight_workflow', 'case_weight_evidence', 'case_weight_material_path',
            'db_weight_mechanical', 'db_weight_biology', 'db_weight_esthetic', 'db_weight_workflow', 'db_weight_evidence', 'db_weight_material_path',
            'mechanical_points', 'biology_points', 'esthetic_points', 'workflow_points', 'evidence_points', 'material_path_points',
            'bonus_points', 'penalty_points', 'target_strength', 'target_esthetic', 'target_adhesion', 'target_workflow'
        ]
        mean_values = {col: round(float(top3[col].mean()), 1) for col in mean_cols if col in top3.columns}

        class_rows.append({
            **best,
            'display_name': examples[0] if examples else best.get('display_name', ''),
            'example_names': examples,
            'example_manufacturers': manufacturers,
            'class_product_count': int(group.shape[0]),
            'final_score': round(_clamp(class_score / 100.0) * 100, 1),
            'pss': round(_clamp(class_pss / 100.0) * 100, 1),
            'structural_fit': round(float(top3['structural_fit'].mean()), 1),
            'esthetic_fit': round(float(top3['esthetic_fit'].mean()), 1),
            'workflow_fit': round(float(top3['workflow_fit'].mean()), 1),
            'bio_fit': round(float(top3['bio_fit'].mean()), 1),
            'evidence_score': round(float(top3['evidence_score'].mean()), 1),
            'scenario_match': round(float(top3['scenario_match'].mean()), 1),
            'source_urls': urls[:5],
            'class_source_count': int(group.get('source_record_count', pd.Series(dtype=float)).fillna(0).astype(float).sum()) if 'source_record_count' in group.columns else len(urls),
            'class_quantitative_observation_count': int(group.get('quantitative_observation_count', pd.Series(dtype=float)).fillna(0).astype(float).sum()) if 'quantitative_observation_count' in group.columns else 0,
            'top_drivers': best.get('top_drivers', ''),
            'class_note': _class_clinical_notes(str(class_name), str(modality)),
            'recommendation_level': 'material_class',
            **mean_values,
        })

    ranked_classes = pd.DataFrame(class_rows).sort_values(['final_score', 'pss', 'evidence_score'], ascending=False).reset_index(drop=True)
    return ranked_classes



def add_material_decision_metadata(ranked: pd.DataFrame) -> pd.DataFrame:
    """Add material-first confidence and near-equivalence metadata.

    Confidence is intentionally based on material score separation, not on the
    restoration envelope. When the top two classes are very close, the output
    should say so explicitly instead of pretending there is a single absolute
    winner.
    """
    if ranked is None or ranked.empty:
        return ranked
    out = ranked.copy().reset_index(drop=True)
    scores = out['final_score'].astype(float)
    next_scores = scores.shift(-1)
    out['score_gap_to_next'] = (scores - next_scores).round(1)
    out.loc[out.index[-1], 'score_gap_to_next'] = None

    labels = []
    messages = []
    for i, row in out.iterrows():
        score = float(row.get('final_score', 0.0))
        gap = row.get('score_gap_to_next')
        gap_val = None if pd.isna(gap) else float(gap)
        if i == 0:
            if gap_val is not None and gap_val < 2.0:
                label = 'Borderline'
                message = 'Prime due classi quasi equivalenti: scelta da rifinire con preferenze operative, anatomia cavitaria e disponibilità del materiale.'
            elif gap_val is not None and gap_val < 6.0:
                label = 'Intermedia'
                message = 'Classe top plausibile, ma con alternativa clinicamente vicina: valutare il secondo materiale come opzione reale.'
            elif score < 72.0:
                label = 'Intermedia'
                message = 'Score non altissimo: raccomandazione utilizzabile, ma condizionata dai limiti del caso e dalla base dati.'
            else:
                label = 'Alta'
                message = 'Vantaggio materiale netto sulle alternative principali nel contesto inserito.'
        else:
            if gap_val is not None and gap_val < 2.0:
                label = 'Vicino alla successiva'
                message = 'Classe vicina alla successiva per score materiale.'
            else:
                label = 'Alternativa'
                message = 'Classe alternativa nel ranking materiale.'
        labels.append(label)
        messages.append(message)
    out['confidence_label'] = labels
    out['confidence_message'] = messages
    return out


def rank_materials(case: Dict[str, object], idx: CaseIndices, materials: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    # Restoration is computed only as a secondary envelope for communication.
    # The material score below does not use the restoration ranking as an input.
    rest_df = recommend_restoration(case, idx)
    direct_fit, indirect_fit = _material_modality_fits(case, idx)
    case_axis_weights = _case_material_axis_weights(case, idx, direct_fit, indirect_fit)

    target_strength = _target_material_strength(case, idx)
    target_esthetic = _target_esthetic_level(case, idx)
    target_adhesion = _target_adhesive_tolerance(case, idx)
    target_workflow = _target_workflow_simplicity(case, idx)

    rows: List[Dict[str, object]] = []
    for _, row in materials.iterrows():
        if str(row.get('include_in_v1_database', 'yes')).lower() not in ['yes', 'sì', 'si', 'true', '1']:
            continue

        class_name = str(row.get('primary_class_name', ''))
        profile = _class_profile(class_name)
        material_type = str(row.get('direct_or_indirect', ''))
        lc = class_name.lower()

        database_axis_weights = _database_axis_weights(row)
        axis_weights = _blend_axis_weights(case_axis_weights, database_axis_weights)

        modality_fit = direct_fit if material_type == 'Direct' else indirect_fit
        mech = _mechanical_from_numeric(row, profile['strength'])
        esthetic = profile['esthetic']
        workflow = profile['workflow']
        adhesive_tolerance = profile['adhesive_tolerance']
        evidence = float(row.get('evidence_score', 0.5))
        completeness = float(row.get('quantitative_completeness_pct', 0.0)) / 100.0

        structural_fit = _clamp(1 - abs(mech - target_strength))
        esthetic_fit = _clamp(1 - abs(esthetic - target_esthetic))
        workflow_fit = _clamp(1 - abs(workflow - target_workflow))
        bio_fit = _clamp(1 - abs(adhesive_tolerance - target_adhesion))

        penalty = 0.0
        bonus = 0.0
        high_load = idx.fsi >= 0.65 or case['occlusal_load'] == 'Alto' or case['bruxism'] == 'Confermata'
        protection_high = idx.ssi >= 0.62 or case['cusp_loss'] == 'Sì' or int(case['involved_cusps']) >= 2 or case['endo_treated'] == 'Sì'
        conservative_low = idx.ssi < 0.43 and int(case['residual_walls']) >= 3 and case['cusp_loss'] == 'No' and int(case['involved_cusps']) == 0
        anterior_sector = _is_anterior(case)
        posterior_sector = _is_posterior(case)

        # Clinical guardrails are material guardrails: they avoid making a class
        # protagonist in contexts where its material behaviour is poorly aligned.
        # Direct classes are not automatically downgraded in every moderate case:
        # the penalty becomes strong only when protection need is truly high.
        strong_protection_need = (
            idx.ssi >= 0.68 or
            int(case['involved_cusps']) >= 2 or
            (case['endo_treated'] == 'Sì' and case['coronal_tissue'] in ['25-50%', '<25%']) or
            (case['cusp_loss'] == 'Sì' and high_load)
        )
        if material_type == 'Direct' and protection_high:
            penalty += 0.04
        if material_type == 'Direct' and strong_protection_need:
            penalty += 0.07
        if material_type == 'Direct' and strong_protection_need and high_load:
            penalty += 0.07
        if material_type == 'Indirect' and conservative_low and case['clinical_priority'] in ['Conservatività', 'Rapidità']:
            penalty += 0.10
        if material_type == 'Indirect' and conservative_low and direct_fit > indirect_fit + 0.18:
            penalty += 0.06
        if material_type == 'Indirect' and anterior_sector and case.get('black_class') in ['III', 'IV', 'V'] and idx.ssi < 0.48:
            penalty += 0.08
        if 'flowable' in lc:
            penalty += 0.08
        if 'bulk-fill flowable' in lc:
            penalty += 0.04
        if case['cavity_size'] == 'Piccola' and 'flowable' in lc:
            penalty += 0.04
        if case['indirect_acceptance'] == 'No' and material_type == 'Indirect':
            penalty += 0.22
        if anterior_sector and material_type == 'Indirect' and ('zirconia 3y' in lc or 'allumina' in lc):
            penalty += 0.12
        if anterior_sector and ('bulk-fill' in lc or 'packable' in lc):
            penalty += 0.08

        anterior_extensive_esthetic = (
            anterior_sector and
            idx.edi >= 0.72 and
            case['indirect_acceptance'] == 'Sì' and
            case['max_sessions'] in ['2', '3+'] and
            case['budget_level'] in ['Medio', 'Alto'] and
            (
                case['cavity_size'] == 'Ampia' or
                case.get('black_class') == 'IV' or
                int(case['involved_cusps']) >= 1 or
                case['coronal_tissue'] in ['50-75%', '25-50%']
            )
        )
        posterior_moderate_direct_viable = (
            posterior_sector and
            0.34 <= idx.ssi <= 0.62 and
            not high_load and
            not strong_protection_need and
            int(case['residual_walls']) >= 2 and
            case['isolation'] != 'Impossibile' and
            case['margin'] != 'Subgengivale'
        )
        posterior_single_cusp_direct = _posterior_direct_cuspal_viability(case, idx)
        posterior_single_cusp_direct_viable = bool(posterior_single_cusp_direct['viable'])

        if anterior_sector and idx.edi >= 0.65 and ('nanofilled' in lc or 'nanoibrido' in lc or 'microibrido' in lc):
            bonus += 0.035
        if anterior_sector and idx.edi >= 0.65 and ('disilicato' in lc or 'lithium disilicate' in lc or 'leucite' in lc or 'feldsp' in lc or 'zls' in lc or 'cad/cam composite' in lc or 'resin nanoceramic' in lc or 'picn' in lc):
            bonus += 0.025
        if anterior_extensive_esthetic and material_type == 'Indirect' and ('disilicato' in lc or 'lithium disilicate' in lc or 'zls' in lc or 'leucite' in lc or 'feldsp' in lc or 'cad/cam composite' in lc or 'resin nanoceramic' in lc or 'picn' in lc):
            bonus += 0.055
        if anterior_extensive_esthetic and material_type == 'Direct' and ('nanofilled' in lc or 'nanoibrido' in lc):
            # Direct composites remain valid, but in extended high-esthetic cases
            # they should no longer block the appearance of glass-ceramic options.
            penalty += 0.015

        if posterior_sector and material_type == 'Direct' and ('bulk-fill sculptable' in lc or 'packable' in lc or 'nanoibrido' in lc or 'microibrido' in lc or 'nanofilled' in lc or 'universale' in lc):
            bonus += 0.020
        if posterior_moderate_direct_viable and material_type == 'Direct' and ('bulk-fill sculptable' in lc or 'packable' in lc or 'nanoibrido' in lc or 'microibrido' in lc or 'nanofilled' in lc or 'universale' in lc):
            bonus += 0.050
        if posterior_single_cusp_direct_viable and material_type == 'Direct' and ('bulk-fill sculptable' in lc or 'packable' in lc or 'nanoibrido' in lc or 'microibrido' in lc or 'nanofilled' in lc or 'universale' in lc):
            bonus += 0.070 * float(posterior_single_cusp_direct['strength'])
        if posterior_sector and material_type == 'Indirect' and (protection_high or high_load or indirect_fit > direct_fit + 0.12) and ('zirconia' in lc or 'disilicato' in lc or 'lithium disilicate' in lc or 'zls' in lc):
            bonus += 0.030
        if posterior_moderate_direct_viable and material_type == 'Indirect' and ('zirconia 3y' in lc or 'alta resistenza' in lc):
            penalty += 0.035
        if posterior_single_cusp_direct_viable and material_type == 'Indirect' and ('disilicato' in lc or 'lithium disilicate' in lc or 'zls' in lc or 'zirconia' in lc or 'cad/cam composite' in lc or 'picn' in lc):
            penalty += 0.060 * float(posterior_single_cusp_direct['strength'])

        if case['tooth_group'] == 'Molare' and high_load and ('feldsp' in lc or 'leucite' in lc or lc == 'flowable composite'):
            penalty += 0.18
        if 'flowable' in lc and high_load and protection_high:
            penalty += 0.15
        if 'feldsp' in lc and protection_high:
            penalty += 0.14
        if 'leucite' in lc and high_load:
            penalty += 0.12
        if 'allumina' in lc and not protection_high:
            penalty += 0.10
        if idx.edi > 0.75 and ('zirconia 3y' in lc or 'alta resistenza' in lc):
            penalty += 0.08
        if case['isolation'] == 'Impossibile' and material_type == 'Indirect' and idx.bri > 0.65:
            penalty += 0.08
        if row.get('site_readiness_tier') == 'catalog_only':
            penalty += 0.04

        mechanical_points = 100 * axis_weights['mechanical'] * structural_fit
        biology_points = 100 * axis_weights['biology'] * bio_fit
        esthetic_points = 100 * axis_weights['esthetic'] * esthetic_fit
        workflow_points = 100 * axis_weights['workflow'] * workflow_fit
        evidence_points = 100 * axis_weights['evidence'] * evidence
        material_path_points = 100 * axis_weights['material_path'] * modality_fit

        base = (
            axis_weights['mechanical'] * structural_fit +
            axis_weights['biology'] * bio_fit +
            axis_weights['esthetic'] * esthetic_fit +
            axis_weights['workflow'] * workflow_fit +
            axis_weights['evidence'] * evidence +
            axis_weights['material_path'] * modality_fit
        )
        final = _clamp(base + bonus - penalty)

        pss = _clamp(
            1 / (1 + pow(2.71828, -(
                1.75 * structural_fit +
                1.35 * bio_fit +
                1.05 * esthetic_fit +
                0.85 * workflow_fit +
                0.60 * modality_fit +
                0.55 * evidence +
                0.45 * bonus -
                0.95 * idx.bri -
                0.90 * idx.fsi -
                0.88 * idx.ssi -
                0.45 * penalty
            )))
        )

        material_sources = sources[sources['material_id'] == row['material_id']].head(3)
        source_urls = [u for u in material_sources['source_url'].dropna().tolist()[:3]]

        drivers = []
        if structural_fit > 0.73:
            drivers.append('fit meccanico coerente')
        if bio_fit > 0.70:
            drivers.append('buona compatibilità biologico-adesiva')
        if esthetic_fit > 0.72:
            drivers.append('profilo estetico adeguato')
        if workflow_fit > 0.72:
            drivers.append('workflow compatibile')
        if modality_fit > 0.70:
            drivers.append('tipo materiale diretto/indiretto compatibile')
        if anterior_sector and esthetic_fit > 0.70:
            drivers.append('coerenza con settore anteriore')
        if posterior_sector and structural_fit > 0.70:
            drivers.append('coerenza con settore posteriore')
        if evidence > 0.75:
            drivers.append('buona affidabilità documentale')
        if 'anterior_extensive_esthetic' in locals() and anterior_extensive_esthetic and material_type == 'Indirect' and bonus > 0.0:
            drivers.append('caso anteriore esteso ad alta richiesta estetica')
        if 'posterior_moderate_direct_viable' in locals() and posterior_moderate_direct_viable and material_type == 'Direct' and bonus > 0.0:
            drivers.append('posteriore moderato ancora gestibile con classe diretta')
        if 'posterior_single_cusp_direct_viable' in locals() and posterior_single_cusp_direct_viable and material_type == 'Direct':
            drivers.append('singola cuspide posteriore ancora gestibile con materiale diretto')
        if bonus > 0.0:
            drivers.append('bonus clinico specifico di settore/materiale')
        if penalty > 0.0:
            drivers.append('penalità cliniche controllate applicate')
        if not drivers:
            drivers.append('bilanciamento complessivo accettabile')

        dominant_axis = max(axis_weights, key=axis_weights.get)
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
            'scenario_match': round(modality_fit * 100, 1),
            'restoration_context_fit': 0.0,
            'clinical_sector': _clinical_sector(case),
            'sector_bonus': round(bonus * 100, 1),
            'target_strength': round(target_strength * 100, 1),
            'target_esthetic': round(target_esthetic * 100, 1),
            'target_adhesion': round(target_adhesion * 100, 1),
            'target_workflow': round(target_workflow * 100, 1),
            'weight_mechanical': round(axis_weights['mechanical'] * 100, 1),
            'weight_biology': round(axis_weights['biology'] * 100, 1),
            'weight_esthetic': round(axis_weights['esthetic'] * 100, 1),
            'weight_workflow': round(axis_weights['workflow'] * 100, 1),
            'weight_evidence': round(axis_weights['evidence'] * 100, 1),
            'weight_material_path': round(axis_weights['material_path'] * 100, 1),
            'case_weight_mechanical': round(case_axis_weights['mechanical'] * 100, 1),
            'case_weight_biology': round(case_axis_weights['biology'] * 100, 1),
            'case_weight_esthetic': round(case_axis_weights['esthetic'] * 100, 1),
            'case_weight_workflow': round(case_axis_weights['workflow'] * 100, 1),
            'case_weight_evidence': round(case_axis_weights['evidence'] * 100, 1),
            'case_weight_material_path': round(case_axis_weights['material_path'] * 100, 1),
            'db_weight_mechanical': round(database_axis_weights['mechanical'] * 100, 1),
            'db_weight_biology': round(database_axis_weights['biology'] * 100, 1),
            'db_weight_esthetic': round(database_axis_weights['esthetic'] * 100, 1),
            'db_weight_workflow': round(database_axis_weights['workflow'] * 100, 1),
            'db_weight_evidence': round(database_axis_weights['evidence'] * 100, 1),
            'db_weight_material_path': round(database_axis_weights['material_path'] * 100, 1),
            'mechanical_points': round(mechanical_points, 1),
            'biology_points': round(biology_points, 1),
            'esthetic_points': round(esthetic_points, 1),
            'workflow_points': round(workflow_points, 1),
            'evidence_points': round(evidence_points, 1),
            'material_path_points': round(material_path_points, 1),
            'bonus_points': round(bonus * 100, 1),
            'penalty_points': round(penalty * 100, 1),
            'dominant_material_axis': dominant_axis,
            'score_model_version': 'material_axis_v2.9.6',
            'top_drivers': '; '.join(drivers),
            'source_urls': source_urls,
            'official_manufacturer_wording': row.get('official_manufacturer_wording', ''),
            'source_record_count': int(float(row.get('source_record_count', 0) or 0)),
            'quantitative_observation_count': int(float(row.get('quantitative_observation_count', 0) or 0)),
        })

    ranked_classes = add_material_decision_metadata(_build_class_ranking(rows))
    return rest_df, ranked_classes

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
    if _is_anterior(case) and idx.edi >= 0.55:
        items.append('settore anteriore con impatto estetico')
    elif _is_posterior(case) and idx.fsi >= 0.45:
        items.append('settore posteriore con impatto funzionale')
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
