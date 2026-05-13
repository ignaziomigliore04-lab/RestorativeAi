from __future__ import annotations

import random
from typing import Dict, List, Tuple

import pandas as pd

from utils.engine import (
    add_material_decision_metadata,
    assess_material_case_feasibility,
    compute_case_indices,
    rank_materials,
    _posterior_direct_cuspal_viability,
)


COVERAGE_OPTIONS: Dict[str, List[object]] = {
    'patient_sex': ['Non specificato', 'Femmina', 'Maschio'],
    'clinical_sector': ['Anteriore', 'Posteriore'],
    'black_class': ['I', 'II', 'III', 'IV', 'V', 'VI'],
    'vitality': ['Vitale', 'Non vitale'],
    'endo_treated': ['No', 'Sì'],
    'clinical_priority': ['Conservatività', 'Durata', 'Estetica', 'Rapidità'],
    'residual_walls': [0, 1, 2, 3, 4],
    'marginal_ridges': [0, 1, 2],
    'cusp_loss': ['No', 'Sì'],
    'involved_cusps': [0, 1, 2, 3, 4],
    'ferrule': ['Presente', 'Parziale', 'Assente'],
    'cavity_size': ['Piccola', 'Media', 'Ampia'],
    'coronal_tissue': ['>75%', '50-75%', '25-50%', '<25%'],
    'wall_thickness': ['Adeguato', 'Sottile', 'Molto sottile'],
    'crack': ['No', 'Sì'],
    'pulp_proximity': ['Bassa', 'Media', 'Alta'],
    'caries_risk': ['Basso', 'Medio', 'Alto'],
    'plaque_control': ['Buono', 'Medio', 'Scarso'],
    'xerostomia': ['No', 'Sì'],
    'isolation': ['Facile', 'Difficile', 'Impossibile'],
    'margin': ['Sovragengivale', 'Juxtagengivale', 'Subgengivale'],
    'periodontal_support': ['Buono', 'Ridotto'],
    'compliance': ['Alta', 'Media', 'Bassa'],
    'adhesive_context': ['Favorevole', 'Intermedio', 'Sfavorevole'],
    'esthetic_demand': ['Bassa', 'Media', 'Alta'],
    'bruxism': ['Assente', 'Sospetta', 'Confermata'],
    'parafunction_severity': ['Assente', 'Lieve', 'Moderata', 'Severa'],
    'occlusal_load': ['Basso', 'Medio', 'Alto'],
    'eccentric_contacts': ['Assenti', 'Presenti'],
    'antagonist': ['Naturale', 'Restauro', 'Protesico'],
    'tooth_wear': ['Assente', 'Lieve', 'Moderata', 'Severa'],
    'cadcam_available': ['No', 'Sì'],
    'indirect_acceptance': ['Sì', 'No', 'Incerta'],
    'max_sessions': ['1', '2', '3+'],
    'budget_level': ['Basso', 'Medio', 'Alto'],
    'workflow_preference': ['Chairside', 'Laboratorio', 'Indifferente'],
}


def _tooth_for(sector: str, rng: random.Random, force_group: str | None = None) -> Tuple[str, str]:
    if force_group == 'Premolare':
        tooth = rng.choice(['14', '15', '24', '25', '34', '35', '44', '45'])
    elif force_group == 'Molare':
        tooth = rng.choice(['16', '17', '26', '27', '36', '37', '46', '47'])
    elif sector == 'Anteriore' or force_group == 'Anteriore':
        tooth = rng.choice(['11', '12', '13', '21', '22', '23', '31', '32', '33', '41', '42', '43'])
    else:
        tooth = rng.choice(['14', '15', '16', '17', '24', '25', '26', '27', '34', '35', '36', '37', '44', '45', '46', '47'])
    second = tooth[1]
    group = 'Anteriore' if second in ['1', '2', '3'] else 'Premolare' if second in ['4', '5'] else 'Molare'
    return tooth, group


def _base_case(sector: str, rng: random.Random, force_group: str | None = None) -> Dict[str, object]:
    tooth, group = _tooth_for(sector, rng, force_group=force_group)
    clinical_sector = 'Anteriore' if group == 'Anteriore' else 'Posteriore'
    return {
        'patient_age': rng.choice([22, 28, 35, 44, 57, 68]),
        'patient_sex': rng.choice(['Non specificato', 'Femmina', 'Maschio']),
        'tooth_number': tooth,
        'tooth_group': group,
        'clinical_sector': clinical_sector,
        'arch': 'Mascellare' if tooth.startswith(('1', '2')) else 'Mandibolare',
        'black_class': rng.choice(['III', 'IV', 'V']) if clinical_sector == 'Anteriore' else rng.choice(['I', 'II', 'V', 'VI']),
        'vitality': 'Vitale',
        'endo_treated': 'No',
        'clinical_priority': rng.choice(['Conservatività', 'Durata', 'Estetica', 'Rapidità']),
        'residual_walls': rng.choice([2, 3, 4]),
        'marginal_ridges': rng.choice([1, 2]),
        'cusp_loss': 'No',
        'involved_cusps': 0,
        'ferrule': 'Presente',
        'cavity_size': rng.choice(['Piccola', 'Media']),
        'coronal_tissue': rng.choice(['>75%', '50-75%']),
        'wall_thickness': rng.choice(['Adeguato', 'Sottile']),
        'crack': 'No',
        'pulp_proximity': rng.choice(['Bassa', 'Media']),
        'caries_risk': rng.choice(['Basso', 'Medio']),
        'plaque_control': rng.choice(['Buono', 'Medio']),
        'xerostomia': 'No',
        'isolation': rng.choice(['Facile', 'Difficile']),
        'margin': rng.choice(['Sovragengivale', 'Juxtagengivale']),
        'periodontal_support': 'Buono',
        'compliance': rng.choice(['Alta', 'Media']),
        'adhesive_context': rng.choice(['Favorevole', 'Intermedio']),
        'esthetic_demand': rng.choice(['Media', 'Alta']) if clinical_sector == 'Anteriore' else rng.choice(['Bassa', 'Media']),
        'bruxism': rng.choice(['Assente', 'Sospetta']),
        'parafunction_severity': rng.choice(['Assente', 'Lieve']),
        'occlusal_load': rng.choice(['Basso', 'Medio']),
        'eccentric_contacts': 'Assenti',
        'antagonist': rng.choice(['Naturale', 'Restauro']),
        'tooth_wear': rng.choice(['Assente', 'Lieve']),
        'cadcam_available': rng.choice(['No', 'Sì']),
        'indirect_acceptance': rng.choice(['Sì', 'Incerta']),
        'max_sessions': rng.choice(['1', '2', '3+']),
        'budget_level': rng.choice(['Basso', 'Medio', 'Alto']),
        'workflow_preference': rng.choice(['Chairside', 'Laboratorio', 'Indifferente']),
    }


def _apply_archetype(case: Dict[str, object], archetype: str, rng: random.Random) -> Dict[str, object]:
    case['validation_archetype'] = archetype
    if archetype == 'anterior_small_direct':
        case.update({
            'black_class': rng.choice(['III', 'V']), 'clinical_priority': rng.choice(['Conservatività', 'Estetica']),
            'residual_walls': 4, 'marginal_ridges': 2, 'cavity_size': 'Piccola', 'coronal_tissue': '>75%',
            'wall_thickness': 'Adeguato', 'cusp_loss': 'No', 'involved_cusps': 0,
            'esthetic_demand': 'Alta', 'isolation': 'Facile', 'margin': 'Sovragengivale',
            'indirect_acceptance': rng.choice(['No', 'Incerta']), 'max_sessions': '1', 'workflow_preference': 'Chairside'
        })
    elif archetype == 'anterior_extended_esthetic':
        case.update({
            'black_class': 'IV', 'clinical_priority': 'Estetica', 'residual_walls': rng.choice([2, 3]),
            'cavity_size': 'Ampia', 'coronal_tissue': rng.choice(['50-75%', '25-50%']),
            'wall_thickness': rng.choice(['Sottile', 'Adeguato']), 'esthetic_demand': 'Alta',
            'isolation': rng.choice(['Facile', 'Difficile']), 'margin': rng.choice(['Sovragengivale', 'Juxtagengivale']),
            'indirect_acceptance': 'Sì', 'cadcam_available': 'Sì', 'max_sessions': rng.choice(['2', '3+']),
            'budget_level': rng.choice(['Medio', 'Alto']), 'workflow_preference': rng.choice(['Laboratorio', 'Indifferente'])
        })
    elif archetype == 'posterior_small_conservative':
        case.update({
            'clinical_priority': rng.choice(['Conservatività', 'Rapidità']), 'residual_walls': 4,
            'marginal_ridges': 2, 'cavity_size': 'Piccola', 'coronal_tissue': '>75%',
            'wall_thickness': 'Adeguato', 'cusp_loss': 'No', 'involved_cusps': 0,
            'occlusal_load': rng.choice(['Basso', 'Medio']), 'bruxism': 'Assente',
            'indirect_acceptance': rng.choice(['No', 'Incerta']), 'max_sessions': '1', 'workflow_preference': 'Chairside'
        })
    elif archetype == 'posterior_moderate_direct_viable':
        case.update({
            'clinical_priority': rng.choice(['Conservatività', 'Durata', 'Rapidità']), 'residual_walls': rng.choice([2, 3]),
            'marginal_ridges': rng.choice([1, 2]), 'cavity_size': rng.choice(['Media', 'Ampia']),
            'coronal_tissue': '50-75%', 'wall_thickness': rng.choice(['Adeguato', 'Sottile']),
            'cusp_loss': 'No', 'involved_cusps': rng.choice([0, 1]), 'endo_treated': 'No',
            'occlusal_load': 'Medio', 'bruxism': rng.choice(['Assente', 'Sospetta']), 'parafunction_severity': rng.choice(['Assente', 'Lieve']),
            'isolation': rng.choice(['Facile', 'Difficile']), 'margin': rng.choice(['Sovragengivale', 'Juxtagengivale']),
            'indirect_acceptance': rng.choice(['Sì', 'Incerta']), 'max_sessions': rng.choice(['1', '2']),
            'workflow_preference': rng.choice(['Chairside', 'Indifferente'])
        })
    elif archetype == 'posterior_single_cusp_direct':
        tooth, group = _tooth_for('Posteriore', rng, force_group='Molare')
        case.update({
            'tooth_number': tooth, 'tooth_group': group, 'clinical_sector': 'Posteriore',
            'black_class': rng.choice(['I', 'II']), 'clinical_priority': rng.choice(['Conservatività', 'Durata', 'Rapidità']),
            'vitality': 'Vitale', 'endo_treated': 'No', 'residual_walls': rng.choice([2, 3]),
            'marginal_ridges': rng.choice([1, 2]), 'cavity_size': rng.choice(['Media', 'Ampia']),
            'coronal_tissue': rng.choice(['>75%', '50-75%']), 'wall_thickness': rng.choice(['Adeguato', 'Sottile']),
            'cusp_loss': 'Sì', 'involved_cusps': 1, 'ferrule': rng.choice(['Presente', 'Parziale']),
            'crack': 'No', 'isolation': rng.choice(['Facile', 'Difficile']),
            'margin': rng.choice(['Sovragengivale', 'Juxtagengivale']), 'adhesive_context': rng.choice(['Favorevole', 'Intermedio']),
            'occlusal_load': rng.choice(['Basso', 'Medio']), 'bruxism': rng.choice(['Assente', 'Sospetta']),
            'parafunction_severity': rng.choice(['Assente', 'Lieve']), 'tooth_wear': rng.choice(['Assente', 'Lieve']),
            'antagonist': rng.choice(['Naturale', 'Restauro']), 'indirect_acceptance': rng.choice(['Sì', 'Incerta', 'No']),
            'max_sessions': rng.choice(['1', '2']), 'budget_level': rng.choice(['Basso', 'Medio']),
            'workflow_preference': rng.choice(['Chairside', 'Indifferente'])
        })
    elif archetype == 'posterior_moderate_indirect_possible':
        case.update({
            'clinical_priority': 'Durata', 'residual_walls': rng.choice([2, 3]), 'marginal_ridges': 1,
            'cavity_size': 'Ampia', 'coronal_tissue': '50-75%', 'wall_thickness': 'Sottile',
            'cusp_loss': rng.choice(['No', 'Sì']), 'involved_cusps': rng.choice([1, 2]),
            'occlusal_load': rng.choice(['Medio', 'Alto']), 'bruxism': rng.choice(['Assente', 'Sospetta']),
            'indirect_acceptance': 'Sì', 'cadcam_available': 'Sì', 'max_sessions': rng.choice(['2', '3+']),
            'budget_level': rng.choice(['Medio', 'Alto']), 'workflow_preference': rng.choice(['Laboratorio', 'Indifferente'])
        })
    elif archetype == 'posterior_high_load':
        case.update({
            'clinical_priority': 'Durata', 'residual_walls': rng.choice([1, 2]), 'marginal_ridges': rng.choice([0, 1]),
            'cavity_size': 'Ampia', 'coronal_tissue': rng.choice(['25-50%', '<25%']), 'wall_thickness': rng.choice(['Sottile', 'Molto sottile']),
            'cusp_loss': 'Sì', 'involved_cusps': rng.choice([2, 3]), 'bruxism': 'Confermata',
            'parafunction_severity': rng.choice(['Moderata', 'Severa']), 'occlusal_load': 'Alto', 'eccentric_contacts': 'Presenti',
            'tooth_wear': rng.choice(['Moderata', 'Severa']), 'indirect_acceptance': 'Sì', 'max_sessions': rng.choice(['2', '3+']),
            'budget_level': rng.choice(['Medio', 'Alto'])
        })
    elif archetype == 'endo_structural_loss':
        case.update({
            'vitality': 'Non vitale', 'endo_treated': 'Sì', 'clinical_priority': 'Durata',
            'residual_walls': rng.choice([1, 2]), 'marginal_ridges': rng.choice([0, 1]), 'cavity_size': 'Ampia',
            'coronal_tissue': rng.choice(['25-50%', '<25%']), 'ferrule': rng.choice(['Parziale', 'Presente']),
            'wall_thickness': rng.choice(['Sottile', 'Molto sottile']), 'cusp_loss': 'Sì', 'involved_cusps': rng.choice([2, 3, 4]),
            'occlusal_load': rng.choice(['Medio', 'Alto']), 'indirect_acceptance': 'Sì', 'max_sessions': rng.choice(['2', '3+'])
        })
    elif archetype == 'high_biologic_risk':
        case.update({
            'caries_risk': 'Alto', 'plaque_control': rng.choice(['Medio', 'Scarso']), 'xerostomia': rng.choice(['No', 'Sì']),
            'isolation': rng.choice(['Difficile', 'Impossibile']), 'margin': rng.choice(['Juxtagengivale', 'Subgengivale']),
            'periodontal_support': rng.choice(['Buono', 'Ridotto']), 'compliance': rng.choice(['Media', 'Bassa']),
            'adhesive_context': rng.choice(['Intermedio', 'Sfavorevole']), 'clinical_priority': rng.choice(['Durata', 'Conservatività']),
            'residual_walls': rng.choice([2, 3]), 'coronal_tissue': rng.choice(['50-75%', '25-50%'])
        })
    elif archetype == 'workflow_constrained':
        case.update({
            'clinical_priority': 'Rapidità', 'cadcam_available': 'No', 'indirect_acceptance': rng.choice(['No', 'Incerta']),
            'max_sessions': '1', 'budget_level': 'Basso', 'workflow_preference': 'Chairside',
            'cavity_size': rng.choice(['Piccola', 'Media']), 'coronal_tissue': rng.choice(['>75%', '50-75%']),
            'residual_walls': rng.choice([3, 4]), 'cusp_loss': 'No', 'involved_cusps': 0
        })
    elif archetype == 'premolar_balanced':
        tooth, group = _tooth_for('Posteriore', rng, force_group='Premolare')
        case.update({
            'tooth_number': tooth, 'tooth_group': group, 'clinical_sector': 'Posteriore',
            'clinical_priority': rng.choice(['Durata', 'Estetica', 'Conservatività']), 'esthetic_demand': rng.choice(['Media', 'Alta']),
            'residual_walls': rng.choice([2, 3]), 'marginal_ridges': rng.choice([1, 2]),
            'cavity_size': rng.choice(['Media', 'Ampia']), 'coronal_tissue': rng.choice(['50-75%', '25-50%']),
            'cusp_loss': rng.choice(['No', 'Sì']), 'involved_cusps': rng.choice([0, 1, 2]),
            'occlusal_load': rng.choice(['Medio', 'Alto']), 'indirect_acceptance': rng.choice(['Sì', 'Incerta']),
            'max_sessions': rng.choice(['1', '2', '3+'])
        })
    return _repair_plausibility(case, rng)


def _repair_plausibility(case: Dict[str, object], rng: random.Random) -> Dict[str, object]:
    """Keep generated scenarios possible in a restorative-material decision context."""
    # Anterior/posterior coherence with Black classes.
    if case['clinical_sector'] == 'Anteriore' and case['black_class'] in ['I', 'II', 'VI']:
        case['black_class'] = rng.choice(['III', 'IV', 'V'])
    if case['clinical_sector'] == 'Posteriore' and case['black_class'] in ['III', 'IV']:
        case['black_class'] = rng.choice(['I', 'II', 'V', 'VI'])
    # Endodontic and vitality coherence.
    if case['endo_treated'] == 'Sì':
        case['vitality'] = 'Non vitale'
    if case['vitality'] == 'Vitale':
        case['endo_treated'] = 'No'
    # Structural coherence.
    if int(case['residual_walls']) >= 3:
        case['ferrule'] = 'Presente' if case['ferrule'] == 'Assente' else case['ferrule']
        case['coronal_tissue'] = rng.choice(['>75%', '50-75%']) if case['coronal_tissue'] == '<25%' else case['coronal_tissue']
    if int(case['residual_walls']) == 0:
        case['cavity_size'] = 'Ampia'
        case['coronal_tissue'] = '25-50%'
        case['ferrule'] = 'Parziale'
        case['cusp_loss'] = 'Sì'
        case['involved_cusps'] = max(int(case['involved_cusps']), 3)
        case['indirect_acceptance'] = 'Sì'
        case['max_sessions'] = rng.choice(['2', '3+'])
    if case['coronal_tissue'] == '<25%' and case['ferrule'] == 'Assente' and int(case['residual_walls']) <= 1:
        case['ferrule'] = 'Parziale'
    if case['cusp_loss'] == 'No':
        case['involved_cusps'] = min(int(case['involved_cusps']), 1)
    if int(case['involved_cusps']) >= 2:
        case['cusp_loss'] = 'Sì'
    # Biological/actionability coherence.
    if case['isolation'] == 'Impossibile' and case['margin'] == 'Subgengivale' and case['adhesive_context'] == 'Sfavorevole':
        case['adhesive_context'] = 'Intermedio'
    if case['caries_risk'] == 'Alto' and case['plaque_control'] == 'Scarso' and case['compliance'] == 'Bassa' and case['xerostomia'] == 'Sì':
        case['compliance'] = 'Media'
    # Functional coherence.
    if case['bruxism'] == 'Confermata':
        case['occlusal_load'] = 'Alto' if case['occlusal_load'] == 'Basso' else case['occlusal_load']
        case['parafunction_severity'] = rng.choice(['Moderata', 'Severa']) if case['parafunction_severity'] in ['Assente', 'Lieve'] else case['parafunction_severity']
    if case['parafunction_severity'] == 'Severa':
        case['bruxism'] = 'Confermata'
        case['occlusal_load'] = 'Alto'
        case['tooth_wear'] = rng.choice(['Moderata', 'Severa']) if case['tooth_wear'] in ['Assente', 'Lieve'] else case['tooth_wear']
    if case['bruxism'] == 'Confermata' and case['parafunction_severity'] == 'Severa' and case['occlusal_load'] == 'Alto' and case['tooth_wear'] == 'Severa' and int(case['involved_cusps']) >= 3 and case['indirect_acceptance'] == 'No':
        case['indirect_acceptance'] = 'Incerta'
    return case


def _make_archetype_case(archetype: str, rng: random.Random) -> Dict[str, object]:
    sector = 'Anteriore' if archetype.startswith('anterior') else 'Posteriore'
    force_group = 'Premolare' if archetype == 'premolar_balanced' else None
    return _apply_archetype(_base_case(sector, rng, force_group=force_group), archetype, rng)


ARCHETYPES = [
    'anterior_small_direct',
    'anterior_extended_esthetic',
    'posterior_small_conservative',
    'posterior_moderate_direct_viable',
    'posterior_single_cusp_direct',
    'posterior_moderate_indirect_possible',
    'posterior_high_load',
    'endo_structural_loss',
    'high_biologic_risk',
    'workflow_constrained',
    'premolar_balanced',
]


def _case_for_target(field: str, value: object, rng: random.Random) -> Dict[str, object]:
    if field == 'clinical_sector' and value == 'Anteriore':
        case = _make_archetype_case(rng.choice(['anterior_small_direct', 'anterior_extended_esthetic']), rng)
    elif field == 'clinical_sector' and value == 'Posteriore':
        case = _make_archetype_case(rng.choice(['posterior_small_conservative', 'posterior_moderate_direct_viable', 'posterior_single_cusp_direct', 'posterior_high_load']), rng)
    elif field == 'black_class' and value in ['III', 'IV']:
        case = _make_archetype_case('anterior_extended_esthetic' if value == 'IV' else 'anterior_small_direct', rng)
    elif field == 'cusp_loss' and value == 'Sì':
        case = _make_archetype_case(rng.choice(['posterior_single_cusp_direct', 'posterior_moderate_indirect_possible', 'posterior_high_load']), rng)
    elif field == 'involved_cusps' and value == 1:
        case = _make_archetype_case(rng.choice(['posterior_single_cusp_direct', 'posterior_moderate_direct_viable']), rng)
    elif field in {'residual_walls', 'ferrule', 'coronal_tissue', 'wall_thickness', 'involved_cusps', 'marginal_ridges', 'endo_treated'} and value in [0, 1, 4, 'Assente', '<25%', 'Molto sottile', 'Sì']:
        case = _make_archetype_case(rng.choice(['endo_structural_loss', 'posterior_high_load', 'posterior_moderate_indirect_possible']), rng)
    elif field in {'isolation', 'margin', 'caries_risk', 'plaque_control', 'compliance', 'adhesive_context', 'xerostomia', 'periodontal_support'} and value in ['Impossibile', 'Subgengivale', 'Alto', 'Scarso', 'Bassa', 'Sfavorevole', 'Sì', 'Ridotto']:
        case = _make_archetype_case('high_biologic_risk', rng)
    elif field in {'bruxism', 'parafunction_severity', 'occlusal_load', 'eccentric_contacts', 'antagonist', 'tooth_wear'} and value in ['Confermata', 'Severa', 'Alto', 'Presenti', 'Protesico']:
        case = _make_archetype_case('posterior_high_load', rng)
    elif field in {'indirect_acceptance', 'max_sessions', 'budget_level', 'workflow_preference', 'cadcam_available'} and value in ['No', '1', 'Basso', 'Chairside']:
        case = _make_archetype_case('workflow_constrained', rng)
    elif field == 'esthetic_demand' and value == 'Alta':
        case = _make_archetype_case('anterior_extended_esthetic', rng)
    else:
        case = _make_archetype_case(rng.choice(ARCHETYPES), rng)

    if field == 'patient_sex':
        case[field] = value
    elif field == 'clinical_sector':
        # Already handled by choosing tooth/sector.
        pass
    elif field == 'black_class':
        case[field] = value
        if value in ['III', 'IV']:
            tooth, group = _tooth_for('Anteriore', rng)
            case.update({'tooth_number': tooth, 'tooth_group': group, 'clinical_sector': 'Anteriore'})
        elif value in ['I', 'II', 'VI']:
            tooth, group = _tooth_for('Posteriore', rng)
            case.update({'tooth_number': tooth, 'tooth_group': group, 'clinical_sector': 'Posteriore'})
    else:
        case[field] = value

    if field == 'ferrule' and value == 'Assente':
        # Feasible but severe: absence of ferrule is represented without also
        # combining it with the extreme non-actionable trio 0 walls + <25% tissue.
        case.update({
            'residual_walls': 2,
            'coronal_tissue': '25-50%',
            'endo_treated': 'Sì',
            'vitality': 'Non vitale',
            'cavity_size': 'Ampia',
            'indirect_acceptance': 'Sì',
            'max_sessions': rng.choice(['2', '3+']),
        })

    return _repair_plausibility(case, rng)


def _is_actionable(case: Dict[str, object]) -> bool:
    idx = compute_case_indices(case)
    return bool(assess_material_case_feasibility(case, idx)['is_actionable'])


def _case_signature(case: Dict[str, object]) -> Tuple[Tuple[str, str], ...]:
    keys = [k for k in COVERAGE_OPTIONS if k in case]
    return tuple((k, str(case[k])) for k in keys)


def _coverage_df(cases: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for field, expected_values in COVERAGE_OPTIONS.items():
        present = {str(case.get(field)) for case in cases}
        for value in expected_values:
            rows.append({
                'input_field': field,
                'level': str(value),
                'covered': str(value) in present,
                'count': sum(1 for case in cases if str(case.get(field)) == str(value)),
            })
    return pd.DataFrame(rows)


def generate_validation_cases(n: int = 200, seed: int = 42) -> List[Dict[str, object]]:
    """Generate deterministic, feasible, clinically plausible material-selection cases.

    The generator first forces coverage of the categorical input levels, then
    fills with balanced archetypes. Non-actionable combinations are repaired or
    discarded: the validation set is meant to represent cases where choosing a
    restorative material is realistically possible.
    """
    rng = random.Random(seed)
    cases: List[Dict[str, object]] = []
    signatures = set()

    def add_case(case: Dict[str, object]) -> bool:
        case = _repair_plausibility(case, rng)
        if not _is_actionable(case):
            return False
        sig = _case_signature(case)
        if sig in signatures:
            # add small variability to avoid exact duplicates
            case['patient_age'] = rng.choice([19, 24, 31, 39, 48, 61, 74])
            sig = _case_signature(case) + (('patient_age', str(case['patient_age'])),)
        signatures.add(sig)
        cases.append(case)
        return True

    # Coverage pass: every input level gets an attempted realistic case.
    for field, values in COVERAGE_OPTIONS.items():
        for value in values:
            for _ in range(12):
                case = _case_for_target(field, value, rng)
                if add_case(case):
                    break

    # Fill with balanced archetypes.
    i = 0
    while len(cases) < n and i < n * 30:
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        add_case(_make_archetype_case(archetype, rng))
        i += 1

    return cases[:n]


def _flag_case(case: Dict[str, object], idx, ranked: pd.DataFrame) -> Tuple[List[str], List[str]]:
    red: List[str] = []
    yellow: List[str] = []
    feasibility = assess_material_case_feasibility(case, idx)
    if not feasibility['is_actionable']:
        red.append('caso non actionable incluso nella validazione')
    if ranked.empty:
        return ['ranking vuoto'], yellow
    top = ranked.iloc[0]
    class_name = str(top.get('primary_class_name', '')).lower()
    top_type = str(top.get('direct_or_indirect', ''))
    second_gap = top.get('score_gap_to_next')
    gap_val = None if pd.isna(second_gap) else float(second_gap)
    high_load = idx.fsi >= 0.65 or case['occlusal_load'] == 'Alto' or case['bruxism'] == 'Confermata'
    strong_protection = idx.ssi >= 0.68 or int(case['involved_cusps']) >= 2 or (case['endo_treated'] == 'Sì' and case['coronal_tissue'] in ['25-50%', '<25%'])
    if 'flowable' in class_name and high_load:
        red.append('flowable come top in caso ad alto carico')
    if case['clinical_sector'] == 'Anteriore' and idx.edi >= 0.72 and ('zirconia 3y' in class_name or 'alta resistenza' in class_name):
        red.append('zirconia 3Y/high-strength top in anteriore ad alta estetica')
    if case['tooth_group'] == 'Molare' and high_load and ('feldsp' in class_name or 'leucite' in class_name):
        red.append('ceramica fragile/top estetica in molare ad alto carico')
    if case['indirect_acceptance'] == 'No' and top_type == 'Indirect':
        red.append('classe indiretta top nonostante mancata accettazione')
    direct_cusp_signal = _posterior_direct_cuspal_viability(case, idx)
    if top_type == 'Indirect' and direct_cusp_signal['viable']:
        yellow.append('classe indiretta top in caso dove la ricostruzione diretta cuspidale è plausibile')
    if top_type == 'Direct' and high_load and strong_protection:
        yellow.append('classe diretta top in caso con alta richiesta protettiva')
    if gap_val is not None and gap_val < 2.0:
        yellow.append('gap <2 punti: classi materialmente quasi equivalenti')
    return red, yellow


def run_material_validation(materials_df: pd.DataFrame, sources_df: pd.DataFrame, n: int = 200, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    cases = generate_validation_cases(n, seed)
    coverage = _coverage_df(cases)
    for case_id, case in enumerate(cases, start=1):
        idx = compute_case_indices(case)
        feasibility = assess_material_case_feasibility(case, idx)
        if not feasibility['is_actionable']:
            rows.append({
                'case_id': case_id,
                'archetype': case.get('validation_archetype'),
                'sector': case.get('clinical_sector'),
                'top_class': 'NO MATERIAL RECOMMENDATION',
                'top_type': None,
                'red_flags': '; '.join(feasibility['reasons']),
                'yellow_flags': '',
            })
            continue
        rest_df, ranked = rank_materials(case, idx, materials_df, sources_df)
        ranked = add_material_decision_metadata(ranked)
        top = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else None
        red, yellow = _flag_case(case, idx, ranked)
        rows.append({
            'case_id': case_id,
            'archetype': case.get('validation_archetype'),
            'sector': case.get('clinical_sector'),
            'tooth_number': case.get('tooth_number'),
            'tooth_group': case.get('tooth_group'),
            'patient_sex': case.get('patient_sex'),
            'black_class': case.get('black_class'),
            'vitality': case.get('vitality'),
            'endo_treated': case.get('endo_treated'),
            'clinical_priority': case.get('clinical_priority'),
            'cavity_size': case.get('cavity_size'),
            'residual_walls': case.get('residual_walls'),
            'marginal_ridges': case.get('marginal_ridges'),
            'cusp_loss': case.get('cusp_loss'),
            'involved_cusps': case.get('involved_cusps'),
            'ferrule': case.get('ferrule'),
            'coronal_tissue': case.get('coronal_tissue'),
            'wall_thickness': case.get('wall_thickness'),
            'crack': case.get('crack'),
            'pulp_proximity': case.get('pulp_proximity'),
            'caries_risk': case.get('caries_risk'),
            'plaque_control': case.get('plaque_control'),
            'xerostomia': case.get('xerostomia'),
            'isolation': case.get('isolation'),
            'margin': case.get('margin'),
            'periodontal_support': case.get('periodontal_support'),
            'compliance': case.get('compliance'),
            'adhesive_context': case.get('adhesive_context'),
            'esthetic_demand': case.get('esthetic_demand'),
            'bruxism': case.get('bruxism'),
            'parafunction_severity': case.get('parafunction_severity'),
            'occlusal_load': case.get('occlusal_load'),
            'eccentric_contacts': case.get('eccentric_contacts'),
            'antagonist': case.get('antagonist'),
            'tooth_wear': case.get('tooth_wear'),
            'cadcam_available': case.get('cadcam_available'),
            'indirect_acceptance': case.get('indirect_acceptance'),
            'max_sessions': case.get('max_sessions'),
            'budget_level': case.get('budget_level'),
            'workflow_preference': case.get('workflow_preference'),
            'actionable': feasibility['is_actionable'],
            'SSI': round(idx.ssi * 100, 1),
            'BRI': round(idx.bri * 100, 1),
            'FSI': round(idx.fsi * 100, 1),
            'EDI': round(idx.edi * 100, 1),
            'WCI': round(idx.wci * 100, 1),
            'top_class': top.get('primary_class_name'),
            'top_type': top.get('direct_or_indirect'),
            'top_score': float(top.get('final_score', 0.0)),
            'top_pss': float(top.get('pss', 0.0)),
            'second_class': None if second is None else second.get('primary_class_name'),
            'second_score': None if second is None else float(second.get('final_score', 0.0)),
            'gap': None if second is None else round(float(top.get('final_score', 0.0)) - float(second.get('final_score', 0.0)), 1),
            'confidence': top.get('confidence_label'),
            'dominant_axis': top.get('dominant_material_axis'),
            'mechanical_weight': top.get('weight_mechanical'),
            'biology_weight': top.get('weight_biology'),
            'esthetic_weight': top.get('weight_esthetic'),
            'workflow_weight': top.get('weight_workflow'),
            'evidence_weight': top.get('weight_evidence'),
            'red_flags': '; '.join(red),
            'yellow_flags': '; '.join(yellow),
        })
    df = pd.DataFrame(rows)
    actionable_df = df[df.get('actionable', True) == True].copy() if 'actionable' in df else df
    summary = {
        'cases': int(len(df)),
        'actionable_cases': int(actionable_df.shape[0]),
        'excluded_cases': int(len(df) - actionable_df.shape[0]),
        'errors': 0,
        'red_flags': int(df['red_flags'].astype(str).str.len().gt(0).sum()),
        'yellow_flags': int(df['yellow_flags'].astype(str).str.len().gt(0).sum()),
        'mean_top_score': round(float(actionable_df['top_score'].mean()), 1) if not actionable_df.empty else 0.0,
        'median_top_score': round(float(actionable_df['top_score'].median()), 1) if not actionable_df.empty else 0.0,
        'mean_gap': round(float(actionable_df['gap'].dropna().mean()), 1) if not actionable_df.empty else 0.0,
        'gap_under_2': int((actionable_df['gap'].dropna() < 2.0).sum()) if not actionable_df.empty else 0,
        'direct_pct': round(float((actionable_df['top_type'] == 'Direct').mean() * 100), 1) if not actionable_df.empty else 0.0,
        'indirect_pct': round(float((actionable_df['top_type'] == 'Indirect').mean() * 100), 1) if not actionable_df.empty else 0.0,
        'classes_top_count': int(actionable_df['top_class'].nunique()) if not actionable_df.empty else 0,
        'coverage_total_levels': int(coverage.shape[0]),
        'coverage_covered_levels': int(coverage['covered'].sum()),
        'coverage_missing_levels': int((~coverage['covered']).sum()),
        'coverage_table': coverage,
    }
    return df, summary
