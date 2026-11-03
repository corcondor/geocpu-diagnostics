# -*- coding: utf-8 -*-
"""
eval_v38_decomposed_state_bench.py: 4-Stage Decomposed State Tracking Evaluator.

Grades 5 distinct metrics for every sample:
1. Stage 1 (Name Extraction Precision & Recall): Did the model extract the exact prompt entities without hallucinating?
2. Stage 2 (Table Bijection): Did the model construct a coherent 1-to-1 mapping between entities and IDs?
3. Stage 3 (CPU Logic Grounding): Did the register hold the ID corresponding to the ground-truth winner, judged by the model's OWN table?
4. Stage 4 (Table Inversion Coherence): Did the prose answer match the entity bound to that register ID in the model's OWN table?
5. End-to-End Exact Match: Final answer string matches ground truth.
"""
import re
import json
from typing import Dict, Any, List, Set, Tuple

def parse_model_table(output_text: str) -> Dict[str, int]:
    """
    Extracts name-to-ID table declared by model in scratchpad.
    Matches formats like:
      登場人物コード: 木村=1, 阿部=2, 橋本=3
      対応表: 田中=1, 鈴木=2
      A=1, B=2, C=3
    """
    table = {}
    # Look for declaration region
    m = re.search(r'(登場人物コード|対応表|コード一覧|所持者コード|割当)[:：]?\s*([^。\n<]+)', output_text)
    candidate_str = m.group(2) if m else output_text[:120]
    
    pairs = re.findall(r'([^\s=,、]+)\s*=\s*(\d+)', candidate_str)
    for name, sid in pairs:
        name = name.strip().lstrip('「【').rstrip('」】')
        # Filter out register names like R0, R1
        if not re.match(r'^R\d+$', name):
            table[name] = int(sid)
    return table

def parse_final_register_state(output_text: str) -> Dict[str, int]:
    """
    Extracts final register values from CPU outputs like => [R0=3] or R0=3.
    """
    regs = {}
    # Find all => [R0=X, R1=Y] or => [R0=X]
    matches = re.findall(r'\[(R\d+)\s*=\s*(\d+)\]', output_text)
    for r, v in matches:
        regs[r] = int(v)
    
    if not regs:
        # Fallback to direct R0=X
        matches2 = re.findall(r'(R\d+)\s*=\s*(\d+)', output_text)
        for r, v in matches2:
            regs[r] = int(v)
    return regs

def parse_location_table(output_text: str) -> Dict[str, int]:
    """Extracts location-to-ID table declared via 場所コード: ..."""
    table = {}
    m = re.search(r'(場所コード|部屋コード|移動先コード)[:：]?\s*([^。\n<]+)', output_text)
    if not m:
        return table
    pairs = re.findall(r'([^\s=,、]+)\s*=\s*(\d+)', m.group(2))
    for name, sid in pairs:
        name = name.strip().lstrip('「【').rstrip('」】')
        if not re.match(r'^R\d+$', name):
            table[name] = int(sid)
    return table

def evaluate_decomposed_item(item: Dict[str, Any], output_text: str) -> Dict[str, Any]:
    expected_entities = set(item.get("entities", []))
    target_object = item.get("target_object", "")
    expected_winner_entity = item.get("final_winner_entity", "")
    expected_final_ans = item.get("final_answer", "").strip()

    # 1. Parse model's entity table (and location table if spatial movement)
    model_table = parse_model_table(output_text)
    loc_table = parse_location_table(output_text)
    extracted_entities = set(model_table.keys())
    grounding_table = loc_table if (loc_table and expected_winner_entity in loc_table) else model_table

    # Stage 1: Name Extraction
    if expected_entities:
        common = extracted_entities & expected_entities
        prec = len(common) / max(1, len(extracted_entities))
        rec = len(common) / max(1, len(expected_entities))
        stage1_pass = (common == expected_entities and len(extracted_entities) == len(expected_entities))
    else:
        prec, rec, stage1_pass = 1.0, 1.0, True

    # Stage 2: Table Bijection
    if grounding_table:
        ids = list(grounding_table.values())
        stage2_pass = (len(ids) == len(set(ids))) and all(i > 0 for i in ids)
    else:
        stage2_pass = False

    # Stage 3: CPU Logic Grounding (relative to model's own table)
    inv_model_table = {v: k for k, v in grounding_table.items()}
    regs = parse_final_register_state(output_text)
    
    target_reg = item.get("target_register", "R0")
    model_final_id = regs.get(target_reg, None)

    stage3_pass = False
    if expected_winner_entity in grounding_table and model_final_id is not None:
        expected_id_in_model_table = grounding_table[expected_winner_entity]
        stage3_pass = (model_final_id == expected_id_in_model_table)

    # Stage 4: Table Inversion (Did output prose map final ID to the entity defined in table?)
    # Look for "答え: ... [名前]"
    stage4_pass = False
    model_winner_from_table = inv_model_table.get(model_final_id, None) if model_final_id is not None else None
    
    ans_match = re.search(r'答え[:：]?\s*([^\n]+)', output_text)
    ans_text = ans_match.group(1) if ans_match else output_text[-80:]
    
    if model_winner_from_table and model_winner_from_table in ans_text:
        stage4_pass = True

    # End-to-End Match
    e2e_pass = False
    if expected_final_ans and expected_final_ans in output_text:
        e2e_pass = True

    return {
        "stage1_name_extraction": stage1_pass,
        "stage1_precision": prec,
        "stage1_recall": rec,
        "stage2_table_bijection": stage2_pass,
        "stage3_cpu_logic": stage3_pass,
        "stage4_table_inversion": stage4_pass,
        "end_to_end_match": e2e_pass,
        "model_table": model_table,
        "final_regs": regs,
        "parsed_ans_text": ans_text[:60]
    }
