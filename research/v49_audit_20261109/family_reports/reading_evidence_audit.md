# Independent audit: GeoCPU v49 family `reading_evidence`

Auditor: independent data auditor (Claude), date 2026-11-09.  All numbers below were produced by scripts in `re/` next to this file (list at the end).  The family's own reports (`manifest.json`, `baselines_*`, `stats_*`, `test_*`, `verify_full_out.txt`) were used only as claims to compare against.  The family's solver modules (`solver_reading.py`, `rops_*.py`, `rcore.py`, `rdata.py`) were NOT imported; the checker `re/rlib.py` was written from the task description and the observed prompt formats.

## 0. Verdict: **ok_with_caveats**

No blocker was found: 0 wrong answers in a full independent re-derivation of all 62,510 rows, 0 shared prompts/sentences/entities between train and heldout, 0 eval-text matches (24-char shingles and >=12-char sentence units) against all 8 eval manifests plus the 4 T8 prompts, 0 conflicting or exact-duplicate rows, 0 over-length rows, files match the manifest byte-for-byte (sha256).

Caveats that matter for how the family should be used (details in sections 5-9):

1. **Low real diversity behind 59,500 rows.** Train has only **9,922 distinct fact sentences** filling 344,512 passage slots (each sentence is reused ~35x).  Worse, the 11,200 date/number rows (18.8% of train) draw on **82 birth-date sentences (80 people) and ~24 numeric entities**: they can be answered from a memorised table, without reading the dates/numbers.  Leave-one-out, the question string alone (no passage) already determines the answer for 99.9% of `date_extract_component`, 99.0% of `number_difference`, ~96% of `number_compare_*` rows.  The 59,500 rows are NOT '数十万 of independent examples'.
2. **One character outside the old-corpus vocabulary**: `將` (from the single entity 琴ノ若晴將) in 177 train rows + 2 dev rows.  It IS in the bundle vocab (`base_metadata.json`, 3,898 entries), so no `<unk>`, but the old 40,500 corpus never contained it, so its embedding has had no training exposure.
3. **Dev is not an independent split**: 92.0% of dev passage sentences also occur in train, 93% of dev subjects occur in train.  Only the heldout split can reveal memorisation.
4. **Few rows really force 'read, don't recall'**: the only counterfactual rows are the 2,800 dated-conflict rows (4.7%), and in about half of them the answer equals the real-world value.  About 1,400 train rows (2.4%) require overriding world knowledge.
5. **Weak format alignment with T5-T8** (section 9): open-book fact reading with `探す/根拠/答え` scaffolding; partial relevance to T6 (post-search reading, 2/3-hop chains), weak to T5/T7 (letter / 3-way verdict), none to T8.
6. 4,200 train rows (7.1%; `count_distinct_values`, `most_frequent_value`) and 240 heldout rows have no `根拠: 「...」` quote at all: they use an `抽出:` list of values instead (verified correct, but not verbatim evidence).  5,133 rows have `根拠: なし` (abstain or zero-count).
7. Licence: 16,800 train rows (28.2%) carry CC-BY-SA-4.0 (JEMHopQA-derived); share-alike/attribution obligations need to be tracked.  All text is templated (`<entity>の<relation>は<value>。`), not human prose.
8. `loss_mask_spans` and `tool_steps` are empty lists in every row: the trainer must derive the supervised span from the prompt/target boundary (`full_text == prompt + target`).

## 1. Inventory (checklist 1)

Streamed line by line; sha256 over raw bytes in 1 MiB chunks (`re/passA.py`).

| split | rows | bytes | sha256 | manifest rows / bytes / sha256 |
|---|---|---|---|---|
| train | 59500 | 147142303 | `c6059cc62c1d9852e2a0687887507355119057b2d95eecfa176338450bc773ec` | identical (rows, bytes, sha256) |
| dev | 850 | 2078039 | `d0f0c3edf0167d02cf06ed59c0341839c4bf3f9c5c403ec7e1fd49c035e84111` | identical (rows, bytes, sha256) |
| heldout | 2160 | 6982846 | `993a2c872015f7f040c4f464da89f0093811193f9ea643a0903758367503962a` | identical (rows, bytes, sha256) |

`gen_all_log.txt` claims (59,500 / 850 / 2,160 rows; 85 / 85 / 18 families) also match.  Manifest `problems: []`, `overlap: families 0, operations 0, heldout_entities_in_train 0`: re-verified independently in section 5.

## 2. Schema (checklist 2)

| check | train | dev | heldout |
|---|---|---|---|
| required fields missing (id, prompt, target, full_text, split, family, source, license, tool_steps, loss_mask_spans) | 0 | 0 | 0 |
| `full_text != prompt + target` | 0 | 0 | 0 |
| id duplicated within split | 0 | 0 | 0 |
| id duplicated across splits | 0 | 0 | 0 |
| `split` field != file | 0 | 0 | 0 |
| empty prompt | 0 | 0 | 0 |
| empty target | 0 | 0 | 0 |
| target does not end with `\n答え: ...。` | 0 | 0 | 0 |
| rows with len(full_text)+1 > 2048 | 0 | 0 | 0 |
| rows with non-empty `tool_steps` | 0 | 0 | 0 |
| rows with non-empty `loss_mask_spans` | 0 | 0 | 0 |
| unparseable JSON lines | 0 | 0 | 0 |

Length of `full_text` (characters = tokens): train mean 344.7, p50 336, p99 564, max 708; dev mean 340.5, p99 563, max 664; heldout mean 454.9, p99 804, max 890.  Longest row +1 = 891, far below 2048.

**Supervised characters (train) = sum(len(target)+1) = 5942707** (dev 83309, heldout 413687).  Every prompt ends with the cue `条件と根拠を順に考えてください。` and every target starts with `\n` (62,510 / 62,510).

## 3. Vocabulary (checklist 3)

Census set: `vocab_chars.json` (3,498 chars, union over the old `train_40500_exact.jsonl`).

| split | rows with >=1 OOV char | OOV characters |
|---|---|---|
| train | 177 (0.30%) | `將` x 332 |
| dev | 2 | `將` x 2 |
| heldout | 0 | - |

All 177 train rows come from the single entity `琴ノ若晴將` (13 distinct sentence/question heads; the char is in the supervised target of 40 rows, in the question of 35).  The family uses 1,645 distinct characters; **all are in the bundle vocabulary** (`v47_handoff/bundle/r6/base_metadata.json`, 3,898 entries; `將` is listed there among the 400 bundle chars absent from the old corpus), so there is no `<unk>` risk, but the `將` embedding is untrained.  Simple fix: drop the 177+2 rows (or the 琴ノ若晴將 facts).  OOV by template: jem_lookup 53, jem_abstain_entity 29, jem_reverse 24, jem_abstain_relation 20, jem_multi_attribute 17, jem_verify 14, number_* 21, date_extract 1.

## 4. Duplicates (checklist 4)

* exact duplicate `full_text` within train / dev / heldout: **0 / 0 / 0**; across splits: **0**.
* same prompt with different target (conflicting answers): **0**.  Prompts shared between any two splits: **0** (`prompt_cross_split_masks` empty).
* distinct prompts per family = rows per family (700 / 10 / 120 each; `min_distinct_prompts_per_family` in the manifest is true).
* template shares (train): the largest `structural_template` is `reading_lookup_mixed_relations` / `reading_count_by_relation` with 2,800 rows = **4.7%**; no template above 15%.  Every one of the 85 families is exactly 700 rows (1.18%).  Heldout: 12 templates, 18 families x 120 rows.
* near-duplication is high in a different sense (fact reuse), see section 7.

## 5. Leakage (checklist 5)

### 5a. Versus the eval manifests (`re/passD.py`)

Method as specified: eval text (prompt + target, plus question/choices for JamC-QA) -> hashed 24-char shingles at stride 8 (**182874 shingles**), then every audited row (prompt+target) slid at stride 1 (62,510 rows).  Separately, eval sentence units (split on newline / `。`, >= 12 chars; **36110 units**) vs the same units of each row.  Sources: `benchmark_8100_manifest.jsonl`, `eval_swe_step_control_heldout.jsonl`, `eval_v33_state_tracking_heldout.jsonl`, `eval_v38_decomposed_state_bench200.jsonl`, `eval_v41_logical_instruction_operators_heldout.jsonl`, `eval_v42_search_retrieval_aligned.jsonl`, `eval_v42_search_retrieval_heldout.jsonl`, `jamcqa_manifest_test.jsonl` plus the 4 fixed T8 prompts parsed from `tier8_readonly_v44fix.py`.

* rows with any matching shingle: **0 / 0 / 0** (train / dev / heldout); distinct matched shingles 0; generic (>500 rows) shingles 0.  Rows with an exact-unit match: **0**.
* reverse view: eval items sharing a non-generic shingle with any audited row: 0 for every eval file (JMMLU/ARC/SWE 8,100; T1 200; T2 188; v38 200; T5 400; T6 aligned 200 + heldout 200; JamC-QA 2,309; T8 4).
* positive control (`re/passD2.py`): an eval text scanned as if it were a row matched 6 and 5 shingles, so the machinery does detect a planted overlap.
* fact-level exposure (not a leak): 39/8,100 benchmark items and 21/2,309 JamC-QA items contain both subject and object of some train fact (e.g. ヴェルサイユ+フランス, 京都府+日本), i.e. generic geography co-occurrence; **0/100** T6 2-hop questions have their first-hop entity as the subject of a train 本社所在地/出生地 fact.
* `forbidden_entities.json` (114 entities >= 4 chars that occur in the eval files; docstring: facts *about* them are left out, city/country structural facts stay): exact-subject check finds only `大和郡山市` as a subject (`大和郡山市の所在国は日本。` = structural city fact, allowed by the docstring; 2 unique train sentences, 1 dev).  32 of the 114 strings still occur inside 100 train sentences as objects or sub-strings (米津玄師 390 rows, ドラゴンボール 151, 松任谷由実 62 ...), e.g. `黒澤明の1954年の映画作品は七人の侍。` vs JamC-QA culture-00027 (answer 七人の侍, but the quote asked about is not in the sentence).  Inspected pairs in `re/passH.txt`: topical co-occurrence only, none answers an eval item.

### 5b. Train vs heldout (and dev)

| check | result |
|---|---|
| heldout vs train: entity strings (WD subjects, WD objects, birth-sentence subjects, all 「..」 in questions) | 0 of 1760 overlap |
| heldout vs train: subjects / objects | 0 of 1637 / 0 of 281 |
| heldout passage sentences that occur in train | 0 of 13431 slots; 0 of 1782 unique sentences share a subject prefix with a train sentence (JEM-style heuristic, `passF`) |
| heldout evidence quotes that occur in train | 0 of 7384 |
| heldout question strings seen in train | 0 of 2,160 |
| heldout prompts equal to a train/dev prompt | 0 |
| 24-char shingles shared train<->heldout | 538 distinct, all boilerplate (`条件と根拠を順に考えてください。\n探す: ` + 1-2 name characters, header + `資料1: ` + 1-2 characters); 0 content shingles (`re/passD2.txt`) |
| heldout `unseen_relation_lookup` relations (49 distinct) appearing as `の<R>は` in any train sentence | 0 of 49 |
| dev vs train | NOT disjoint by design: 2410 of 2638 unique dev sentences share a subject prefix with train; 92.0% of dev sentences occur verbatim in train; 2103/2257 dev subjects occur in train; dev prompts are all new (0 shared) |

The manifest claim `heldout_entities_in_train: 0` is confirmed.  Heldout is also domain-shifted: its WD facts are Nordic (Norway/Sweden/Finland) cities, persons and organisations (train: 571 distinct WD objects, none Nordic) and its JEM facts are 1,782 unique sentences, of which only **25 distinct birth-date sentences** (1,354 slots) serve all heldout date families.  Heldout operations are new compositions of train primitives (e.g. `year_span` ~ `year_gap`, `compare_frequency` ~ `most_frequent_value`, `negated_list` ~ `list_all`, `count_through_two_hops` ~ `count_matching`+`two_hop`).

## 6. Independent answer verification (checklist 6)

`re/rlib.py` (own parser + solver for all 54 structural templates), `re/passC.py` (all rows), `re/passE.py` (intermediate lines).  For each row: parse `資料N[（YYYY年M月更新）]: sentence` lines and the `質問:` line; derive the answer from the question text + passage sentences only; compare with the final `答え: X。` (trailing `。` stripped) and with `reference_final_answer`.

* **Full census: 62510 rows checked, 62481 agree, 0 mismatches, 29 conservative flags, 0 unparsed, 0 errors.**  The 29 flags are all `date_before_threshold` rows where the birth year equals the threshold year; the target says `年: Y 基準: Y → 前ではない`, answer `いいえ`, which is logically correct (not before that year), so they are agreements.  Counting them: 62,510/62,510 correct.
* Seeded random sample (seed 20261109, p=1%): n=621 rows over 53 templates, mismatches 0 (the sample is a subset of the full census above).
* **Quotes**: 102938 `根拠: 「...」` lines; every one is an exact passage sentence of its own prompt (0 not exact, 0 not a substring of the prompt, 0 date-suffix mismatches); `quotes` field == quotes parsed from the target in all rows; `reference_final_answer` == parsed answer in all rows.
* **Derivable from the quoted sentences alone** (re-solving with only the quoted sentences as passages): ok for all templates that have quotes except `h_odd_one_out` (240 heldout rows: one quoted sentence cannot show that it is the odd one; by design) and the three quote-less templates; 0 wrong answers (column `reduced ok/not-derivable/na` below).
* **Intermediate scaffold lines** (`数値化`, `年`, `月`, `月日`, `年代`, `数値`, `差`, `抽出`, `出現`, `重複なし`, `国`, `並び`, `最多`, `値 -> 選択肢`, `比較`, `判断`) were also re-derived: all consistent in all rows (`re/passE.py`, `il` block: no `_bad` counter anywhere).  Two apparent failures during development were bugs in my checker (the data uses `異なる`, and `差:` prints larger-minus-smaller), fixed and re-run.
* **Negative control** (`re/passI.py`): 1,047 deliberately corrupted answers over 51 templates -> 1,047 flagged (100%); 400/400 corrupted quotes detected as non-verbatim.  So the checker has power.
* **Counterfactual rows** (`counterfactual: true` = the 2,800 train + 40 dev dated-conflict rows: `conflict_latest` birth/hq/country and `conflict_earliest` hq).  For each row the entity's real value was taken from the non-counterfactual rows (Wikidata sentences of the same entity+relation elsewhere in the data): in 2,837 of 2,840 rows (the other 3 entities occur nowhere else) exactly one of the two conflicting sentences states the real value and the other a value that is not real for that entity (e.g. アイザウル県の本社所在地は果川市, 2012; real: アイゾール), so the planted statement does contradict world knowledge in every row.  But the answer is the real value in only ~50% of rows (table), so the rows do NOT consistently contradict world knowledge at the answer level, and world knowledge is not a shortcut either (balanced).

| family:relation | rows (train+dev) | entity known elsewhere | exactly one value real | answer == real value | real value is the later-dated | real sentence listed first |
|---|---|---|---|---|---|---|
| reading_conflict_earliest:本社所在地 | 710 | 709 | 709 | 355 | 354 | 353 |
| reading_conflict_latest:出生地 | 710 | 709 | 709 | 354 | 354 | 328 |
| reading_conflict_latest:所在国 | 710 | 710 | 710 | 355 | 355 | 367 |
| reading_conflict_latest:本社所在地 | 710 | 709 | 709 | 354 | 354 | 347 |

Dates (`2013年8月更新` ...) are synthetic; the real/fake assignment is independent of date order (354/355) and of passage position (answer sentence listed first of the pair in 49-52%).

### Per-template verification table

Columns: rows per split, rows where my independent answer == target answer (`ok`), mismatches, conservative ambiguity flags (all resolved as correct, see above), rows with a bad quote or `reference_final_answer` mismatch, and the reduced-prompt check `ok/not-derivable/na` (the 29 `not-derivable` in `date_before_threshold` are the equal-year conservative flags, not wrong answers; `na` = template has no quote lines by design).

| template | train | dev | heldout | ok | mismatch | flag | quote/ref bad | reduced ok/not-derivable/na |
|---|---|---|---|---|---|---|---|---|
| reading_abstain_chain_broken | 1400 | 20 | 0 | 1420 | 0 | 0 | 0 | 1420/0/0 |
| reading_abstain_entity_missing | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_abstain_relation_missing | 1400 | 20 | 0 | 1420 | 0 | 0 | 0 | 1420/0/0 |
| reading_conflict_earliest | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_conflict_latest | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_count_by_relation | 2800 | 40 | 0 | 2840 | 0 | 0 | 0 | 2840/0/0 |
| reading_count_distinct_values | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 0/0/2130 |
| reading_count_matching_value | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_date_before_threshold | 700 | 10 | 0 | 681 | 0 | 29 | 0 | 681/29/0 |
| reading_date_compare_two | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_date_compare_two_later | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_date_count_before_threshold | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_date_extract_component | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_date_extreme_earliest | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_date_extreme_latest | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_date_same_month | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_date_sort | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_date_year_gap | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_h_compare_frequency | 0 | 0 | 240 | 240 | 0 | 0 | 0 | 0/0/240 |
| reading_h_count_through_two_hops | 0 | 0 | 240 | 240 | 0 | 0 | 0 | 240/0/0 |
| reading_h_cross_relation_chain_compare | 0 | 0 | 120 | 120 | 0 | 0 | 0 | 120/0/0 |
| reading_h_date_calendar_order | 0 | 0 | 120 | 120 | 0 | 0 | 0 | 120/0/0 |
| reading_h_date_decade | 0 | 0 | 120 | 120 | 0 | 0 | 0 | 120/0/0 |
| reading_h_date_year_span | 0 | 0 | 120 | 120 | 0 | 0 | 0 | 120/0/0 |
| reading_h_distinct_countries_two_hops | 0 | 0 | 240 | 240 | 0 | 0 | 0 | 240/0/0 |
| reading_h_four_hop_compare | 0 | 0 | 120 | 120 | 0 | 0 | 0 | 120/0/0 |
| reading_h_negated_list | 0 | 0 | 240 | 240 | 0 | 0 | 0 | 240/0/0 |
| reading_h_odd_one_out | 0 | 0 | 240 | 240 | 0 | 0 | 0 | 0/240/0 |
| reading_h_reverse_two_hop | 0 | 0 | 240 | 240 | 0 | 0 | 0 | 240/0/0 |
| reading_h_unseen_relation_lookup | 0 | 0 | 120 | 120 | 0 | 0 | 0 | 120/0/0 |
| reading_jem_abstain_entity_missing | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_jem_abstain_relation_missing | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_jem_lookup | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_jem_multi_attribute | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_jem_reverse_lookup | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_jem_verify_claim | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_list_all_matching | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_lookup | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_lookup_mixed_relations | 2800 | 40 | 0 | 2840 | 0 | 0 | 0 | 2840/0/0 |
| reading_lookup_multiple_choice | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_lookup_near_name | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_lookup_two_entities | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_most_frequent_value | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 0/0/2130 |
| reading_number_compare_larger | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_number_compare_smaller | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_number_difference | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_number_max_of_n | 700 | 10 | 0 | 710 | 0 | 0 | 0 | 710/0/0 |
| reading_reverse_lookup | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_same_country_pair | 1400 | 20 | 0 | 1420 | 0 | 0 | 0 | 1420/0/0 |
| reading_same_value_pair | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_three_hop | 1400 | 20 | 0 | 1420 | 0 | 0 | 0 | 1420/0/0 |
| reading_two_hop | 1400 | 20 | 0 | 1420 | 0 | 0 | 0 | 1420/0/0 |
| reading_verify_claim | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |
| reading_which_of_two | 2100 | 30 | 0 | 2130 | 0 | 0 | 0 | 2130/0/0 |

## 7. Shortcut / bias audit (checklist 7)

### 7a. Label balance (by construction)

* yes/no templates: exactly 50/50 (e.g. `same_value_pair` 1,050/1,050, `same_country_pair` 700/700, `date_same_month` 350/350, heldout 60/60 in each of 3 templates).
* verdict (`verify_claim`): 正しい 702 / 誤り 699 / わからない 699 (train); JEM verify 234/233/233.  `わからない` rows always carry a `判断:` line; `正しい/誤り` carry `比較: ... → 一致/不一致`: perfectly consistent with the answer.
* MC letters: A/B/C/D = 525 each (train).  Counts: `count_matching_value` 0-4 = 420 each, `count_by_relation` 0-4 = 560 each, `distinct_values` 2-5 = 525 each, heldout counts 60 each (0-3) and 80 each (1-3).
* `which_of_two`: answer is the first-named entity in 1,050/2,100 and the earlier-listed passage in 1,079/2,100 (position unbiased).  Mean relative position of the answer-bearing sentence in the passage: 0.44-0.53 for all lookup templates.
* abstain: `わからない` is exactly 50% of the abstain templates; the 3 header variants (one of them mentions `わからない`) are independent of the answer (e.g. abstain_entity: 344/358/363 abstain per header).

### 7b. Majority-answer baseline

Share of the single most frequent answer string per template (train and heldout):

| template | split | n | distinct answers | top answer (count) | majority baseline |
|---|---|---|---|---|---|
| reading_h_compare_frequency | heldout | 240 | 100 | ボードー (7) | 2.9% |
| reading_h_count_through_two_hops | heldout | 240 | 4 | 0 (60) | 25.0% |
| reading_h_cross_relation_chain_compare | heldout | 120 | 2 | はい (60) | 50.0% |
| reading_h_date_calendar_order | heldout | 120 | 22 | 徳川家広 (16) | 13.3% |
| reading_h_date_decade | heldout | 120 | 2 | はい (60) | 50.0% |
| reading_h_date_year_span | heldout | 120 | 48 | 55 (9) | 7.5% |
| reading_h_distinct_countries_two_hops | heldout | 240 | 3 | 1 (80) | 33.3% |
| reading_h_four_hop_compare | heldout | 120 | 2 | はい (60) | 50.0% |
| reading_h_negated_list | heldout | 240 | 236 | ロゲル・ルート (2) | 0.8% |
| reading_h_odd_one_out | heldout | 240 | 220 | フォレカ (3) | 1.2% |
| reading_h_reverse_two_hop | heldout | 240 | 209 | スパーレバンク・エンSRバンク (3) | 1.2% |
| reading_h_unseen_relation_lookup | heldout | 120 | 58 | 2009年1月25日 (4) | 3.3% |
| reading_abstain_chain_broken | train | 1400 | 14 | わからない (700) | 50.0% |
| reading_abstain_entity_missing | train | 2100 | 197 | わからない (1050) | 50.0% |
| reading_abstain_relation_missing | train | 1400 | 185 | わからない (700) | 50.0% |
| reading_conflict_earliest | train | 700 | 185 | トロント (58) | 8.3% |
| reading_conflict_latest | train | 2100 | 282 | ニューヨーク (112) | 5.3% |
| reading_count_by_relation | train | 2800 | 5 | 0 (560) | 20.0% |
| reading_count_distinct_values | train | 2100 | 4 | 2 (525) | 25.0% |
| reading_count_matching_value | train | 2100 | 5 | 0 (420) | 20.0% |
| reading_date_before_threshold | train | 700 | 2 | はい (350) | 50.0% |
| reading_date_compare_two | train | 700 | 74 | 桐野利秋 (27) | 3.9% |
| reading_date_compare_two_later | train | 700 | 74 | 森山未來 (38) | 5.4% |
| reading_date_count_before_threshold | train | 700 | 5 | 0 (153) | 21.9% |
| reading_date_extract_component | train | 2100 | 92 | 5月 (85) | 4.0% |
| reading_date_extreme_earliest | train | 700 | 60 | チェスター・カールソン (37) | 5.3% |
| reading_date_extreme_latest | train | 700 | 58 | 清原果耶 (41) | 5.9% |
| reading_date_same_month | train | 700 | 2 | はい (350) | 50.0% |
| reading_date_sort | train | 700 | 700 | チェスター・カールソン、宮迫博之、井上芳雄 (1) | 0.1% |
| reading_date_year_gap | train | 700 | 104 | 9 (27) | 3.9% |
| reading_jem_abstain_entity_missing | train | 700 | 272 | わからない (350) | 50.0% |
| reading_jem_abstain_relation_missing | train | 700 | 202 | わからない (350) | 50.0% |
| reading_jem_lookup | train | 2100 | 601 | 5歳 (32) | 1.5% |
| reading_jem_multi_attribute | train | 700 | 262 | 秋元康 (13) | 1.9% |
| reading_jem_reverse_lookup | train | 700 | 399 | 三國連太郎 (9) | 1.3% |
| reading_jem_verify_claim | train | 700 | 3 | 正しい (234) | 33.4% |
| reading_list_all_matching | train | 2100 | 2097 | ニューヨーク、マイアミ (2) | 0.1% |
| reading_lookup | train | 2100 | 257 | ニューヨーク (101) | 4.8% |
| reading_lookup_mixed_relations | train | 2800 | 294 | ニューヨーク (115) | 4.1% |
| reading_lookup_multiple_choice | train | 2100 | 4 | A (525) | 25.0% |
| reading_lookup_near_name | train | 2100 | 237 | メキシコ (162) | 7.7% |
| reading_lookup_two_entities | train | 2100 | 1048 | ニューヨーク、ニューヨーク (43) | 2.0% |
| reading_most_frequent_value | train | 2100 | 237 | イタリア (74) | 3.5% |
| reading_number_compare_larger | train | 700 | 21 | トルコ (83) | 11.9% |
| reading_number_compare_smaller | train | 700 | 22 | 板橋区 (59) | 8.4% |
| reading_number_difference | train | 700 | 45 | 10 (73) | 10.4% |
| reading_number_max_of_n | train | 700 | 18 | トルコ (106) | 15.1% |
| reading_reverse_lookup | train | 2100 | 1579 | 曽坪郡 (6) | 0.3% |
| reading_same_country_pair | train | 1400 | 2 | はい (700) | 50.0% |
| reading_same_value_pair | train | 2100 | 2 | はい (1050) | 50.0% |
| reading_three_hop | train | 1400 | 13 | 東京都 (140) | 10.0% |
| reading_two_hop | train | 1400 | 13 | 日本 (141) | 10.1% |
| reading_verify_claim | train | 2100 | 3 | 正しい (702) | 33.4% |
| reading_which_of_two | train | 2100 | 1600 | 江陵市 (6) | 0.3% |

Highest majority baselines: 50% (yes/no, abstain templates), 20-33% (count templates; heldout 25-33%), <= 10% for entity/span answers (`three_hop`/`two_hop` ~10% because only 13 countries/capitals exist), and ~12% for `number_compare_larger` (トルコ).

### 7c. Surface baselines without doing the task (train)

| baseline | accuracy | chance |
|---|---|---|
| `lookup` (plain): most frequent passage value of the asked relation (random tie-break) | 34.7% | 26.5% (pick a random value of that relation) |
| `lookup_near_name`: same | 49.5% | 29.0% |
| `lookup_mixed_relations`: same | 44.5% | 48.2% |
| abstain_entity (non-abstain rows): same | 38.6% | 26.3% |
| abstain_relation (non-abstain rows): same | 33.8% | 31.0% |
| `multiple_choice`: option whose value is most frequent in the passages (random tie-break) | 36.4% | 25.0% |
| `multiple_choice`: longest option | 17.0% (358 unique-longest hits / 2,100) | 25.0% |
| first / last matching-relation passage (lookup) | 28.6% / 28.5% | 26.5% |

So the answer is mildly over-represented among repeated values (a 'frequent value' prior worth about +8 points on `lookup`, +20 on `near_name`, +12 on `abstain_entity`, +11 on `multiple_choice`; roughly zero or negative on `mixed_relations` and `abstain_relation`), by design of the distractors (same-value / look-alike passages).  The family's `leaky()` filter already drops facts whose value is a substring of the entity name or whose name contains a country name.  The family's own `baselines_*.txt` report the same ballpark (majority 0.31-0.41 on lookup families, heldout span majority 0.40 including the trivially 100% `compare_frequency`, where the 'shortcut' is the task).  No baseline reaches high accuracy on the lookup/hop families.

### 7d. Memorisation without reading (the main shortcut)

Train facts are few: **9,922 unique passage sentences** (dev 2,638, heldout 1,782) fill 344,512 / 4,900 / 13,431 passage slots; 571 distinct WD objects (top-10 objects = 35.7% and top-50 = 71.3% of 264,258 WD object slots: ニューヨーク 12,515, メキシコ 11,340, トロント 10,378 ...).  `passage_slots` per unique sentence = 34.7 in train.

Date families use **80 people / 82 distinct birth sentences** across 8,400 rows (長澤まさみ and 森山未來 appear in ~750 rows each); number families use 24 (compare) / 18 (difference) entities with fixed values.  Because every person/entity has one fixed value, a model that memorises the table solves these rows while only reading the *names*.  Leave-one-out within train (question string seen in another train row, answer = majority answer of those rows):

| template | train rows | distinct question strings | rows whose question repeats | leave-one-out: majority answer of the other rows with the same question string is correct |
|---|---|---|---|---|
| reading_abstain_chain_broken | 1400 | 1215 | 358 | 168 (12.0%) |
| reading_abstain_entity_missing | 2100 | 1824 | 510 | 268 (12.8%) |
| reading_abstain_relation_missing | 1400 | 1355 | 89 | 89 (6.4%) |
| reading_conflict_earliest | 700 | 585 | 214 | 70 (10.0%) |
| reading_conflict_latest | 2100 | 1606 | 848 | 257 (12.2%) |
| reading_count_by_relation | 2800 | 4 | 2800 | 0 (0.0%) |
| reading_count_distinct_values | 2100 | 3 | 2100 | 0 (0.0%) |
| reading_count_matching_value | 2100 | 261 | 2003 | 396 (18.9%) |
| reading_date_before_threshold | 700 | 611 | 168 | 168 (24.0%) |
| reading_date_compare_two | 700 | 656 | 86 | 86 (12.3%) |
| reading_date_compare_two_later | 700 | 650 | 96 | 96 (13.7%) |
| reading_date_count_before_threshold | 700 | 114 | 672 | 241 (34.4%) |
| reading_date_extract_component | 2100 | 240 | 2098 | 2098 (99.9%) |
| reading_date_extreme_earliest | 700 | 1 | 700 | 37 (5.3%) |
| reading_date_extreme_latest | 700 | 1 | 700 | 41 (5.9%) |
| reading_date_same_month | 700 | 590 | 201 | 201 (28.7%) |
| reading_date_sort | 700 | 1 | 700 | 0 (0.0%) |
| reading_date_year_gap | 700 | 664 | 71 | 71 (10.1%) |
| reading_jem_abstain_entity_missing | 700 | 573 | 237 | 129 (18.4%) |
| reading_jem_abstain_relation_missing | 700 | 635 | 122 | 122 (17.4%) |
| reading_jem_lookup | 2100 | 1026 | 1518 | 1518 (72.3%) |
| reading_jem_multi_attribute | 700 | 456 | 414 | 414 (59.1%) |
| reading_jem_reverse_lookup | 700 | 470 | 403 | 361 (51.6%) |
| reading_jem_verify_claim | 700 | 581 | 218 | 131 (18.7%) |
| reading_list_all_matching | 2100 | 222 | 2040 | 2 (0.1%) |
| reading_lookup | 2100 | 1811 | 538 | 538 (25.6%) |
| reading_lookup_mixed_relations | 2800 | 1840 | 1234 | 1234 (44.1%) |
| reading_lookup_multiple_choice | 2100 | 2100 | 0 | 0 (0.0%) |
| reading_lookup_near_name | 2100 | 1468 | 972 | 972 (46.3%) |
| reading_lookup_two_entities | 2100 | 2096 | 8 | 8 (0.4%) |
| reading_most_frequent_value | 2100 | 3 | 2100 | 74 (3.5%) |
| reading_number_compare_larger | 700 | 155 | 669 | 669 (95.6%) |
| reading_number_compare_smaller | 700 | 157 | 677 | 677 (96.7%) |
| reading_number_difference | 700 | 137 | 693 | 693 (99.0%) |
| reading_number_max_of_n | 700 | 3 | 700 | 154 (22.0%) |
| reading_reverse_lookup | 2100 | 296 | 1972 | 146 (7.0%) |
| reading_same_country_pair | 1400 | 1399 | 2 | 2 (0.1%) |
| reading_same_value_pair | 2100 | 2090 | 20 | 20 (1.0%) |
| reading_three_hop | 1400 | 1215 | 350 | 350 (25.0%) |
| reading_two_hop | 1400 | 1203 | 373 | 373 (26.6%) |
| reading_verify_claim | 2100 | 1864 | 421 | 218 (10.4%) |
| reading_which_of_two | 2100 | 2100 | 0 | 0 (0.0%) |

Reading guide: whenever a question string repeats, the repeated rows always carry the same answer for fact-lookup templates (e.g. `lookup` 538/538), so the share of repeating rows is the no-reading-needed share.  For templates with only 4-5 answer classes (`count_*`, `date_count_before`, `number_max_of_n`) the percentages are class priors (chance), not memorisation.  The real memorisation signal is where the answer space is large and the question repeats: `date_extract_component` 99.9%, `number_difference` 99.0%, `number_compare_*` 96%, `jem_lookup` 72%, `jem_multi_attribute` 59%, `jem_reverse_lookup` 52%, `near_name`/`mixed` 44-46%.

(`date_compare_two`/`gap` show low repeat rates only because the pair differs; with 80 people the whole pair space is memorisable, so the question-string number understates the risk.  `count_*`, `most_frequent`, `sort`, `extreme` have constant question strings, so the table is meaningless for them; their answers require reading the passage.)  The heldout date families use 25 other birth sentences, so heldout does expose memorisation for these ops.

WD-based families are better: e.g. `lookup` has 1,603 distinct question entities in 2,100 rows, question-only memorisation gives 25.6%; `near_name` 46.3%, `mixed` 44.1%, `two_hop`/`three_hop` 25-27%.

## 8. Provenance and licence (checklist 8)

| source | rows (train / dev / heldout) | licence field | nature |
|---|---|---|---|
| Wikidata truthy triples verbalised as `<e>の<rel>は<v>。` | 42,700 / 610 / 1,680 (71.8%) | CC0-1.0 | real facts, templated sentences |
| JEMHopQA triples (jem) | 16,800 / 240 / 480 (28.2%) | CC-BY-SA-4.0 | real facts, templated sentences |

* Real/human data: all fact *content* is from real knowledge bases (100%); 0% is human-written prose; 100% of questions and scaffolds are self-authored templates (42 train + 12 heldout operations; 3 header variants; 2 phrasings for lookups).  4.7% of train rows (2,800) contain one deliberately substituted value plus synthetic update dates.
* `source` strings: `derived:geocpu/v49-reading-evidence (real fact sentences: Wikidata CC0 truthy)` / `(... JEMHopQA triples)`.  Both fact sources are read from the retrieval files `combined_train_wikidata_first.jsonl` (train/dev pools) and the extra facts of `combined_dev_wikidata_first.jsonl` (heldout pool), per `rdata.py`.
* A trained model is probably not a derivative work, but the CC-BY-SA-4.0 share (28.2% of train, 22.2% of heldout) should be recorded in the release notes.

## 9. Format alignment with T5-T8 (checklist 9)

Family format (train example, `lookup`):

```
問題: 次の資料だけを根拠に答えてください。\n資料1: ...の出生地は...。\n資料2: ...\n質問: 「X」の出生地はどこですか。\n条件と根拠を順に考えてください。
-> \n探す: Xの出生地\n根拠: 「Xの出生地はサンパウロ。」\n答え: サンパウロ。
```

| tier | eval format (quoted) | what this family can plausibly train | format gap |
|---|---|---|---|
| T5 6-operator logical contrast, 3 options, JNLI | `タスク: 自然言語指示の論理演算子変換・3値論理推論 (JNLI Grounded)\n【本文】\n（1）テーブルの上に花が活けられています。 （2）... \n【指示】\n次のA〜Cのうち、前提から論理的に【正しい（確実に導ける）】ものを選べ。\n【選択肢】\nA. （3）に関して、...\nB. ...\nC. ...\n思考の連鎖と回答:` -> `B` | weakly: `verify_claim` (3-way 正しい/誤り/わからない ~ entailment/contradiction/unknown, 2,800 train rows incl. JEM) and `lookup_multiple_choice` (letter) | header `問題:` vs `タスク:`; `資料N:` vs `【本文】（N）`; evidence = templated `Xの<rel>はY。` vs natural JNLI captions; 4 options inline `A=..., B=...` vs 3 lines `A. ...`; claim matching is exact string / value substitution, not paraphrase NLI; target `答え: A。` vs bare letter after a free chain of thought; no 6-operator (negation, quantifier ...) contrast |
| T6 search-augmented multi-hop retrieval | `質問: 通信機器大手ノキア（Nokia）が本社を置く国の首都はどこか？\n思考: 外部検索エンジンを使用して多段階で調査します。\n` -> `ヘルシンキ` (expected queries `ノキア 本社`, `フィンランド 首都`); 1-hop items are Python-stdlib questions (`math.isqrt`) | best fit: `two_hop` / `three_hop` (本社所在地 -> 所在国 -> 首都, 2,800 train rows) with `探す:` ~ query and `根拠: 「...」` ~ observation; `abstain_*` (わからない), `conflict_*` (newest source) | passages are given in the prompt (open book) while T6 has none and the observation comes from a tool; `探す: ノキアの本社所在地` vs query `ノキア 本社`; `答え: ヘルシンキ。` vs bare `ヘルシンキ`; no Python-API content for 1-hop items; no tool-call/observation turn format; prompt prefix `問題:`/`資料N` vs `質問:`/`思考:` |
| T7 JMMLU-style closed/open-book MC | `分野: abstract_algebra (math)\n問題: ...\nA: 8\nB: 2\nC: 24\nD: 120\n正解:` -> `C`; JamC-QA `質問: ...\nA: ...\n答え:` -> `D` | only the letter-selection mechanics (`lookup_multiple_choice`, 2,100 rows, letters balanced); no academic knowledge | inline `選択肢: A=..., B=...。記号で答えてください。` vs one option per line `A: ...`; `答え: A。` vs `C`; open-book fact lookup vs closed-book subject knowledge; no `分野:` line; the family cannot raise knowledge accuracy |
| T8 four fixed Python/C++/Rust code+explanation prompts | `ユーザー: 関数 f(x) = 3x^3 + 2sin(x) - 5 について、導関数 f'(x) を求め...\nアシスタント: ` (and C++ binary search, Rust memo-DP) | none | no code, no `ユーザー:/アシスタント:` format |

## 10. Plain answer to the user's question (dataset management / scale / ingenuity, T5-T8)

* **Managed**: yes, for this family: sha256-pinned files, manifest, per-family 700/10/120 rows, deterministic seeds, forbidden-entity blocklist, own baselines, 33 unit tests (claimed in `test_out.txt`), separate disjoint heldout.
* **Scaled to hundreds of thousands**: no.  This family has 59,500 train rows (5.94 M supervised characters, 20.5 M characters in total), but only 9,922 distinct fact sentences behind them; the date/number part (18.8% of rows) is effectively 82 + ~24 facts.
* **Ingenuity**: real. Verbatim-quote scaffolding (`探す/根拠/答え`), 85+18 families with exact class balance, abstention and chain-broken rows, answer-leak filter (`leaky()`), heldout with new operations *and* a disjoint (Nordic) entity domain plus 49 unseen relations; all verified here.
* **Relevance to T5-T8**: partial for T6, weak for T5/T7, none for T8 (section 9).

## 11. Not run / limits

* No model was trained or evaluated; baselines are statistics, not learned models.
* Real-world truth of each Wikidata/JEM triple was not re-checked against the live sources (no network); correctness is relative to the passage text.  Known Wikidata quirks remain (e.g. an organisation with an `出生地`).  Real-value checks for counterfactual rows use the same corpus (non-counterfactual rows), not an external KB.
* The eval-leak method used 183K shingles (not ~1M) because the eight eval files are smaller than assumed; the positive control shows detection works.
* Entity overlap for JEM distractor subjects relies on sentence identity and a prefix-before-`の` heuristic, not a full entity linker.

## 12. Scripts and output files

All in `C:/Users/81808/AppData/Local/Temp/claude/ff74aa4e8a98970d/f80753de-c283-4765-a931-f974c2b54295/scratchpad/audit_v49/re/` (run one at a time with `python -I`, streaming, < 400 MB):

| script | purpose | output |
|---|---|---|
| `passA.py` | inventory, sha256, schema, ids, OOV, dups, lengths, distributions, sample per key | `passA.json`, `samples_by_key.txt` |
| `passB_skel.py` | question / target skeletons per template | `skeletons.txt` |
| `rlib.py` | independent parser + solver (54 templates) | - |
| `passC.py` | full verification, quotes, reduced-prompt check, baselines, counterfactual check, entity/sentence overlap, forbidden entities | `passC.json`, `summaryC.txt`, `summaryC2.txt` |
| `passD.py`, `passD2.py` | eval leakage (24-char shingles + units), positive control, train-heldout shingles | `passD.json`, `passD2.txt` |
| `passE.py` | intermediate-line audit, diversity, memorisation, fact-level exposure | `passE.json`, `summaryE.txt` |
| `passF.py`, `passG.py`, `passH.py` | forbidden-entity subject check, JEM prefix overlap, OOV detail, tie-aware baselines, entity pairs with eval | `passF.json`, `passG.txt`, `passH.txt` |
| `passI.py` | negative controls | `negcontrol.txt` |
| `make_report.py` | assembles this report from the JSON files | this file |

