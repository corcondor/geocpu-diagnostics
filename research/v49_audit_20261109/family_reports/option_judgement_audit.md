# GeoCPU v49 data audit: family `option_judgement`

Auditor: independent re-verification (no family solver modules imported; every check below was written from the task text and visible surface patterns).
Family dir: `.../scratchpad/wt_v48/research/v49_data_20261108/families/option_judgement`
Scripts and raw outputs: `C:/Users/81808/AppData/Local/Temp/claude/ff74aa4e8a98970d/f80753de-c283-4765-a931-f974c2b54295/scratchpad/audit_v49/oj_*.py|json|txt`

## 0. Verdict and one-paragraph summary

**Verdict: ok_with_caveats** (no hard blocker; one real defect to fix, several design caveats that decide how much this family is worth for T5).

What is good: labels are 100% correct against an independent checker (all 50,625 rows, 0 mismatches, 0 unparsed, 0 ties); every trace line is consistent with the label; no duplicates, no conflicting prompts, no over-length rows, no OOV characters; zero 24-char shingle overlap with all 8 eval manifests (positive control 100/100); heldout shares no entity, sentence, relation or template with train; the T5 "premise number" rule ((1)=T,(2)=F,(3)=U) is at chance (0.31 to 0.37).

What is not good (the answer to "did you scale to hundreds of thousands and craft it?"):
1. Scale: this family is **46,008 train rows, not hundreds of thousands**, and those rows are built from only **5,223 distinct facts** (mean 33.7 uses per fact; max 895). The three date relations of 7 templates (29.6% of train rows) use **only 75 distinct birth-date facts**.
2. The single-rule shortcuts the generator targeted (difflib, Jaccard, length, position) are at chance, **but the shortcut is not gone**: a learned classifier on surface-overlap features reaches 0.79 (dev) / 0.84 (train sample) / 0.50 (heldout) against 0.333 chance, and "which option names an entity absent from the premise" solves und/notund rows with ~100% accuracy wherever it gives a unique answer. By construction U always means "entity not in premise", so U is a name-lookup, not a judgement.
3. Format and content differ from T5 on every axis (header, slot names, option form, cue line, target form, instruction wording, content domain, operator set). Only the pointer template (8.45% of train, 0% of heldout) carries the T5-style "（k）に関して、" citation.
4. dev is not independent of train: 100% of dev premise facts also occur in train (heldout is clean).
5. Naming defect: `train.jsonl.gz` is plain JSONL.

## 1. Inventory

| split | file | lines | bytes | sha256 | log claim (`_build_all.log`) |
|---|---|---|---|---|---|
| train | train.jsonl.gz | 46,008 | 122,756,415 | e9d6b7264d1919d9bae0097eab8a365307fa54495645a93b8eed3ba67c61ed82 | rows 46008, same sha: MATCH |
| dev | dev.jsonl | 1,917 | 5,097,957 | 2a6926e10ba04cc9f75d37ea3ce0a2df971637ecaa79b9e39cca86c07b9e7734 | rows 1917, same sha: MATCH |
| heldout | heldout.jsonl | 2,700 | 7,852,276 | 64ae6de49b08b54040a9ce674ee00b2f960a3881a8a9cf4c80e90fe5535408f1 | rows 2700, same sha: MATCH |

No separate manifest file exists. `_gen_hd.log` in the same folder lists older, different sha256 values for heldout (0a86fbcf...) and dev (27d983eb...); these are stale leftovers from an earlier build, not the final files (the final files match `_build_all.log`).

**Naming defect**: first four bytes of `train.jsonl.gz` are `{"id` (not the gzip magic 1f 8b). It is plain JSONL. Any loader that calls `gzip.open` raises `BadGzipFile`. Fix: rename to `train.jsonl` (or actually gzip it) and update `_build_all.sh`, which writes it with `--out train.jsonl.gz`.

## 2. Schema

Script: `oj_pass1.py` (streams each file; blake2b-8 digests only).

| check | train | dev | heldout |
|---|---|---|---|
| JSON parse failures | 0 | 0 | 0 |
| required fields id,prompt,target,full_text,split,family,source,license,tool_steps,loss_mask_spans missing | 0 | 0 | 0 |
| full_text != prompt+target | 0 | 0 | 0 |
| duplicate id within split | 0 | 0 | 0 |
| id seen in an earlier split | 0 | 0 | 0 |
| split field != file | 0 | 0 | 0 |
| empty prompt / empty target | 0 / 0 | 0 / 0 | 0 / 0 |
| target not ending `答え: [ABC]。` | 0 | 0 | 0 |
| reference_final_answer != letter in target | 0 | 0 | 0 |
| len(full_text) min / mean / p99 / max | 286 / 420.17 / 566 / 710 | 300 / 419.98 / 564 / 612 | 327 / 450.90 / 584 / 611 |
| rows with len+1 > 2048 | 0 | 0 | 0 |
| loss_mask_spans non-empty | 0 | 0 | 0 |
| tool_steps non-empty | 0 | 0 | 0 |

Supervised characters in train (sum of len(target)+1): **5,137,820** (mean target about 110.7 chars; prompt about 310 chars). `loss_mask_spans` is `[]` in every row, so whether loss is restricted to the target depends on the trainer using the prompt/target split rather than spans (not verified here).

Terminal pattern: every target ends with `...\n選ぶのは{真|偽|不明|真か不明|偽か不明|真か偽}のもの → X\n答え: X。`. Target layout (verified by regex on all rows, 0 violations): line 1 `根拠:`, three lines `L: <evidence> → 真|偽|不明`, the 選ぶのは line, the 答え line.

## 3. Vocabulary

Vocabulary file read: `vocab_chars.json` (3,498 characters, source: union of characters of train_40500_exact.jsonl full_text). Script `oj_pass1.py`.
OOV rows: **train 0, dev 0, heldout 0**; no OOV characters at all.

## 4. Duplicates, conflicts, structure

| check | result |
|---|---|
| exact duplicate full_text within train / dev / heldout | 0 / 0 / 0 |
| same prompt with different target (conflicts) | 0 in every split (every prompt is unique: 46,008 / 1,917 / 2,700 distinct) |
| exact prompt overlap train-dev / train-heldout / dev-heldout | 0 / 0 / 0 |
| identical premise block reused by several rows | train: 45,909 blocks used once, 48 twice, 1 three times; dev and heldout: all unique. So there are **no T5-style contrast groups** (same premise, several instructions) in training. |
| template share (max) | train 8.45% (8 templates), dev same; heldout 12.0% (7 templates). None above 15%. |
| families | train/dev 71 families (16 templates x operators), heldout 50 families (11 templates, all new names `ojh_*`) |
| operator mix (train) | ent 10,368; con 10,368; und 9,720; notcon 5,184; notent 5,184; notund 5,184 |
| pointer-number (T5-style) template | train 3,888 rows (8.45%), dev 162, heldout 0 |

Template list. Train/dev: atom, pointer, negation, conj, disj, same, differ, crossrel, exists, date_exact, date_year, date_month, date_decade, date_before, date_sameyear, date_gap. Heldout: inlist, notin, xor, notboth, namedcount, alldiff, hassame, count, atleast, ndistinct, majority.

**Fact diversity (script `oj_trace.py`, `oj_leak.py`)**:
- train premise facts: 5,223 distinct (entity, relation, value) facts for 175,880 premise mentions (mean 33.7 uses, median 22, min 6, max 895).
- by relation: 出生地 2,639 facts (143 distinct values), 本社所在地 2,509 facts (372 distinct values), **生年月日 only 75 facts**. Date templates = 13,608 train rows (29.6%).
- dev: 3,253 distinct facts, **all 3,253 are also in train** (every dev row has all premise sentences seen in train; 1,331 of 5,526 distinct dev option bodies, 24%, also occur in train).
- heldout: 287 distinct facts, relation 所在国 only (Nordic places, 3 countries: スウェーデン / ノルウェー王国 / フィンランド). Train and dev contain **no 所在国 fact at all**. Overlap heldout-train: 0 entities, 0 premise sentences, 0 entity-relation pairs, 0 option bodies, 0 rows.
- So heldout is a genuine relation-OOD and template-OOD test (good), but it rests on only 287 facts and 3 country values; dev only tests recombination of memorised facts.

## 5. Leakage

### 5.1 Versus eval manifests (script `oj_leak.py`, `oj_leak2.py`)
Method as specified: all string leaves of each eval JSON row; hashed 24-char shingles at stride 8 (167,245 distinct shingles over 8 files; chars per file: benchmark_8100 2,243,214; swe 38,579; v33 112,020; v38 37,668; v41 (T5) 160,752; v42 aligned 18,077; v42 heldout 23,277; jamcqa 420,497); stride-1 sliding over each audited row full_text (prompt+target); generic threshold >500 rows.

| item | result |
|---|---|
| rows with any shingle hit (train / dev / heldout) | **0 / 0 / 0** |
| generic shingles (>500 rows) | 0 |
| content-overlap rows | 0 |
| positive control (T5 eval prompts scanned with the same code) | **100/100 detected**, so the method works |
| exact match of T5 option sentences (299 distinct) and T5 premise sentences (598 distinct) against audited option/premise sentences | 0 hits |
| exact match of JamC-QA / T6 short items (9,507 and 281) against audited options | 0 hits |

Entity-name overlap (names of at least 4 characters, 5,192 distinct audited entities), substring search in the full eval text: benchmark_8100 33, jamcqa 11, T5 2, all others 0. Inspected all hits (`oj_leak2_out.txt`): they are incidental. JMMLU hits are substrings of unrelated words (ブロック, アルファ, コンパクト, ダーファ, ヘンリー of "ヘンリーの法則", ディマジオ, クオピオ in a medical item). T5 hits are substrings of unrelated words (アルファ inside アルファベット, ストラン inside レストラン). JamC-QA hits are person or company names appearing as answer choices or question words (手塚治虫, 石原裕次郎, 小西克幸, 明治天皇, 村田新八, 大隈重信, 朝日新聞, 読売新聞, ...): the family's fact about them is a birth date or HQ city, which is never the answer to those questions. No answer-bearing leakage found. T6 (Python/API questions), T1/T2 (code and state tracking), T8 (code) share nothing.

### 5.2 Train versus dev/heldout
See section 4: heldout shares nothing with train; dev shares all facts with train (not a leak into an eval that matters, but dev cannot measure generalisation).

## 6. Independent answer verification

Checker: `oj_check.py` + `oj_checkrun.py`. Written from the task text: strong-Kleene closed-premise semantics (a claim about an (entity, relation) is T if the premise states that value, F if it states another, U if the (entity, relation) is not mentioned), with my own regex parser for every one of the 27 option templates (filler prefix, "とは別に" distractor prefix and pointer prefix stripped; compound claims via Kleene and/or/not, counts, comparisons, date arithmetic, majority, distinct countries). It does not import any family module.

I ran it on **all rows**, not a sample (n = 50,625, far above the 400 required, all six operators, all 27 templates):

| split | rows | unparsed | tie/ambiguous | checker-unique answer equals reference letter | per-option T/F/U equals family `option_labels` |
|---|---|---|---|---|---|
| train | 46,008 | 0 | 0 | 46,008 | 46,008 |
| dev | 1,917 | 0 | 0 | 1,917 | 1,917 |
| heldout | 2,700 | 0 | 0 | 2,700 | 2,700 |

Per-operator counts (all 100%): train ent 10,368, con 10,368, und 9,720, notcon 5,184, notent 5,184, notund 5,184; dev 432/432/405/216/216/216; heldout 594/594/378/378/378/378. **Mismatching rows: none**, so there are no concrete mismatch examples to list. Two parse gaps found and fixed in my own checker during development (negation phrase "…の本社は…にはない", date phrase "…に生まれた"); they were checker gaps, not data errors.

Trace verification (`oj_trace.py`, all 50,625 rows): every target has the exact 6-line layout; each evidence line's final 真/偽/不明 equals the option label; the 選ぶのは phrase matches the operator class and the chosen letter; every cited premise number exists, the cited value equals the premise value (year-only citations in sameyear/gap are consistent), every `前提にXなし` names a string that is in the option and not a premise entity. 0 violations.

Semantic-validity probe (`oj_compat.py`): of 24,534 option pairs whose claimed value differs from the premise value (atom, pointer, negation, same, differ, crossrel), 0 are plausibly compatible in the real world under my containment / shared-prefix / mixed-admin-level heuristics, so the "F" labels are not hiding place-granularity ambiguities (e.g. city vs prefecture).

Caveat on what this proves: the checker confirms labels and traces match the stated closed-world semantics. It does not (and cannot) show the semantics train T5-style judgement; see sections 7 and 9.

## 7. Shortcut / bias audit

### 7.1 Answer and label balance
- Answer letters are exactly balanced: train A=B=C=15,336; dev 639 each; heldout 900 each. Majority-answer baseline = 0.3333 exactly. Balanced within every operator (e.g. train ent 3,456 per letter).
- Option label classes in a train sample: T 5,808, F 5,832, U 5,613 (balanced). No yes/no or numeric answers.

### 7.2 Single-rule surface solvers (script `oj_shortcut.py`)
Accuracy of picking the option with the highest / median / lowest value of a feature (ties to earliest letter). "oracle/op" = best of the three ranks chosen per operator in hindsight (optimistic: under a null the best-of-3 over 6 operators already adds about +0.03 to +0.05).

| feature (all computed from prompt only) | dev max/med/min | oracle/op | heldout max/med/min | oracle/op | train-sample max/med/min | oracle/op |
|---|---|---|---|---|---|---|
| position (A/B/C) | .333 each | .333 | .333 each | .333 | .333 each | .333 |
| len_raw (option chars) | .333/.333/.333 | .333 | .333/.333/.333 | .333 | .333/.333/.333 | .333 |
| len_body (filler stripped) | .330/.315/.361 | .378 | .336/.310/.350 | .366 | .337/.312/.347 | .369 |
| difflib ratio to premise (raw) | .348/.346/.306 | .405 | .329/.348/.324 | .367 | .332/.349/.319 | .399 |
| difflib ratio (body) | .355/.353/.292 | .454 | .354/.308/.335 | .430 | .335/.339/.325 | .461 |
| difflib best premise sentence | .373/.321/.306 | .538 | .358/.298/.344 | .457 | .351/.305/.344 | .548 |
| Jaccard char-2gram | .348/.359/.296 | .551 | .336/.306/.356 | .536 | .328/.343/.332 | .576 |
| coverage of option 2-grams by premise | .360/.347/.298 | .569 | .333/.307/.357 | .534 | .328/.337/.336 | .596 |
| uncovered chars (3-gram) | .294/.323/.377 | .589 | .346/.299/.349 | .526 | .344/.322/.338 | .616 |
| copy score (entity and its own value both in option) | .395/.304/.320 | .548 | .375/.267/.358 | .488 | .358/.296/.347 | .531 |
| number of premise entity names in option | .401/.300/.298 | .609 | .379/.246/.381 | .648 | .370/.267/.367 | .678 |
| number of premise values in option | .375/.317/.314 | .436 | .352/.291/.353 | .401 | .348/.311/.344 | .424 |
| has filler prefix (ちなみに/なお/また/ところで) | .365/.271/.322 | .424 | .365/.304/.324 | .410 | .371/.302/.326 | .430 |
| odd-one-out (least similar to other options) | .282/.334/.381 | .410 | .303/.317/.383 | .383 | .295/.316/.389 | .397 |

Global single rules (difflib, Jaccard, length, position) are indeed at chance. Entity-aware rules are not: ranking by "how many premise entity names does the option contain" with a per-operator rank reaches .61 / .65 / .68. The family's own `_audit_*.txt` also reports a "copy" oracle of .537 (heldout) and .591 (dev).

### 7.3 Citation-number rule (the T5 shortcut) (`oj_shortcut.py`)
- Only the pointer template cites premise numbers: train 3,888 rows (8.45%), dev 162, **heldout 0**. All other templates have no number in the options.
- T5 rule ((1)=T, (2)=F, (3)=U, pick the option whose cited number maps into the operator's class): dev 0.370 (n=162), train sample 0.311 (n=486). Chance 0.333. **Removed.**
- Learned number-to-label table from train pointer rows: cited k=1: U132/F128/T142; k=2: F133/U122/T125; k=3: T148/U158/F149; k=4: U65/T46/F55; k=5: U12/T22/F21 (uniform); learned rule: dev 0.358, train sample 0.292. **At chance.**
- But in pointer rows "does the cited premise sentence's entity occur in the option" separates U from T/F **100%** (dev 162/162, train sample 486/486). The number is no longer a label cue, but the pointer still hands over a deterministic U detector.

### 7.4 Learned surface classifiers (the real test) (`oj_shortcut.py`, `oj_shortcut2.py`)
Gradient-boosting on per-option surface features (length, difflib x3, Jaccard, coverage, uncovered chars, entity hits, value hits, copy score, filler flags, digit count, similarity to the other options), trained on 5,751 train rows (every 8th row, offset 1), row decision = option maximising the summed probability of the operator's class set. Chance = 0.333.

| model | dev (n=1,917) | heldout (n=2,700) | train-sample, disjoint (n=5,751) |
|---|---|---|---|
| GBM all surface features | **0.793** | **0.501** | **0.841** |
| GBM without entity-hit / value-hit / copy features (char overlap only) | 0.729 | 0.471 | 0.754 |
| GBM on filler flags + raw length + digit count only | 0.432 | 0.391 | 0.457 |
| logistic regression on option text only (char 1-3-grams, **no premise at all**) | 0.510 | 0.386 | 0.535 |
| same, filler-stripped body only | 0.477 | 0.352 | 0.492 |

Per operator (GBM all features): dev und .978, notund .977, notcon .667, notent .611, ent .752, con .722; heldout und .741, notund .675, notcon .558, notent .500, ent .325, con .377. By template (GBM, dev): atom .895, pointer .988, date_exact 1.0, date_year 1.0, date_decade .975, conj .870, disj .883, negation .698, same .642, differ .605, exists .519. Heldout by template: xor .747, inlist .691, notboth .639, notin .583, hassame .410, alldiff .389, namedcount .367, ndistinct .343, atleast .278, count .241, majority .176 (the last three are anti-correlated, i.e. the learned cues do not transfer there).

### 7.5 Why: the entity-novelty cue
Rule "und: pick the option with the fewest premise entity names; notund: pick the most", applied only where the minimum/maximum is unique:
- dev: accuracy 1.000 on all 15 templates (coverage 100% for atom/pointer/negation/same/differ/crossrel/date_*, 76% conj, 37% disj);
- heldout: 1.000 on inlist, notin, xor, notboth, namedcount, alldiff, hassame (coverage 69% to 100%);
- train sample: 1.000 (one disj miss 0.989).
So und and notund rows (14,904 train rows = 32.4%) are solvable by checking whether a name appears in the premise. By construction every U option names an entity that is absent from the premise (the generator never uses an entity that is present with an unmentioned relation).
The T and F options differ by whether the (entity, value) pair is copied from the premise; a "copy" feature gives most of the ent/con accuracy on in-distribution templates (atom .895, pointer .988).

### 7.6 Fact memorisation versus premise reading (`oj_memory.py`)
Using only the train fact table (no premise at all), the claim in atom/pointer options is judged: true-T options are memory-T 378/378 (dev), true-F options are memory-F 324/324; true-U options come out memory-T 140, memory-F 114, memory-U 16. Because every premise fact is a real-world truth (Wikidata/JEMHopQA) and the false claims are real-world false, **a model that memorises the 5,223 facts needs no premise for T and F**, and premises are never counterfactual. This explains the option-only logistic regression at 0.51 on dev (it falls to 0.386 on heldout where the facts are new). Dev accuracy is therefore inflated by memorisation; heldout is the only honest number.

### 7.7 Surface bias in option wording
Filler prefixes (ちなみに / なお / また / ところで) are less frequent on U options: dev U 370/1,647 = 22% versus T 825/2,160 = 38% and F 700/1,944 = 36%; heldout U 559/2,268 = 24.6% versus T 38.4% and F 35.7%. A classifier on filler flags + length + digits gets 0.43 to 0.46 in-distribution and 0.391 on heldout (n=2,700, about 6 standard errors above chance). Small, but a free prior independent of the premise.

### 7.8 Conclusion on the shortcut question
- Premise-number shortcut: removed (rule at chance; heldout contains no citations at all).
- Cheap single-feature overlap/length/position rules: neutralised (the generator forces their rank).
- Not removed: entity-presence (novelty) cue, copy cue, memorised-fact cue, filler prior. A learned surface solver gets 0.73 to 0.84 in-distribution and 0.47 to 0.50 on heldout. These cues are inherent to the closed-world U definition, so they cannot be fully removed without changing the task (see fixes).

## 8. Provenance and licence

- 100% of rows are self-authored templates over real facts: `selfauthored:geocpu/v49-option-judgement+facts:Wikidata_CC0_truthy` 32,400 train rows (70.4%), licence CC0-1.0; `...+facts:JEMHopQA_derivation_triple` 13,608 train rows (29.6%), licence **CC-BY-SA-4.0** (share-alike attaches to the facts-derived rows). Dev: 1,350 CC0 / 567 CC-BY-SA. Heldout: 2,700 CC0 (Wikidata only).
- Human-written natural sentences: 0 rows. The only human/real component is the fact table (5,223 distinct facts in train, 287 in heldout); all wording comes from a few dozen templates (about 14 surface frames per template, 3 prompt headers, 3 cue lines, 24 instruction sentences).
- training_category is "japanese" for every row.

## 9. Format alignment with T5 to T8

### 9.1 T5 (6-operator logical contrast, 3 options, JNLI-grounded)
Note: the T5 eval file has **4** operator_types x 100 items (positive_entailment, negation_contradiction, non_entailment_undeducible, not_necessarily_true), 400 items, targets A 135 / B 126 / C 139. The harness text says "6-operator" but only 4 occur. The family's 6 classes (ent, con, und, notcon, notent, notund) cover those 4 (not_necessarily_true is U-class) plus 2 operators T5 never tests; the same-premise instruction-contrast structure of T5 (4 instructions on one context) never occurs in training.

T5 eval prompt (`eval_v41_logical_instruction_operators_heldout.jsonl`), target is just the letter:
```
タスク: 自然言語指示の論理演算子変換・3値論理推論 (JNLI Grounded)
【本文】
（1）テーブルの上に花が活けられています。 （2）パーマをした女性がこちらを向いています。 （3）部屋の手前に女性が二人立っています。
【指示】
次のA〜Cのうち、前提から論理的に【正しい（確実に導ける）】ものを選べ。
【選択肢】
A. （3）に関して、リビングでは、二人の女性が立食しています。
B. （1）に関して、テーブルの上に、花が活けられた花瓶が置かれています。
C. （2）に関して、パーマをした女性がむこうを向いています。
思考の連鎖と回答:
```
target: `B`. The scorer (`eval_logical_instruction_operators.extract_choice_abc`) first looks for `正解[:：]\s*([ABC])`, else the last standalone A/B/C in the final 40 chars, else the last A/B/C character.

Family prompt and target (train, `v49oj_train_oj_atom_ent_0`):
```
問題: 次の【前提】だけを根拠にして、【指示】に合う選択肢を一つ選んでください。
【前提】
（1）ハリー・キューウェルの出生地はシドニー。
（2）高松寿嗣の出生地は明石市。
...
【指示】前提から論理的に導けるものはどれですか。一つ選んでください。
【選択肢】
A. リライアンス・ペトロリアムとハリー・キューウェルとは別に、明石市は高松寿嗣の出生地である。
B. ...
選択肢ごとに根拠を書いてから答えてください。      (or: 一つずつ確かめてから答えてください。 / 考え方と答え:)
```
target:
```
根拠:
A: (2)と一致=真 → 真
B: (5)はロアンヌ=偽 → 偽
C: 前提にコラスFCなし=不明 → 不明
選ぶのは真のもの → A
答え: A。
```
Differences: header (`タスク: ...(JNLI Grounded)` vs `問題: ...` x3), slot name (`【本文】` vs `【前提】`), premise layout (all on one line vs one-per-line or one line), instruction wording (T5 `【正しい（確実に導ける）】ものを選べ` vs 24 different `...一つ選んでください。` sentences), option form (T5 always `（k）に関して、` + free caption sentence; family: pointer prefix only in 8.45% of train, 0% of heldout, other options are name-value facts with "X とは別に、" / ちなみに distractor prefixes), cue line (`思考の連鎖と回答:` vs `考え方と答え:` or a sentence), target (letter only vs a 6-line closed-world trace ending `答え: X。` instead of `正解: X`), content (JNLI image captions with semantic neutral/contradiction vs Wikidata/JEMHopQA name-value facts). The scorer's fallback (last standalone letter) would still read a trace that ends `答え: A。`, but a model must first emit the trace after a cue it has never seen. Trace vocabulary (`一致`, `前提にXなし=不明`) has no counterpart in caption NLI.
Plausibility: it can teach the instruction-to-class mapping (ent/con/und), three-way answer discipline and letter selection; it cannot teach caption-level entailment, and U in this family means name-absent, which is not how T5 neutral options work.

### 9.2 T6, T7, T8
- T6 (search-augmented multi-hop retrieval; e.g. `質問: Pythonで大きな整数 ... 標準関数は何か？\n思考:` with search queries): this family has no search, no tool steps (`tool_steps` empty) and different content. Cannot train T6.
- T7 (closed/open-book JMMLU-style 4-choice): eval format `分野: abstract_algebra (math)\n問題: ...\nA: 8\nB: 2\nC: 24\nD: 120\n正解:` with target `C` (JamC-QA: `質問: ...\nA: ..\n答え:`). Family: 3 options, `A. ` with a dot, premise-grounded, answer inside a trace. It trains no knowledge. At most a letter-selection habit.
- T8 (4 fixed Python/C++/Rust code plus explanation prompts): no code in this family. Cannot train T8.

## 10. Fixes recommended (in order of value)

1. Rename or gzip `train.jsonl.gz`; delete or regenerate the stale `_gen_hd.log`.
2. Break the entity-novelty cue: add U options whose entity IS in the premise but whose relation is not stated (e.g. ask the HQ of a person premise entity, a country for a birth-place entity), and F options that reuse entities absent from the premise only in distractor prefixes, so "name absent" no longer implies U.
3. Break memorisation: counterfactual premises (swap values among entities, or use invented entities/values) so the premise is the only source of truth; that also makes dev independent of train facts.
4. Scale and diversify: 5,223 facts (75 date facts) is the limit, not the 46k row count. Draw facts from the full Wikidata/JEMHopQA tables; hold out entities (not only relations) in dev.
5. If the purpose is T5: emit the T5 header, `【本文】`, `（k）に関して、` options, `思考の連鎖と回答:` cue and a `正解: X` ending for a share of rows, include same-context 4-instruction groups, and use caption-style premises; otherwise treat this family as general closed-world logic data, not T5 data.

## 11. Commands and scripts (all `python -I`, one process at a time, streaming)

| script | purpose |
|---|---|
| `oj_pass1.py` | inventory, sha256, schema, OOV, duplicates, conflicts, distributions (`oj_pass1_out.json`, `oj_pass1_view.txt`) |
| `oj_variants.py`, `oj_samples.py` | option surface frames per template, heads/tails/instructions (`oj_variants_out.txt`, `oj_samples_out.txt`) |
| `oj_check.py`, `oj_checkrun.py` | independent three-valued checker over all rows (`oj_checkrun_out.json`, `oj_checkrun_view.txt`) |
| `oj_trace.py` | trace consistency and fact-reuse statistics (`oj_trace_out.json`) |
| `oj_shortcut.py`, `oj_shortcut2.py` | rank rules, pointer number rules, learned surface classifiers, novelty rule (`oj_shortcut_out.json`, `oj_shortcut2_out.json`) |
| `oj_memory.py` | memory-only (no premise) fact solver (`oj_memory_out.json`) |
| `oj_leak.py`, `oj_leak2.py` | shingle leak versus 8 eval files, exact short-item match, entity overlap, train/dev/heldout overlap, positive control (`oj_leak_out.json`, `oj_leak2_out.txt`) |
| `oj_compat.py`, `oj_groups.py`, `oj_len.py` | F-label real-world compatibility probe, premise-block reuse, option length spread |

Not run: none of the checklist items were skipped. Limits: learned classifiers were fit on a 5,751-row train sample (not all 46,008) and evaluated on dev, heldout and a disjoint 5,751-row train sample; entity-name search used names of at least 4 characters; T8's four prompts were not opened (the family has no code content, so format comparison is by task description).
