# Audit: family `jglue_human` (GeoCPU v49 training corpus)

Auditor: independent data auditor, 2026-11-09. Everything below was recomputed from the data files and the JGLUE source files. The family builder was read only to learn the claims. None of its modules was imported.

**VERDICT: blocked.** One blocker: T5 eval sentences are in train and dev (68 JNLI rows). The fix is a drop list of 68 ids, written to `jg_t5_nfkc_drop_ids.txt`. Everything else checked out, with caveats listed in section 11.

Paths (all under `C:/Users/81808/AppData/Local/Temp/claude/ff74aa4e8a98970d/f80753de-c283-4765-a931-f974c2b54295/scratchpad/`):
- Family: `wt_v48/research/v49_data_20261108/families/jglue_human/` (`data/train.jsonl`, `data/dev.jsonl`, `data/heldout.jsonl`, `data/manifest.json`)
- Audit dir (scripts and raw results): `audit_v49/`. Scripts are `jg_common.py`, `jg_a1.py`, `jg_a2_dups.py`, `jg_a3_source.py`, `jg_a3b_examples.py`, `jg_a4_leak.py`, `jg_a4_leak_nfkc.py`, `jg_a4b_t5detail.py`, `jg_a4c_nfkc_t5.py`, `jg_a4d_t5items.py`, `jg_a5_baselines.py`, `jg_a6_misc.py`, `jg_a7_evalfmt.py`, `jg_a8_negctl.py`. Results are in the matching `jg_a*_result.json` / `.txt` files.
- Run command for every script: `python -I <script>.py`. One process at a time, every JSONL file streamed line by line, longest run 35 s.

---

## 1. Inventory (streamed; `jg_a1.py`)

| split | rows (newline count) | bytes | sha256 | vs manifest |
|---|---:|---:|---|---|
| train | 73,863 | 168,782,864 | `33b913c34633bda2504d87f499483157259451a1b683547d0792f0b44d85435b` | rows, bytes and sha256 all match |
| dev | 2,043 | 5,796,514 | `ebcdb623aadefbc1b1c01a7916c4ebea08d09960721bee3681483e522db02a14` | all match |
| heldout | 2,457 | 5,654,869 | `5bd2b2826efc76b285044793bfe064f5c493f8ab95e9eff83caa0c48c95480c9` | all match |

- Total is 78,363 rows. Each file ends with a newline.
- The three JGLUE source files match the sha256 values recorded in `external_data/jglue_v1.3/PROVENANCE.md`.
  - `jsquad` = `3879e9df...40e36d0`
  - `jnli` = `c506354d...90d338d1c4`
  - `jcommonsenseqa` = `9b55fae5...48229e`
- Source record counts: JSQuAD 62,697 questions, JNLI 20,073, JCQA 8,939.

## 2. Schema (`jg_a1.py`)

- **Required fields.** All ten (id, prompt, target, full_text, split, family, source, license, tool_steps, loss_mask_spans) are present in every row. Missing count is 0.
- **full_text.** `full_text == prompt + target` exactly in all 78,363 rows (0 violations).
- **ids.** 78,363 distinct ids, none repeated within or across splits.
- **split field.** It equals the file name in every row.
- **Empty fields.** No empty prompt or target.
- **Terminal pattern.** Every row's target ends with `答え: …。`. Each family's own regex parsed all rows, with 0 prompt or target parse failures:
  - jsquad: `\n根拠: 「<quote>」\n答え: <span>。`
  - jnli_label: `\n答え: (含意|矛盾|中立)。`
  - jnli_letter: `\n答え: [ABC]。`
  - jcqa: `\n答え: [A-E]（<choice>）。`
- **reference_final_answer.** It equals the answer parsed from the target in all rows (0 mismatches).
- **tool_steps and loss_mask_spans.** Both are `[]` on all 78,363 rows. No explicit loss mask exists, so the trainer has to mask the prompt itself.
- **Extra fields.** `structural_template`, `answer_type`, `difficulty`, `training_category`="evidence", `_v46_group`="v49_human_reading". `nli_label` is on 13,733 rows and `quotes` on 55,692.

Length of `full_text` (chars; 1 char = 1 token, so a row needs len+1 <= 2048):

| split | n | min | mean | p50 | p99 | max | rows with len+1 > 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 73,863 | 92 | 292.66 | 281 | 672 | 1,245 | 0 |
| dev | 2,043 | 97 | 374.44 | 357 | 813 | 1,246 | 0 |
| heldout | 2,457 | 91 | 294.96 | 267 | 703 | 810 | 0 |

- Overall mean is 294.87.
- Mean by family (train): jsquad 350.7, jnli_label 177.8, jnli_letter 173.3, jcqa 115.0.
- **Supervised chars** (sum of len(target)+1) are **train 4,652,214**, dev 173,551, heldout 147,719.
- Family counts, train: human_jsquad 52,348, human_jcqa 8,599, human_jnli_label 6,470, human_jnli_letter 6,446. These match the manifest.

## 3. Vocabulary (`jg_a1.py`, `jg_a6_misc.py`)

The set is `vocab_chars.json`: 3,498 characters, the union of characters in `full_text` of `train_40500_exact.jsonl`.

| split | rows with at least 1 OOV char | rows with OOV in the target |
|---|---:|---:|
| train | 9,105 / 73,863 (12.3%) | 3,525 |
| dev | 359 / 2,043 (17.6%) | 143 |
| heldout | 302 / 2,457 (12.3%) | 88 |

- Train OOV rows by family: jsquad 7,568/52,348 (14.5%), jnli_label 560/6,470, jnli_letter 504/6,446, jcqa 473/8,599.
- 1,107 distinct OOV characters.
- Most frequent OOV characters (occurrences / rows):
  - `＝` U+FF1D 1,481 / 807
  - `％` 711 / 324
  - `２` 671 / 549
  - `１` 658 / 506
  - `℃` 342 / 144
  - `鄧` 302 / 91
  - `３` 290 / 247
  - `袁` 259 / 80
  - `蔣` 232 / 110
  - `款` 232 / 86
  - `ā` 196 / 59
  - `ا` 196 / 38
  - `嶋` 186 / 98
  - `阜` 181 / 92
  - `４` 175 / 140
- Applying NFKC (full-width digits, `＝`, `％` to ASCII) removes some of this:
  - 6,811 rows are still OOV afterwards, with 1,006 distinct characters left.
  - Target OOV still affects train jsquad 2,863 rows, train jcqa 68, dev jsquad 107, heldout jsquad 71 and heldout jcqa 1.
  - What remains is mostly rare kanji from Wikipedia names.
- The `ā` and Arabic letters are also OOV.
- To train this family as is, the vocabulary has to be extended or the rows filtered or normalised.

## 4. Duplicates (`jg_a1.py`, `jg_a2_dups.py`)

- **Exact duplicate `full_text`:**
  - Within train: 15 rows. Within heldout: 1 row. Within dev: 0. Across splits: 0.
  - Ids per group are in `jg_a2_dups.txt`. They are 11 JNLI label-form groups, 1 JNLI letter-form group and 4 JSQuAD groups. The JNLI ones are identical sentence pairs from different images, for example `v49human_jnli_7140`/`7143`. The JSQuAD ones are duplicated questions with the same answer.
- **Same prompt, different target: 19 prompts (38 rows), all JSQuAD, all in train.**
  - All 19 are annotator span-granularity variants, where one answer string contains the other. Example: `…a14170p0q4` answer "クテシフォン" versus `…a14170p0q0` "クテシフォン（現在のイラク）".
  - They are not factual contradictions (checked: 19 of 19 nested).
  - Still, an identical prompt with two different supervised answers is label noise. The cleanest fix is to keep one per prompt.
- **Prompt overlap, train vs dev/heldout:** 0 shared prompts. Dev vs heldout: 0.
- **Template share (train):**
  - human_jsquad 70.9%
  - human_jnli 17.5% (label form 8.8%, letter form 8.7%)
  - human_jcqa 11.6%
  - Two structural templates are over 15%, and JSQuAD dominates at 70.9%. The content is diverse human text, but the instruction format is one fixed line, "条件と根拠を順に考えてください。".

## 5. Leakage against the eval manifests

### 5a. Shingle method (`jg_a4_leak.py`)
Method as specified:
- Eval text is every string value of every row in the 8 eval files, plus the 4 T8 prompts and their aligned variants (8 strings), parsed from `tier8_readonly_v44fix.py` with `ast.literal_eval`. In total 3,054,754 chars.
- 167,308 hashed 24-char shingles at stride 8.
- Stride-1 windows slid over `prompt+target` of all 78,363 audited rows.
- A shingle is generic if it occurs in more than 500 audited rows.

Result:

| measure | value |
|---|---|
| distinct eval shingles hit | 5 |
| generic shingles (>500 rows) | 0 |
| rows with content overlap | 12 (train only; dev 0, heldout 0) |

- 2 of the 12 rows are `v49human_jsquad_a44508p0q0` and `…q1`, which overlap `benchmark_8100_manifest.jsonl` on `": Transport Layer Securi"`. That is the JMMLU option "Transport Layer Security" against a Wikipedia TLS passage. It is term overlap, not an eval item. Not leakage.
- 10 of the 12 rows are JNLI train rows, `v49human_jnli_903`/`904`/`905`/`907`/`908`/`909` and `931`/`932`/`933`/`934`. They overlap `eval_v41_logical_instruction_operators_heldout.jsonl` (T5). Spans:
  - "オレンジ色の長椅子のついたテーブルに並んで座って"
  - "字型のキッチンのカウンターにレンジや棚があります"

Re-running the same method on NFKC-normalised text (`jg_a4_leak_nfkc.py`, 167,341 shingles) gives 24 rows with content overlap:
- 22 in train, 2 in dev.
- 22 rows are JNLI-vs-T5 (20 in train, 2 in dev) and 2 are the TLS JSQuAD rows. Heldout: 0.
- The match is lost without NFKC because the T5 file has half-width characters ("1頭", "C字型") where JNLI has full-width ones.

No other eval file (T1, T2, T6, JMMLU or JamC-QA content, T8) has a content shingle in any audited row.

### 5b. The specific T5 claim (`jg_a3_source.py`)
- JNLI source pairs where sentence1 or sentence2 (at least 14 chars) appears verbatim in the T5 file: **1,158** (equals the builder's claim).
- Of these, 0 appear in any output split. The specified check passes.
- Any length: 1,172 source pairs, of which 14 are in output (train). All 14 are short generic fragments of 7 to 12 chars, for example "立っています。" (`v49human_jnli_1183`/`1184`), "トラックが走っています。", "男性が歩いています。". They are not leakage.
- Output JNLI rows that share a sentence with T5: 0 at 14 chars or more.

### 5c. BLOCKER: near-verbatim T5 leakage that the exact check misses (`jg_a4b_t5detail.py`, `jg_a4c_nfkc_t5.py`, `jg_a4d_t5items.py`)

The builder's blocklist is an exact substring test of the JNLI sentence inside the raw eval text, in one direction only. Two normal variants slip through:

1. **NFKC variant.** At least one JNLI sentence of the row equals a T5 sentence once full-width digits and letters are normalised. 32 train rows and 2 dev rows. Example rows:
   - `v49human_jnli_2310`: "黄色いバスが１０台とまっています。" versus T5 "…10台…".
   - `v49human_jnli_2312`: "赤いバスが１０台走っています。" and "赤いバスが１０台とまっています。".
   - `v49human_jnli_4141`: "シマウマが２頭、草を食べています。".
   - `v49human_jnli_16960`: "１頭の象が水の近くの土の上を歩いており、奥にもう１頭の象がいます。".
   - dev `v49human_jnli_2879`/`2880`: "2階建てバスの前の乗降口から、乗り込んでいる人が…".
2. **Suffix variant.** A T5 sentence (14 chars or more) is a verbatim suffix of a longer JNLI sentence, because T5 shortened the original caption. 34 train rows. Examples:
   - `v49human_jnli_7140`/`7141`/`7143`: JNLI "黒いウエットスーツを着た男性が、海でサーフィンをしています。" contains T5 "男性が、海でサーフィンをしています。".
   - `v49human_jnli_2895`/`2901`/`2902`: "道路脇の歩道にパーキングメーターが設置されています。" contains T5 "パーキングメーターが設置されています。".
   - `v49human_jnli_10165`: "雪山の斜面をスキーヤーが滑っています。" contains T5 "斜面をスキーヤーが滑っています。".
   - `v49human_jnli_4410`: JNLI "茶色いTシャツを着た男性がスケートボードに乗っています。" contains T5 "男性がスケートボードに乗っています。".
   - `v49human_jnli_9319`: JNLI "波しぶきを上げてサーフィンをしている男性がいます。" contains T5 "サーフィンをしている男性がいます。".

Totals:
- **68 output rows: 66 train and 2 dev** (`jg_a4c_result.json`). Ids are in `jg_t5_nfkc_drop_ids.txt`.
- Under the NFKC rule, 1,238 source pairs are flagged (versus 1,158 by the exact rule), and 68 of the flagged pairs are present in the output.
- These 68 rows touch **68 of the 400 T5 items, which is 17 of the 100 T5 context groups** (17 groups times 4 operators, `jg_a4d_result.json`).
- The leaked rows show the model T5 sentences together with a JNLI relation label for a pair containing them.

Related and softer: 182 train rows and 12 dev rows come from the 372 image groups that T5 touches (sibling captions of T5 sentences). Excluding them is optional and conservative.

Recommended fix:
- Drop the 68 ids.
- Rebuild the blocklist with NFKC normalisation and a two-way containment test (JNLI sentence inside T5, and T5 segment inside JNLI sentence).
- Optionally drop whole T5-touched image groups.

### 5d. Exact eval short items (`jg_a4_leak.py`)
- 79,027 distinct eval segments of 10 chars or more, with 10-char prefixes indexed, were searched as exact substrings of every audited row.
- 64 distinct items hit, in 690 rows (train 651, dev 26, heldout 13).
- All 64 are entity names or option strings from JMMLU/JamC-QA that Wikipedia passages naturally contain. Examples: "ミハイル・ゴルバチョフ" in 160 JSQuAD rows, "ミルトン・フリードマン" in 68, "私的独占の禁止及び公正取引の確保に関する法律", "こちら葛飾区亀有公園前派出所".
- The remaining hits are T5 JNLI sentences, already counted in 5c ("パーキングメーターが設置されています。" and similar).
- No eval question stem and no full-sentence option from JMMLU or JamC-QA occurs inside any JSQuAD or JCQA row. Knowledge overlap by entity name is unavoidable and not an item leak.

### 5e. Train vs dev/heldout (`jg_a3_source.py`, `jg_a3b_examples.py`)
- No group appears in two splits, with the group taken from the SOURCE record, not from the builder:
  - JSQuAD title: 708 groups
  - JNLI image id (`yjcaptions_id` split on "-"): 5,941 groups
  - JCQA q_id: 8,938 groups
  - 0 groups are in more than one split.
- JNLI: 0 identical (sentence1, sentence2) pairs across splits, and 0 pairs with conflicting labels.
- Sentence-level overlap that remains:

| item | dev | heldout |
|---|---|---|
| JNLI rows sharing a full sentence with a train row (different image, identical caption) | 8/239 (3.3%) | 21/578 (3.6%) |
| JSQuAD passages equal to a train passage | 0 | 0 |
| JSQuAD quote sentences also a train quote | 0 | 2 (e.g. "血液型はB型。") |
| JSQuAD question strings equal to a train question | 0 | 3 (e.g. "フランスの首都は。" ×2, "任期は何年か？") |
| JCQA question strings equal to a train question | 1 | 6 (e.g. "生き物はどれ？", "甘いのは？") |

  These are short or generic strings. They are minor but real, so heldout is not perfectly disjoint at the sentence level.

## 6. Independent answer verification (`jg_a3_source.py`, `jg_a8_negctl.py`)

**Every output row was checked against the source files by id, not a sample.** n = 78,363 (JSQuAD 55,692, JNLI 13,733, JCQA 8,938), which is more than the 600 rows per task asked for. The checker was written from the visible row format.

Mismatches:

| check | mismatches |
|---|---:|
| JSQuAD id not in source | 0 |
| passage differs from source context (after "title [SEP] ") | 0 |
| title differs | 0 |
| question differs | 0 |
| answer differs from `answers[0].text` | 0 |
| source offset inconsistent | 0 |
| quote is not a substring of the prompt passage | 0 |
| answer not contained in quote | 0 |
| quote does not cover the source answer span (`answer_start` minus title length) | 0 |
| `quotes` field differs from quote | 0 |
| quote has more than one "。" | 0 |
| JNLI sentence1/sentence2 differ | 0 |
| JNLI gold label differs from source (label form) | 0 |
| JNLI letter-form option set is not a permutation of the 3 glosses | 0 |
| JNLI letter-form gold letter points at a gloss with the wrong label | 0 |
| JNLI `nli_label` field differs | 0 |
| JCQA question differs | 0 |
| JCQA option multiset differs from the 5 source choices | 0 |
| JCQA gold letter points at a choice other than the original gold | 0 |
| JCQA parenthesised text differs from original gold choice | 0 |
| `reference_final_answer` letter differs | 0 |

- 71 JSQuAD rows are flagged only by a strict "quote starts right after 。" test. In each, the preceding character is a space (56), U+3000 (11) or NBSP (4) that the builder stripped. This is benign and the quote is still verbatim.
- Not a vacuous check. Negative controls on heldout (`jg_a8_result.json`):
  - JNLI letter: 287/287 agree with source as built, but 0/287 after shifting the letter.
  - JCQA: 250/250 versus 0/250.
  - JSQuAD answer-in-quote: 1,629/1,629 as built, 13/1,629 when the answer is swapped with another row's.
- Source-to-output coverage reconciles exactly:
  - JSQuAD 62,697 questions minus 6,986 with the answer in the title, minus 10 passages over 700 chars, minus 9 sentence-span or blocklist cases gives 55,692.
  - JNLI 20,073 minus 1,158 T5 blocklist minus 5,182 neutral under-sampling (train only) gives 13,733.
  - JCQA 8,939 minus 1 duplicate-choices case gives 8,938.

## 7. Shortcut and bias audit (`jg_a5_baselines.py`, `jg_a6_misc.py`)

**JNLI**
- Gold label in train:

| label | count | share |
|---|---:|---:|
| contradiction | 5,186 | 40.2% |
| neutral | 5,186 | 40.2% |
| entailment | 2,544 | 19.7% |

  This confirms "neutral under-sampled in train only". Entailment is still half as frequent as the others.
- Dev: entailment 36, contradiction 84, neutral 119.
- Heldout: entailment 84, contradiction 152, neutral 342.
- **The label prior shifts between train and heldout.** The always-neutral baseline gets 59.2% on heldout and 49.8% on dev.
- Letter form, gold letter distribution:

| split | A | B | C |
|---|---:|---:|---:|
| train | 2,149 | 2,193 | 2,104 |
| dev | 42 | 36 | 37 |
| heldout | 87 | 107 | 93 |

  Majority letter on heldout is B at 37.3%. Letter is independent of label (each label splits near-uniformly over A/B/C).
- Surface baselines trained on train (character-bigram naive Bayes, tested on heldout, with dev in parentheses):

| baseline | heldout | dev |
|---|---:|---:|
| hypothesis only | 49.8% | 51.9% |
| premise only | 52.8% | 48.1% |
| both sentences | 52.6% | 50.6% |
| symmetric-difference bigrams | 60.0% | 59.8% |
| Jaccard of bigrams × length-difference lookup table | **66.8%** | 62.8% |

  - No hypothesis-only shortcut: it is below the majority baseline.
  - Lexical overlap alone is 7.6 points above the heldout majority baseline of 59.2%.
- `sentence1 == sentence2` occurs in 114 rows, and all 114 are entailment.

**JCQA**
- Gold letter in train: A 1,703, B 1,779, C 1,771, D 1,669, E 1,677 (chi-square 6.29, df 4, p about 0.18).
- Dev: chi-square 1.06.
- Heldout: A 58, B 68, C 47, D 31, E 46, chi-square 15.48, p about 0.004. The majority-letter baseline there is 27.2% (B), versus 20% chance.
- Other surface baselines: longest choice 21.5% (train) / 23.6% (heldout), shortest 17.8% / 18.8%, maximum bigram overlap with the question 23.2% / 26.8%.
- A choice-only prior (how often a string was gold in train) scores 6.8% on heldout and 11.2% on dev, which is below chance. Inverting it scores 29.6% on heldout and 29.2% on dev, about 3 standard errors above 20% on n=250. This is a mild anti-correlation artifact: strings that were gold in train tend to be distractors elsewhere.

**JSQuAD**
- Gold evidence sentence is the first sentence in 58.8% of train and 57.5% of heldout. Chance for picking the sentence is about 37.5%.
- A selector that takes the sentence with the highest bigram overlap with the question picks the gold sentence 86.1% (train) and 84.5% (heldout) of the time. The "根拠" quote is therefore mostly recoverable by lexical match.
- 24.6% of train questions have more than 80% of their bigrams inside the quote (near-copy questions).
- The answer string appears inside the question in 1.5% of train rows (790). Answers have mean length 6.3 chars (max 19), and 29% contain a digit.
- **Statistical power of dev/heldout is poor.** Heldout JSQuAD has only 19 articles, and 野村克也 alone is 455 of 1,629 rows (27.9%). Dev has 10 articles, and 太平洋戦争 is 901 of 1,715 rows (52.5%). Heldout accuracy is effectively an average over about 19 independent clusters.

**Claims checked**
- "Neutral under-sampled in train only": true.
- "JCQA re-lettered so the gold letter is uniform": true for train and dev, skewed by chance in heldout.
- "Group split": true.
- No family claim of shortcut removal is contradicted.

## 8. Provenance and licence

- 100% of rows are human-written JGLUE v1.3 text. There are no self-authored content rows.
- The only text the builder adds: the fixed instruction sentences, the JNLI relation glosses and the "根拠:" quote. The quote is mechanically cut from the passage and verified verbatim (section 6). JCQA options are re-lettered by a seeded shuffle.
- `license` = `CC-BY-SA-4.0` on all 78,363 rows.
- `source` = "JGLUE v1.3 (Kurihara et al., Yahoo Japan / Waseda Univ.) - CC BY-SA 4.0 - https://github.com/yahoojapan/JGLUE" on all 78,363 rows. `manifest.json` repeats both. PROVENANCE.md pins commit `6f071c09316baae89c3d083a90985b4b1cb9968c`.
- ShareAlike obligations apply to redistribution of the data. JSQuAD itself derives from Japanese Wikipedia, so attribution at row level is only the article title inside the prompt. Whether ShareAlike reaches model weights is unsettled.

## 9. Format alignment (T5 to T8)

T5 eval (`eval_v41_logical_instruction_operators_heldout.jsonl`): 400 items = 100 context groups × 4 operators (positive_entailment, negation_contradiction, non_entailment_undeducible, not_necessarily_true). The file has 4 operators, not 6. Targets A 135 / B 126 / C 139. Example prompt:
`タスク: 自然言語指示の論理演算子変換・3値論理推論 (JNLI Grounded)\n【本文】\n（1）テーブルの上に花が活けられています。 （2）パーマをした女性がこちらを向いています。 （3）部屋の手前に女性が二人立っています。\n【指示】\n次のA〜Cのうち、前提から論理的に【正しい（確実に導ける）】ものを選べ。\n【選択肢】\nA. （3）に関して、リビングでは、二人の女性が立食しています。\nB. （1）に関して、…\nC. （2）に関して、…\n思考の連鎖と回答:` → target `B`.

This family, JNLI letter form: `問題: 前提の文と仮説の文を読み、2つの関係を判断してください。\n前提: …\n仮説: …\n正しい関係の記号を答えてください。\nA. どちらとも言えない\nB. 前提と仮説は両立しない\nC. 前提から仮説が言える\n条件と根拠を順に考えてください。` → target `\n答え: A。`

| | T5 eval | jglue_human |
|---|---|---|
| context | 3 captions | 1 premise + 1 hypothesis |
| options | three JNLI sentences tagged `（k）に関して、` | three fixed relation glosses |
| question | which sentence is entailed / contradicted / undeducible / not necessarily true | which relation holds |
| prompt ends with | `思考の連鎖と回答:` | `条件と根拠を順に考えてください。` |
| target | bare letter | `\n答え: A。` |

- **T5:** the skill (3-way NLI over JNLI captions) is the same and the 13.7k rows are the right kind of material. The format differs, so it supports skill transfer, not format learning.
- **T6** (search-augmented multi-hop; prompt `質問: …\n思考: 外部検索エンジンを使用して必要な事実を確認します。\n` → `math.isqrt`): this family cannot train it. Every row has `tool_steps: []` and there is no search-query step. JSQuAD's passage-grounded `根拠: 「…」` quote is not retrieval.
- **T7:** JCQA (5-choice commonsense) is the closest.
  - JMMLU eval: `分野: …\n問題: …\nA: 8\nB: 2\nC: 24\nD: 120\n正解:` → `C` (4 options, `A: ` separator).
  - JamC-QA eval: `質問: …\nA: 影武者\n…\nD: 乱\n答え:` → `D`.
  - Here: 5 options, `A. ` separator, the prompt ends with an instruction sentence, and the target is `\n答え: B（絵本）。` (letter plus choice text).
  - JSQuAD is open-book span extraction, not multiple choice.
- **T8** (4 fixed code + explanation prompts: python_derivative, python_projectile, cpp_search, rust_fibonacci): not trainable. There is no code and no explanation text.
- T1 and T2 do not apply.

## 10. Blocker list
1. **T5 eval sentences are in train and dev.** 68 JNLI rows (66 train, 2 dev) carry T5 sentences, either NFKC-equal (34 rows) or as a verbatim suffix of a longer JNLI sentence (34 rows). They affect 68 of 400 T5 items (17 of 100 groups). The builder's blocklist is one-directional and not NFKC-normalised. Drop the ids in `jg_t5_nfkc_drop_ids.txt` and rebuild the blocklist as in 5c.

## 11. Caveats (usable after the blocker is fixed)
- 9,766 rows contain characters outside the 3,498-char vocabulary (train 9,105, dev 359, heldout 302). Target-side OOV is in 3,756 rows. NFKC repairs only part of it (6,811 rows remain). Extend the vocabulary, normalise or filter.
- 19 JSQuAD prompts (38 train rows) carry two different answers (nested spans). Keep one per prompt. There are also 15 exact duplicate `full_text` groups in train and 1 in heldout.
- JSQuAD is 70.9% of train rows. Mixing weights need care against the T5/T7 targets.
- JNLI prior shift: train is 40/40/20 (neutral/contradiction/entailment) and heldout is 59/26/15. Report per-class accuracy, since the always-neutral baseline is 59.2%. Lexical overlap alone reaches 66.8%.
- JCQA heldout gold letters are skewed (p about 0.004, majority B 27.2%) and a mild inverted choice-prior exists (29.6%).
- JSQuAD dev (10 articles) and heldout (19 articles) are highly clustered. Treat their accuracy as noisy.
- A small amount of sentence-level overlap between heldout and train remains (JNLI 21/578 heldout rows, JCQA 6/250 questions).
- `loss_mask_spans` is empty on every row, so the trainer must mask the prompt.
- The "根拠" quote is easy to copy (lexical overlap picks it 85% of the time), so the evidence step teaches copying more than reasoning.
- Format mismatch with T5/T7. T6 and T8 are not covered.
- The CC BY-SA 4.0 ShareAlike notice must travel with the data.

## 12. Counts at a glance
- Rows: train 73,863 / dev 2,043 / heldout 2,457.
- Supervised chars (train): 4,652,214.
- Over-length (len+1 > 2048): 0. Answer mismatches against source: 0 of 78,363.
- sha256 and row counts: match the manifest in all three files.
- Not run: nothing from the checklist was skipped. The one difference is that "seeded random sample of at least 600 rows per task" was replaced by verifying every row.
