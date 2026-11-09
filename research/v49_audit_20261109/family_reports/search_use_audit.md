# Independent audit: family `search_use` (GeoCPU v49 corpus)

Auditor run: 2026-11-09. Verdict: **ok_with_caveats** (no blocker found; five caveats below, two of them material for how the data is used).

Family dir: `.../wt_v48/research/v49_data_20261108/families/search_use` (data in `data/`). Manifest/claims file: none, so the only claims are the generator logs (`gen_*.log`, `test_full.log`) and the docstrings in `baselines.py` / `gen_search.py`. Everything below was recomputed from the three JSONL files with my own scripts (no import of `solvers_search.py`, `fam_*.py`, `core.py`, `baselines.py`). All scripts are in `.../scratchpad/audit_v49/su/` and were run one at a time, streaming line by line.

## 0. Headline numbers

| item | train | dev | heldout |
|---|---|---|---|
| rows | 49,004 | 1,362 | 1,892 |
| structural templates / families | 29 / 57 | 29 / 57 | 12 / 12 |
| bytes | 139,758,291 | 3,850,391 | 7,656,272 |
| mean / p50 / p99 / max len(full_text) | 387.7 / 377 / 709 / 965 | 384.7 / 374 / 704 / 816 | 591.4 / 565 / 929 / 1,149 |
| rows with len+1 > 2048 | 0 | 0 | 0 |
| search calls (sum n_search) | 83,731 | 2,318 | 5,117 |
| rows with n_search = 0 | 3,592 (7.3%) | 96 | 0 |
| OOV rows (vs old 40,500 vocab) | 10 | 14 | 0 |
| answers checked by my verifier | 49,004 | 1,362 | 1,892 |

Supervised characters (train): `sum(len(target)+1)` = **16,124,119**. After removing `loss_mask_spans` (masked observations) the loss-bearing part is 9,988,650 chars (61.9%). Total `full_text` chars in train: 19,000,129.

Scale note for the question "did you grow it to hundreds of thousands?": this family has 49,004 train rows, but they are built from only **6,633 distinct subject entities** (mean 7.9 rows per subject, 91.8% of rows belong to a subject with >= 5 rows) and **5,097 distinct retrievable facts** (plus 4,389 fact keys that appear as in-prompt documents; 6,442 distinct valid `(entity|relation)` keys overall). Row count is therefore not a measure of knowledge diversity (see caveat C1).

## 1. Inventory (checklist 1)

Commands: `su_a_core.py` (stream, `hashlib.sha256`, count `\n`).

| split | newline count = rows | sha256 | claim (gen_*.log) |
|---|---|---|---|
| train | 49,004 | `4227d7cc8456449d75a7e889abe22835ce37e48d5b175d57812928acd84fa0d9` | identical |
| dev | 1,362 | `a7447b09e5e01b1835d5112d550e79c6f207176d96fe21422cdd5b76829f13ac` | identical |
| heldout | 1,892 | `e243e833c767a56bac8b08f2299e6b5959ff64513bdffa97a1fac6957cddbe90` | identical |

The generator's own full test run (`data/test_full.log`) ends `Ran 25 tests ... FAILED (failures=1)`; the failing test is `test_no_overlap_between_splits` (train subjects intersect dev subjects). I reproduced the underlying fact independently (section 5, 49 shared subjects).

## 2. Schema (checklist 2)

All 10 required fields present in every row of every split (`id, prompt, target, full_text, split, family, source, license, tool_steps, loss_mask_spans`); extra fields `structural_template, difficulty, reference_final_answer, world, subject_entities, n_search, training_category` plus template-specific flags (`claim_true`, `masked_first_query`, `docs_given`, ...).

Violation counts (all zero): JSON errors 0; blank lines 0; missing fields 0; `full_text != prompt + target` 0; duplicate ids within a split 0; duplicate ids across splits 0; `split` field != file 0; empty prompt 0; empty target 0; target not ending with `答え: ...。` 0; more than one `答え:` per target 0; `n_search != len(tool_steps)` 0; number of `<SEARCH q>` in target != tool_steps (or query text differs) 0; `<SEARCH` inside a prompt 0; observation not of the exact form ` => [RESULT (src: local, date: 常駐): ... ] ` 0; stray `<` or `>` characters (count of `<` == n_search and count of `>` == 2*n_search; checked on all train rows) 0; train prompts not ending with `\n思考:` 0; any `<SEARCH q>` longer than 64 chars (the T6 engine drops commands longer than 64 chars) 0 in all splits (longest 57 train, 41 dev, 55 heldout).
`reference_final_answer` equals the parsed answer line in 52,258/52,258 rows; `claim_true` consistent with the answer in 8,136/8,136 claim rows; heldout `count` field consistent in 160/160.

Note: `world` is `"train"` for all 1,362 dev rows (dev is generated from the train-world engine, not from a separate world); `world` equals the split for train and heldout.

## 3. Vocabulary (checklist 3)

Vocabulary: `audit_v49/vocab_chars.json`, 3,498 characters, union of `train_40500_exact.jsonl full_text`.
* train: 10 rows (28 characters), only `％` (U+FF05), in `search_jl_mcq` (5), `search_jl_other_free` (3), `search_jl_verify_yn` (2). 6 rows OOV on the prompt side, 7 on the target side.
* dev: 14 rows (98 characters), only `將` (U+5C07), in `search_jl_other_free` (7), `search_jl_mcq` (5), `search_jl_verify_yn` (2).
* heldout: 0.
The T6 engine silently drops OOV characters (`input_ids = [vocab[c] for c in prompt if c in vocab]`), so these 24 rows would be tokenised with a missing character; trivial to drop or replace.

## 4. Duplicates and template shares (checklist 4)

* exact duplicate `full_text` within train/dev/heldout: 0; across splits: 0 (train/dev, train/heldout, dev/heldout).
* distinct prompts in train: 49,004 (= rows). Same prompt with different target: 0.
* exact prompt overlap train/dev, train/heldout, dev/heldout: 0.
* template shares in train: largest is `search_post_string_op` 4,500 rows = 9.2%; next `search_verify_yn` 4,500 (9.2%), `search_mcq4` 3,600 (7.3%). Largest family 900-ish rows (1.84%). **No template above 15%.** Heldout: 12 templates of 132-160 rows (largest 8.5%).
* Heldout templates and families are disjoint from train templates by design (12 compositional/aggregate operations: count, compare dates, min of three, difference, odd one out, ordering, same city/country, which-of-two (+missing), year difference, larger number).
* Answer "見つかりませんでした" is the overall majority answer in train: 7,240 rows = 14.8% (templates `search_not_found_*` 6,813 + 427 `search_docs_only` rows).

## 5. Leakage (checklist 5)

### 5a. Versus the eval manifests (method as specified)
Script `su_c_leak.py`. Eval text = every string leaf of the 8 manifests (benchmark_8100 56,453 strings, jamcqa 23,090, swe_step 1,033, v33 state 1,316, v38 2,260, v41 logic 3,600, T6 aligned 1,100, T6 heldout 1,100) plus the 5 prompt strings of `tier8_readonly_v44fix.py` (found by `ast`, no execution). 3,054,439 eval characters -> **167,278 distinct hashed 24-char shingles at stride 8**. Slid stride 1 over `full_text` (prompt+target) of all 52,258 audited rows.
* rows containing >= 1 eval shingle: **0** (train 0, dev 0, heldout 0), so no GENERIC shingles (> 500 rows) and no content overlap either. Positive control: the same code on a fake row with an embedded T6 question detects it (`positive_control_detects_embedded_eval_text: true`).
* exact-sentence overlap (sentences >= 12 chars split at 。/newline; 48,000 eval sentences): **0** shared sentences.
* T6 exact checks: 21 distinct T6 questions, 25 distinct expected queries, 21 distinct targets. Rows containing a T6 question verbatim: **0**. Rows containing `<SEARCH q>` for any of the 25 expected T6 queries: **0**. Rows whose answer line equals a T6 target: only the target `イギリス` (train 632 rows, dev 20), a generic country answer, not an item leak. T6 keyword presence in train: `ボルボ` 9 rows and `慶応` 13 rows (incidental), none of `ノキア, フレミング, ペニシリン, 大政奉還, math.isqrt, ヘルシンキ, ストックホルム, フィンランド, スウェーデン, 民法第709条, 憲法第25条, RuBisCO, ...`. (Heldout world contains Nordic cities: `フィンランド` 152 rows, `スウェーデン` 217, `ヘルシンキ` 2, `ストックホルム` 2; this is not train and not T6.)
Conclusion: no eval item leaks into train/dev/heldout.

### 5b. Train versus heldout (must be disjoint)
* subjects: train/heldout 0 shared, dev/heldout 0 shared. Heldout subject strings that occur anywhere as text in train: 17 of 951 (1.8%), touching 269 of 1,892 rows (14.2%); all are JEM date/number style entities (e.g. 仲代達矢, 黒澤明, 宮脇咲良) that occur incidentally as values in `search_jl_*` rows. Fact level: **0 of 4,371 heldout head facts and 0 of 640 non-head facts** (entity|relation|value) appear in train (neither as retrieved observation nor as in-prompt document), and 0 of 5,011 heldout `資料: 「...」` sentences appear in train. Heldout is a separate fact world (different cities/countries) plus unseen operations.
* A raw 24-char shingle comparison heldout->train gives 1,129 "content" shingles (<= 500 train rows) in 1,337/1,892 heldout rows, but inspection shows they are lead-sentence + first four characters of a name (e.g. `資料を引いて確かめます。<SEARCH ロビー・`), i.e. boilerplate coincidences, not shared facts; I do not use this as a leak metric.

### 5c. Train versus dev (not required to be disjoint, but the generator claims it)
* dev/train shared subjects: **49** of 403 dev subjects (the generator's own failing test lists e.g. コヨアカン, サカテカス, 金堤市, 城南市). Dev rows with a train subject: 133/1,362 (9.8%).
* the real contamination is larger because train prompts of the docs families (`dc_*`, `ps_*`) list facts about other entities as in-prompt documents: 892 of 1,166 dev head facts (76.5%) occur in train (805 only via train prompt documents, 87 via train search results); 582/582 dev non-head hops (city->country, country->capital) occur in train; 796/1,362 dev rows (58.4%) have **all** their head facts already shown in train. Dev is therefore an in-distribution, in-KB split and will overestimate generalisation to new entities (caveat C3).

## 6. Independent answer verification (checklist 6)

Script `su_b_verify.py` (own parser, own question-cue chain derivation, own replay simulation, nothing imported from the family). Coverage is **every row of every split** (52,258 rows, all 29 train templates and all 12 heldout templates), which is a superset of the requested >= 400-row seeded sample.

What is checked per row:
* observation grammar: ` => [RESULT (src: local, date: 常駐): TITLE | ENT の REL は VAL。] ` parsed; a hop is "valid" only if title == query and text restates `ENTのRELはVAL。`.
* relation chain derived from the question wording (birth / 本社 / 国 / 首都 cues after removing the entity name; family name for claim families; relations present in the question for the JEMHopQA `jl_*` families). The simulator walks the chain using valid search steps (or in-prompt documents for the docs families) and requires that every valid step is consumed (no unused valid evidence), that wrong/mismatched hops are flagged as non-valid, and that "見つかりませんでした" is the expected answer exactly when the chain cannot be resolved.
* template operations recomputed: claim compare (はい/いいえ, 正しい / 誤り、正しくは...), MCQ letter by option text, multipart `(1) ... (2) ...`, string operations (length, k-th, first/last n chars, contains), date part/decade/plus-minus years, docs-only / docs+search decision, reverse lookup, and all 12 heldout operations (count, earlier date, min of three, numeric difference with unit, larger number, odd one out, ordering, same city / same country, which-of-two with missing, year difference).
* every intermediate line (`su_c8_cot.py`): 75,270 `資料: 「...」` copy lines are retrieved texts or cited documents; 3,577 document citation lines equal the cited prompt document; 8,136 `比較:` lines (word vs values, retrieved value); 4,440 `選択肢と照らすと` lines (letter matches option); 1,169 arithmetic lines (Decimal re-computation); 4,620 name+`文字:` listings; 1,110 n-th character lines; 1,107 character counts; 924 decade lines; 2,114 year lines; 160 `数える:` mark counts; 159 year comparisons; 320 sorted lists. **0 inconsistencies** in any category.
* `照合:` lines (91,166): `と一致。` for every valid step, 不一致/ではありません for every distractor, 一致する資料はありません for every not-found: 0 bad.

Result: **52,258 / 52,258 rows consistent** (49,002 train rows + all 1,362 dev + all 1,892 heldout automatically; 2 train rows `v49search_train_jl_date_part_203/331` have a half-width `(昭和24年)` era annotation my date parser did not handle, and I checked them by hand: `1949年(昭和24年)9月21日` -> `9月` is correct). **Mismatches: 0.** (My first runs reported mismatches/unverifiable rows, all traced to bugs in my own parser: greedy quotes in claim sentences, ambiguous claim patterns, `(1)`-style dates; they are fixed in the shipped script and I list this so the history is transparent.)

Positive control (`su_b2_mutation.py`): replacing the final answer by a different answer of the same template in 3,082 sampled rows (6% of train + 30% of heldout) is flagged as a mismatch **3,082/3,082** times, so the verifier is not vacuous.

Consistency of the knowledge base across rows: 6,442 valid `(entity|relation)` facts with **0** conflicting values; 18,475 distinct queries with **0** queries mapped to two different observations; the 3 relation-absent / unknown-entity not-found families contain **0** contradictions (no "unknown" entity or "absent" relation ever appears as a valid fact elsewhere).

Replay vs the shipped engine (`su_c4_engine.py`, my own re-implementation of `LocalSearchEngineV42._search_local` scoring: tokens from `re.sub(r"[^\w\s぀-ゟ゠-ヿ一-鿿]"," ",q)`, +5 per token in title, +1 in text, snippet <= 130 chars): over the 6,449 documents observed in the data, all 15,122 queries that returned a document returned a maximum-score document; 3,351 not-found queries have zero score on all observed documents except 2 (`ギルバート+トビン`, `ギルバート+トビン 年収`) whose scoring document (`マツ・ギルバート 出生地`) exists only in the heldout world, so this is a world-separation artefact, not an error. Longest snippet 61 chars (engine limit 130), no `...` truncation.

## 7. Loss mask (the task asks whether `loss_mask_spans` mask the injected observations)

`su_c5_mask.py`: for every row, the span list equals exactly `[(call_start if step.query_masked else obs_start, obs_end) for each step]` where `obs` is the injected ` => [RESULT ...] ` text: **52,258/52,258 rows** (train 49,004, dev 1,362, heldout 1,892). In 5,158 steps (train 5,020 rows of the `rf_*` families, dev 138) `query_masked` is true and the span starts at the `<SEARCH` of a deliberately wrong query (entity-with-suffix, relation-only, wrong hop), so the model is not trained to emit the wrong query; all 5,158 of those steps return a mismatching or not-found result. The call `<SEARCH q>` of ordinary steps, the lead sentence, `照合:`, `資料:` and `答え:` lines are supervised; the observation text is never supervised. Spans are sorted, non-overlapping and lie inside the target region in every row.

Composition of the loss-bearing (unmasked) characters in train: `照合:` lines 36.4%, lead sentence + `<SEARCH q>` 29.0%, `資料:` copy lines 18.0%, other reasoning lines 8.8%, `答え:` line 5.3%, newlines 2.0%, EOS 0.5%. Over half of the loss is the deterministic restatement of what was just retrieved.

## 8. Shortcut and bias audit (checklist 7)

Majority-answer baselines (train): overall majority answer ("見つかりませんでした") 14.8%; template-conditional majority 29.4%; family-conditional majority 31.6%; 4,084 distinct answers. Dev: 15.0% / 32.7% / 38.1%.
Answer distributions: MCQ letters in train A 910+166, B 922+211, C 884+173, D 884+170 (per-family shares 22-27%; best always-one-letter accuracy 26.2%); yes/no: verify_yn 2,281 はい / 2,219 いいえ (50.7%), jl verify 360/360, claim correction 1,347 正しい / 1,353 誤り; heldout same_city 76 はい / 84 いいえ, same_country 79/81. Not-found is 14.8% of train rows.
Answer-in-prompt (raw substring test, `su_b_verify.py`): 0 rows for every lookup / chain / not-found / date / reformulation / reverse-lookup template. Non-zero only where the answer is a quoted label (`正しい` in claim correction 1,347/1,347), a character or count of a quoted string (string ops 713/892), text of provided documents (docs families, 900/900 by design), or in `search_post_string_op` 1,744/4,500 rows (the 900 `はい` answers are quoted in the prompt, the rest are single characters that happen to occur in the question).

Heldout choose-one baselines (own computation): first / last / longest / shortest entity are at chance in every family (e.g. earlier-of-two 0.512 / 0.487 / 0.494 / 0.506 vs 0.5; min-of-three 0.356 / 0.325 / 0.338 / 0.325 vs 0.333; odd-one-out 0.338 / 0.344 / 0.312 / 0.331 vs 0.333). No order or length shortcut.

Shortcuts that DO exist and are not covered by the generator's own tests (leave-one-out, fitted on train answers only):
* **MCQ option prior** (pick the option that is frequently a correct answer): `mc_birth_city` 70.6%, `mc_hq_city` 66.3% (chance 25%); `jl_mcq` 38.6%; `mc_birth_country` 31.2%; `mc_hq_capital` 31.3%. Overall 48.0% LOO on train, 42.9% on dev (n=120). Distractor cities are not drawn from the answer distribution.
* **Claim-value prior** in claim/verify families (predict yes/no from the claimed value alone): `cc_hq_city` 73.4%, `cc_birth_city` 72.6%, `vf_birth_city` 71.8%, `vf_hq_city` 64.1% (chance ~50%); country/capital claims 53-61% (only 13 values); JEM verify 44.6% (no frequency signal, but its false claims are type-implausible, e.g. 「コロンビア大学のノーベル賞受賞者は2歳です」).
* **Not-found surface cue**: 2,700 of the 7,240 not-found rows (37%) are questions about one of 12 closed-set absent attributes (体重, 初恋の相手, 好きな色, 年収, 座右の銘, 従業員の平均年齢, 愛犬の名前, 最寄りの郵便局, 血圧, 趣味, 電話番号, 靴のサイズ), and none of those words occurs in an answerable row (precision 100%); the other 4,540 not-found rows need real evidence (unknown entity, missing hop, docs lacking the fact).
* **Search/no-search decision** is determined by prompt surface (quoted string or `資料N:` block means no search, except `dc_need_it` where entity absence from the docs decides).
* Heldout entity prior (LOO within heldout, informational only; a model trained on train has never seen these entities): earlier-of-two 81.2%, min-of-three 90.6%, larger-number 100%, odd-one-out 83.9%, which-born-in-city 99.1%, because each heldout entity has one fixed value and recurs. This only matters if heldout entities are ever used for training or few-shot priming.

## 9. Provenance and licence (checklist 8)

* `source` is the same string on every row: `selfgen:geocpu/v49-search-use (corpus: Wikidata CC0 truthy + JEMHopQA corpus_ver1.2 triples, local search replay)`. **Real/human content = the facts** (Wikidata truthy triples, JEMHopQA triples); **100% of questions, search traces, reasoning lines and answers are self-authored templates** (about 10 paraphrases per relation, 5 lead sentences, 4-12 closed-set absent attributes). 0 rows of human-written natural-language text.
* licence field: `CC0-1.0` 42,907 train rows (87.6%, 49 families) and `CC-BY-SA-4.0` 6,097 train rows (12.4%, the 8 JEMHopQA-derived `search_jl_*` families); dev 1,170 / 192, heldout 960 / 932 (heldout is ~49% CC-BY-SA). The share-alike/attribution obligation of CC-BY-SA-4.0 has to be carried to any mix that contains these rows.

## 10. Format alignment with T5-T8 (checklist 9)

Quotes are exact. Train rows (search needed):
`質問: 五井孝蔵が生まれた都市の所在国を答えてください。\n思考:` -> target ` 必要な事実を調べます。<SEARCH 五井孝蔵 出生地> => [RESULT (src: local, date: 常駐): 五井孝蔵 出生地 | 五井孝蔵の出生地は秋田県。]  照合: 見出し「五井孝蔵 出生地」は「五井孝蔵 出生地」と一致。\n資料: 「五井孝蔵の出生地は秋田県。」\n次に秋田県の所在国を調べます。<SEARCH 秋田県 所在国> => [RESULT ...]  ...\n答え: 日本。`

* **T6 (search-augmented multi-hop retrieval): trainable, protocol matches.** The T6 evaluator (`eval_v42_search_retrieval.py`) rebuilds every prompt to end with `"\n思考:"` (both manifests), the model must emit `<SEARCH query>`, the engine (`integrated_model_v42.py`) injects `f" => {res_snippet} "` where the snippet is `[RESULT (src: local, date: 常駐): {title} | {snippet}]`, and the item is correct if `target in out_text`. Train rows use the identical prompt ending, the identical `<SEARCH q>` syntax (`SEARCH_PATTERN` accepted; no stray angle brackets; all commands < 64 chars), the identical injected string (` => [RESULT (src: local, date: 常駐): TITLE | TEXT] ` incl. the leading and trailing space), and multi-hop chains of 2 and 3 searches (`search_chain2/3`, `mc_hq_capital`, `ps_*`). Differences: (1) T6 aligned/heldout prompts contain Python/law/science/Nokia-Volvo-style questions with queries such as `math isqrt`, `憲法 25条`, `ノキア 本社`, `フィンランド 首都`; train queries are always `ENTITY RELATION` with a small relation vocabulary (出生地, 本社所在地, 所在国, 首都 plus JEM relations), and every train question is about a Wikidata person/organisation/city. (2) T6 snippets look like `math.isqrt(n) returns the exact integer square root ...` or `ノキア 本社: 所在地・関連先は フィンランド です。`; train snippets are all `XのRはV。` (the `照合` copy loop is learned on one sentence shape). (3) The T6 heldout manifest primes the prompt with ` 外部検索エンジンを使用して必要な事実を確認します。\n` (a newline before `<SEARCH`); train never has `。\n<SEARCH` (0 rows) and never contains `外部検索エンジン` (0 rows). The evaluator strips that priming (`prompt.split("\n思考:")[0] + "\n思考:"`), so this only matters for other harnesses. (4) T6 scoring is `target in out_text` over text that includes the injected observation, so a correct `<SEARCH>` can score without any answer being written; the train data teaches both the call and the answer.
* **T8 (4 fixed code+explanation prompts): not trainable by this family.** T8 prompts are `ユーザー: C++ を用いて、効率的な『二分探索 (Binary Search)』の完全な実装コードを作成し、その計算量と動作原理を解説してください。\nアシスタント: ` (and Python/Rust analogues); train rows use `質問: ...\n思考:` with `答え: X。` terminals and contain no code fences. 0 overlap with the 5 T8 prompt strings.
* **T7 (JMMLU-style multiple choice): partially related.** `search_mcq4` and `search_mcq4_jem` (4,320 train rows) teach choosing a letter after retrieving a fact. Format differs: train `選択肢: A=スーザ, B=グラスゴー, C=アフマダーバード, D=キングストン。\n一つ選び、最後に記号で答えてください。\n思考:` -> reasoning with search -> `答え: B。`; T7 JMMLU `分野: abstract_algebra (math)\n問題: ...\nA: 8\nB: 2\nC: 24\nD: 120\n正解:` -> target `C`, and JamC-QA `質問: ...\nA: ...\nB: ...\nC: ...\nD: ...\n答え:` -> `D`. Options are inline `A=..., B=...` in train versus one per line `A: ...` in eval, the trigger is `思考:` versus `正解:` / `答え:`, the train content is entity facts (birth places, HQ cities, capitals) whereas JMMLU/JamC-QA is academic/culture knowledge; the train letter answer follows a retrieval chain, T7 expects a direct letter. Also note MCQ option-prior shortcut (section 8).
* **T5 (6-operator logical contrast, 3 options, JNLI-grounded): not trainable by this family.** T5 prompts are `タスク: 自然言語指示の論理演算子変換・3値論理推論 (JNLI Grounded)\n【本文】...【選択肢】\nA. ...\nB. ...\nC. ...\n思考の連鎖と回答:` -> `B`; nothing in `search_use` is a logical-operator task or uses `【本文】/【指示】`.

## 11. Verdict and caveats

**ok_with_caveats.** Blockers checked and absent: wrong answers (0 of 52,258), eval-item leakage into train/dev/heldout (0), train-heldout fact leakage (0), conflicting duplicates (0), exact duplicates (0), over-length rows (0; max 1,149), broken/changed files (sha256 equal to the generator logs), mask not covering observations (0 failures).

Caveats (ranked):
1. **C1 Low diversity behind a large row count.** 49,004 train rows come from 6,633 subjects / ~5.1k searchable facts; the same subject recurs ~8 times with different templates and also as in-prompt documents. A small model can memorise `entity -> value` instead of copying from the observation (the `資料:` copy lines and the answer are then predictable without reading). Heldout (new fact world, new operations) is the real test of reading; growing to "tens of thousands more" rows from the same KB will not add diversity, new facts/entities will.
2. **C2 Unremoved value-prior shortcuts** (section 8): city-valued MCQ 66-71% from option frequency alone, claim families 64-73% from the claimed value alone, 37% of not-found rows decided by a 12-word surface cue. The generator's tests only check letter balance, longest/shortest option and yes-share, so these are not claimed to be removed, but they weaken the training signal. Fix: draw MCQ distractors and false claims from the empirical answer distribution; mix absent attributes into answerable templates.
3. **C3 Dev is not held out** (generator's own test fails): 49 shared subjects, 58% of dev rows have all head facts already in train (mostly via train prompt documents). Use heldout, not dev, for generalisation claims.
4. **C4 Format gap to T6/T7/T8** (section 10): protocol identical for T6, but queries are all `ENTITY RELATION` over Wikidata entities, snippets are one template, and T6 items (Python stdlib, law, science, 2-hop Nokia/Volvo) are out of the train domain; T7 MCQ layout differs; T5 and T8 are not covered.
5. **C5 Minor:** 24 rows contain a character outside the 3,498-char vocabulary (10 train `％`, 14 dev `將`); 12.4% of train rows (JEMHopQA, `search_jl_*`) are CC-BY-SA-4.0; `jl_verify_yn` false claims are type-implausible; 36% of loss characters are the `照合:` restatement and 18% `資料:` copies.

## 12. Scripts and outputs (all in `.../scratchpad/audit_v49/su/`)

`su_a_core.py` (inventory, schema, OOV, duplicates, mask, syntax) -> `su_a_out.json`; `su_b_verify.py` (answer verification, KB consistency, shortcuts) -> `su_b_out.json`; `su_b2_mutation.py` (positive control) -> `su_b2_out.json`; `su_c_leak.py` (eval shingles, T6 exact) -> `su_c1_out.json`; `su_c2_split.py` (train vs dev/heldout, composition, MCQ prior, not-found cue, format counts) -> `su_c2_out.json`; `su_c3_exposure.py` (fact exposure) -> `su_c3_out.json`; `su_c4_engine.py` (replay vs engine scoring) -> `su_c4_out.json`; `su_c5_mask.py` (mask semantics) -> `su_c5_out.json`; `su_c6_baselines.py`, `su_c7_entprior.py` (baselines); `su_c8_cot.py` (intermediate lines) -> `su_c8_out.json`. Not run: nothing from the checklist was skipped; the only approximations are (i) the engine check uses the corpus of documents observed in the data, not the generator's full corpus (not shipped), and (ii) the MCQ/claim priors are simple frequency baselines, not a trained model.
