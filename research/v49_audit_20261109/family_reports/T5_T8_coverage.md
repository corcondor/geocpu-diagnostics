# GeoCPU T5-T8 dataset coverage: managed? grown to hundreds of thousands? engineered?

Auditor run 2026-11-09. Every number below was either recomputed by me from the files (marked **[mine]**), taken from the census/auditor/v48 documents and not re-run (marked **[claim]**), or measured by the auditors and reproduced by me (marked **[repro]**). Scripts and raw outputs: `.../scratchpad/audit_v49/t58/`. All data files were streamed line by line; no data file was modified.

## 0. Bottom line (for the owner)

Short answers:

1. **Managed?** Partly. v48 and three of the v49 families have a manifest plus sha256, and the census/auditors pinned row counts and hashes. python_a, search_use, option_judgement have no manifest (only generator logs). python_b has **no data file at all** (a generator with a capacity of 72,550 rows; only a 1,750-row sample was materialised for the audit).
2. **Grown to hundreds of thousands?** In raw rows, yes, but the growth is not in T5-T8 skill content.
   * The OLD corpus (`train_40500_exact.jsonl`) is already **423,348 rows** (40,500 is the checkpoint step, not the row count) **[claim census]**. For T5-T8 its unique, on-format content is small: T5 5,552 rows with only 6 distinct target texts; T6 18,090 distinct search sequences; T7 634 distinct knowledge-MC rows (15 questions x 1,000 copies + 619 legal); T8 zero C++/Rust code.
   * v48 added **4,096** rows (+5,936 replay copies of 848 rows), 100% self-authored.
   * v49 candidates: **378,295 train rows materialised** (6 families) **[mine]**, plus **70,000** python_b rows that exist only as a generator (448,295 if generated). Of the 378,295, **73,863 (19.5%) are human-written text** (JGLUE), **154,512 (40.8%) real Wikidata/JEMHopQA facts in self-written templates**, **149,920 (39.6%) pure self-authored synthetic**.
   * Rows that feed T5-T8 at all (non-exclusive primary mapping): T5 63,324, T6 51,804, T7 10,699 (closed-book academic knowledge: **0**), T8 65,920 (Python only; **C++/Rust: 0**). One pass over all 378,295 rows fills **34.4%** of a 20,000-update stage (55 rows/update, 1.1M row-presentations); T5+T6+T7+T8 together (191,747 rows, no double counting) fill **17.4%** of one stage on a single pass.
3. **Engineered?** Yes for correctness and leakage (independent solvers, 0 eval-shingle hits, sha256 match, loss masks on tool outputs), no for diversity (rows are re-parameterisations of a few hundred templates and about 5-10k real facts each) and no for shortcut removal (value priors, name-absence cues, T5 number-citation shortcut replaced by other shortcuts). One family (jglue_human) is **blocked** by T5 leakage that I reproduced independently.

日本語要約: 行数は数十万に達しているが、T5〜T8 に効く中身は薄い。T8 の C++/Rust は旧教材・v48・v49 のどれにも 0 行。T7 の閉じた知識（解剖・抽象代数）の教材も 0 行で、v48 の教材も検索コーパス（25 文書）もこの分野を含まない。T7 の 5% は、同じ 200 問を選択肢ロジットで測ると 19.0%（偏り除去後 27.0%）なので、生成採点の問題（出力が文字として取れていない可能性が高い）。ただし生の生成ログはディスクに無いので、原因は推定。

## 1. Old corpus, v48, v49: what exists (counts)

### 1.1 OLD corpus (census, `baseline_corpus_census.json`) **[claim]**

* 423,348 rows, 140,616,585 full_text chars, supervised chars (len(target)+1) 52,716,411; completion chars incl. EOS 71,070,889 of which 6,071,928 are loss-masked (tool outputs). 32,273 exact-duplicate rows (distinct full_text 391,075). Vocabulary 3,498 distinct chars, 5 chars outside the 3,898 bundle vocab (rows_with_oov 5,000, all in `v43_relational_causal_cot`).
* Code-fence labels in the whole corpus: `""` 185,345 (closing fences), `python` 108,634, `output` 76,347, `lean4` 150, `lean` 121, `text` 90. **No `cpp`, no `rust`, no `c`.**

### 1.2 v48 additive data **[mine]** (`v48_balanced_augmentation_20261108/data`)

* 4 files x 1,024 train rows (python, logic, evidence, japanese) = 4,096 rows, 1,079,318 full_text chars, 348,392 supervised chars (len(target)+1). Source on every row: `selfauthored:geocpu/v48-additive`, CC0-1.0.
* Per the v48 README/analysis **[claim]**: 4,096 template rows = 1.1897% of supervised tokens (2,079,114 of 174,759,265); replay x8 of the 848 practical rows adds 5,936 copies; the run was 21,000 updates, 55.0 rows/update measured, 1,155,008 row-presentations. Derived by me: 2,079,114 / 348,392 is about 6, so each v48 row was shown roughly 6 times (approximate; the trainer's token count differs from len(target)+1 by masked spans).
* Content: `python` = Python tool use ("道具: Python"); `logic` = the same numeric filter/sum problems **without** a tool; `evidence` = fictional documents ("研究2号12の登録値は44。"), no real-world fact; `japanese` = 4 numeric options ("選択肢: A=63, B=72, C=77, D=99。... 「偶数である」に該当するものを一つ選び"). v48 held-out result **[claim]**: new families 8-11%, the 4-option family at its 25% chance level.

### 1.3 v49 candidate families (train rows, streamed by me) **[mine]**

| family | train | dev | heldout | supervised chars (train) | content provenance | licence rows (train) | auditor verdict |
|---|---:|---:|---:|---:|---|---|---|
| logic_trace | 84,000 | 1,680 | 1,800 | 8,857,360 | 100% self-authored rule templates | CC0 84,000 | ok_with_caveats |
| python_a | 65,920 | 1,030 | 1,500 | 13,528,184 | 100% self-authored, code really executed | CC0 65,920 | ok_with_caveats (audit incomplete) |
| python_b | 70,000 capacity (**not on disk**; 1,000-row sample) | 1,200 cap. | 1,350 cap. | about 27.9M extrapolated | 100% self-authored, code really executed | CC0 | ok_with_caveats |
| search_use | 49,004 | 1,362 | 1,892 | 16,124,119 | real facts (Wikidata CC0, JEMHopQA) + self-written traces | CC0 42,907 / BY-SA 6,097 | ok_with_caveats |
| reading_evidence | 59,500 | 850 | 2,160 | 5,942,707 | real facts + self-written templates | CC0 42,700 / BY-SA 16,800 | ok_with_caveats |
| option_judgement | 46,008 | 1,917 | 2,700 | 5,137,820 | real facts + self-written templates | CC0 32,400 / BY-SA 13,608 | ok_with_caveats |
| jglue_human | 73,863 | 2,043 | 2,457 | 4,652,214 | 100% human text (JGLUE v1.3) | BY-SA 73,863 | **blocked** (T5 leakage) |
| **materialised total (6 families, without python_b)** | **378,295** | | | **54,242,404** | | BY-SA 110,368 (29.2%) | |

* Exact-format check **[mine]** over all six materialised families plus the python_b sample: rows whose prompt starts with `ユーザー:` = 0; contains `【本文】` = 0; starts with `分野:` = 0; ends with `正解:` = 0; ends with `思考の連鎖と回答:` = 0. **No v49 row has a T5, T7 or T8-original eval prompt format.** The T6 `質問: ...\n思考:` ending and `<SEARCH q>` protocol are present in search_use (45,412 train rows contain `<SEARCH`).
* Code fences **[mine]**: python_a and python_b sample contain only `python` and `output` fences; the other five families contain no fence. `C++`/`#include`/`std::` appears in 139 jglue train rows, all JSQuAD Wikipedia passages about C++ (e.g. "文章（C++）: 長年にわたる作業の後、ANSIとISOの合同委員会は..."), no code, no fence. Rust syntax: 0 rows.
* Distinctness **[mine]**: every full_text is distinct (jglue 73,848 of 73,863). Digit-masked prompts: logic_trace 35,038 of 84,000, python_a 33,525 of 65,920, search_use 48,834 of 49,004 (names vary). Distinct answer strings: option_judgement **3** (A/B/C), search_use 4,084, reading_evidence 7,770, logic_trace 9,585, python_a 14,781, jglue 37,482.
* Real-fact breadth **[mine, partly repro]**: option_judgement 175,880 premise occurrences = **5,223 distinct premise sentences** (exactly the auditor's 5,223); reading_evidence 327,753 document-line occurrences = 7,186 distinct lines ending in 。 (auditor: 9,922; I did not reproduce the larger figure); search_use loose regex: 6,274 distinct entities, 8,751 distinct entity|relation|value facts (auditor: 6,633 subjects, ~5.1k searchable facts). So 49-60k rows per family sit on roughly 5-10k facts each.

## 2. T5 (logical operators / JNLI-grounded, 400 items)

**(1) Exact eval format.** 100 contexts x 4 operators (file name and docstring say 6-operator; the real file has 4) **[mine]**. Prompt (quoted from row `logic_op_eval_2776_positive_entailment`):

```
タスク: 自然言語指示の論理演算子変換・3値論理推論 (JNLI Grounded)
【本文】
（1）テーブルの上に花が活けられています。 （2）パーマをした女性がこちらを向いています。 （3）部屋の手前に女性が二人立っています。
【指示】
次のA〜Cのうち、前提から論理的に【正しい（確実に導ける）】ものを選べ。
【選択肢】
A. （3）に関して、リビングでは、二人の女性が立食しています。
B. （1）に関して、...
C. （2）に関して、...
思考の連鎖と回答:
```

Target `B` (a bare letter; targets A 135 / B 126 / C 139). Scorer `extract_choice_abc`: `正解[:：]\s*([ABC])`, else last standalone A/B/C in the last 40 chars, else last A/B/C character; generation `max_tokens=None`. The OLD-corpus response is a ~421-char chain ending `正解: C` (sections 【1. 指示の論理演算子への変換】, 【2. 各選択肢の論理状態検証】 "- C: 本文から導出可能で真 (P |- Q)", 【3. ALU論理実行】 `<SET R0 1> => [R0=1] ...`, "求める論理条件 ... に合致するのは C である。\n正解: C").
**Shortcut (verified by me over all 400 items):** the gold option cites premise (1) for positive_entailment 100/100, (2) for negation_contradiction 100/100, (3) for non_entailment_undeducible 100/100 and (3) for not_necessarily_true 100/100. A number rule scores 100%. T5 therefore measures reliance on the "（k）に関して" index, not reading (v48 analysis says the same for 5,552/5,552 training rows **[claim]**).
Scores: 61,500 = 39.50%, 30,500 = 54.5%, 40,500 never measured **[claim]**; the Kaggle T5 rerun (cap 192 chars) is invalid (400/400 byte-identical outputs, 100% hit the cap) **[claim]**.

**(2) OLD corpus rows.** Exact-format JNLI-operator rows: **5,552** (1.31% of rows; supervised chars 2,343,851 = 4.45% of the formula-A total, summed by me from census group figures); six groups of 925-926 rows; 3,702 rows for the 4 eval operators, 1,850 for 2 extra operators; 2,776 distinct JNLI contexts; **6 distinct target texts**. Adjacent same-skill other-format rows: 45,643 (10.78%): Yahoo_JAPAN_JNLI_Full 19,992 (3 distinct targets 含意/矛盾/中立), FOLIO 1,204, LogiQA2 12,047 (answer letter A in 12,047/12,047), bridge_py logic generators 12,400 (Python tool use, not T5 format).

**(3) v48 added.** Nothing in T5 format. `logic` and `japanese` use "条件と根拠を順に考えてください。" and "答え: X。". No JNLI text, no 【本文】, no operator instruction.

**(4) v49 candidates feeding T5 (train rows, provenance, verdict).**

| family | rows | what | human vs self-authored | verdict |
|---|---:|---|---|---|
| option_judgement | 46,008 | 3-option "【指示】前提から論理的に導けるものはどれですか" true/false/unknown selection over Wikidata/JEM facts, 6 operators | facts real (5,223 distinct sentences), wording and options self-authored; 0 human prose | ok_with_caveats; labels 100% correct (50,625 rows); "unknown" = name absent from premise (rule near 100% where unique) |
| jglue_human JNLI | 12,916 (6,470 label form + 6,446 letter form) | premise/hypothesis → 含意/矛盾/中立 | 100% human (JGLUE, CC BY-SA 4.0) | **blocked**: 66 train + 2 dev rows contain a T5 sentence (see below) |
| reading_evidence verify | 2,800 (reading_verify_claim 2,100 + jem 700) | 正しい/誤り/わからない against a document | facts real, templates self-authored | ok_with_caveats; exact-match claims, not paraphrase NLI |
| logic_trace (weak) | 1,600 (syl_three_valued 800 + trans_cmp_3valued 800) | symbolic 3-valued syllogism with invented 〇〇族 terms | 100% self-authored | ok_with_caveats |
| **T5 total** | **63,324** | human 12,916 (20.4%); real-fact templated 48,808; synthetic 1,600 | | |

**Independent leakage replication [mine].** I NFKC-normalised the 588 distinct T5 sentences (≥10 chars, from 【本文】 and from the option text after "に関して、") and compared them to the 前提/仮説 sentences of every JNLI row. Train: 32 exact-sentence matches + 34 rows whose sentence ends with a T5 sentence = **66 rows**; dev 2 exact; heldout 0. This reproduces the auditor's 68 (66 train + 2 dev). The jglue verdict "blocked" stands until those ids (`jg_t5_nfkc_drop_ids.txt`) are dropped and the blocklist is made NFKC and two-way.

**(5) Format mismatches (quotes).**
* eval: `タスク: 自然言語指示の論理演算子変換・3値論理推論 (JNLI Grounded)\n【本文】...【選択肢】\nA. （3）に関して、...\n思考の連鎖と回答:` → `B`
* option_judgement: `問題: 次の【前提】だけを根拠にして、【指示】に合う選択肢を一つ選んでください。\n【前提】\n（1）ハリー・キューウェルの出生地はシドニー。...\n【指示】前提から論理的に導けるものはどれですか。一つ選んでください。\n【選択肢】\nA. リライアンス・ペトロリアムとハリー・キューウェルとは別に、...\n選択肢ごとに根拠を書いてから答えてください。` → `\n根拠:\nA: (2)と一致=真 → 真\nB: (5)はロアンヌ=偽 → 偽\nC: 前提にコラスFCなし=不明 → 不明\n選ぶのは真のもの → A\n答え: A。` (【前提】 not 【本文】; Wikidata name-value facts, not captions; target "答え: A。" not `正解: A`/bare letter; 24 instruction sentences, none equal to T5's.)
* JNLI row: `問題: 前提の文と仮説の文を読み、2つの関係を判断してください。\n前提: 茅葺き屋根の古民家の軒先に犬が1頭と...\n仮説: ...\n正しい関係の記号を答えてください。\nA. どちらとも言えない\nB. 前提と仮説は両立しない\nC. 前提から仮説が言える\n条件と根拠を順に考えてください。` → `\n答え: B。` (one hypothesis, no instruction switch, no per-context 3-option set.)
* logic_trace: `問題: 次の前提から結論の正誤を判断してください。...前提1: あるナス族はチツ族である。...結論は「正しい」「誤り」「どちらとも言えない」のどれですか。` → `\n導出: ...\n→ 正しい\n答え: 正しい。`

**(6) Honest gaps.** (a) No v49 family contains a T5-format row (0 rows). (b) The T5 number shortcut was removed from option_judgement (pointer template 8.45% of train, 0% of heldout) but replaced by an entity-absence shortcut (auditors: a learned surface model reaches 0.79 dev / 0.50 heldout vs 0.333 chance). (c) JNLI-style human NLI is only 12,916 rows and 66 of them leak. (d) The scorer, with `max_tokens=None` and a 421-char CoT, makes T5 sensitive to output format changes (the "正解:" vs "答え:" competition, v48 analysis section 5.4 **[claim]**). (e) No row teaches the 4-instruction switch on one context (auditors: no same-context contrast groups).

## 3. T6 (search-augmented retrieval, 200 rows = 21 distinct questions)

**(1) Exact eval format.** `eval_v42_search_retrieval_aligned.jsonl` (200 rows, 100 "1_hop" + 100 "2_hop") and `_heldout.jsonl` (same 200 questions, prompt additionally primed with "思考: 外部検索エンジンを使用して必要な事実を確認します。\n") **[mine]**. 21 distinct prompts: 17 one-hop (8 Python stdlib e.g. `質問: ...正確な整数平方根を...計算する標準関数は何か？\n思考:` → `math.isqrt`, expected query `math isqrt`; 5 law articles → `民法第709条`; 4 science terms → `RuBisCO`, `ヘスの法則`, `光電効果`, `ミトコンドリア`) and 4 two-hop (Nokia→country→`ヘルシンキ`, Volvo→`ストックホルム`, 大政奉還→era `慶応`, Fleming→`イギリス`). Evaluator rebuilds the prompt to end in `\n思考:`, model emits `<SEARCH q>`, engine injects ` => [RESULT (src: local, date: 常駐): TITLE | TEXT] `; correct iff `target in out_text` (the injected result is part of out_text, so a correct search can score without a written answer). Independent samples: 21, not 200.
The search backend in the A/B and T6 evaluation is `LocalSearchEngineV42`; its default corpus is `knowledge_corpus_v42.jsonl` = **25 documents** (the v43 variant has **15**) **[mine]**. Scores: v42 100%, 61,500 = 0.00%; Kaggle (cap 256): searched 55/200 (30,500) and 50/200 (61,500) row-weighted, answer written 5/200 vs 0/200 **[claim]**.

**(2) OLD corpus.** Rows containing `<SEARCH`: **35,378** (8.36%), of which only **18,090 are distinct** (17,288 are re-splits of 11,460 sequences); groups: Wikidata_CC0_search_depth1 5,920 / depth2 5,738 / depth3 5,978 / post_result 17,288 / search_evidence_sufficiency 454. All Wikidata-CC0 templated relations (本社所在地, 出生地, 所在国, 首都, 生年月日), `src: local` only; no-tool depth0 adjacent 6,384; `prompt_asks_search_rows` 341 **[claim]**.

**(3) v48 added.** None. The `evidence` family is open-book over fictional documents with no search call.

**(4) v49 candidates feeding T6.**

| family | rows | detail | provenance | verdict |
|---|---:|---|---|---|
| search_use | 49,004 | 45,412 with ≥1 `<SEARCH` (n_search 1: 17,553; 2: 17,399; 3: 10,460; 0: 3,592); heldout 1,892 rows with 2-4 searches on a disjoint fact world | facts real (Wikidata CC0 87.6%, JEMHopQA CC BY-SA 12.4%); all traces/answers self-authored; 0 human prose | ok_with_caveats; 52,258/52,258 answers independently consistent; 0 eval shingle hits |
| reading_evidence two_hop + three_hop | 2,800 | open-book (資料 in prompt), "探す: ... 根拠: ..." | facts real, templates self-authored | ok_with_caveats |
| **T6 total** | **51,804** | real-fact templated 51,804; human prose 0 | | |

**(5) Format mismatches.**
* eval prompt `質問: 通信機器大手ノキア（Nokia）が本社を置く国の首都はどこか？\n思考:` (expected queries "ノキア 本社", "フィンランド 首都"; target `ヘルシンキ`).
* search_use row: `質問: 五井孝蔵が生まれた都市の所在国を答えてください。\n思考:` → ` 必要な事実を調べます。<SEARCH 五井孝蔵 出生地> => [RESULT (src: local, date: 常駐): 五井孝蔵 出生地 | 五井孝蔵の出生地は秋田県。]  照合: ...\n資料: 「...」\n次に秋田県の所在国を調べます。<SEARCH 秋田県 所在国> => ...\n答え: 日本。`.
* Protocol (prompt ending, `<SEARCH q>` syntax, injected string, <64-char commands, 2-3 hop chains) matches. Mismatches: queries are always "ENTITY RELATION" over Wikidata persons/orgs/cities, whereas T6 queries are keyword-style ("math isqrt", "憲法 25条", "RuBisCO 光合成"); all train snippets have one shape "XのRはV。" whereas T6 snippets look like "math.isqrt(n) returns the exact integer square root ..."; answer line "答え: 日本。" vs bare target; the heldout priming sentence "外部検索エンジンを使用して..." never occurs in train (the evaluator strips it).

**(6) Honest gaps.** 17 of 21 T6 questions (Python stdlib, law, science) are outside the v49 domain (Wikidata persons/orgs/cities); only the 4 two-hop items resemble 本社所在地→所在国→首都 chains. The eval's own search corpus has 25 documents, so T6 measures "emit a query and copy a snippet from a tiny index", not retrieval from the real world. search_use rests on about 5.1k searchable facts (auditor) and has unremoved value-prior shortcuts (MCQ city options 66-71% from option frequency alone; 37% of not-found rows decided by 12 absent-attribute words). T6 sample size is 21, so any difference is not testable.

## 4. T7 (JMMLU-style multiple choice, closed vs open book, 200 items)

**(1) Exact eval format and scorer.** Items = the first 200 multiple-choice rows of `benchmark_8100_manifest.jsonl` by the registry rule `tier7_first_mc` **[mine]**: **99 abstract_algebra + 101 anatomy** (not "231 anatomy+abstract_algebra"; the manifest holds 99 + 132, only the first 101 anatomy rows are in). Gold letters A 38 / B 52 / C 66 / D 44. Model input is `prompt.strip() + "\n思考: "`, i.e.

```
分野: abstract_algebra (math)
問題: S_5 において p = (1, 2, 5, 4)(2, 3) とする。S_5における<p>の指数を求めよ。
A: 8
B: 2
C: 24
D: 120
正解:
思考: 
```

Closed arm: `max_tokens=60`, `enable_search=False`; open arm: `max_tokens=120`, `enable_search=True`. Scorer `extract_answer_choice` (source `eval_v43_ab_benchmark.py`): regex `(正解|回答|答え|選ぶのは|合致するのは|結論として)\s*[:：は]?\s*([A-D])\b`; else the first standalone `[A-D]` followed by `.`/`)`/end/space; else the **first of "A","B","C","D" that occurs anywhere in the text (so any "A", e.g. in "ATP", wins)**; else `text.strip()[:1]` (a non-letter, always wrong). Target is the bare letter.
Reported scores: closed 5.00% (10/200, Wilson 95% 2.7-9.0%), open 6.50% (13/200) **[claim]**.

**Is 5% < chance a scorer artefact? (answer, with evidence level).** Chance is 25% for a uniformly random letter; the best constant letter (C) gets 33.0% and constant "A" gets 19.0% on these 200. 10/200 is far below every one of those (binomial P(X<=10) = 4.6e-14 at p=0.25 and 7.2e-9 at p=0.19, computed by me), so the model's generations cannot be "a letter chosen at random": they are mostly **not parseable as a letter** (fallback returns a non-letter, counted wrong) or systematically wrong. Independent evidence that the underlying letter preference is at chance **[mine, from the Kaggle teacher-forced logits `mcq.json`, exact same 200 ids, 61,500 checkpoint]**: rotation 0 (original option order) 38/200 = 19.0% (A picked 188 times, B 12; equals the share of gold A), rotations 1/2/3: 27.5% / 31.0% / 21.5%, debiased 54/200 = **27.0%** (Wilson 21.3-33.5%); the 30,500 checkpoint gives 18.5% / 29.0%. Over all 7,534 JMMLU items the debiased accuracy is 24.9% (23.9-25.9%), A picked 80.9% **[claim, Kaggle report]**. So "knowledge" is at chance and heavily position-biased; the 5% is mostly a format/extraction failure of the generation arm. **Not verified**: the raw T7 generations are not on disk (I searched the scratchpad for tier_7 outputs and found none), so I cannot show what the 200 texts look like.

**(2) OLD corpus.** Rows in exact JMMLU format (分野 header or ending `正解:`): **0**. Knowledge MC: 15,619 rows but only **634 distinct** (`v43_relational_causal_cot` 15,000 rows = 15 questions x 1,000 copies, answer letters A 6,000 / B 4,000 / C 3,000 / D 2,000 only because of the repetition; `Government_Legal_Full` 619). Adjacent: SoftBank_JCommonsenseQA_Full 8,876 (answer is option text), LogiQA2 12,047 (always A) **[claim]**.

**(3) v48 added.** `japanese` 1,024 rows: 4 numeric options in a "A=63, B=72, ..." layout with "答え: B。"; this teaches the 4-letter format on arithmetic properties, not knowledge. No academic knowledge.

**(4) v49 candidates feeding T7.**

| family | rows | what | provenance |
|---|---:|---|---|
| search_use search_mcq4 + search_mcq4_jem | 4,320 | `選択肢: A=スーザ, B=グラスゴー, ...。\n一つ選び、最後に記号で答えてください。\n思考:` → retrieve → `答え: B。` (entity birth/HQ facts) | real facts, templated; counted under T6 above |
| reading_evidence reading_lookup_multiple_choice | 2,100 | open-book MC over documents in the prompt | real facts, templated |
| jglue_human JCQA | 8,599 | 5-option commonsense, e.g. `質問: 建物をたてれる地面のことをなんという？\nA. 土地 ... E. 農地` → `答え: A（土地）。` | 100% human (JGLUE) |
| **academic closed-book knowledge (anatomy, abstract algebra, any JMMLU subject)** | **0** | | |

T7-only partial total excluding search_use's 4,320 (already in T6): 10,699 (human 8,599 + templated facts 2,100).

**(5) Format mismatches.** eval `分野: abstract_algebra (math)\n問題: ...\nA: 8\nB: 2\nC: 24\nD: 120\n正解:` → `C` (JamC-QA: `質問: ...\nA: ...\n答え:` → `D`). v49: inline options `A=..., B=...。`, five options `A. ` in JCQA, trigger `思考:`/"条件と根拠を順に考えてください。" instead of `正解:`, target `答え: B。` instead of a bare letter.

**(6) Honest gaps.** (a) No closed-book academic knowledge exists in any corpus (old, v48, v49): the 634 distinct knowledge-MC rows are 15 history questions + 619 legal. (b) The open-book arm retrieves from a 25-document index (15 in the v43 variant) with 0 documents on anatomy or abstract algebra (regex over all 25 + 15 docs: 0 hits), so "open book" cannot help on these 200 items by construction. (c) The scorer's fallback chain makes the generation arm unreliable; a logit/letter-probability scorer on the same 200 items gives 19.0% (27.0% debiased). (d) JMMLU's abstract_algebra and anatomy are translations of MMLU test subjects (JMMLU README: translated from MMLU's 57 subjects) **[web, verified]**, so Japanese MMLU translations (Global-MMLU, MMLU-ProX and similar) would **contaminate T7** and must not be used as training data.

## 5. T8 (code + explanation, 4 prompts, scored for form only)

**(1) Exact eval format.** `tier8_readonly_v44fix.py` **[mine]**: 4 prompts x 2 profiles = 8 generations. `original_v44`: `ユーザー: 関数 f(x) = 3x^3 + 2sin(x) - 5 について、導関数 f'(x) を求め、x = 0 における微分係数 f'(0) を計算してください。\nアシスタント: ` (python_derivative); `ユーザー: 初速 20 m/s、仰角 45° で斜方投射された物体の滞空時間と水平到達距離を求めてください（重力加速度 g = 9.8 m/s^2、空気抵抗は無視）。\nアシスタント: ` (python_projectile); `ユーザー: C++ を用いて、効率的な『二分探索 (Binary Search)』の完全な実装コードを作成し、その計算量と動作原理を解説してください。\nアシスタント: ` (cpp_search); `ユーザー: Rust を用いて、効率的な『フィボナッチ数列のメモ化DP』の完全な実装コードを作成し、その計算量と動作原理を解説してください。\nアシスタント: ` (rust_fibonacci). `aligned_v43`: `問題: ...\n思考:` for Python, `指示: ...\n回答:` for C++/Rust. Scoring is form only: `has_code` = "```" or "def " or "int " or "fn " in the text; `has_narrative` = any of 起承転結求 in the text; `expected_language_fence` = a fence labelled python/py, cpp/c++/cxx or rust/rs; `has_prose_outside_code` = 4+ kana/kanji outside fences; python invocation counts. With n = 4 per profile every rate moves in 25-point steps; no answer is executed or checked (C++/Rust are never run). Status of 61,500: per the task, the T8 log has only the START line; I found no T8 output file in the scratchpad, so I report **no T8 result**.

**(2) OLD corpus.** C++ fenced code rows **0**; Rust fenced code rows **0**; rows with C++-like syntax anywhere 9 (incidental text: python-docs-ja 5, cpython 2, tldr 1, ichikara 1); rows mentioning Rust in a short prompt 236 (tldr rustup/cargo command lines, 2 Dolly); rows whose prompt starts `ユーザー:` **0**; code-writing request in the prompt 146 (mostly JSQuAD passages, none an "implement X and explain" instruction); Python-fence + output + prose tool-use rows 62,805 (14.84%; NuminaMath-TIR 10,204, gen_prog 14,000, gen_word 12,000, gen_expr 8,000, ...); code-only continuation (no instruction, no explanation) python_cad 1,115 + python_doc_code 1,060 + others **[claim]**.

**(3) v48 added.** `python` 1,024 rows (Python tool use, short plan + code + output + "答え: 256。実行結果を確認しました。"), 0 non-Python fences **[mine]**.

**(4) v49 candidates feeding T8.**

| family | rows | provenance | note |
|---|---:|---|---|
| python_a | 65,920 | 100% self-authored, executed for real | fences `python`+`output` only; `道具: Python` line; one-line plan/check; 0 C++/Rust, 0 complexity/principle explanation |
| python_b | 70,000 (generator only; 1,000-row sample read) | 100% self-authored, executed for real | 100 task types x 700; 30% are repair rows with a deliberately wrong first program in the target; auditor: 0 cases mention 導関数/斜方投射/C++/Rust |
| **T8 total** | **65,920 materialised (135,920 if python_b is generated)** | human 0 | **C++ 0, Rust 0** |

**(5) Format mismatches.** `ユーザー: ...\nアシスタント: ` appears in 0 training rows of any corpus (old census, v48, v49 families: all `問題: ...\n思考:` or `質問:`/`指示:`). v49 Python row: `問題: [63, 9, 88, ...]はお店の日ごとの来客数を並べたものです。 全体のうち45未満であるものを集めて、最大値を計算してください。\n道具: Python\n思考:` → `\n計画: 条件に合う要素を選んで最大値を計算する。\n```python\nL=[...]\nb=[x for x in L if x<45]\nprint(max(b))\n```\n```output\n41\n```\n確認: ...\n答え: 41。`. The T8 aligned Python prompt lacks the `道具: Python` line, and the C++/Rust prompts ask for a complete implementation plus complexity and principle explanation, which no training row contains.

**(6) Honest gaps.** (a) **There is no C++ or Rust code (fenced or otherwise) in the old corpus, v48 or any v49 family** (old census 0/0; my scan: python/output fences only). (b) No calculus (導関数) or projectile/trig tasks in python_a/b (python_b auditor: 0 cases; python_a families are list/text/number-theory/sim/dict/calendar/matrix/set). (c) No row teaches "implementation + complexity + principle" prose. (d) T8 is 4 prompts with a presence-of-form score, so even a perfect result would not show programming ability; the C++/Rust fence rate is the only measurable thing and the data to move it does not exist yet.

## 6. Totals and "is it hundreds of thousands?"

Stage budget: 55.0 rows/update measured in v48 **[claim]** x 20,000 updates = **1,100,000 row-presentations**; supervised about 8,322 tokens/update x 20,000 = 166.4M **[claim, derived]**.

| quantity | train rows | share of one 20k-update stage (one pass, rows) |
|---|---:|---:|
| v49 materialised, 6 families | **378,295** | **34.4%** |
| + python_b generator (70,000, not yet written) | 448,295 | 40.8% |
| T5 primary (OJ 46,008 + JNLI 12,916 + verify 2,800 + weak logic 1,600) | 63,324 | 5.8% |
| T6 primary (search_use 49,004 + two/three-hop reading 2,800) | 51,804 | 4.7% |
| T7 partial (JCQA 8,599 + MC reading 2,100); closed-book academic = 0 | 10,699 (+4,320 inside T6) | 1.0% |
| T8 Python-only (python_a 65,920) | 65,920 (135,920 with python_b) | 6.0% (12.4%) |
| T5+T6+T7+T8 together | 191,747 | 17.4% |
| not mapped to T5-T8 (logic_trace rest 82,400, JSQuAD 52,348, reading_evidence rest 51,800) | 186,548 | 17.0% |

(The partition sums to 378,295 rows: 191,747 mapped to T5-T8 plus 186,548 unmapped. T7 excludes the 4,320 search MC rows already inside T6.) By supervised chars, the six materialised families sum to 54.24M chars (32.6% of a 166.4M stage; 49.3% with python_b's extrapolated 27.9M). That is the same size as the entire old corpus (52.7M supervised chars by formula A). Provenance split of the 378,295: human text 73,863 (19.5%); real facts + templated wording 154,512 (40.8%); pure synthetic 149,920 (39.6%). Share-alike (CC BY-SA 4.0) rows: 110,368 (29.2%).
Context: old corpus 423,348 rows; v48 added 4,096 (1.19% of supervised tokens) and was shown 21,000 updates. So "hundreds of thousands of rows" is true in the row count of v49 (378k-448k), but (i) T5-T8-specific on-format rows are 63k/52k/11k/66k, (ii) real, human-written content for T5-T8 is 12,916 (T5) + 8,599 (T7-adjacent) rows and 0 for T6/T8, (iii) distinct facts per family are 5-10k and answer-string vocabularies are tiny (option_judgement: 3 distinct answers), and (iv) C++/Rust and closed-book academic knowledge have no data at all.

## 7. Candidate real, openly licensed Japanese datasets for each gap (not self-authored; no downloads performed)

Licence evidence level: **V** = I read it on the dataset's own page this session; **M** = mirror/secondary source; **U** = not verified by me. Sizes are as stated on the pages.

**T5 gap (human/real NLI without the number shortcut; JNLI is the T5 source, so leak-check everything with NFKC):**
1. JSNLI (Kyoto Univ.; SNLI machine-translated, train auto-filtered 533,005 pairs, eval 3,916 crowd-checked); CC BY-SA 4.0 (**M**: HF loader script `_LICENSE`; official page not fetched). https://nlp.ist.i.kyoto-u.ac.jp/ (JSNLI page), mirror https://huggingface.co/datasets/shunk031/jsnli . Caveat: machine-translated captions (Flickr30k-derived), different images from the STAIR/COCO captions behind JNLI.
2. JGLUE JNLI (Yahoo Japan; CC BY-SA 4.0; already used as `jglue_human`; 12,916 train rows in v49): reuse only after dropping the 66+2 T5-overlap ids. https://github.com/yahoojapan/JGLUE (licence string is on every v49 row; page not re-fetched).
3. JSICK (human-translated SICK, NLI + STS): README body says CC BY 4.0 while the repo sidebar says CC-BY-SA-4.0 (**V, conflicting; treat as BY-SA**); size not on the page. https://github.com/verypluming/JSICK
4. JaNLI (adversarial, 2 labels entailment/non-entailment): README text "CC BY 4.0" with a by-sa/4.0 link, sidebar CC-BY-SA-4.0 (**V, conflicting**); example counts not on the page. https://github.com/verypluming/JaNLI
5. JFLD (Hitachi, LREC-COLING 2024; HF `hitachi-nlp/JFLD`: 4 subsets, 30k train / 5k val / 5k test each; programmatically generated formal-logic deduction): licence not found on the official card (**U**; a third-party JINIAC mirror says apache-2.0, **M**). https://huggingface.co/datasets/hitachi-nlp/JFLD . Lowest rank because licence unclear and the text is synthetic (though not authored by us).

**T6 gap (real queries/documents beyond Wikidata persons/orgs; a real index instead of a 25-doc corpus):**
1. Japanese Wikipedia, `wikimedia/wikipedia` config `20231101.ja` (1.39M articles; cc-by-sa-3.0 + GFDL, **V**). https://huggingface.co/datasets/wikimedia/wikipedia . As the retrieval index and snippet source.
2. MIRACL ja (human queries with relevance judgments over Wikipedia; apache-2.0 **V**; ja: 3,477 train queries / 34,387 judgments, 860 dev queries **V**). https://huggingface.co/datasets/miracl/miracl (check the corpus's own Wikipedia licence).
3. JAQKET / AI王 (quiz questions with Wikipedia-title answers): CC BY-SA 4.0 (**M**: derivative dataset README; official terms not retrieved); derivative `hotchpotch/jaqket_v1_qa_wikija_context`: 2,939 train / 980 validation (**V** on that card). https://sites.google.com/view/project-aio/dataset . Note a derivative README reports test items mixed into train in some copies: check splits.
4. JaQuAD (Skelter Labs; 39,696 QA pairs: 31,748 train / 3,939 validation / 4,009 test not public; cc-by-sa-3.0 **V**). https://huggingface.co/datasets/SkelterLabsInc/JaQuAD
5. TyDi QA (apache-2.0 on the card **V**; Japanese rows exist in `primary_task`; Japanese counts not on the page **U**). https://huggingface.co/datasets/google-research-datasets/tydiqa
6. JEMHopQA (CC BY-SA 4.0; already used for facts in search_use/reading_evidence; licence from the family provenance fields, page not re-fetched).

**T7 gap (closed-book academic knowledge: anatomy, abstract algebra, other JMMLU subjects):**
There is no openly licensed Japanese multiple-choice set that is both topical and not derived from MMLU/JMMLU. Ranked options:
1. Japanese Wikipedia (as above; cc-by-sa-3.0 + GFDL, **V**) for anatomy and algebra articles, with retrieval rather than memorisation (a real index would also make the open-book arm meaningful).
2. JaQuAD / JAQKET / TyDi as above for Wikipedia-grounded questions (general knowledge, not these two subjects).
3. **Do not use**: Japanese translations of MMLU (Global-MMLU, MMLU-ProX, etc.): JMMLU's abstract_algebra and anatomy are translated from MMLU's 57 subjects (**V**, JMMLU README: "translated from MMLU ... machine-translated and reviewed"), so these would leak T7. JMMLU itself is the eval (CC BY-SA 4.0 for 53 tasks / 7,097 questions, CC BY-NC-ND 4.0 for 3 tasks / 439 questions, **V**). https://github.com/nlp-waseda/JMMLU
4. Wikibooks JA / Wikiversity JA textbooks (CC BY-SA family; not fetched this session, **U**).

**T8 gap (C++/Rust implementation + Japanese explanation):**
1. The Rust Programming Language, Japanese translation `rust-lang-ja/book-ja`: Apache-2.0 and MIT (**V**, repo shows LICENSE-APACHE and LICENSE-MIT). https://github.com/rust-lang-ja/book-ja . Rust code with Japanese prose (content not displayed on the fetched page).
2. Rust by Example, Japanese `rust-lang-ja/rust-by-example-ja`: Apache-2.0 and MIT (**V**). https://github.com/rust-lang-ja/rust-by-example-ja
3. `p1atdev/ja-stackoverflow` (Japanese Stack Overflow Q&A, code blocks fenced; 30,551 question rows; cc-by-sa-4.0 on the card **V**; per-post original licences vary, older posts CC BY-SA 3.0 **M**; languages seen: Ruby, Go, Python, Swift, Objective-C, C, Android; C++/Rust presence not confirmed **U**). https://huggingface.co/datasets/p1atdev/ja-stackoverflow
4. cppreference (English prose): CC BY-SA 3.0 (**V**, licence page https://en.cppreference.com/w/Cppreference:Copyright/CC-BY-SA); Japanese edition not verified (my licence URL for ja.cppreference.com returned 404, **U**); would need Japanese explanations, so lowest value.
5. TheAlgorithms/Rust: MIT (**V**); contains Rust implementations (binary search / Fibonacci not confirmed on the fetched page, **U**); English comments only. https://github.com/TheAlgorithms/Rust . TheAlgorithms/C-Plus-Plus: fetch failed (504), licence **U**.

## 8. What I did not run or could not verify

* No re-verification of answer correctness in any v49 family (I rely on the auditors' independent solvers; python_a's audit is incomplete per its own caveats: exact stdout equality, independent solvers and the shingle leakage scan were not finished).
* python_b: no data on disk, so all python_b row counts are generator capacity; fence/provenance statements are from the 1,750-row sample.
* T7 and T8 raw generations and the T8 log for the 61,500 checkpoint are not on disk; the T7 artefact diagnosis is an inference from the scorer source plus teacher-forced logits.
* T5/T6/T7/T8 scores for 30,500, 40,500 and 61,500 are taken from the task text and the v48 analysis; I did not re-measure.
* My own regex fact counts for reading_evidence (7,186) do not reproduce the auditor's 9,922; the search_use loose regex gives 8,751 facts vs the auditor's ~5.1k searchable/6,442 keys (definitions differ).
* Licence statements marked M or U are not verified against the official page; do not rely on them without checking before any data is used. No dataset was downloaded.

Scripts and outputs: `.../audit_v49/t58/` (`scan_family.py`, `scan_*.json`, `scan_summary.txt`, `t7_logit_on_200.py/.txt`, `t5_leak_check.py/.json`, `t5_numberrule.txt`, `diversity*.py/.json`, `kc_peek.txt`, `peek_evals.txt`, `v48_scan_summary.txt`).
