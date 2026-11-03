# v43と純Python v44 Hotfixの生成経路

読取専用のコード照合と保存済み回答の確認。モデル読込、生成、学習、既存評価の再実行はしていない。ここでいう「純v44」はALUを外したPython TIR Hotfixであり、Drive上の別枝 `v44fix` ではない。

| 段階 | v43 | 純v44 Hotfix | 今回の判断 |
|---|---|---|---|
| 入力の文字化 | [v43](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v43_source_readonly_pure_audit_20261103/integrated_model_v30.py:661)は記号を正規化し、語彙外字を除いてID化（同681行） | [v44](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/tir_engine_v44r1.py:97)は入力を変更せず、語彙外字でエラー | 保存4問はエラーなく生成されたため、語彙外エラーは当該4問の崩れの説明ではない。記号正規化の実寄与は未測定。 |
| 質問の末尾 | v43のDolly元学習例は `指示: …\n文脈: …\n回答:` の次に半角スペースを学習 | [教材作成](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/prepare_public_v44.py:233)は公開3群を `ユーザー: …\nアシスタント: ` へ変更（同244、251行） | 保存4問のprompt長は144、117、38、47字で、すべて新しい末尾。形式の寄与は高優先の比較対象だが、この監査だけでは因果量を確定できない。 |
| prefillと最初の文字 | v43は[bf16 autocast](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v43_source_readonly_pure_audit_20261103/integrated_model_v30.py:698)で `prefill`、最終logitsをfloatのCPU上でargmax（同704、720行） | v44は[autocastなし](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/tir_engine_v44r1.py:118)で `prefill`、最終logitsをGPU上でargmax（同121–133行） | 両者とも最終語彙logitsから選ぶ。実重みでの最初のlogit差と選択差は未測定。 |
| 状態更新とEOS | 通常の選択文字を1回 `step_state` に渡す。EOSは状態へ渡さず終了（[v43](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v43_source_readonly_pure_audit_20261103/integrated_model_v30.py:720)、同785行） | 通常文字を1回渡す。EOSも `advance` してから終了（[v44](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/tir_engine_v44r1.py:178)、同182–184行） | EOSを先に更新する差は終了以前の文字選択を説明しない。12位置のランダム小模型による全文・逐次比較は最大3.58e-7・argmax差0だが、実重み/長文/GPUは対象外。 |
| 外部実行と結果 | `<...>`をCPU回路で実行し ` => [R..] ` を注入（[v43](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v43_source_readonly_pure_audit_20261103/integrated_model_v30.py:729)） | 行頭の ` ```python` を実行し ` ```output` を注入（[v44](/C:/Users/81808/.openclaw/reports/jmmlu_v1/ag_fused_fullqa_training_v1/v44_hotfix_source_readonly_20261103/tir_engine_v44r1.py:150)、同187–216行） | [保存済み前後8回答](<G:/マイドライブ/GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103/run_1/training/before_public_answers.json>)のtool呼出しは全0。旧ALUが使えなくなったことは確定した設計変更だが、Pythonが呼ばれなかった直接原因は未確定。 |
| 反復停止 | v43には同じ8または16 token列の4連続を検出する停止条件（同776–782行） | 同じ条件はない | [保存token列への旧条件再適用](/C:/Users/81808/.openclaw/workspace/geocpu-diagnostics/reports/v44fix/v43_guard_replay_saved_pure_hotfix.json)では8/8回答で発火0。旧停止処理の復元だけでは、この保存済み反復を短縮しない。 |

## 追加資料との照合

[Claudeの診断メモ](<G:/マイドライブ/GeoCPU_v44fix_20261103/v44_diagnosis_20261103.md>)に対応する実コード、結果JSON、exit 0 のログを、Driveとは別のClaude一時作業領域で確認して同一バイトで保全した。詳細は[原票追補](2026-11-03-claude-v43-v44-evidence-addendum.md)。v43重み・v43生成関数を固定した6件の旧風/新形式比較で、自然EOSは3/6対1/6、CPU呼出しは2/6対0/6だった。また、同じv43実装・同じheld-out SWE先頭12問を解いた保存メトリクスは、v43 step15,000が12/12・CPU12/12、純v44 step25,000が0/12・CPU0/12だった。前者の回答は先頭400字、後者のSWE回答は先頭160字で切り詰められており、全文原回答は未確認。checkpointのSHAは保存側の原票値とファイルパスを照合し、巨大実体の再ハッシュはしていない。隣接する `run_1/gate_baseline_v43.json` と `gate_16000.json` は別の **v44fix混合型** の200問結果であり、この比較には使っていない。

短文4問が学習前から崩れたこと自体は純v44の[保存前回答](<G:/マイドライブ/GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103/run_1/training/before_public_answers.json>)で確認できる。ただし、その回答は別Release版からコピーされ、学習後はHotfix版で生成されている。重み更新・入力形式・生成コードの寄与は、同じ重みと同じ生成器で一条件ずつ比べない限り分離できない。
