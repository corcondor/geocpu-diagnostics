# Claude の v43 / 純 v44 比較の原票追補

この追補は既存コード・保存 JSON・実行ログの読取専用監査である。モデル実行、再評価、学習、保存重みの変更は行っていない。元ファイルが一時領域にあったため、原物 6 点を `claude_v43_v44_readonly_evidence/` に同一バイトで保全した。

使用したClaude一時領域の `v43/` にある `integrated_model_v42.py`、`GeoCPU_CED_v1.py`、`GeoCPU_CED_v2.py`、`local_search_engine_v42.py`、`base_metadata.json` は、Drive の `GeoCPU_Frontier_CoTraining_v43/Grand_cotraining_v43.zip` 内の同名5ファイルと個別SHA256がすべて一致した。この照合はコードと語彙の同一性を裏づける。checkpointの実体は以下のとおり再ハッシュ対象外とした。

Claude の実tool記録では、同一の `scratchpad/v44diag` を作業ディレクトリとして、先に `PYTHONIOENCODING=utf-8 python run_v43_baseline.py`、次に `PYTHONIOENCODING=utf-8 python compare_indomain.py` が実行された。各コマンドに対応する保全ログ末尾はともに exit 0 である。

## 重みを固定し、入力形式だけを変えた検査

[`run_v43_baseline.py`](claude_v43_v44_readonly_evidence/run_v43_baseline.py) SHA256 `159f81fe659b2f770c4f1861904cdb29d720a9fb561c3e5fbf5c8d18c13c5c2e` の 21、53–70、75–96 行は、v43 `checkpoint_step_15000.pt` を v43 `IntegratedModelV42` に読み、同一の生成関数 `generate_with_qwenputer_v42` で形式だけを変える。読み込み結果は内部 step 15000、missing/unexpected とも 0。6 件は純 v44 公開例 4 件、算数 1 件、歴史 1 件。旧風の `質問: …\n思考: ` または `問題: …\n思考: ` と、新しい `ユーザー: …\nアシスタント: ` を各々反復ペナルティ 1.0、最大 300 token で比較している。別途、旧風形式のペナルティ 1.12 も実行した。旧風形式は全 4 公開例に対する元 v43 教材の厳密な入力形式とは限らない。

[`v43_baseline_result.json`](claude_v43_v44_readonly_evidence/v43_baseline_result.json) SHA256 `ce075a38c983c93a09269d73948f252e1457fc3514831c578356bb6b4cfa6235` と実行ログ [`bh2o8ven1.output`](claude_v43_v44_readonly_evidence/bh2o8ven1.output) SHA256 `8de1e0976cee1c3b6c0eb9628804f9a7f6f6a1d061233f82a6389cc8c3606503` によると、ペナルティ 1.0 の 6 件で、旧風形式は自然 EOS 3 件・CPU 呼出し 2 件・300 token 到達 3 件、新形式はそれぞれ 1・0・5 件。したがって入力形式は実出力に影響した。ただし、この 6 件だけから日本語性能低下の全原因を特定できない。JSON の `text` はソース 96 行で先頭 400 文字へ切り詰められ、全回答原文ではない。

## v43 実装と入力を固定し、重みだけを替えた検査

[`compare_indomain.py`](claude_v43_v44_readonly_evidence/compare_indomain.py) SHA256 `7939a2ccb5bcc1063d3eb2cc37a4ea304997bd17c660e1000998098d6b88aaac` の 16–19 行は次の 2 ファイルを指定する。

- v43: `G:/マイドライブ/GeoCPU_Frontier_CoTraining_v43/checkpoints/checkpoint_step_15000.pt`。保存側 `strict_neural_migration.json` に記録された親 SHA256 は `a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5`。この監査では 990,545,719 byte の実ファイルを再ハッシュしていない。
- 純 v44 Hotfix: `G:/マイドライブ/GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103/run_1/training/checkpoint_step_25000_aa6c44ab47a14297b5c6481a5fb1374d_27867b30ef624c8985a22b81ae7789fa.pt`。保存側 `result.json` に記録された SHA256 は `28308d0445829b8bf137f801f98b7573f80e699fb00e97d56a15523158ee984a`。この監査では 2,971,632,900 byte の実ファイルを再ハッシュしていない。

同スクリプト 33–59 行は v43 `IntegratedModelV42` 1 個へ順に重みをロードする。v43 は missing/unexpected 0、純 v44 は missing 30（すべて固定 ALU/RAM buffer）、unexpected 0。純 v44 の固定回路は初期値と一致した記録がある。46、63–71 行は v43 由来の held-out SWE 先頭 12 問を、各重みで **同一 prompt、同一 v43 生成関数** `generate_with_qwenputer`、最大 400 token、反復ペナルティ 1.0 で解く。採点は CPU trace に R0 があれば最後の R0、なければ出力から命令・結果表示を除いた最後の語を、保存 target と文字列比較する。74–78 行の算数 4 問は別の v43 `generate_with_qwenputer_v42`、最大 300 token、最終 `答え:` 数字で採点する。

[`compare_indomain_result.json`](claude_v43_v44_readonly_evidence/compare_indomain_result.json) SHA256 `6553f37af38f93886dcd91b09b8fcc4791403f9c9fb6234e57b4712f4d958851` と完了ログ [`bp7d7dsxz.output`](claude_v43_v44_readonly_evidence/bp7d7dsxz.output) SHA256 `d463bee1fd43a3f137fa147129332ca1f341759071393c953f58ce0831b88498` には exit 0 と以下の保存値がある。

| 重み | SWE 先頭 12 問 | SWE で ALU 呼出し | SWE 自然 EOS | 算数 4 問 |
|---|---:|---:|---:|---:|
| v43 step 15000 | 12/12 | 12/12 | 12/12 | 2/4、ALU 4/4 |
| 純 v44 step 25000 | 0/12 | 0/12 | 0/12 | 0/4、ALU 2/4 |

これは **保存された採点基準上、同じ旧実装・同じ 12 問で 12/12→0/12** という狭い範囲の実測である。新しい Python TIR 実装での総合性能や 8,100 問の低下をこの数字へ読み替えない。重み更新後の旧 ALU 利用が崩れたことは、純 v44 の生成コードだけでは説明できない。どの教材・損失・学習率・最適化変更が寄与したかはこの比較では分離していない。

保存コードは SWE 原回答を先頭 160 文字、算数を先頭 200 文字へ切り詰める（71、78 行）。完了ログも各回答の先頭 90 文字だけを表示する。この 6 原物と隣接する Drive 診断フォルダでは、SWE 12 問の全回答・全 CPU trace を保存した別ファイルを確認できなかった。したがって **12 問の保存メトリクスと先頭原文は確認済み、全文原回答の監査は未完** とする。
