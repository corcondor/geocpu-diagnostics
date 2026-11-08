# 元v43 SWE200 原回答取得用の隔離パッケージ

このパッケージは**準備のみ**です。Kaggleのversion作成、GPU起動、provider claim、元重み更新は行っていません。

`staged/bootstrap.py` と `staged/kernel-metadata.json` が、新規private Kaggle scriptに渡す2ファイルです。`private_kaggle_source_package.zip` は同内容と照合票をまとめた保管用ZIPです。既存のprivate入力データセット `geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103` を読みます。入力から使うのは元v43の `v43_source.zip`（または同byteの `.zip.bin`）と `v43_step15000.pt` だけです。v44fix重みは読みません。

起動前の固定条件は、元v43 ZIP 67,976,709 B / SHA256 `0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198`、step15,000重み 990,545,719 B / SHA256 `a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5`、ZIP内metadata `7efa8dea9a0100e2250f72805c89cca7a281efec30b4479795743814756cdcd5`、SWE200 `0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd`、元採点器 `c3589b97f531b6460a95907e56284df5f20be990eff1a4d2e56a921b4c705899` です。起動時にも全入力を再照合します。

期限は**2026-11-03 18:00 UTC**、生成/採点の壁時計上限は30分、保存確認の余裕は2分です。期限まで15分を切る場合は起動前に拒否します。終了できなければ途中原物を未完として残し、合格・自然EOS・旧技能保持に補完しません。期限はこのパッケージ内で延長しません。

成功時は `/kaggle/working/geocpu_v43_native_swe200_once_20261103/actual/` に、200問の全文原回答、全CPU trace、元採点器のauditとgrade、正答かつ自然EOSの旧技能subset、`COMPLETE.json` を保存します。出力は**元v43の読取専用評価**で、Python移行版の性能や新学習の結果ではありません。自然EOSと元評価器の正誤を別々に数えます。更新数は0です。

静的検証は `package_verification.json` に保存済みです。元v43ローカル入力の全byte SHA、ZIP内部のmetadata/SWE/採点器、200件/2048位置、埋込sourceとarchiveを照合しました。Kaggle上のmounted byte、実GPU実行時間、実回答は未測定です。現在の実行中GPU jobと公式fresh quota/内部holdは起動前にRootが別途確認します。
