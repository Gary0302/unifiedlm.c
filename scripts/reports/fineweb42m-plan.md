# FineWeb-Edu 42M：S 對 T（PoC，2026-11-09）

Gary 決定先做 42M，資料用 FineWeb-Edu，跑在 A6000 上。這是 PoC：每個模型只跑一次，一組 LR、一個 seed，不 sweep。

## 要回答的問題

- 42M 的 S 能不能在大一點的真實語料上正常學（loss 一路下降、沒有發散）。
- S 和 T 的 val loss 差多少。BabyLM 15M 時 S 比 T 高 0.10，但那是不同的資料和詞表，不能直接當成放大的趨勢。要看趨勢，得在同一份資料上補 15M 的 S 和 T（還沒排，要 Gary 同意）。
- 超過訓練長度（1024 → 4096）時 S 還撐不撐得住，T 是否照樣崩。

## 資料

- FineWeb-Edu sample-10BT 14 個 parquet 中的前 2 個（`000_00000`、`001_00000`），由沒改過的 `fineweb.py` 用 Llama 2 32K 詞表切成約 1 億 token 一個的 uint16 shard。
- 照 `fineweb.py` 的規則，排序後最後一個 shard 是驗證集，訓練不會讀到。
- 程式在 `scripts/fineweb/fineweb_subset.py`，在 NRP 的 CPU Job `ulm-fw-prep` 上建好，放在 PVC `ulm-storage` 的 `/workspace/data/FineWebEdu_2of14`，附 `SHA256SUMS`。

## 設定（`scripts/nrp/fw42m.txt`）

| | S | T |
| --- | --- | --- |
| 結構 | 16 個 UnifiedBlock，dim 512，每 block 1280 格 | 16 個 block = 8 個 attention + 8 個 FFN 交替，dim 512，SwiGLU 1376，head dim 64（8 個 head，同 llama2.c 42M） |
| 參數量 | 41,632,288 | 41,689,600 |

兩者共用：

- 序列長度 1024，Llama 2 詞表 32000。
- 每步 32 × 4（梯度累積）× 1024 = 131,072 token，練 6400 步，共 8.39 億 token，約參數量的 20 倍。資料不重複。
- LR 5e-4（沿用 15M 的配方，`docs/models.md` 42M 欄寫「待調參」），warmup 1000 步，再 cosine 衰減到 0。weight decay 0.1，β2 0.95，dropout 0，bf16 + `torch.compile`，seed 1337。
- 每 500 步用 50 個 batch 估一次 val loss。

風險：LR 沒調。如果 S 和 T 對 LR 的敏感度不同，比較可能不公平。結果只當作單次 PoC 觀察。

## 評估（`scripts/fineweb/eval_fineweb.py`）

- 最後一個 checkpoint 在驗證 shard 上的逐位置 NLL：長度 1024 取 2048 個窗口（約 210 萬 token），長度 4096 取 512 個窗口。
- 沒有 BLiMP：手上的 BLiMP 是用 BabyLM 詞表切的。

## 預估成本

- A6000 測速（`scripts/bench/train_speed.py`，bf16，T=1024）：
  - 42M 的 S 在 batch 32 時每秒 8.0 萬 token，顯存 22GB，練完約 2.9 小時。
  - 42M 的 T 在 batch 32 時每秒 21.4 萬 token，約 1.1 小時。
- 兩個 pod 各一張 A6000，每個 pod 都在 6 小時上限內。
- A6000 節點的空閒 CPU 只有 0.1–0.3 顆，所以 pod 只要 150m CPU，裝環境和編譯會比較慢。
