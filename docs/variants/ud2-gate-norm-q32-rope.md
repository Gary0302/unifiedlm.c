# UD2-gate-norm-q32-rope（代號 `UD2-gate-norm-q32-rope`）

**15M 目前最好（1 個 seed）。** UD2-gate-norm-q32 的一半格子兩兩成對旋轉（週期 4–256 個 token），讓記憶的比對看相對距離；不加參數。

- **類別**：主線（統一 block，不含 attention）
- **基於**：[UD2-gate-norm-q32](ud2-gate-norm-q32.md)
- **程式碼**：`scripts/delta.py` 的 `RopeReadNormQueryGateAnchorDeltaBlock`，訓練時 `--variant=UD2-gate-norm-q32-rope`

## 怎麼做的

- UD2-gate-norm-q32：讀出先 RMSNorm，讀位址加秩 32 的修正，hidden_dim 662。
- UD2-gate-norm-q32-rope：一半的格子兩兩成對，寫入鑰匙按寫入的位置 s 旋轉、讀位址按讀的位置 t 旋轉（角度 = 位置 × 2π／週期，週期 4–256 個 token，對數間隔），所以這些格子的比對 q_t · k_s 只看 t − s；同一對的衰減綁在一起（取兩個 log λ 的平均），衰減才跟旋轉可交換。另一半格子和其他部分不變。
- 不加參數（15,457,884，跟 UD2-gate-norm-q32 一樣）。

**問題**：記憶只用內容比對，不知道距離。加上相對位置，會不會更好？

```text
一半格子：k_s ← R(s) k_s，q_t ← R(t) q_t，R 是成對旋轉，週期 4–256
同一對共用衰減
y = μ z_t + o_t · RMSNorm(E_tᵀ q_t)
```

**設定**：15M（12 層、2,300 步），hidden_dim 662，跟 UD2-gate-norm-q32 同設定；1 個 seed（1337），3090 1.6 小時（5,861 秒）；fork session 跑（2026-11-15）；第一次跑時程式有 bug（讀位址修正沒作用），這是修正後從頭重跑的。

## 結果

FineWeb-Edu 驗證集，NLL 越低越好；括號裡是 seed 範圍。設定見 [實驗報告索引](../experiments.md#設定)。

| 規模 | run | seed 數 | 參數量 | 驗證 NLL @1024 | @4096 | 前 64 個 token | 512–1,024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15M | `ud2_gate_norm_q32_rope_15m` | 1 | 15.46M | 3.716 | 3.671 | 4.104 | 3.668 |
| 42M | `ud2_gate_norm_q32_rope_42m` | — | — | 訓練中 |  |  |  |

讀到 20K（訓練長度 20 倍）的 16K–20K 段：3.632。

每層有效記憶長度（遺忘門衰減到 1/e 的 token 數，按格子平均，會被快格子拉低，第 0 → 11 層）：3 / 3 / 1 / 2 / 2 / 1 / 2 / 2 / 5 / 3 / 3 / 2。

15M 按寫入量加權、只算衰減，過了 256 個 token 還留下的寫入量（第 0 → 11 層，[每個格子記多久](../experiments/slot-retention.md)）：20% / 13% / 0% / 1% / 1% / 0% / 2% / 1% / 27% / 34% / 9% / 14%。

## 結論

15M（1 個 seed）：NLL 3.716（長度 1,024）、3.671（4,096），比 UD2-gate-norm-q32（3.727、3.683）好 0.011，是目前 15M 最好的統一 block，贏 Transformer（3 個 seed 平均 3.945）0.229。每一段都好：長度 1,024 的五段好 0.010–0.016，前 64 個 token 最多（4.104 對 4.120）；長度 4,096 也每段都好 0.009–0.016，超過訓練長度的 2K–4K 好 0.012，所以旋轉在沒訓練過的長度也沒壞事。同樣加旋轉、但讀位址修正用秩 128（格子 538）的 UD2-gate-norm-q128-rope 是 3.732，比這個差。讀到 20K 的 16K–20K 3.632，是 15M 模型裡這段最好的（UD2-gate-norm-q32 3.645）：旋轉的週期最長只有 256 個 token，讀到訓練長度 20 倍也沒壞事。逐 token 對 Transformer 全部 −0.202（UD2-gate-norm-q32 −0.197），進步在新 token：新 token −0.057 對 −0.040，位置 64–256 −0.052 對 −0.039，256–1,024 −0.052 對 −0.032；照抄反而略差（−0.619 對 −0.637）。截掉遠處前文，新 token 還是變好 0.006（UD2-gate-norm-q32 0.004），所以旋轉讓新 token 在每個距離都更好，但更遠的前文還是沒幫上新 token。按寫入量加權，第 0、8、9、11 層過了 256 個 token 還留 20%／27%／34%／14%，比 UD2-gate-norm-q32 少。42M 已開始（ud2_gate_norm_q32_rope_42m，hidden_dim 1,195，41.77M 參數，跟 U-Delta-rawFFN+λlr1000 42M 對齊）。

## 相關實驗

- [讀出端：門、縮放、RMSNorm、讀位址修正、旋轉相對位置](../experiments/read-side.md)
- [長上下文與外推（讀到 20K）](../experiments/extrapolation.md)
- [逐 token 差距：輸在哪些 token](../experiments/token-gap.md)
- [遠處的前文值多少](../experiments/reach.md)
- [每個格子記多久（按寫入量加權）](../experiments/slot-retention.md)

## 重現

在倉庫根目錄執行（需要 CUDA；FineWeb 子集見 `scripts/fineweb/fineweb_subset.py`）：

```bash
# 15M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_rope_15m --variant=UD2-gate-norm-q32-rope --hidden_dim=662 --dataset=fineweb --dim=288 --n_layers=12 --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --batch_size=16 --gradient_accumulation_steps=8 --learning_rate=5e-4 --warmup_iters=360 --max_iters=2300 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_rope_15m
# 42M（scripts/nrp/fwsdfk.txt）
python scripts/train_variant.py --out_dir=outputs/ud2_gate_norm_q32_rope_42m --variant=UD2-gate-norm-q32-rope --hidden_dim=1195 --dim=512 --n_layers=16 --batch_size=8 --gradient_accumulation_steps=16 --warmup_iters=1000 --max_iters=6400 --dataset=fineweb --max_seq_len=1024 --vocab_source=llama2 --vocab_size=32000 --learning_rate=5e-4 --weight_decay=0.1 --beta2=0.95 --dropout=0.0 --eval_interval=500 --eval_iters=50
python scripts/fineweb/eval_fineweb.py --out_dir=outputs/ud2_gate_norm_q32_rope_42m
```
