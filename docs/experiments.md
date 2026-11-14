# 统一性实验报告（FineWeb-Edu，15M / 42M）

截至 2026-11-13。除非表里写了 seed 数，每个数字都是单次运行（1 个 seed）；所有模型用同一个学习率，没有调参。主线问题见 [`research_goal.md`](research_goal.md)：同一个原语能不能同时承担 token mixing 和 channel mixing，并达到同规模 Transformer 的语言建模质量。目标是**不含 attention 的统一 block**；含 attention 的版本只作对照。

## 命名

代码、run 名（如 `fwsdu15m`）和 `variant.json` 里用的是代号；文档里用下面的名字。`U-` 表示统一 block（同一组格子既是 FFN 又是上下文记忆）。

| 名字 | 代号 | 是什么 |
| --- | --- | --- |
| U-Avg | `S` | 原版 UnifiedLM（`model.py` 现在的标准模型）：格子存"先验 + 上下文证据"的后验均值，遗忘率固定 |
| U-Gate | `SG` | U-Avg 加上按内容决定的遗忘门：每格、每 token 的遗忘率由地址算出（低秩 16，softplus） |
| U-Gate-fastinit | `SH` | U-Avg 只把遗忘率起点调快到 1/64，U-Gate 的对照 |
| U-Gate-rank64 / -fromh / -silu | `SG64` / `SGh` / `SGs` | U-Gate 的门：低秩 64 / 从 block 输入读 / 中间加 SiLU |
| U-Gate-nogate | `SR` | U-Gate 的消融：每格各自的遗忘率，没有门 |
| **U-Delta** | `SD` | delta rule 记忆，以 FFN 为锚：记忆只存 FFN 没预测到的部分 `v − μk`，同一个地址的新值按比例 β 覆盖旧值；遗忘门沿用 U-Gate |
| **U-Delta-rawFFN** | `SDu` | U-Delta 的 FFN 那一半改读未归一化的 relu²（就是标准的 ReLU² FFN），记忆那一半不变 |
| U-Delta-λlr10 / λlr100 / λlr1000 | `SDl` / `SDl100` / `SDl1000` | 每格基础遗忘率 `log_rate` 的有效学习率 ×10 / ×100 / ×1000（重参数化，起点不变） |
| **U-Delta-rawFFN+λlr1000** | `SDul` | U-Delta-rawFFN 加上 λlr1000 |
| U-Delta-spreadλ | `SDw` | 遗忘率初始值分散在 1/(L/4)～1/(16L)：同一份记忆里有快有慢 |
| U-Delta-diverseλ | `SDv` | 基础记忆长度从 4 到 65,536 个 token 分散，门最多只能让遗忘快 2 倍 |
| U-Delta-docreset | `SDb` | 遇到文档开头（BOS）就清空记忆 |
| U-Delta-learnP | `SDp` | 每层的地址次方 p 可学（从 2 开始） |
| U-Delta-ReLU4 | `SDr` | 前 4 个 block 用 ReLU 地址（p = 1） |
| U-Delta+slow | `SD2` | U-Delta 再加一份慢记忆和读出混合 ρ |
| U-Delta+fast2 | `SD2c` | U-Delta+slow 的同容量对照：两份都是快记忆 |
| U-Delta+attn | `SDA` | 每 4 个 U-Delta block 后插一个窗口 128 的局部 attention（含 attention，不算统一 block） |
| Split | `P` | 静态格子和上下文格子用各自的地址（不共享），统一的对照 |
| Transformer | `T` | llama2.c 式 Transformer，同 dim、参数量对齐 |
| Transformer+conv / +SSMax | `TC` / `TS` | Transformer 在 attention 前加同样的宽 4 因果卷积 / attention 换成 Scalable-Softmax |
| GDN / GDN-bigstate / GDN-2 / Mesa | `GDN` / `GDNs` / `GDN2` / `MESA` | 线性递归基线：Transformer 的 attention 换成 Gated DeltaNet / 状态放大 3 倍的 GDN / NVIDIA Gated DeltaNet-2 / Mesa layer |

## 设置

- **数据**：FineWeb-Edu sample-10BT 的前 2 个 parquet（`FW_SUBSET=2of14`，见 `scripts/fineweb/fineweb_subset.py`），Llama 2 32K 词表；验证集是 `001_00000_008.bin`。110M 用前 4 个（`4of14`），验证集相同。
- **15M**：dim 288，12 个 block，`hidden_dim` 712，`max_seq_len` 1024，每步 131,072 tokens，2,300 步（约 3.0 亿 tokens）；lr 5e-4，warmup 360，cosine 到 0，weight decay 0.1，β₂ 0.95，dropout 0。
- **42M**：dim 512，16 个 block，`hidden_dim` 1248（U-Delta 系列；U-Avg 1280），每步 131,072 tokens，6,400 步（约 8.4 亿 tokens），warmup 1000，其余同 15M。
- **评测**：验证 NLL，长度 1,024（2,048 个窗口，2,097,152 tokens）和 4,096（512 个窗口）。读到 20K：64 个长 20,480 的窗口按位置分段（`scripts/fineweb/long_eval.py`）。Transformer 超过训练长度时报两种：完整上下文（RoPE 直接外推）和滑动窗口 1,024。
- **训练命令**：`scripts/nrp/*.txt` 每行一个 run（run 名 + `train_variant.py` 的参数）；在 NRP Nautilus 上一次一张卡（A40、A6000、A10、L40 等）。

## 主要结论

1. **15M：统一 block 赢 Transformer。** U-Delta 三个 seed 3.895–3.901，Transformer 三个 seed 3.928–3.957。最好的是 U-Delta-rawFFN+λlr1000，3.793，连含 attention 的 U-Delta+attn（3.840）都赢（1 个 seed）。要打折扣的地方：15M 的 Transformer 不太会用前文，第 64–128 段到第 512–1,024 段 NLL 只降 0.06（3.979 → 3.918），U-Delta-rawFFN 降 0.12，所以 15M 的领先有一部分来自小 Transformer 的弱点。
2. **42M：差一点追上，还没赢。** Transformer 3.214，U-Delta-rawFFN 3.228（输 0.014），U-Delta 3.350，U-Gate 三个 seed 3.328–3.333。U-Delta 在 42M 落后的主因是 FFN 读了归一化的地址：改成标准 ReLU² FFN 补回 0.122。剩下的差距不在开头（前 64 个 token 还略赢），而在 64–1,024 之间（每段 +0.013～+0.016）。42M 时 Transformer 也学会用前文了（同样两段降 0.129，U-Delta-rawFFN 0.128），15M 那种优势消失。U-Delta-rawFFN+λlr1000 的 42M 在排队。
3. **110M（只有原版）：U-Avg 2.994，Transformer 2.822。** 原版输很多；U-Delta 系列还没跑 110M。
4. **记忆的写法：delta rule 远好于平均。** 15M 时 U-Delta 3.898，U-Gate 3.998，U-Avg 4.112（各 3 个 seed）。
5. **记忆短一点更好。** 放开基础遗忘率的学习率，模型会一路把记忆缩短：λlr100 3.851、λlr1000 3.843，都跟 U-Delta+attn 打平；λlr1000 除了第 1–3 层，每层都忘到上限（每个 token 最多忘 63%，约 1 个 token）。和 rawFFN 合并只叠加了一部分（1,024 内再好 0.019，不是 0.055），但外推的好处留住了：读到 20K 时比 U-Delta-rawFFN 好的幅度从开头的 0.017 增加到 16K–20K 的 0.047。
6. **强迫留住慢记忆没用，读得很长时还更差。** U-Delta+slow（慢记忆几乎不忘）和 U-Delta-diverseλ（门有上限，慢格子留得住）在 1,024 内只小赢，读到 4K 以后都比 U-Delta 差，16K–20K 是 3.856、3.859，U-Delta 3.810–3.820。
7. **外推：统一 block 过了训练长度不会崩。** 完整上下文的 Transformer 在 16K–20K 是 6.09（15M）、6.25（42M）。跟滑动窗口的 Transformer 比：15M 统一 block 赢（滑动窗口 Transformer 3.874，U-Delta 三个 seed 3.810–3.820，U-Delta-rawFFN+λlr1000 3.713），42M U-Delta-rawFFN 输 0.01～0.03（16K–20K 3.166 对 3.142）。
8. **纯记忆题 m2kv（宽度 32）：只有 U-Delta 外推。** 训练时间隔最多 256；间隔 4,096 时 U-Delta 100%，U-Gate 39.6%，Transformer 和 U-Avg 0%。不重训再拉长：U-Delta 8,192 95.4%、16,384 55.1%、20,000 33.9%；U-Delta+slow 16,384 77.0%、20,000 66.9%，慢记忆在真的需要记很久时有用。
9. **U-Delta 赢 GDN 不是因为卷积、写入规则或状态大小。** Transformer+conv 只比 Transformer 好 0.007；GDN-2、Mesa 和 GDN 同一档；GDN 状态放大 3 倍没变。统一 block 原本的弱点是开头 64 个 token（比 GDN 差约 0.09），来自 FFN，U-Delta-rawFFN 补平了（4.170 对 4.166）。

## 重要观察

- **每一层自己决定记多久。** 从同一个起点（每层记 4,096 个 token、地址次方 2）出发，训练完第 0 层只记几个 token，第 1–4、11 层记几百个；可学次方时，记得越久的层地址越尖（U-Delta-learnP）。
- **给它选，记忆会一路变短，短到上限还在变好**（见结论 5）。U-Delta-λlr10 和 U-Delta-spreadλ 用不同方法学到几乎一样的分工：中间层（第 6–10 层）从 65–95 降到 15–31 个 token。
- **内容丢失主要来自遗忘，不是被新写入覆盖。** 把每格的流失拆成遗忘门衰减和 delta rule 覆盖（`scripts/fineweb/overwrite.py`）：U-Delta、U-Delta-spreadλ、U-Delta-λlr10 几乎所有格子里覆盖占不到 1%；遗忘门让实际遗忘比基础速率快 10–30 倍（基础记 1K–4K 的格子实际约 113 个 token）。只有基础速率非常慢的少数格子，覆盖才成为主因。
- **记忆偏向写意外的 token，但预测到的、模型自己生成的也照写**（`scripts/fineweb/writeloss.py`）。每个 token 写入量 β·|目标 − 记忆已有| 和该 token loss 的 Spearman 相关：U-Delta 0.15（15M）/ 0.14（42M），U-Delta-rawFFN 0.20 / 0.21。loss 最低的一成 token 写入量是平均的 0.67–0.83 倍，最高的一成 1.11–1.35 倍。模型自己 greedy 生成时写入量仍有真实文字的 0.52–0.74 倍；按概率采样时反而多 4–10%。逐 token 重算的记忆与 kernel 一致（相对误差 2×10⁻⁶）。
- **U-Delta+slow 的慢记忆没有学**：慢记忆的速率停在初始值附近（约 16,000 个 token，几乎不忘）；第 3、4 层读出比重 ρ 高但几乎不写，等于关掉记忆；第 11 层只用快记忆。
- **15M → 42M 变化的是 Transformer**：U-Delta-rawFFN 用前文的能力没变，Transformer 在 42M 学会了用前文（见结论 2）。

## 结果

### 15M（验证 NLL，越低越好）

| 模型 | 代号 | 参数量 | seed 数 | 验证 NLL @1024（范围） | @4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| **统一 block（主线）** | | | | | |
| U-Delta-rawFFN+λlr1000 | `SDul` | 15.45M | 1 | 3.793 | 3.750 |
| U-Delta-rawFFN | `SDu` | 15.45M | 1 | 3.812 | 3.774 |
| U-Delta-λlr1000 | `SDl1000` | 15.45M | 1 | 3.843 | 3.796 |
| U-Delta-λlr100 | `SDl100` | 15.45M | 1 | 3.851 | 3.809 |
| U-Delta-diverseλ | `SDv` | 15.45M | 1 | 3.877 | 3.845 |
| U-Delta-λlr10 | `SDl` | 15.45M | 1 | 3.887 | 3.846 |
| U-Delta+slow | `SD2` | 15.47M | 1 | 3.888 | 3.851 |
| U-Delta-docreset | `SDb` | 15.45M | 1 | 3.890 | 3.843 |
| U-Delta-spreadλ | `SDw` | 15.45M | 1 | 3.890 | 3.848 |
| U-Delta-learnP | `SDp` | 15.45M | 1 | 3.892 | 3.851 |
| U-Delta-ReLU4 | `SDr` | 15.45M | 1 | 3.894 | 3.852 |
| U-Delta | `SD` | 15.45M | 3 | 3.898（3.895–3.901） | 3.857 |
| U-Delta+fast2 | `SD2c` | 15.75M | 1 | 3.897 | 3.853 |
| U-Gate-rank64 | `SG64` | 16.28M | 1 | 3.977 | 3.942 |
| U-Gate | `SG` | 15.46M | 3 | 3.998（3.988–4.003） | 3.963 |
| U-Gate-silu | `SGs` | 15.46M | 1 | 4.002 | 3.966 |
| U-Gate-fromh | `SGh` | 15.38M | 1 | 4.012 | 3.978 |
| U-Gate-fastinit | `SH` | 15.17M | 3 | 4.051（4.049–4.052） | 4.029 |
| U-Avg | `S` | 15.17M | 3 | 4.112（4.107–4.116） | 5.658 |
| U-Gate-nogate（SR） | `SR` | 15.18M | 1 | 4.124 | 5.886 |
| **基线** | | | | | |
| Transformer | `T` | 15.19M | 3 | 3.945（3.928–3.957） | 5.016 |
| GDN | `GDN` | 15.23M | 1 | 3.964 | 3.948 |
| GDN-bigstate | `GDNs` | 15.22M | 1 | 3.964 | 3.948 |
| Mesa | `MESA` | 15.24M | 1 | 3.969 | 3.951 |
| GDN-2 | `GDN2` | 15.21M | 1 | 3.970 | 3.954 |
| Split（P） | `P` | 17.64M | 3 | 4.038（4.034–4.040） | 6.233 |
| **含 attention 的对照（不算统一 block）** | | | | | |
| U-Delta+attn | `SDA` | 15.45M | 1 | 3.840 | 3.798 |
| Transformer+SSMax | `TS` | 15.19M | 1 | 3.913 | 5.101 |
| Transformer+conv | `TC` | 15.20M | 1 | 3.921 | 4.910 |

### 42M

| 模型 | 代号 | 参数量 | seed 数 | 验证 NLL @1024（范围） | @4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| Transformer | `T` | 41.69M | 1 | 3.214 | 4.871 |
| U-Delta-rawFFN | `SDu` | 41.77M | 1 | 3.228 | 3.183 |
| U-Gate | `SG` | 42.33M | 3 | 3.330（3.328–3.333） | 3.288 |
| U-Delta | `SD` | 41.77M | 1 | 3.350 | 3.303 |
| U-Gate-fastinit | `SH` | 41.63M | 1 | 3.361 | 3.327 |
| Split（P） | `P` | 52.14M | 1 | 3.368 | 7.561 |
| U-Avg | `S` | 41.63M | 1 | 3.423 | 8.708 |

按位置（长度 1,024）：

| 位置 | 0-64 | 64-128 | 128-256 | 256-512 | 512-1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 3.628 | 3.293 | 3.233 | 3.182 | 3.164 |
| U-Delta-rawFFN | 3.623 | 3.307 | 3.245 | 3.196 | 3.179 |
| U-Delta | 3.757 | 3.437 | 3.369 | 3.317 | 3.299 |
| U-Delta-rawFFN − Transformer | -0.005 | +0.014 | +0.013 | +0.014 | +0.016 |

### 110M（FineWeb-Edu 4of14，16,700 步）

| 模型 | 代号 | 参数量 | seed 数 | 验证 NLL @1024 | @4096 |
| --- | --- | ---: | ---: | ---: | ---: |
| Transformer | `T` | 109.53M | 1 | 2.822 | 4.614 |
| U-Avg | `S` | 109.40M | 1 | 2.994 | 6.635 |

### 读到 20K（按位置分段的 NLL）

| 模型（15M） | 0-64 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-20480 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T（完整上下文） | 4.106 | 3.842 | 4.784 | 5.603 | 5.875 | 6.074 | 6.088 |
| T（滑动窗口 1,024） | 4.106 | 3.842 | 3.886 | 3.903 | 3.923 | 3.911 | 3.874 |
| GDN | 4.052 | 3.872 | 3.927 | 3.948 | 3.965 | 3.960 | 3.922 |
| U-Delta | 4.114 | 3.812 | 3.836 | 3.876 | 3.875 | 3.841 | 3.817 |
| U-Delta-spreadλ | 4.111 | 3.800 | 3.827 | 3.865 | 3.870 | 3.832 | 3.807 |
| U-Delta+slow | 4.125 | 3.796 | 3.826 | 3.876 | 3.889 | 3.870 | 3.856 |
| U-Delta-diverseλ | 4.107 | 3.789 | 3.818 | 3.869 | 3.886 | 3.873 | 3.859 |
| U-Delta-docreset | 4.129 | 3.806 | 3.825 | 3.861 | 3.858 | 3.822 | 3.797 |
| U-Delta-λlr100 | 4.081 | 3.763 | 3.786 | 3.830 | 3.828 | 3.793 | 3.769 |
| U-Delta-λlr1000 | 4.085 | 3.753 | 3.776 | 3.814 | 3.813 | 3.776 | 3.752 |
| U-Delta-rawFFN | 4.024 | 3.725 | 3.748 | 3.796 | 3.803 | 3.780 | 3.760 |
| U-Delta-rawFFN+λlr1000 | 4.007 | 3.699 | 3.725 | 3.766 | 3.768 | 3.736 | 3.713 |
| U-Delta+attn | 4.061 | 3.747 | 3.778 | 3.814 | 3.819 | 3.785 | 3.762 |

| 模型（42M） | 0-64 | 512-1024 | 1024-2048 | 2048-4096 | 4096-8192 | 8192-16384 | 16384-20480 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T（完整上下文） | 3.533 | 3.120 | 4.703 | 5.810 | 5.991 | 6.221 | 6.251 |
| T（滑动窗口 1,024） | 3.533 | 3.120 | 3.148 | 3.198 | 3.196 | 3.161 | 3.142 |
| U-Delta | 3.663 | 3.262 | 3.280 | 3.325 | 3.322 | 3.286 | 3.269 |
| U-Delta-rawFFN | 3.521 | 3.139 | 3.158 | 3.208 | 3.206 | 3.176 | 3.166 |

### 记忆任务 m2kv（宽度 32，正确率；训练时间隔 ≤ 256）

| 模型 | 间隔 0 | 间隔 64 | 间隔 256 | 间隔 1024 | 间隔 4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Avg | 99.9% | 100.0% | 99.9% | 18.5% | 0.0% |
| U-Gate | 99.6% | 99.7% | 99.5% | 98.5% | 39.6% |
| U-Gate-fastinit | 90.2% | 90.8% | 90.7% | 44.0% | 0.0% |
| Transformer | 99.6% | 99.8% | 99.7% | 0.0% | 0.0% |
| U-Delta | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| U-Delta+slow | 100.0% | 100.0% | 100.0% | 100.0% | 98.3% |

## 进行中与待决定

- **进行中**：U-Delta-rawFFN+λlr1000 42M（约 9.5 小时，第 1,000 步验证 loss 4.137，同一步 U-Delta-rawFFN 4.137、Transformer 4.155）；U-Delta-rawFFN+λlr1000 15M 补 seed 1338、1339。
- **待决定**：记忆完全拿掉的对照（U-Delta-λlr1000 几乎把记忆关掉，记忆还剩多少作用）；42M 的学习率（U-Delta-rawFFN 只差 0.014）；U-Delta-λlr10、U-Delta-spreadλ 补 seed；Transformer 加 NoPE 层作完整上下文的外推对照；U-Delta 的两个归因实验（写入目标去掉 −μk 锚、μ 和记忆用各自的地址）；Transformer 学习率 1e-3；Transformer 和 U-Delta 各跑 2 倍步数。

## 数据与代码

- `scripts/reports/fineweb/runs.json`：所有 run 的验证 NLL（含按位置）、训练曲线、每层记忆长度（gates）、训练时间；`sgvt20.json`：读到 20K；`mem.json`：m2kv；`writeloss.json`：写入量分析。
- `scripts/variants.py` 是变体注册表（代号 → block），`scripts/delta.py` 是 U-Delta 系列，`scripts/gated.py` 是 U-Gate 系列，`scripts/mesa.py` 是 Mesa；训练入口 `scripts/train_variant.py`；评测和诊断在 `scripts/fineweb/`（`eval_fineweb.py`、`long_eval.py`、`gates.py`、`overwrite.py`、`writeloss.py`）。
- 依照 `AGENTS.md`，变体在赢之前都放在 `scripts/`，`model.py` 仍然是 U-Avg。

## 各实验说明

### U-Delta-rawFFN+λlr1000：U-Delta-rawFFN 加上 log_rate 学快 1,000 倍

把两个各自有效、都不加参数的改动合在一起：FFN 读没范式的地址（U-Delta-rawFFN），加上遗忘率的 log_rate 学快 1,000 倍（U-Delta-λlr1000）。

- **对照**：U-Delta-rawFFN（同一个 seed）
- **问题**：U-Delta-rawFFN（15M 好 0.086，每个位置差不多）和 U-Delta-λlr1000（好 0.055，越后面越多）进步的位置不一样。合在一起能不能叠加？U-Delta-rawFFN 42M 只输 Transformer 0.014，而且差距在 64 之后、越后面越大，正好是 U-Delta-λlr1000 进步的地方。
- U-Delta-rawFFN：y = μ z + Eᵀq，FFN 读没除总和的 relu²，记忆跟 U-Delta 一样。
- U-Delta-λlr1000：log_rate 存成 θ = log_rate ／ 1000，Adam 的步伐等于让 log_rate 学快 1,000 倍；起点一样。
- U-Delta-rawFFN+λlr1000：两个都做，其他跟 U-Delta-rawFFN 一样。参数、状态跟 U-Delta 一样。
- **设置**：15M（12 层、2,300 步），跟 U-Delta-rawFFN 15M 同设置；比 U-Delta-rawFFN 好 0.01 以上就接着跑 42M（16 层、6,400 步，约 9.5 小时）；参数：15.45M（同 U-Delta）
- **代码**：`scripts/delta.py` 里的 `RawFFNFastRate1000DeltaBlock`

```text
y = μ z + E_tᵀ q_t，z = relu(K u + b)²
log λ = expm1(−exp(1000 θ) · g)
```

**结论**：15M：NLL 3.793（长度 1,024）、3.750（4,096），比 U-Delta-rawFFN（3.812）再好 0.019，是目前 15M 最好的统一 block，也赢 U-Delta+attn（3.840）0.047、Transformer（3 个 seed 平均 3.945）0.152（1 个 seed）。但只叠加了一部分：U-Delta-λlr1000 单独比 U-Delta 好 0.055，加在 U-Delta-rawFFN 上只再好 0.019。每个位置都比 U-Delta-rawFFN 好 0.017–0.020（前 64 个 token 4.153 对 4.170，512–1,024 3.748 对 3.768）。记忆几乎被关掉：有效记忆长度 12 层里有 10 层只有 1.1–2.2 个 token，只剩第 2、3 层记 38、22 个 token（U-Delta-rawFFN 是 5–509）。比 U-Delta-rawFFN 好 0.01 以上，照规则接着跑 42M（18:08 开始）；读到 20K：每一段都是 15M 最好的，16K–20K 3.713（U-Delta-λlr1000 3.752、U-Delta-rawFFN 3.760、U-Delta+attn 3.762、U-Delta 3.817、滑动窗口的 Transformer 3.874）。比 U-Delta-rawFFN 好的幅度越读越大：前 64 个 token 0.017、512–1,024 0.026、16K–20K 0.047，U-Delta-λlr1000 那一半带来的外推好处留住了。

### U-Delta-rawFFN：FFN 那一半读没范式的地址

统一 block 的 FFN 那一半（μ）改读没除总和的 relu²，像一般的 ReLU² FFN；记忆那一半不变。看 42M 输 Transformer 是不是因为地址范式让 FFN 变弱。

- **对照**：U-Delta（同一个 seed）
- **问题**：U-Delta 15M 赢 Transformer 0.047，42M 却输 0.136，而且每个位置都输一样多，像是基本建模能力（FFN）落后。FFN 那一半读范式过的地址，是不是放大后的瓶颈？
- U-Delta：y = μ r + Eᵀq，r = relu(K u + b)² ／ (ε + Σ)，FFN 和记忆共用范式过的地址。除以总和让 FFN 输出跟「这个 token 有多强烈地触发 slot」无关，一般的 FFN（ReLU²、SwiGLU）都保留这个大小。
- U-Delta-rawFFN：y = μ z + Eᵀq，z = relu(K u + b)²（不除总和）。记忆的读写地址、写入目标 v − μ k（k = r_{t−1}）都跟 U-Delta 一样。
- 还是同一组 slot、同一个 K：只改 FFN 读的是 z 还是 r。
- 参数、状态跟 U-Delta 一样；初始时 FFN 输出比 U-Delta 大（约跟一般 FFN 一样）。
- **设置**：15M 先跑，没变差（NLL 不高于 U-Delta 三个 seed 的范围太多）就接着跑 42M（16 层、6,400 步，约 12 小时）；参数：15.45M（同 U-Delta）
- **代码**：`scripts/delta.py` 里的 `RawFFNDeltaBlock`

```text
z = relu(K u + b)²，r = z ／ (ε + Σ_a z_a)
y = μ z + E_tᵀ q_t（U-Delta：μ r + E_tᵀ q_t）
```

**结论**：15M 大幅变好：NLL 3.812（长度 1,024）、3.774（4,096）。U-Delta 三个 seed 是 3.895–3.901，所以好了 0.086，约是 seed 间差距的 10 倍（1 个 seed）。它也赢 Transformer（3 个 seed 平均 3.945）0.133、赢含注意力的 U-Delta+attn（3.840）0.028，是目前 15M 最好的模型。进步在每个位置都差不多（0.08–0.09），前 64 个 token 4.170，追平 GDN（4.166；U-Delta 4.258），所以是基本建模能力变好，不是记忆，符合假设：FFN 读范式过的地址会让它变弱。学到的记忆长度也变了：长记忆层变少，第 1–2 层记约 460–510 个 token，第 7–8 层约 135，其他层 5–68（U-Delta 是第 1–4、11 层记 259–434）。读到 20K 也不崩，每一段都最好，连 U-Delta+attn 都赢（16K–20K 3.760 对 3.762，打平；U-Delta 3.817），但比 U-Delta 好的幅度越远越小：1,024 内约 0.09，16K–20K 0.056，可能因为记得久的层变少。42M：NLL 3.228（1,024）、3.183（4,096），Transformer 42M 3.214，只输 0.014；U-Delta 42M 3.350，好 0.122；U-Gate 42M 3.328（各 1 个 seed）。U-Delta 从 15M 赢 Transformer 0.047 变成 42M 输 0.136，U-Delta-rawFFN 从赢 0.133 变成输 0.014，FFN 是 U-Delta 在 42M 落后的主因。对 Transformer：前 64 个 token 略赢（3.623 对 3.628），之后每段落后 0.013–0.016，剩下的差距在比较远的位置。读到 20K（42M）：每一段都比 U-Delta 42M 好 0.10–0.14，但都输只看最近 1,024 个 token 的 Transformer（滑动窗口）0.01–0.03，16K–20K 3.166 对 3.142（U-Delta 42M 3.269）；原本的 Transformer 过了 1,024 就崩（16K–20K 6.251）。

### U-Delta-λlr10：基础遗忘率学习率 10 倍

U-Delta 的每个 slot 基础遗忘率（log_rate）用 10 倍学习率训练，其他全部不变，看 U-Delta 能不能自己学出 U-Delta-spreadλ 靠初始值得到的慢 slot。

- **对照**：U-Delta（同一个 seed）
- **问题**：U-Delta-spreadλ 追平 U-Delta+slow，但它的基础遗忘率在 2,300 步内几乎没动，等于靠初始值。U-Delta 的 λ 本来就能学，是学不出慢 slot，还是只是学得太慢？
- log_rate 改存成 θ＝log_rate／10，用的时候乘回 10。Adam 每步大约移动 lr 个单位（1 维参数没有 weight decay），所以 log_rate 每步移动变成 10 倍；已在 CPU 上量过，正好 10 倍。
- 初始值跟 U-Delta 一样：全部 slot 从记 4,096 个 token 开始。初始输出跟 U-Delta 完全相同。
- 参数、状态跟 U-Delta 一样。
- **设置**：参数：15.45M（同 U-Delta）；状态：2.46M（同 U-Delta）
- **代码**：`scripts/delta.py` 里的 `FastRateDeltaBlock`

```text
λ = exp(−exp(10·θ) · g(u))，θ 初始 = log(1/(4L))／10
其他同 U-Delta
```

**结论**：学得出来，但方向跟预期相反：U-Delta 没有学出慢 slot，而是把记忆缩短。NLL 3.887（长度 1,024）、3.846（4,096），比 U-Delta 三个 seed（3.895–3.901）都好，跟 U-Delta-spreadλ（3.890）、U-Delta+slow（3.888）同级（1 个 seed）。基础记忆长度中位数从 4,096 学成 667–3,332 个 token，p90 都低于 4,040；有效记忆长度在中间层（第 6–10 层）从 U-Delta 的 65–95 降到 18–31，跟 U-Delta-spreadλ 一样短。所以 U-Delta-spreadλ 的进步可能主要来自记得较短，不一定是慢 slot；读到 20K 确认了这点：U-Delta-λlr10 跟 U-Delta-spreadλ 从第 128 个 token 起每段差不到 0.004（16K–20K 3.806 对 3.807，U-Delta 三个 seed 3.810–3.820，U-Delta+slow 3.856）。U-Delta-spreadλ 有记上万 token 的 slot、U-Delta-λlr10 没有，结果一样，所以慢 slot 没用上。

### U-Delta-λlr100 和 U-Delta-λlr1000：基础遗忘率学习率 100 倍、1000 倍

跟 U-Delta-λlr10 一样，只是 log_rate 的学习率乘 100 和 1000，看把它放得更自由时 U-Delta 会把各层的记忆长度推到哪里。

- **对照**：U-Delta 和 U-Delta-λlr10（同一个 seed）
- **问题**：U-Delta-λlr10（10 倍）把记忆缩短，没学出慢 slot。再放大 100、1000 倍，记忆长度会停在同一个地方（代表那是它想要的），还是一路往短（或长）冲、NLL 变差？
- log_rate 改存成 θ＝log_rate／100（或／1000），用的时候乘回去；Adam 每步让 log_rate 移动约 100（1000）个学习率。学习率 5e-4，所以每步最多约 0.05（0.5），在 log 尺度上，2,300 步可以走很远。CPU 上量过：每步平均移动 0.050 和 0.500，是 U-Delta 的 100 和 1000 倍。
- 初始值、初始输出、参数、状态都跟 U-Delta 一样。
- 遗忘率有上限：每个 token 最多忘 63%（expm1 的形式），所以 log_rate 很大也不会出数值问题。
- **设置**：参数：15.45M（同 U-Delta）；状态：2.46M（同 U-Delta）
- **代码**：`scripts/delta.py` 里的 `FastRate100DeltaBlock、FastRate1000DeltaBlock`

```text
λ = exp(−exp(c·θ) · g(u))，c＝100 或 1000，θ 初始 = log(1/(4L))／c
其他同 U-Delta
```

**结论**：U-Delta-λlr100：一路往短冲，而且变好很多。NLL 3.851（长度 1,024）、3.809（4,096），比 U-Delta 三个 seed（3.895–3.901）好约 0.047，比 U-Delta-λlr10（3.887）好 0.036，每个位置都好 0.04–0.05（1 个 seed）。第 5–10 层的基础记忆长度中位数学到 1–5 个 token，有效记忆长度 1–2，已经碰到上限（λ ≥ e⁻¹，每个 token 最多忘 63%）；第 0 层 2，第 3、4、11 层 19–34，只有第 1–2 层还记 83–139 个。所以记忆长度没有停在 U-Delta-λlr10 的位置：中间层想要的是「几乎不记」，短到上限还在变好，那几层的记忆等于只看前一两个 token 的局部运算。读到 20K 也不崩，每一段都比 U-Delta 好约 0.048（前 64 个 token 0.033），而且不随距离缩小，16K–20K 3.769（U-Delta 3.817，U-Delta-rawFFN 3.760，U-Delta+attn 3.762）：中间层几乎不记，远距离靠第 1–2 层就够。 U-Delta-λlr1000：再往短冲，又更好。NLL 3.843（长度 1,024）、3.796（4,096），比 U-Delta-λlr100 再好 0.008、0.013；长度 4,096 跟含注意力的 U-Delta+attn（3.798）一样，每一段差不到 0.006。除了第 1–3 层（有效记忆长度 53、14、9 个 token），每层都忘到上限（1.0–1.9）。所以记忆长度没有停在某个想要的值：给它多少自由，它就把长记忆关掉多少，NLL 一路变好。可是位置越后面 NLL 还是越低，长距离的信息可能靠第 2–3 层少数慢 slot（基础记忆长度 p90 417、983）。U-Delta-λlr1000 读到 20K 越远越好：比 U-Delta 好的幅度从前 64 个 token 的 0.029 变大到 8K 之后的 0.065，16K–20K 3.752，是所有模型里这段最好的（U-Delta-rawFFN 3.760，U-Delta+attn 3.762，U-Delta 3.817）。长记忆几乎关掉，外推反而最好。

### U-Delta-spreadλ：单一记忆，遗忘率初始值分散

不另开一份慢记忆，而是让单一记忆里的 slot 一开始就有快有慢，验证「单一记忆靠 λ 的设计就能做出快慢」。

- **对照**：U-Delta 和 U-Delta+slow（同一个 seed）
- **问题**：U-Delta+fast2 显示 U-Delta+slow 的进步不是容量，而是那份「几乎不忘、很少写」的慢记忆；U-Delta+slow 诊断也显示慢记忆的速率停在初始值没动。如果 U-Delta-spreadλ 追上 U-Delta+slow（3.888），就代表统一 block 维持单一记忆、只要把 λ 初始值设好就够了。
- U-Delta 的每个 slot 基础遗忘率都从 1/(4L) 开始（记忆约 4,096 个 token）。
- U-Delta-spreadλ 改成在 slot 之间以对数均匀分散，从 1/(L/4) 到 1/(16L)：最快的 slot 约 256 个 token，最慢的约 16,384 个（跟 U-Delta+slow 慢记忆的初始值一样）。
- 其他（依内容的 gate、delta rule、β）完全不变，参数和状态都跟 U-Delta 一样。
- **设置**：参数：15.45M（同 U-Delta）；状态：2.46M（同 U-Delta，U-Delta+slow 是 4.92M）
- **代码**：`scripts/delta.py` 里的 `WideRateDeltaBlock`

```text
log_rate_a = linspace(log(4/L), log(1/(16L)), m)   # U-Delta：全部 = log(1/(4L))
λ_{t,a} = exp(−exp(log_rate_a) · g_{t,a})，g 是依内容决定的 gate
```

**结论**：单一记忆就追平 U-Delta+slow：U-Delta-spreadλ 3.890，比 U-Delta（3.897，同一个 seed；U-Delta 三个 seed 3.895–3.901）好 0.007，跟 U-Delta+slow（3.888）只差 0.001，而 U-Delta+slow 多一份记忆、状态是两倍。4,096 评估 2K–4K 3.833；读到 20K 16K–20K 段 3.807，U-Delta+slow 在这段变差到 3.856。但进步不是来自慢 slot：U-Delta-spreadλ 训练完基础记忆长度 p10／中位数／p90 约 330／1,900／10,000，跟初始值差不多，慢 slot 一直都在；对照组 U-Delta-λlr10（从 U-Delta 的单一初始值出发、log_rate 学快 10 倍）把记忆缩短、没有慢 slot，结果在 1,024 内和读到 20K 都跟 U-Delta-spreadλ 一样。两者共同的地方是中间层（第 6–10 层）记得较短：有效记忆长度约 15–31 个 token，U-Delta 是 65–95。1 个 seed；要不要补 seed 在「待决定」。

### U-Delta-diverseλ：非常分散的 λ（基础记忆 4 到 65,536 个 token，gate 有上限）

每个 slot 的基础记忆长度从 4 个到 65,536 个 token（训练长度的 64 倍）对数均匀分散，而且遗忘 gate 最多只能把遗忘加快 2 倍，让这个分散在训练后还留着。

- **对照**：U-Delta-spreadλ（同一个 seed）
- **问题**：λ 真的很分散（从记几个 token 到记几万个）时，单一记忆能不能同时拿到 U-Delta-spreadλ 在一般文本上的好处和 U-Delta+slow 在记忆题上撑得久的好处？还是 gate 被限制住反而变差？
- 初始记忆长度范围从 U-Delta-spreadλ 的 256–16,384 放宽到 4–65,536 个 token。
- gate 从 softplus（没有上限）改成 2·sigmoid，值在 0 到 2 之间：可以让 slot 记得更久（g→0 就完全不忘），但最多只能忘快 2 倍。初始 g＝1，跟 U-Delta 一样。
- 原因：U-Delta-spreadλ 的 gate 平均把慢 slot 的遗忘加快约 25–30 倍，初始记上万 token 的 slot 实际只记约 350 个，分散被 gate 收回去了。
- 代价：慢 slot 在文档开头也没办法很快清空。
- 参数、状态跟 U-Delta 一样。
- **设置**：参数：15.45M（同 U-Delta）；状态：2.46M（同 U-Delta）
- **代码**：`scripts/delta.py` 里的 `DiverseRateDeltaBlock`

```text
λ_{t,a} = exp(−rate_a · g_{t,a})，g = 2·sigmoid(P Q rms(r_t)) ∈ (0, 2)
rate_a 初始从 1/4 到 1/(64 L) 对数均匀（U-Delta：全部 1/(4L)，g = softplus(·)/ln 2）
```

**结论**：分散留下来了，NLL 小幅变好：3.877（长度 1,024）、3.845（4,096），比 U-Delta 三个 seed（3.895–3.901）好约 0.021，比 U-Delta-spreadλ（3.890）、U-Delta-λlr10（3.887）好，但不如把记忆缩短的 U-Delta-λlr100（3.851）、U-Delta-λlr1000（3.843）（1 个 seed）。训练后每层的基础记忆长度几乎没动：p10 约 9、中位数约 430–510、p90 约 21,000–23,000 个 token，跟初始一样分散；gate 最多只能忘快 2 倍，所以慢 slot 真的留着。有效记忆长度每层都是 17–47 个 token（被快 slot 主导），中间层跟 U-Delta-λlr1000 一样短，所以进步可能来自一开始就有很快的 slot，而不是慢 slot。 读到 20K：2K 以内每段比 U-Delta 好约 0.02，2K–4K 打平，4K 之后变差，16K–20K 3.859，比 U-Delta 三个 seed（3.810–3.820）差约 0.04，跟完全不忘的 U-Delta+slow（3.856）一样。两个保留慢 slot 的做法（U-Delta+slow、U-Delta-diverseλ）读超过 4K 都变差，把记忆缩短的 U-Delta-λlr100、U-Delta-λlr1000 反而越读越好。

### U-Delta-docreset：遇到文档开头就清空记忆

在每个 BOS（文档开头）把记忆清空（λ＝0、β＝0），上一篇文章写进去的东西不会被下一篇读到。

- **对照**：U-Delta（同一个 seed）
- **问题**：跨文档的记忆是杂讯吗？明确清空之后，gate 不必学这件事，NLL 会不会变好？之后如果要用很慢的 slot（U-Delta-diverseλ），这个清空是不是必要的？
- U-Delta 的遗忘 gate 只学会在文档开头多忘一点（大部分层是平常的 1–4 倍），慢 slot 会把上一篇带到下一篇。
- U-Delta-docreset 在 BOS 那一步让记忆归零：λ＝0，而且那一步不写入（β＝0），从下一个 token 开始写新文档。
- 卷积（看前 3 个 token）没有清空。
- 实作：分块计算里把跨过 BOS 的项遮掉；已跟逐 token 的参考算法比过，误差 4e-16。
- 没有新参数，参数、状态跟 U-Delta 一样。
- **设置**：参数：15.45M（同 U-Delta）；状态：2.46M（同 U-Delta）
- **代码**：`scripts/delta.py` 里的 `DocResetDeltaBlock、doc_reset_forward`

```text
E_t = 0（x_t = BOS）
否则 E_t = (I − β_t k_t k_tᵀ) diag(λ_t) E_{t−1} + β_t k_t u_tᵀ（同 U-Delta）
```

**结论**：小幅变好：NLL 3.890（长度 1,024）、3.843（4,096）；U-Delta 三个 seed 是 3.895–3.901、3.854–3.860，所以好约 0.008、0.014，跟 U-Delta-spreadλ、U-Delta-λlr10 同级（1 个 seed）。长度 4,096 的 1K–4K 段好 0.014–0.016，比 1,024 内多，表示上一篇文档留在记忆里的东西确实有点干扰。清空之后模型让少数层记更久：第 1、3 层 572、650 个 token（U-Delta 373、434），第 4 层反而变短（33，U-Delta 259）。读到 20K 时好得更多：1,024 内跟 U-Delta 三个 seed 重叠，2K 之后每段都比三个 seed 好 0.01–0.02，16K–20K 3.797（U-Delta 3.810–3.820，U-Delta-λlr10 3.806，U-Delta-spreadλ 3.807），是不改 FFN 的 U-Delta 变体里最好的。文档越多、越远，清空越有用。

### U-Delta-learnP：地址尖锐度每层可学

U-Delta 的地址是 ReLU² 再除以总和，尖锐度固定在 2 次方；U-Delta-learnP 让每一层自己学次方 p。

- **对照**：U-Delta（同一个 seed）
- **问题**：blog（alexwang.ai/posts/associative-memory）说回想函数越尖，干扰越少、容量越大（线性约 d，ReLU^p 约 d^p，softmax 指数级）。U-Delta 的尖锐度是写死的，让它可学就知道模型想不想要更尖。
- 每层多一个纯量 a，p = 1 + eᵃ，从 0 开始（p＝2，跟 U-Delta 完全一样）；a 限制在 [−5, 2]，所以 p 在 1.007 到 8.4 之间。
- 地址 r = ReLU(Ku + b)^p ÷ (ε + Σ)，同一个 r 同时给 FFN、写入钥匙和读取查找，所以还是统一 block。
- p 的梯度：∂s^p/∂p = s^p · log s，整层所有 slot 和 token 加总到同一个 a；AdamW、学习率 5e-4、不做 weight decay。
- **设置**：参数：跟 U-Delta 一样（每层只多 1 个数）；状态：2.46M（同 U-Delta）
- **代码**：`scripts/delta.py` 里的 `PowerDeltaBlock`

```text
r = ReLU(K u + b)^p / (ε + Σ_a ReLU(K u + b)_a^p),   p = 1 + exp(a)
ReLU² 除以总和对 K 的尺度不变：K 放大 c 倍，结果不变，所以原本学不到尖锐度
```

**结论**：每一层都学得比 2 尖一点（2.02–2.35），没有一层想变钝；记忆长的层（第 1–4、11 层）比较尖，第 10 层是例外。NLL 只好 0.005（约 U-Delta seed 标准差的 1.5 倍），前 64 个 token 没有改善。尖锐度不是瓶颈，改成 softmax 的必要性不高。

### U-Delta-ReLU4：前 4 个 block 改用 ReLU

前 4 个 block 的地址改成普通 ReLU（p＝1，比较钝、比较密），其余维持 ReLU²，看快记忆所在的前几层用钝的地址会怎样。

- **对照**：U-Delta（同一个 seed）
- **问题**：U-Delta-learnP 学到的 p 都 ≥ 2，前几层也接近 2。直接把前几层压成 p＝1 是反方向的检查：钝的地址会让更多 slot 同时被读写，快记忆会变好（容易混合近处信息）还是变差（干扰变多）？
- 第 0–3 个 block：r = ReLU(Ku + b) ÷ (ε + Σ)；第 4–11 个 block 跟 U-Delta 一样是 ReLU²。
- p 是固定的，不学。参数跟 U-Delta 完全一样。
- U-Delta 里第 0 层的记忆最短（约 7 个 token），是最像「快记忆」的一层。
- **设置**：参数：15.45M（同 U-Delta）；状态：2.46M（同 U-Delta）
- **代码**：`scripts/delta.py` 里的 `FixedPowerDeltaBlock(p=1)`

```text
block 0–3：r = ReLU(K u + b) / (ε + Σ)
block 4–11：r = ReLU(K u + b)² / (ε + Σ)（同 U-Delta）
```

**结论**：跟 U-Delta 打平：3.894 对 3.897（同一个 seed，在 U-Delta 三个 seed 的差距内）。近处略差（前 64 个 token 4.264 对 4.258），远处略好（512–1K 3.847 对 3.851）。改成 ReLU 的第 3 层记忆从约 430 个 token 掉到 22，长记忆改由第 4、5、10 层负责；第 0 层本来就是快记忆（约 6 个 token）。所以钝的地址没有让快记忆变好，模型只是把长记忆搬到 ReLU² 的层。跟 U-Delta-learnP 学出来的 p 都 ≥ 2 一致：前几层不需要比较钝的地址。

### U-Delta+fast2：U-Delta+slow 的同容量对照

把 U-Delta+slow 的慢记忆换成第二份一般 U-Delta 记忆，状态一样大，但没有快慢之分，看 U-Delta+slow 的进步是分工还是容量。

- **对照**：U-Delta+slow（同一个 seed）
- **问题**：U-Delta+slow 比 U-Delta 好 0.009，但它同时多了一份状态。同容量对照能分开两种解释。
- U-Delta+slow：一份 U-Delta 记忆（依内容的 gate、从 1/(4L) 开始）＋一份慢记忆（固定速率、从 1/(16L) 开始），用 ρ 混合。
- U-Delta+fast2：两份都是 U-Delta 记忆，各自有 gate 和写入强度 β，一样用 ρ 混合。
- 状态都是 4.92M；U-Delta+fast2 参数多 1.8%（第二份 gate）。
- **设置**：参数：15.75M（U-Delta+slow 15.47M）；状态：4.92M（同 U-Delta+slow）
- **代码**：`scripts/delta.py` 里的 `TwoFastDeltaBlock`

```text
y = μ r + (1 − ρ) E₁ᵀ q + ρ E₂ᵀ q,   ρ = sigmoid(W_ρ h)
U-Delta+slow：E₂ 的遗忘率固定、很慢；U-Delta+fast2：E₂ 跟 E₁ 一样由内容决定
```

**结论**：多一份容量没有用：U-Delta+fast2（3.897）跟 U-Delta 一样，U-Delta+slow（3.888）比它好 0.009（约 U-Delta seed 标准差的 3 倍，各 1 个 seed）。所以 U-Delta+slow 的进步不是容量。不是容量的话，原本推测是那份几乎不忘的慢记忆；但 U-Delta-spreadλ（3.890）和 U-Delta-λlr10（3.887）用单一记忆就追平 U-Delta+slow，而且 U-Delta-λlr10 没有慢 slot，所以慢记忆是不是原因还不确定。 每层记忆长度：第 3 层拉到约 1,600 个 token，其他层跟 U-Delta 差不多。

### U-Delta+slow 的记忆任务 m2kv

U-Delta+slow 在记忆任务上跑跟其他模型一样的设置，看多一份慢记忆在纯记忆题上有没有差。

- **对照**：U-Delta（同一个任务、同样设置）
- **问题**：U-Delta 已经外推到 4,096。U-Delta+slow 的慢记忆几乎不忘，可能在更长的间隔（20,000）比 U-Delta 撑得久。
- 宽度 32、4 个 block、40,000 步、训练时间隔最多 256，跟其他模型一样。
- 只用 CPU 训练。
- **设置**：宽度 32、1 个 seed
- **代码**：`scripts/memory/mqar.py --model=SD2`

**结论**：训练 40,000 步（第一次在约 29,500 步被 OOM 砍掉，从头重跑）。间隔 1,024 以内 100%，4,096 98.3%（U-Delta 100%），8,192 95.3%（U-Delta 95.4%），16,384 77.0%（U-Delta 55.1%），20,000 66.9%（U-Delta 33.9%）。间隔越长，U-Delta+slow 的慢记忆越有用：20,000 时答对率是 U-Delta 的两倍。但一般文本（FineWeb）读到 20K 时 U-Delta+slow 比 U-Delta 差（16K–20K 3.856 对 3.816）：纯记忆题需要把很远的东西留住，一般文本不太需要，反而被占位置。1 个 seed。

### 记忆任务 m2kv 拉到间隔 20,000

U-Delta 在间隔 4,096 已经 100%，把间隔拉到 8,192、16,384、20,000，看记忆什么时候开始饱和或被洗掉。

- **对照**：同一批训练好的模型（训练时间隔最多 256）
- **问题**：线性注意力类的记忆外推不是自动就好：状态可能饱和、衰减可能把要找的东西洗掉。U-Delta 在 4,096 还完美，要看它的极限在哪。
- 不重训：直接加载原本的权重，用原本测试集的抽样种子，每个间隔 256 题。
- 题目：开头 16 组钥匙→值，中间塞 G 个也是钥匙或值的填充 token，最后问那 16 个钥匙。
- 只评 U-Delta、U-Gate、U-Gate-fastinit；U-Avg 和 Transformer 在间隔 1,024 就已经失效。
- **设置**：宽度 32、1 个 seed；在 CPU 上跑（越长越慢）
- **代码**：`scripts/memory/mem_long.py`

**结论**：U-Delta 外推得很远但不是无限：间隔 4,096 100%、8,192 95.4%、16,384 55.1%、20,000 33.9%（训练时最多 256）。U-Gate 在 4,096 只剩 39.5%，8,192 3.7%，20,000 0.3%。U-Gate-fastinit 在 4,096 已经是 0%（8,192 0.05%），所以更长的不跑。U-Delta 是唯一撑过训练长度 30 倍的。

### U-Delta 放大到 42M

同一个 U-Delta 架构，宽度和层数放大到 42M，看 15M 赢 Transformer 的优势会不会随规模消失。

- **对照**：Transformer 42M（同样的数据、步数和学习率）
- **问题**：U-Delta 在 15M 赢 Transformer 0.048。如果到 42M 输 Transformer 0.05 以上，就代表优势只在小规模成立；如果还赢或差不多，统一 block 的主张就强很多。42M 目前 Transformer 领先 U-Gate 0.116，这是要追的距离。
- 架构跟 15M 的 U-Delta 完全一样，只改尺寸：d 288→512、block 12→16、slot 712→1,248。
- slot 数 1,248 是为了让参数跟 Transformer 42M 对齐（只多 0.2%）。
- 训练 6,400 步、每步 13 万 token（约 8.4 亿 token），跟 Transformer 42M、U-Gate 42M 一样。
- **设置**：参数：约 41.8M（Transformer 42M 41.69M）；状态：每条串行 1,248 × 512 × 16 ≈ 10.2M 个数（固定）；GPU：A10，一步约 6.5 秒
- **代码**：`scripts/delta.py` 里的 `DeltaBlock（跟 15M 同一個類別）`

**结论**：优势放大后消失：U-Delta 42M 3.350，输 Transformer（3.214）0.136，也输 U-Gate（3.330）。差距在每个位置都差不多（前 64 个 token ＋0.129，512–1K ＋0.135），所以不是上下文记忆，而是基本建模能力放大后落后。第 500 步时 U-Delta 还领先，1,500 步之后被 Transformer 超过。超过训练长度仍然稳（2K–4K 3.286，Transformer 崩到 5.79）。1 个 seed、没调学习率；下一步要分开学习率和 FFN 那半的影响。 读到 20K：U-Delta 42M 不崩（16K–20K 3.269），但开窗口的 Transformer 是 3.142，差距跟 1,024 内一样。每层记忆长度：12、18、200、509、167、133、27、22、12、20、42、16、16、323、22、626 个 token。

### Transformer+conv：Transformer 加上 U-Delta 的卷积

在 Transformer 的注意力前面加上跟 U-Delta 一模一样的短卷积，看 U-Delta 赢 Transformer 是不是只是因为多了卷积。

- **对照**：Transformer（同一个 seed）
- **问题**：U-Delta 的地址前面有卷积，Transformer 没有。卷积本身让每个 token 混到前几个 token 的信息，可能就足以解释 U-Delta 的优势。
- 每个注意力半层：范式后先过一个宽 4 的因果 depthwise 卷积，再算 q、k、v。
- 卷积初始化成恒等（一开始等于没有卷积），没有激活函数，跟 U-Delta 的卷积完全相同。
- 其他（RoPE、FFN、宽度）都跟 Transformer 一样，参数只多 6.9K。
- **设置**：参数：15.20M（Transformer 15.19M）；状态：跟 Transformer 一样，KV cache 随长度增加
- **代码**：`scripts/variants.py` 里的 `AttentionHalf(conv=True)、CausalConv`

```text
a_t = conv4(RMSNorm(x))_t        # 看前 3 个 token 加自己
q, k, v = W_q a_t, W_k a_t, W_v a_t  # 之后跟 Transformer 一样
```

**结论**：卷积解释不了 U-Delta 的优势。Transformer+conv 只比 Transformer 好 0.007，大约是 U-Delta 领先幅度（0.031）的四分之一，而且在 Transformer 三个 seed 的标准差（0.016）之内；前 64 个 token 完全没改善。

### GDN-bigstate：状态放大 3 倍的 GDN

我们自己做的 GDN 变体（不是 NVIDIA 的 GDN-2）：把 6 个 48 维的 head 改成 2 个 144 维，状态变 3 倍，看 GDN 输 U-Delta 是不是因为状态太小。

- **对照**：GDN（6 head × 48，同一个 seed）
- **问题**：U-Delta 每条串行的状态约 2.46M 个数，GDN 只有 83K。U-Delta 赢 GDN 可能只是因为能存的东西多 30 倍。
- head 数 6→2，每个 head 的维度 48→144。每个 head 的状态是 d_k × d_v，所以总状态 6×48² → 2×144²，变 3 倍。
- 衰减 log α 的下限从硬性截断（−2）改成平滑的 2·expm1(log α / 2)，让靠近下限的 head 还有梯度。
- FFN 宽度 672，参数 15.22M，跟 GDN 对齐。
- **设置**：参数：15.22M（GDN 15.23M）；状态：249K（GDN 83K，U-Delta 2.46M）
- **代码**：`scripts/variants.py` 里的 `GatedDeltaHalf(soft=True)，訓練時 --head_dim=144`

```text
S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ,   o_t = S_tᵀ q_t
每个 head：U-Avg 是 144 × 144（GDN 是 48 × 48）
```

**结论**：状态放大 3 倍完全没进步（3.964 对 3.964，各位置差都在 0.01 内）。所以「GDN 缺状态」的说法变弱了，但还不能排除：U-Delta 的总状态仍是 GDN-bigstate 的 10 倍。

### GDN-2：NVIDIA Gated DeltaNet-2

GDN 的后续版本：把写入强度 β 拆成抹除和写入两个逐信道的 gate，衰减也逐信道。

- **对照**：GDN（同一个 seed）
- **问题**：论文（Hatamizadeh、Choi、Kautz，NeurIPS 2026）在 1.3B、100B token 赢 GDN 和 KDA，长文检索进步最多。在我们的规模是否也成立，它会不会是最强的基准。
- GDN：一个纯量 β 同时决定抹掉多少、写进多少；衰减 α 是每个 head 一个纯量。
- GDN-2：抹除 gate b（钥匙的每一维）、写入 gate w（值的每一维）分开；衰减 α 是每个钥匙维度一个（沿用 KDA）。
- 跟论文不同：三个 gate 用低秩投影（秩 48，照 fla 的 KDA）；FFN 宽度 672→576 让参数跟 GDN 对齐。
- **设置**：参数：15.21M（GDN 15.23M）；状态：83K（同 GDN）
- **代码**：`scripts/variants.py` 里的 `GatedDeltaNet2Half、erase_write_memory`

```text
S_t = (I − k_t (b_t ⊙ k_t)ᵀ) diag(α_t) S_{t−1} + k_t (w_t ⊙ v_t)ᵀ
b_t = σ(W_b h), w_t = σ(W_w h), log α_t = −exp(a) · softplus(W_f h + δ)
```

**结论**：在 15M 没有进步：3.970 对 GDN 3.964，在杂讯内，各位置几乎重叠。规模（15M、3 亿 token 对 1.3B、100B）、低秩 gate、FFN 变窄都可能是原因，所以不代表论文错，只说明在这个规模把抹除和写入拆开没有帮助。对 U-Delta 的意义：近距离的弱点不是因为 β 只有一个纯量。

### Mesa：Mesa layer

每个 token 都精确解一次加权最小平方的记忆。GDN 和 U-Delta 的 delta rule 每个 token 只往这个解走一步。

- **对照**：GDN（同一个 seed）
- **问题**：blog 说最小平方是「最佳线性记忆」，容量约 2d（一般外积记忆约 d）。如果 Mesa 明显赢 GDN，代表 delta rule 一步近似浪费了容量，U-Delta 也可能从精确解受益。
- 跟 GDN 一样是把 Transformer 的注意力换掉，q、k、v、卷积、输出 gate 都相同。
- 记忆不是一步步更新 U-Avg，而是存 H（kkᵀ 的加权和）和 G（vkᵀ 的加权和），读取时解 (H + λI) x = q。
- 用共轭梯度法解 24 步（相关钥匙的难例误差约 1e-3），梯度用隐式微分，不需要存下每个 token 的 H。
- **设置**：参数：15.24M（GDN 15.23M）；状态：166K（H 和 G 各一份，GDN 83K）；速度：一步约 10 秒，是 GDN 的约 4 倍
- **代码**：`scripts/mesa.py` 里的 `MesaHalf、MesaSolve、gla`

```text
H_t = γ_t H_{t−1} + β_t k_t k_tᵀ,   G_t = γ_t G_{t−1} + β_t v_t k_tᵀ
o_t = G_t (H_t + diag(λ))⁻¹ q_t
```

**结论**：没有比 GDN 好：Mesa 3.969，GDN 3.964、GDN-2 3.970，都输 Transformer（3.928）和 U-Delta（3.897）。近处（前 64 个 token）4.188 比 GDN 4.166 还差；训练时间约 GDN 的 4 倍。在 15M，把 delta rule 的单步近似换成每个 token 精确解最小平方没有帮助，U-Delta 赢 GDN 的原因不在写入规则多精确。1 个 seed。

### Transformer+SSMax：Transformer 加 Scalable-Softmax，以及 log-n 缩放

让 Transformer 读超过训练长度时注意力不会被摊平，给 Transformer 一个公平的长文本对照。

- **对照**：Transformer（同一个 seed）
- **问题**：纯 RoPE 的 Transformer 读到 2 倍长度以上就崩（20K 时 NLL 6.05），拿它跟 U-Delta 比是稻草人。滑动窗口已经是一个公平的对照；SSMax 和 log-n 测的是能不能让 Transformer 用完整上下文也不崩。
- Transformer+SSMax（要重训）：一个看得到 n 个 key 的 query，logits 乘上 s·log n；s 每个 head 可学，从 1/log 1,024 开始，所以在 n＝1,024 时等于普通 softmax。
- log-n 缩放（不用重训，Qwen 的做法）：只在评估时，超过训练长度 L 之后 query 乘上 log n / log L；在 L 内输出完全不变。
- 两者都不改参数量（Transformer+SSMax 每层多 6 个数）。
- **设置**：参数：跟 Transformer 一样；状态：跟 Transformer 一样，KV cache 随长度增加
- **代码**：`scripts/variants.py` 里的 `AttentionHalf(ssmax=True)、AttentionHalf._scale、LOGN；scripts/fineweb/long_eval.py --logn`

```text
Transformer+SSMax：softmax(s · log n · q kᵀ / √d)
log-n：q_t ← q_t · max(1, log n_t / log L)
```

**结论**：训练长度内 Transformer+SSMax 比 Transformer 好 0.015（3.913 对 3.928，在 Transformer 的 seed 标准差附近，1 个 seed），仍输 U-Delta（3.897）。超过训练长度照样崩，比 Transformer 更差：读到 20K 的 16K–20K 段 Transformer+SSMax 6.52、Transformer 6.05（3 个 seed）、U-Delta 3.82。SSMax 解决的是注意力被摊平，不是 RoPE 没看过的位置。Transformer 的 log-n 缩放也没用（16K–20K 6.19）。Transformer+SSMax 加窗口 1,024 是 3.857，比 Transformer 加窗口（3.893）好一点。

