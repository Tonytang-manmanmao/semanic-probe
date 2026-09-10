# Semantic-Probe: 超边驱动源消融（Route B · soft-hyperedge）

> MGSSM 双路线第一阶段 · **路线 B（SSM → HyperGraph）** · PEDRo 事件相机行人检测
> 探针设计：**四源消融**，唯一变量 = 超边归属矩阵 `H` 的输入

---

## 1. 这个探针在问什么

路线 B 的核心主张是：**用 SSM 的隐状态去构造超图**（`H = AdaptiveHyperedge(h_ssm)`）。

为了判断这个主张是否成立，我们做了一个**四源消融**：把超边归属矩阵 `H` 的输入分别换成四种特征，**其它一切不变**（同一训练脚本、同一 LR 0.01、同一数据划分、同一预算）。

| 源 | `H` 的输入 | 含义 |
|---|---|---|
| `ssm` | SSM 隐状态 `h_ssm` | 路线 B 的原始主张 |
| `semantic` | 图节点特征 `data.x` | 通用特征基线 |
| `motion` | 位置/时间 `[x, y, t]` | 物理先验 |
| `joint` | `concat(h_ssm, data.x, motion)` | 三者联合 |

> 探针只做**方向筛选**（3000 步 ≈ 10 epoch），不做有效性结论。

---

## 2. 结果（epoch 5，val 集，seed 42）

| 源 | epoch0 mAP | **epoch5 mAP** | AP50 | AP75 |
|---|---:|---:|---:|---:|
| B0（无超图对照） | .2209 | .4090 | 未记录 | 未记录 |
| `ssm` 探针 | .1978 | .3998 | .8241 | .3093 |
| **`semantic` 探针** | .1828 | **.4220** | **.8269** | **.3644** |

**Δ（epoch5 mAP）**

- `semantic` − B0 = **+1.30pp** ← 唯一越过 +1pp 门槛的配置
- `semantic` − `ssm` = **+2.22pp**
- `ssm` − B0 = **−0.92pp**

**AP50 / AP75**（B0 未记录，只能 ssm vs semantic 对比）

- AP50：`.8269` vs `.8241` → **+0.28pp**，基本持平（检出目标数量不变）
- AP75：`.3644` vs `.3093` → **+5.51pp**，明显提升（框贴得更紧）

→ **增益性质**：AP50 持平 + AP75 大涨 = **定位收紧**，而不是"检出更多目标"。

---

## 3. 结论

1. **超边操作有早期正信号**：`semantic` 驱动超边比 B0 高 **+1.30pp**。
2. **SSM 隐状态是坏的超边源**：比 `semantic` 低 2.22pp，比 B0 低 0.92pp
   → **路线 B「用 SSM 隐状态驱动超图」的核心主张在软超边版本下被否定**。
3. 应转向 **`semantic` / `motion` 驱动超边**。

> 诚实版汇报措辞：
> 在软关联（softmax 软超边）+ 门控残差版本下，SSM 隐状态作为超边归属来源，epoch5 已低于语义特征来源 2.22pp（AP75 差 5.5pp）、低于无超边 B0 基线 0.92pp。

---

## 4. 方法细节

```
graph_stage2 之后插入：

  h_ssm = NodeSSM(x, t)                    # 节点级选择性扫描（Mamba S6）
  h_src = <source>(h_ssm, x, pos, t)       # ★ 探针唯一变量
  H     = AdaptiveHyperedge(h_src)         # 软超边归属矩阵 [N, num_hyperedges]
  h_hyp = HyperConv(x, H)                  # 超图卷积 V -> E -> V（HGNN 式）
  out   = h_ssm + h_hyper                  # 残差融合
```

- **软超边**：`H = softmax(proj(h_src) / temperature)`，**无硬 top-k 选择**；`H` 按行归一化（每个节点在超边上分配单位质量）。
- **门控残差**：`data.x = h_ssm + h_hyper`。
- **三项软正则**（加进 `total_loss`）：
  - `ssm_hyper_entropy_weight = 0.02` —— 超边负载均衡（**抗坍缩**）
  - `ssm_hyper_sharpness_weight = 0.02` —— 抑制单节点在超边上均匀散开
  - `ssm_hyper_consistency_weight = 0.1` —— 同一超边内节点特征拉近

配置见 `configs/probe-{ssm,semantic,motion,joint}-p3000-seed42.yaml`。

---

## 5. 目录结构

```
.
├── README.md
├── code/
│   └── dagr_env/src/dagr/model/
│       ├── layers/ssm_hyper.py        # NodeSSM / AdaptiveHyperedge / HyperConv / 正则
│       └── networks/
│           ├── net_vss.py             # 插入点 + hyperedge_source 开关（探针唯一改动）
│           └── dagr_vss.py            # 三项正则接入 total_loss
├── configs/                           # 四份探针配置（各 3000 步 / seed 42）
├── tests/
│   └── test_source_switch.py          # 四源形状/行为自检（CPU，合成张量，无需数据集）
├── results/
│   └── probe_results.md               # 结果与判读
└── docs/
    └── 探针设计与结论.md                # 汇报用整理稿
```

自检（无需数据集，CPU 上秒级跑完）：

```bash
python tests/test_source_switch.py
```

```
  ssm       in_dim= 64  H=(97, 16)  row_sum~1  loss=+0.0691  OK
  semantic  in_dim= 64  H=(97, 16)  row_sum~1  loss=+0.0681  OK
  motion    in_dim=  3  H=(97, 16)  row_sum~1  loss=+0.0540  OK
  joint     in_dim=131  H=(97, 16)  row_sum~1  loss=+0.0707  OK
```

---

## 6. 复现方式

本仓库**只包含探针相对基础版的那部分改动**（一个源码文件的小改 + 配置）。

```bash
# 1) 取基础工程
git clone https://github.com/Tonytang-manmanmao/ssm-hypergraph.git
cd ssm-hypergraph
git checkout 2a5ffbf        # 探针所基于的提交

# 2) 覆盖探针版源码（本仓库 code/ 下的 3 个文件）
cp -r <this-repo>/code/dagr_env/src/dagr/model/layers/ssm_hyper.py \
      dagr_env/src/dagr/model/layers/
cp -r <this-repo>/code/dagr_env/src/dagr/model/networks/net_vss.py \
      dagr_env/src/dagr/model/networks/
cp -r <this-repo>/code/dagr_env/src/dagr/model/networks/dagr_vss.py \
      dagr_env/src/dagr/model/networks/

# 3) 放配置并训练（以 semantic 源为例）
cp <this-repo>/configs/*.yaml .
python train_pedro_baseline.py \
  --config probe-semantic-p3000-seed42.yaml \
  --dataset_directory . --output_directory output
```

**环境**：Python 3.13.9 / PyTorch 2.6.0+cu124 / CUDA 12.4 / PyG 2.8.0 / RTX 4060 Laptop 8GB（单卡）。
**数据**：PEDRo，`numpy/{train,val,test}` + `yolo/{train,val,test}`（train 19228 / val 3950 / test 3823）。

---

## 7. ⚠️ 诚实声明（重要）

1. **本仓库是重建版，不是当年跑出结果的原始代码。**
   原始探针代码在工程整理阶段被删除，且**从未提交过 git**（`git stash` 与 `git fsck --lost-found`
   的 dangling 对象里均无残留），因此无法逐字节还原。本仓库按 `docs/探针设计与结论.md` 记录的
   设计**重建**：保留"唯一变量 = `H` 的输入"这一关键性质，其余模块与基础版一致。
2. **第 2 节的所有数值来自原始运行的记录**（epoch5 val mAP / AP50 / AP75），不是重建后跑出来的。
   重建版尚未重跑验证。
3. **探针只判方向，不判有效性**：3000 步 ≈ 10 epoch，单 seed（42），落在 ±2pp 噪声带内；
   `+1.30pp` 只能说明"语义源是值得继续的方向"，**不是**已验证的收益。
4. **`motion` / `joint` 两源没有记录结果**——原始记录只留下了 `ssm` 与 `semantic` 两行。
5. `semantic` 源的 `H` 输入维度 = `node_dim`；`motion` 为 3（`[x,y,t]`）；`joint` 为 `2*node_dim+3`。
   原实现如何把 3 维 `motion` 升到 `num_hyperedges` 已不可考，本重建统一用 `nn.Linear(in_dim, num_hyperedges)`，
   等价于"仅改变 `H` 的输入"。

---

## 8. 后续（当时记录的方向）

- 选项 1（推荐）· 转向 `semantic` / `motion` 驱动超边，跑完剩下两个源、补多种子
- 选项 2（不优先）· 回到 SSM 源，改硬选择

> 实际上后续路线改为「去超图、只留图节点双向 Mamba」，本探针的线索（语义源有效）未被继续。
