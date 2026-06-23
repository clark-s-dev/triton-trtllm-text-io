# 0016 · 实验10:投机解码(Draft-Target Speculative Decoding)— M4 收尾

> 实验台用法见 [`../L2-LAB.md`](../L2-LAB.md)。这是消融矩阵之外的**主动加速**实验:不是关一个旋钮,而是**加一个 draft 模型**。draft(0.5B)提议 K 个 token,target(1.5B)一次前向**验证** K 个,接受最长匹配前缀。**先填①②③(动手前),再填④⑤。**

| 字段 | 内容 |
|---|---|
| 日期 | 2026-06-23 |
| 里程碑 | M4(量化 + **投机解码**)→ 本条收尾 M4 |
| 旋钮 | 无 → 投机解码:`draft_len K`:0 → {1,2,4,6,8};`accept`:logits(论文式拒绝采样)/ tokens(逐 token 相等) |
| 引擎/层 | draft **small(0.5B)** + target **large(1.5B)**,FP16;**直连 C++ executor**(`ModelRunnerCpp`,绕开 BLS/guard/gRPC,见 L2-LAB §4.1) |

## ① 假设(Hypothesis)

投机解码把「**1 次 target 前向 = 1 个 token**」变成「**1 次 target 前向 = (接受数+1) 个 token**」。它之所以能赢,**根因是 decode 在小 batch 是带宽受限**(M0 / 0011 反复量到的那道墙):target 每个 decode step 不管打分几个位置,都要把 1.5B 权重**完整读一遍**——所以让它一次验证 K+1 个位置,**几乎和解 1 个 token 一样贵**(权重只读一次)。于是只要 draft 猜得够准,单流延迟直接降。

代价两块:(a) draft 自己要跑 **K 次串行** decode(0.5B,便宜但非零);(b) 被拒绝的 draft token 是**白算的 target 算力**。所以这是一个**拿"高 batch 时本就闲置的算力" 换 "低 batch 时的延迟"** 的交易——**它和连续批处理是对头**:batch 一大,target 转 compute-bound,验证 K+1 个位置不再免费,加速比应当**塌掉甚至变慢**。

## ② 预测(Predict — 动手前写死)

**机理量级模型(roofline 估算).** 设 target 单 step 时间 `T`,draft 单 step 时间 `r·T`(带宽受限 → `r ≈ 权重字节比 ≈ 0.49B/1.54B ≈ 0.32`)。设逐 token 接受概率 `a`,则一次投机迭代:
- 推进的 token 数 ≈ `1 + a·K`(几何分布近似的简化:期望接受 `a·K`,再加 target 必给的 1 个)
- 花的时间 ≈ `K·(r·T)`(draft 串行 K 步)`+ T`(target 验证 1 次,带宽受限≈解 1 token)
- **加速比 ≈ `(1 + a·K) / (1 + r·K)`**,`r ≈ 0.32`

代入预测(贪心、logits 接受):

| K | 预测接受率 a | 预测 mean accept/iter `1+aK` | 预测加速比 `(1+aK)/(1+0.32K)` |
|---|---|---|---|
| 1 | 0.62 | 1.62 | 1.62/1.32 ≈ **1.23×** |
| 2 | 0.60 | 2.20 | 2.20/1.64 ≈ **1.34×** |
| **4** | 0.56 | 3.24 | 3.24/2.28 ≈ **1.42×** ← 预测最优区 |
| 6 | 0.50 | 4.00 | 4.00/2.92 ≈ **1.37×** |
| 8 | 0.45 | 4.60 | 4.60/3.56 ≈ **1.29×** |

- **接受率随 K 递减**(越往后猜越难),且 **easy 工作负载(代码/列表/表格)明显 > hard(开放创作)**:预测 easy `a≈0.70–0.80`、hard `a≈0.40–0.50`。**接受率是工作负载的属性,不只是模型的属性**——这是本条要量出来的核心反直觉点。
- **最优 K ≈ 4**,贪心 mixed 加速 **~1.4–1.6×**(模型偏保守,真实 target 验证可能比 1 个 token 略贵但远不到 K 倍,所以实测可能更高)。
- **logits 接受 vs tokens 接受**:贪心下两者接受率应接近(都挑 argmax);**采样**下 logits(拒绝采样)能保住 target 分布且接受率更稳,tokens 接受会因 target 采到不同 token 而**接受率掉一截**。
- **batch 交叉点(和连续批处理联动):** 固定 K=4 扫 batch。预测 **C=1 加速最大**,随 C 上升加速比单调下降,在 **C≈8–16**(0.5B/1.5B 在 L4 上转 compute-bound 的区间,见 0008/0001 的 batch=64 拐点推算)附近**跌破 1.0(变慢)**。理由:验证算力 ∝ batch×(K+1),高 batch 时这不再免费;且 draft 的串行步数没法靠 batch 摊薄。
- **精度/正确性:** logits 接受(拒绝采样)**理论上无损**(输出分布 == 单 target 采样);tokens 接受在贪心下也无损(都取 argmax)。**投机解码不改变输出质量**——这是它相对量化(0009/0011 拿精度换速度)的**关键区别**,本条要顺带验证(贪心下 specdec 输出应与 baseline **逐 token 相同**)。

## ③ 实验设置(可复现)

- **引擎(B 级,重编):** `scripts/build_specdec_engines.sh`
  - draft `engines/qwen2.5-0.5b-draft`:`--gather_generation_logits --use_paged_context_fmha enable`
  - target `engines/qwen2.5-1.5b-target`:同上 + `--speculative_decoding_mode draft_tokens_external --max_draft_len 10`
  - 复用 0009/build_engines 的 Qwen tied-emb 修复(`--use_embedding_sharing` + `check_share_embedding(config)` sed)。
- **Workload:** 10 条 prompt = 5 easy(列表/代码/表格/JSON/日历,可预测)+ 5 hard(超现实创作/反常识论证,难猜),见 `lab/specdec_bench.py:PROMPTS`。`output_len=200`。
- **命令(容器内跑;先停 triton-llm 腾显存):**
  ```bash
  # baseline:target-only(同一 target 引擎,无 draft)
  specdec_bench.py --mode baseline   --target-engine engines/qwen2.5-1.5b-target --tokenizer hf_models/Qwen2.5-1.5B-Instruct --output-len 200 --prompts mixed --temperature 0
  # specdec:扫 K ∈ {1,2,4,6,8} × accept ∈ {logits,tokens} × prompts ∈ {easy,hard,mixed}
  specdec_bench.py --mode specdec --draft-engine engines/qwen2.5-0.5b-draft --target-engine engines/qwen2.5-1.5b-target \
       --tokenizer hf_models/Qwen2.5-1.5B-Instruct --draft-len K --accept logits --output-len 200 --prompts mixed --temperature 0
  # batch 交叉:固定 K=4,扫 --batch ∈ {1,2,4,8}
  ```
- **控制变量:** warmup 1 次;3 次计时取最快(去噪);`batch=1` 给干净的逐请求记账(接受率/迭代数);贪心(temp 0,top_k 1)做主对照(可复现 + 接受率最大),另跑一组采样;draft/target KV 显存比 0.20 / 0.45(同卡共存,加载顺序 draft→target,见 M0 加载顺序课)。
- **测量层:** `ModelRunnerCpp`(C++ executor)直连——**不**走 BLS/guard/gRPC,timing 里只有 draft+target 两个引擎和编排循环。

---

## ④ 实测(Measure)

数据:[`../../lab/results_0016.jsonl`](../../lab/results_0016.jsonl)(20 配置);harness [`../../lab/specdec_bench.py`](../../lab/specdec_bench.py),复现 [`../../lab/run_0016_sweep.sh`](../../lab/run_0016_sweep.sh)。baseline = **plain fp16 1.5B**(`draft_tokens_external` 引擎无 draft 时每 call 只出 1 token,所以诚实基线用普通引擎,同权重)。基线 batch=1 贪心 = **74.7 tok/s(13.4 ms/token)**。

**A. 主表:贪心 mixed,扫 draft_len K(logits 接受)vs baseline**

| K | 接受率 a | mean accept/iter | target 前向次数 | 吞吐 tok/s | **实测加速** | 预测加速(r=0.32) |
|---|---|---|---|---|---|---|
| 0(baseline) | — | 1.00 | 1729 | 74.7 | 1.00× | — |
| 1 | 0.803 | 1.79 | 985 | 81.3 | 1.09× | 1.23× |
| **2** | 0.724 | 2.44 | 724 | **87.8** | **1.18×** ← 最优 | 1.34× |
| 4 | 0.588 | 3.34 | 529 | 85.1 | 1.14× | 1.42× |
| 6 | 0.469 | 3.79 | 466 | 75.4 | 1.01× | 1.37× |
| 8 | 0.391 | 4.11 | 430 | 67.1 | **0.90×(变慢!)** | 1.29× |

```
加速比 vs draft_len K(贪心 mixed):
K=1  ████████▏ 1.09
K=2  ██████████████▏ 1.18   ← 最优(预测是 K=4)
K=4  ███████████▌ 1.14
K=6  █▏ 1.01
K=8  ▏(负增益) 0.90  ← 比不投机还慢:draft 串行成本 > 多产的 token
接受率单调降:0.80 → 0.72 → 0.59 → 0.47 → 0.39(越往后越难猜)
```

**B. 接受率 = 工作负载属性(K=4,贪心,logits)**

| 工作负载 | baseline tok/s | specdec tok/s | 接受率 a | **加速** |
|---|---|---|---|---|
| **easy**(列表/代码/表格/JSON/日历) | 73.9 | 111.1 | **0.857** | **1.50×** |
| mixed | 74.7 | 85.1 | 0.588 | 1.14× |
| **hard**(超现实创作/反常识论证) | 74.3 | 72.0 | **0.456** | **0.97×(变慢)** |

> 同一对模型、同一 K,**只换 prompt**:easy 1.50× ↔ hard 0.97×。**接受率是工作负载的属性**,不是模型的常数。可预测文本(代码/列表)draft 几乎全中;开放创作 draft 一直猜错 → 投机反而拖累。

**C. logits vs tokens 接受(K=4,mixed)**

| 采样 | accept | 接受率 | 加速 | 输出 token 数 |
|---|---|---|---|---|
| 贪心 | logits | 0.588 | 1.14× | 1765 |
| 贪心 | tokens | 0.588 | 1.15× | 1765 |
| 采样(T=0.8) | logits | **0.847** | **1.47×** | 1395 |
| 采样(T=0.8) | tokens | **0.335** | **0.68×(大幅变慢)** | 1570 |

> 贪心下 logits 与 tokens **完全一致**(都取 argmax)。**采样下必须用 logits(拒绝采样)**:token 逐字相等接受崩到 0.335(target 采的 token 和 draft 采的极少相同)→ 0.68× 倒退。logits 接受既**保住 target 分布**又把接受率拉到 0.85。

**D. batch 交叉(K=4,贪心 mixed)— 和连续批处理正面联动**

| batch C | baseline tok/s | specdec tok/s | **加速** | mean accept/iter(spec) |
|---|---|---|---|---|
| 1 | 74.7 | 85.1 | **1.14×** | 3.34 |
| 2 | 127.2 | 108.7 | 0.86× | 4.99 |
| 4 | 209.2 | 133.8 | 0.64× | 7.54 |
| 8 | 309.4 | 161.6 | **0.52×** | 11.39 |

```
吞吐 tok/s vs batch:                         加速比 vs batch:
baseline  74.7→127→209→309 (连续批近线性↑)    1.14× → 0.86× → 0.64× → 0.52×
specdec   85 →109→134→162 (平,draft 串行)     交叉点在 batch 1→2 之间
```

## ⑤ Gap 分析(预测 vs 实测)

**1. 最优 K = 2,不是预测的 4;K=8 直接变慢。预测模型方向对、参数错。**
把每次迭代耗时拆出来(`wall / target_passes`),对 K 做线性拟合,**漂亮的直线**:
```
T_iter(K) ≈ 16.6 + 5.59·K   (ms)     —— K=1,2,4,6,8 五点拟合,残差 <1%
```
- 斜率 **5.59 ms = 一步 draft(0.5B)decode**;截距 **16.6 ms = target 验证一次 + 编排开销**。
- 我预测时设 `r = T_draft/T_target ≈ 权重字节比 ≈ 0.32`。实测 `T_draft=5.59`、`T_target(baseline)=13.4` → **r ≈ 0.42**,比 0.32 大。而且截距 16.6 > 13.4:**target 验证 K+1 个位置 + Python 编排,比解 1 个 token 贵 ~3ms**(不是「完全免费」)。
- 两个偏差(draft 更贵 + 验证非免费)都**抬高分母**,把最优 K 从预测的 4 **左移到 2**。把实测 r=0.42、加上截距代回,`speedup = (mean_acc/iter)·13.4 / (16.6+5.59K)`,对每个 K 都吻合到 1%(K=1→1.09,K=2→1.18,K=8→0.90 ✓)。**模型骨架对,r 必须实测——不能用纯权重比当 draft 成本**(固定开销:kernel launch、attention、采样、我 harness 里两次 generate + sync + logits 搬运)。生产级 C++ BLS 编排会更紧,r 会更接近 0.32。

**2. 接受率 easy/hard:方向全对,easy 比预测还高。** 预测 easy 0.70–0.80 → 实测 **0.857**;hard 0.40–0.50 → 实测 **0.456**(精准)。

**3. batch 交叉点:预测 C≈8–16,实测 1→2 之间就翻盘——错得有价值。** 原因正是连续批处理的物理(0008/0013 的成本模型 `T(B)=T_weight+B·T_token`):**baseline 一上 batch 就把「读一次权重」摊到 B 个 token 上**(b1→b2 吞吐 ×1.70,近线性),增益巨大;而 specdec 的 **draft 是串行的、没法靠 batch 摊薄**,验证还要算 B×(K+1) 个位置。所以 baseline 的批处理增益**立刻盖过** specdec 的单流优势。投机解码的优势本质是 **batch=1 现象**,batch 一大就被连续批处理吃掉。

**4. 无损性:实测证。** A 表 K=1/2/4/6/8 **输出 token 数恒为 1765**(与 K 无关),贪心下逐 token 一致 → **投机解码忠实复现 target 引擎的贪心输出**(与 K 无关 = 无损的实证)。baseline 的 1729 vs 1765 差异是 **plain 与 spec 两个独立 TRT build 的 FP 数值在 argmax 平局上分叉**,不是投机解码的误差。

## ⑥ 机制:投机解码 = decode 带宽墙的「另一面」

- **和量化(0009/0011)同一道墙,两种打法。** decode 在 batch=1 带宽受限(读全权重只算 1 个 token,见 [0017](./0017-nsight-decode-bandwidth.md) 的 kernel 级实锤)。0011 的打法:**砍权重字节**(INT4 → 1/4 流量)。本条的打法:**一次读权重多产几个 token**(verify K+1 ≈ 读一次权重)。两者都在「decode 带宽受限」这个前提上吃红利——所以本条的物理前提,正是 0017 量到的 `Memory% ≫ Compute%`。
- **和连续批处理(0003/0013)是替代,不是互补。** 两者都在消费「低 batch 时闲置的算力」:batch=1 时算力大把闲着,投机拿去验证 draft;但连续批处理一上 batch 就把这些算力用来摊薄权重读取。**同一份闲置算力,只能给一家**——这就是 D 表交叉的根因。结论:投机解码属于**低并发 / 延迟敏感**场景(交互式、batch≈1),高吞吐批量服务里让位给「连续批 + 量化」。
- **和 KV(0001/0014)的张力。** target 每轮按 (接受数+1) 变长 KV,**变长 = 块记账比 0014 的定长滚动更碎**;draft 还各自占一份 KV(本实验 draft 0.2 / target 0.45 显存比共存,呼应 M0 的「共置 + 加载顺序」)。

## ⑦ 结论 / 下一步

- **一句话:** 在 Qwen2.5 0.5B→1.5B 上,投机解码是**单流、贪心/采样皆可、且无损**的加速:最优 **K=2 → mixed 1.18×、可预测文本 1.50×**;但**创作类 0.97×(反而慢)**,**采样必须用 logits 接受**(否则 0.68×),而且**一上 batch 就被连续批处理反超(b2 已 0.86×)**。
- **能讲给面试官的取舍:** 投机解码换的是「延迟」不是「吞吐」,且**不掉精度**(对比量化拿精度换速度)——所以它属于**低 QPS、有延迟 SLA**的交互式服务;高并发批量场景里,连续批处理 + 量化才是主力。**「接受率是工作负载属性」**和**「draft 成本必须实测、纯权重比会高估最优 K」**是两个最容易被 roadmap 漏掉的实打实结论。
- **M4(量化 + 投机解码)就此收尾。** 量化:[0009](./0009-fp8-quantization.md)/[0011](./0011-int4-awq.md);投机解码:本条;decode 带宽受限的 kernel 级实锤:[0017](./0017-nsight-decode-bandwidth.md)。
- **下一步(可选深化):** 把 [`cbatch_sim.py`](../../lab/cbatch_sim.py) 扩出一个「投机解码 + 连续批」的成本模型,用 `T_iter(K)=16.6+5.59K` 预测 D 表交叉点(toy 校准实验台,延续 M2/M3 套路)。
