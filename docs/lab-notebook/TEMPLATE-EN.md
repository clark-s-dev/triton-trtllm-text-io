# NNNN · <one-line title: which knob / which question are you testing>

> 🌐 **中文版 (Chinese):** [`TEMPLATE-CN.md`](./TEMPLATE-CN.md)

> Lab usage: see [`../L2-LAB-EN.md`](../L2-LAB-EN.md). **Fill in ①② BEFORE you touch anything, then ④⑤.** The gap between ①② and ④⑤ is the learning signal.

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Milestone | M? |
| Knob | `<param>`: `<from>` → `<to>` (R/B) |
| Engine/layer | small (0.5B) / large (1.5B); direct-to-engine / via BLS |

## ① Hypothesis
<one line: if I turn this knob off / change it, what do I expect to move, and in which direction.>

## ② Predict (write it down BEFORE you act)
- **Number:** <predicted TTFT / ITL / throughput / concurrency / memory, with an order-of-magnitude estimate>
- **Mechanism:** <why this direction, this magnitude. Cite roofline / the KV formula / scheduler behavior.>

## ③ Setup (reproducible)
- **Workload:** concurrency C=?, input-len distribution?, output-len?, shared-prefix ratio? (designed specifically to expose this knob — see L2-LAB §4.2)
- **Commands:**
  ```bash
  # change param → restart → wait for ready → measure
  ```
- **Controls:** warmup ?; N=?; did you `docker stop triton-fused`?; all other knobs left at default.

## ④ Measure
| Metric | baseline (knob ON) | ablation (knob OFF/changed) | delta |
|---|---|---|---|
| TTFT P50 / P99 |  |  |  |
| ITL P50 / P99 |  |  |  |
| throughput (tok/s) |  |  |  |
| max concurrency / memory |  |  |  |
| corroborating (DCGM SM% / power, KV metrics) |  |  |  |

## ⑤ Gap analysis (predicted vs measured)
<did it match? if not, where's the gap? was my mechanism wrong, or did the real system do something I didn't model (chunked / recompute / fragmentation / gateway serialization)?>

## ⑥ vLLM mechanism (the explanation layer)
<which file did you read, how does it implement this behavior, does it match the delta you measured?> `vllm/v1/core/...`

## ⑦ Conclusion / next step
<one interview-ready sentence + the next experiment this one points to.>
