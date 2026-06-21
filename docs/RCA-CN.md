# Root Cause Analysis — Local L4 bring-up (engine build, serving, observability)
# 根因分析 —— 本地 L4 端到端启动（引擎构建、推理服务、可观测性）

> 🌐 **English version:** [`RCA-EN.md`](./RCA-EN.md)

This document records every issue hit while bringing `triton-trtllm-text-io` up
end-to-end on a single **NVIDIA L4** with the **NGC `tritonserver:24.10-trtllm-python-py3`**
image (which bundles **TensorRT-LLM 0.14.0**), and how each was fixed. Each entry has
**Symptom / 现象**, **Root cause / 根因**, and **Fix / 解决方案** in English and Chinese.

本文件记录在单卡 **NVIDIA L4** 上、使用 **NGC `tritonserver:24.10-trtllm-python-py3`**
镜像（内置 **TensorRT-LLM 0.14.0**）将 `triton-trtllm-text-io` 端到端跑通过程中遇到的每个问题
及其修复。每条包含 **现象**、**根因**、**解决方案**（中英文对照）。

---

## 1. Engine build: TensorRT-LLM examples/library version mismatch
## 1. 引擎构建：TensorRT-LLM 示例与库版本不匹配

**Symptom / 现象**
- EN: `scripts/build_engines.sh` cloned the TensorRT-LLM **examples at `v0.13.0`**, but the
  build container logged `[TensorRT-LLM] TensorRT-LLM version: 0.14.0`. The `convert_checkpoint.py`
  from one release was run against the library of another.
- 中文：`scripts/build_engines.sh` 克隆的是 **`v0.13.0`** 的 TensorRT-LLM 示例，但构建容器打印
  `[TensorRT-LLM] TensorRT-LLM version: 0.14.0`。即用某个版本的 `convert_checkpoint.py` 去调用
  另一个版本的库。

**Root cause / 根因**
- EN: The script's defaults were mispaired: `TRITON_TAG=24.10-trtllm-python-py3` ships
  TRT-LLM **0.14.0**, but `TRTLLM_REF` defaulted to `v0.13.0`. The example scripts and the
  installed library must be the same version.
- 中文：脚本默认值搭配错误：`TRITON_TAG=24.10-trtllm-python-py3` 内置的是 TRT-LLM **0.14.0**，
  而 `TRTLLM_REF` 默认却是 `v0.13.0`。示例脚本与已安装库必须版本一致。

**Fix / 解决方案**
- EN: Set `TRTLLM_REF=v0.14.0` in `build_engines.sh` to match the image.
- 中文：在 `build_engines.sh` 中将 `TRTLLM_REF` 改为 `v0.14.0`，与镜像保持一致。

---

## 2. Engine build: Qwen2.5 tied word embeddings → `None` lm_head
## 2. 引擎构建：Qwen2.5 共享词嵌入导致 lm_head 为 `None`

**Symptom / 现象**
- EN: `convert_checkpoint.py` crashed in `tensorrt_llm/layers/linear.py:407` with
  `AttributeError: 'NoneType' object has no attribute 'to'` while loading weights. It happened
  with **both** v0.13 and v0.14 examples.
- 中文：加载权重时 `convert_checkpoint.py` 在 `tensorrt_llm/layers/linear.py:407` 抛出
  `AttributeError: 'NoneType' object has no attribute 'to'`。在 v0.13 与 v0.14 示例下**都会**发生。

**Root cause / 根因**
- EN: Qwen2.5-0.5B/1.5B set `tie_word_embeddings: true` — the checkpoint has **no separate
  `lm_head.weight`** (it is tied to the input embedding). The default converter path tried to
  load `lm_head.weight`, got `None`, and called `.to()` on it.
- 中文：Qwen2.5-0.5B/1.5B 设置了 `tie_word_embeddings: true`——权重中**没有独立的
  `lm_head.weight`**（与输入词嵌入共享）。默认转换路径尝试加载 `lm_head.weight`，得到 `None`，
  再对其调用 `.to()` 即报错。

**Fix / 解决方案**
- EN: Pass `--use_embedding_sharing` to `convert_checkpoint.py`. This sets
  `share_embedding_table=True`, so the loader remaps `lm_head` to the vocab-embedding weight
  instead of expecting a separate tensor.
- 中文：给 `convert_checkpoint.py` 加上 `--use_embedding_sharing`，使
  `share_embedding_table=True`，让加载器把 `lm_head` 重映射到词嵌入权重，而不再期望单独的张量。

---

## 3. Engine build: NGC 0.14.0 library bug in `check_share_embedding()`
## 3. 引擎构建：NGC 0.14.0 库中 `check_share_embedding()` 的缺陷

**Symptom / 现象**
- EN: After enabling embedding sharing, the converter failed earlier in `qwen/model.py:346`:
  `TypeError: ModelWeightsLoader.check_share_embedding() missing 1 required positional argument: 'config'`.
- 中文：启用 embedding 共享后，转换在更靠前的 `qwen/model.py:346` 处失败：
  `TypeError: ModelWeightsLoader.check_share_embedding() missing 1 required positional argument: 'config'`。

**Root cause / 根因**
- EN: The `tensorrt_llm` **bundled in the NGC 24.10 image is internally inconsistent**:
  `models/qwen/model.py` calls `loader.check_share_embedding()` with no argument, but the
  installed `models/model_weights_loader.py` defines `check_share_embedding(self, config)`
  (the helper that actually remaps the tied lm_head). The image bundles **no** examples, so the
  GitHub example scripts must be used and cannot avoid this library call.
- 中文：NGC 24.10 镜像内**自带的 `tensorrt_llm` 自身不一致**：`models/qwen/model.py` 以无参方式调用
  `loader.check_share_embedding()`，但已安装的 `models/model_weights_loader.py` 的签名是
  `check_share_embedding(self, config)`（正是它负责重映射共享的 lm_head）。镜像**不含**示例，
  因此必须使用 GitHub 上的示例脚本，无法绕开此库调用。

**Fix / 解决方案**
- EN: Patch the call inside the build container before converting:
  `sed -i 's/loader.check_share_embedding()/loader.check_share_embedding(config)/' <path>/qwen/model.py`.
  The library path is **hardcoded** because deriving it via `python -c "import tensorrt_llm…"`
  prints a version banner to **stdout** that polluted the captured path. The build container runs
  as **root**, so stale `ckpt/` left by a failed run must be cleaned with a throwaway root
  container (`docker run --rm -v $PWD:/work … rm -rf /work/ckpt`), not a host `rm`.
- 中文：转换前在构建容器内打补丁：
  `sed -i 's/loader.check_share_embedding()/loader.check_share_embedding(config)/' <路径>/qwen/model.py`。
  库路径采用**硬编码**，因为用 `python -c "import tensorrt_llm…"` 推导路径时，导入会向 **stdout**
  打印版本横幅，污染了捕获到的路径。构建容器以 **root** 运行，失败残留的 `ckpt/` 需用一次性 root
  容器清理（`docker run --rm -v $PWD:/work … rm -rf /work/ckpt`），主机 `rm` 无权删除。

---

## 4. Model load: "trimmed" `tensorrt_llm` configs declare no I/O tensors
## 4. 模型加载：被“精简”的 `tensorrt_llm` 配置未声明 I/O 张量

**Symptom / 现象**
- EN: The `tensorrt_llm_{small,large}/config.pbtxt` declared only `parameters` — no `input`/`output`
  blocks — which the `tensorrtllm` backend cannot load.
- 中文：`tensorrt_llm_{small,large}/config.pbtxt` 只声明了 `parameters`，没有 `input`/`output` 块，
  而 `tensorrtllm` 后端无法加载这样的配置。

**Root cause / 根因**
- EN: The repo configs were intentionally "trimmed to the Part II knobs," omitting the required
  tensor set (`input_ids`, `input_lengths`, `request_output_len`, … → `output_ids`,
  `sequence_length`) that the BLS calls and the backend requires.
- 中文：仓库配置被有意“精简为 Part II 调参项”，省略了 BLS 调用、后端必需的张量集合
  （`input_ids`、`input_lengths`、`request_output_len`…→ `output_ids`、`sequence_length`）。

**Fix / 解决方案**
- EN: Merge the full `input[]` set verbatim from the tensorrtllm_backend **v0.14** template, keep
  `output_ids` + `sequence_length` as `output[]`, and re-attach the project's Part II parameters.
- 中文：从 tensorrtllm_backend **v0.14** 模板原样合入完整 `input[]`，`output[]` 保留
  `output_ids` 与 `sequence_length`，并补回项目的 Part II 参数。

---

## 5. Serving: only the first token of text is returned
## 5. 推理：只返回了首个 token 的文本

**Symptom / 现象**
- EN: A 128-token request streamed back only `"GPU"` then `finish_reason: length`. The model
  generated the full response but the client saw one token of text.
- 中文：一次 128 token 的请求只流式返回了 `"GPU"`，随后 `finish_reason: length`。模型生成了完整
  回复，但客户端只看到一个 token 的文本。

**Root cause / 根因**
- EN: The 24.10 TRT-LLM backend streams the **new token(s) per decoupled response** (a delta),
  not the full running sequence. `_stream_engine` assumed cumulative output and did
  `new = seq[emitted:]; emitted = seq.shape[0]`. With per-response deltas, `emitted` became the
  delta length (1) and every subsequent `seq[1:]` was empty → all tokens after the first were dropped.
- 中文：24.10 的 TRT-LLM 后端在每个 decoupled 响应里只流式返回**新增 token（增量）**，并非累计完整
  序列。`_stream_engine` 误以为是累计序列，执行 `new = seq[emitted:]; emitted = seq.shape[0]`。
  当响应是增量时，`emitted` 变成增量长度（1），之后 `seq[1:]` 恒为空 → 首个之后的 token 全部被丢弃。

**Fix / 解决方案**
- EN: Yield each response's tokens directly (each response is already the delta).
- 中文：直接产出每个响应的 token（每个响应本身就是增量）。

---

## 6. Serving: generation never stops at end-of-turn
## 6. 推理：生成不会在轮次结束时停止

**Symptom / 现象**
- EN: Every request ran to `max_tokens` (`finish_reason: length`), even short answers like
  "The capital of France is Paris."
- 中文：每次请求都跑满 `max_tokens`（`finish_reason: length`），即便是“法国的首都是巴黎”这类短答案。

**Root cause / 根因**
- EN: `_stream_engine` did not pass an `end_id`, so the engine had no end-of-turn token to stop on.
- 中文：`_stream_engine` 未传入 `end_id`，引擎没有可据以停止的轮次结束 token。

**Fix / 解决方案**
- EN: Pass `end_id = tokenizer.eos_token_id` as an engine input. (At this point the BLS still
  mislabeled an EOS stop as `finish_reason: length` — generation was correct, only the label;
  that labeling is fixed in §9 below.)
- 中文：将 `end_id = tokenizer.eos_token_id` 作为引擎输入传入。（此时 BLS 仍会把 EOS 停止误标为
  `finish_reason: length`——生成行为正确，只是标签不对；该标签问题已在下文第 9 节修复。）

---

## 7. Observability: Prometheus & OTel collector crash — config permission denied
## 7. 可观测性：Prometheus 与 OTel collector 崩溃 —— 配置文件权限被拒

**Symptom / 现象**
- EN: `prometheus` exited(2) and `otel-collector` exited(1) with `open …: permission denied`
  on their mounted config files; Grafana/Jaeger/DCGM stayed up.
- 中文：`prometheus` 退出码 2、`otel-collector` 退出码 1，对各自挂载的配置文件报
  `open …: permission denied`；Grafana/Jaeger/DCGM 正常。

**Root cause / 根因**
- EN: The repo was cloned with mode `640` (`-rw-r-----`, umask 027). These containers run as
  **non-root** users, which fall under "other" and cannot read the mounted config files.
- 中文：仓库以 `640`（`-rw-r-----`，umask 027）模式被克隆。这些容器以**非 root** 用户运行，属于
  “其他人(other)”，因而无法读取挂载的配置文件。

**Fix / 解决方案**
- EN: `chmod -R o+rX observability/` to make configs world-readable. (Git tracks only the
  executable bit, so `640→644` does not appear as a repo change.)
- 中文：执行 `chmod -R o+rX observability/` 让配置可被任意用户读取。（Git 只跟踪可执行位，
  因此 `640→644` 不会体现为仓库变更。）

---

## 8. Observability: OTel collector crash — duplicate spanmetrics dimension
## 8. 可观测性：OTel collector 崩溃 —— spanmetrics 维度重复

**Symptom / 现象**
- EN: After the permission fix, `otel-collector` still exited:
  `connectors::spanmetrics: failed validating dimensions: duplicate dimension name service.name`.
  No traces reached Jaeger and the `otel-spanmetrics` Prometheus target was `down`.
- 中文：修复权限后，`otel-collector` 仍退出：
  `connectors::spanmetrics: failed validating dimensions: duplicate dimension name service.name`。
  Jaeger 收不到链路，`otel-spanmetrics` 这个 Prometheus 目标为 `down`。

**Root cause / 根因**
- EN: In the spanmetrics connector (contrib 0.110.0), `service.name`/`span.name`/`span.kind`/
  `status.code` are **implicit** dimensions. The config re-declared `service.name`/`span.name`,
  which fails validation.
- 中文：在 spanmetrics 连接器（contrib 0.110.0）中，`service.name`/`span.name`/`span.kind`/
  `status.code` 是**隐式**维度。配置又显式声明了 `service.name`/`span.name`，导致校验失败。

**Fix / 解决方案**
- EN: Remove the explicit `dimensions` block (rely on the implicit ones; add only extra dimensions).
  After this, traces flow Triton → collector → Jaeger and all Prometheus targets are `up`.
- 中文：删除显式的 `dimensions` 块（使用隐式维度；仅在需要额外维度时添加）。修复后链路
  Triton → collector → Jaeger 正常，所有 Prometheus 目标均为 `up`。

---

## 9. Serving: finish_reason mislabeled "length" on a natural (EOS) stop
## 9. 推理：自然结束（EOS）时 finish_reason 被误标为 “length”

**Symptom / 现象**
- EN: A short answer the model finished on its own — e.g. "The capital of France is Paris." in
  ~8 tokens against a 256-token budget — still returned `finish_reason: length`. The generation
  was correct; only the reported reason was wrong.
- 中文：模型自行写完的短答案——例如用约 8 个 token 在 256 预算下回答“法国的首都是巴黎。”——
  仍返回 `finish_reason: length`。生成本身正确，只是上报的结束原因错误。

**Root cause / 根因**
- EN: In `text_pipeline_bls`, `finish` was initialized to `"length"` and only ever flipped to
  `"stop"` when a client STOP string matched. Nothing detected an end-of-turn (EOS) stop, so every
  EOS-terminated generation fell through to the `"length"` default. (Follow-up to §6: §6 made the
  engine stop at EOS via `end_id`; the BLS still mislabeled the reason.)
- 中文：在 `text_pipeline_bls` 中，`finish` 初始化为 `"length"`，仅当客户端 STOP 字符串命中时才
  改为 `"stop"`。没有任何逻辑识别轮次结束（EOS）停止，因此所有以 EOS 结束的生成都落到 `"length"`
  默认值。（第 6 节的后续：第 6 节用 `end_id` 让引擎在 EOS 停止，但 BLS 仍误标结束原因。）

**Fix / 解决方案**
- EN: Count the tokens the engine emits and classify the reason explicitly — a STOP-string match or
  an early halt (`generated < max_tokens`, i.e. EOS) → `"stop"`; only exhausting the budget →
  `"length"`, matching OpenAI semantics. The decision is a pure function `classify_finish_reason()`
  extracted to `src/text_io/finish.py` (the single source of truth), imported by the BLS and covered
  by a GPU-free unit test (`tests/test_finish.py`, wired into `make test`). Verified live on the L4:
  EOS→`stop`, tiny-budget→`length`, stop-string→`stop`, CJK+emoji→`stop`.
- 中文：统计引擎产出的 token 数并显式判定结束原因——命中 STOP 字符串或提前停止
  （`generated < max_tokens`，即 EOS）→ `"stop"`；仅当跑满预算 → `"length"`，与 OpenAI 语义一致。
  该判定被抽取为纯函数 `classify_finish_reason()`，置于 `src/text_io/finish.py`（单一事实源），
  由 BLS 导入并有免 GPU 单元测试覆盖（`tests/test_finish.py`，已接入 `make test`）。已在 L4 上实测：
  EOS→`stop`、极小预算→`length`、停止串→`stop`、中文+emoji→`stop`。

---

## 10. Serving/DevX: edits to the `src/` "single source of truth" silently don't take effect
## 10. 服务/开发体验：对 `src/`「单一事实源」的修改静默不生效

**Symptom / 现象**
- EN: After adding `src/text_io/finish.py` and importing it from the BLS, `docker restart triton-llm`
  came up with `text_pipeline_bls` **UNAVAILABLE** — `ImportError: No module named 'text_io.finish'` —
  and Triton's exit-on-error then made the whole container **Exit(1)**. The other three models
  (guardrail, both engines) loaded fine, which is what made it confusing.
- 中文：在新增 `src/text_io/finish.py` 并从 BLS 导入后，`docker restart triton-llm` 启动时
  `text_pipeline_bls` 处于 **UNAVAILABLE**——`ImportError: No module named 'text_io.finish'`——
  随后 Triton 的 exit-on-error 使整个容器 **Exit(1)**。其余三个模型（guardrail、两个引擎）均正常
  加载，正是这点让问题容易误判。

**Root cause / 根因**
- EN: `src/` was **only** baked into the image (`Dockerfile: COPY src /workspace/src`), while
  `model_repository/` was bind-mounted. So editing the BLS `model.py` took effect on restart
  (it lives under the mount), but a **new file** under `src/text_io/` stayed invisible until an
  image rebuild. The project documents an "edit and `docker restart`" workflow and calls
  `src/text_io` the single source of truth — yet that workflow silently did not apply to `src/`.
  The existing `detokenize_incremental`/`stop` imported only because they were present at the last
  image build.
- 中文：`src/` **仅**被烤进镜像（`Dockerfile：COPY src /workspace/src`），而 `model_repository/`
  是 bind 挂载。因此修改 BLS 的 `model.py` 重启即生效（位于挂载目录），但 `src/text_io/` 下的
  **新文件**在重建镜像前不可见。项目宣称的工作流是“编辑后 `docker restart`”，并称 `src/text_io`
  为单一事实源——但该工作流对 `src/` 实际并不生效。现有的 `detokenize_incremental`/`stop` 能导入，
  仅因为它们在上次构建镜像时已存在。

**Fix / 解决方案**
- EN: Bind-mount the host `src/` over `/workspace/src` in `start_server.sh`
  (`-v "$PWD/src:/workspace/src"`), so the single source of truth is live-on-restart exactly like
  `model_repository`. The `Dockerfile` still `COPY`s `src/` so the image stays self-contained for a
  fresh clone (the runtime mount shadows the baked copy). Rebuilt the image and recreated the
  `triton-llm` container with the mount; all four models then **READY**.
- 中文：在 `start_server.sh` 中将主机 `src/` bind 挂载到 `/workspace/src`
  （`-v "$PWD/src:/workspace/src"`），使单一事实源像 `model_repository` 一样重启即生效。
  `Dockerfile` 仍 `COPY` `src/`，以便全新克隆时镜像自包含（运行时挂载会覆盖烤入的副本）。
  重建镜像并带挂载重建 `triton-llm` 容器，之后四个模型全部 **READY**。

---

## Appendix: host environment notes (not code changes)
## 附录：主机环境说明（非代码改动）

- EN: System Python 3.12 has **no pip** → bootstrap a venv (`python3 -m venv .venv --without-pip`
  then `get-pip.py`). `huggingface_hub` is now **1.x** — the CLI is `hf` (`hf download …`); the
  `[cli]` extra and `huggingface-cli` name are gone. The base image numpy is **1.26.4** and
  `pip install` (no `--upgrade`) keeps it, so the NumPy-2.x BYTES-tensor bug does not bite here.
  Before launching, the co-located CV server (`triton-fused`) must be stopped — it shares ports
  8000–8002 and GPU memory.
- 中文：系统 Python 3.12 **没有 pip** → 需自举 venv（`python3 -m venv .venv --without-pip` 再
  `get-pip.py`）。`huggingface_hub` 现为 **1.x**——命令行是 `hf`（`hf download …`）；`[cli]` 额外项
  与 `huggingface-cli` 名称已移除。基础镜像 numpy 为 **1.26.4**，`pip install`（无 `--upgrade`）会保留它，
  因此此处不会触发 NumPy-2.x 的 BYTES 张量缺陷。启动前必须停止同机的 CV 服务（`triton-fused`）——
  它会占用 8000–8002 端口与显存。
