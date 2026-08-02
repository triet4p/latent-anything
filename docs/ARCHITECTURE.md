# Latent-Anything — Architecture

> Tài liệu kiến trúc kỹ thuật của dự án
> Phiên bản: 0.1 — May 2026

---

## 1. Core primitives

Toàn bộ framework xây trên năm primitive. Nếu các primitive này đúng, mọi thứ khác emerge tự nhiên. Nếu sai, refactor toàn bộ framework. Đây là phần quan trọng nhất cần thiết kế cẩn thận nhất.

**`LatentSpace`** — abstraction mô tả một không gian tiềm ẩn cụ thể. Bao gồm: dimensionality (có thể không flat — sequence, grid, Gaussian set, structured), geometry hint (Euclidean, manifold-constrained, discrete code, Riemannian), distribution prior nếu có, metadata về model nguồn. Đây là *handle*, không phải data — data sống trong `Trajectory`.

**`Trajectory`** — sequence latent state qua thời gian hoặc qua step, với operation cơ bản (slice, interpolate, compare, concat, rollout). Cầu nối tự nhiên đến world model rollout, agent execution, và bất kỳ thứ gì có temporal/sequential structure. Trajectory một điểm vẫn hợp lệ cho static latent analysis. Immutable — mọi operation trả về `Trajectory` mới.

**`ModelAdapter`** — interface để bất kỳ model nào expose latent space của nó. Tối thiểu: `encode`, `latent_space`. Adapter có decoder implement thêm `DecodableAdapter` với `decode`; decoder-free adapters như hidden-state và JEPA/LeWM không bị buộc phải giả tạo decoder. Một file = một adapter.

**`Method`** — interface chung cho mọi A/B/C method. Stateful method có thêm `fit`, `save`, `load`. Stateless method chỉ có `__call__`. Cả hai là first-class.

**`Pipeline`** — composition của ModelAdapter + Method(s) + execution config. Entry point user-facing chính. Config-driven: swap method mà không sửa code.

---

## 2. Ba layer

### Layer A — Introspection

**Mục tiêu.** Hiểu cái gì đang xảy ra bên trong latent space. User chính: researcher debug model, người explore pretrained model, mechanistic interpretability work.

**Scope.**

- *Dimensionality reduction & visualization*: PCA, UMAP, t-SNE, PaCMAP, projection lên direction tùy chỉnh.
- *Clustering & structure discovery*: K-means, DBSCAN, hierarchical, Gaussian mixture trong latent space.
- *Probing*: linear probe, MLP probe, concept activation vector (TCAV), causal probe.
- *Feature attribution*: integrated gradients, attention rollout cho transformer-based latent.
- *Sparse decomposition*: sparse autoencoder, dictionary learning để decompose latent thành interpretable feature.
- *Trajectory analysis*: smoothing, segmentation, change point detection, similarity metric giữa trajectory.
- *Visualization layer*: 2D/3D projection với trajectory overlay, interactive cluster inspection, concept atlas. Notebook-first (jupyter widget), web app sau.

**Implement approach.**

Bắt đầu hardcoded hai method có triết lý khác nhau: PCA (linear, deterministic, stateful sau fit) và UMAP (non-linear, stochastic, stateful sau fit). Sau khi chạy end-to-end qua `LatentSpace` + `Trajectory`, extract `Method` interface từ code thật.

Method thứ ba để verify interface: sparse autoencoder (neural network nhỏ, training process khác hoàn toàn). Nếu SAE fit vào interface mà không phải sửa interface, layer A đủ vững.

Sau đó thêm dần: linear probe (cần label — stress test cho data dependency), trajectory segmentation (operate trên `Trajectory` không phải flat latent — stress test temporal API), TCAV (cần concept dataset).

**Tech stack.**

- `numpy` cho array operation cơ bản. Public API trả về numpy, không leak PyTorch tensor.
- `scikit-learn` cho PCA, clustering, baseline probe.
- `umap-learn`, `pacmap` cho non-linear reduction.
- `torch` cho SAE, neural probe, attribution. Internal, không leak ra public API.
- `plotly` cho interactive visualization, `matplotlib` cho static export.
- `ipywidgets` + `anywidget` cho notebook integration.

**Technical decisions.**

- *Lazy vs eager*: eager ở 0.x, reconsider khi có trajectory >10k step.
- *Visualization*: protocol là plugin (mỗi method expose `to_plot()`), có default renderer built-in.
- *Cache cho fitted state*: filesystem cache, key = input data hash + config hash. Pluggable backend (local, S3, HF Hub).

---

### Layer B — Manipulation

**Mục tiêu.** Tác động lên latent space: edit, interpolate, compose, steer. User chính: người build application trên pretrained model, agent developer compose skill trong latent, researcher causal intervention.

**Scope.**

- *Basic transform*: linear interpolation (lerp), spherical interpolation (slerp), latent arithmetic (a − b + c), projection lên/khỏi direction.
- *Steering*: steering vector từ contrast pair, activation patching, hook-based intervention.
- *Composition*: chain transform, conditional transform, batch transform với broadcasting rule rõ ràng.
- *Constrained edit*: project lên manifold, project lên feasible set, edit với regularization về original.
- *Latent diffusion edit*: nếu adapter là diffusion model, support edit qua noise level (SDEdit-style).
- *Skill composition cho embodied context*: trajectory blending, sequential composition, parallel composition.

**Implement approach.**

Bắt đầu hardcoded hai transform: linear interpolation (stateless, pure function) và steering vector (stateful — cần fit từ contrast dataset). Stress-test stateful vs stateless trong cùng interface.

Method thứ ba: activation patching. Khác hai cái trên vì *intervene trong forward pass*, không chỉ post-hoc transform. Stress test cho hook-into-model-execution.

Sau đó mở rộng theo use case thật: latent arithmetic, SDEdit, trajectory blending khi tích hợp world model.

**Tech stack.**

- `numpy` + `torch`. Torch internal cho hook-based method, không leak ra interface.
- `einops` cho shape manipulation rõ ràng và an toàn.
- Custom hook manager cho activation patching, build trên `torch.nn.Module` hook nhưng abstract đi.

**Technical decisions.**

- *Immutable trajectory*: mọi transform trả về `Trajectory` mới. Dễ reason, dễ cache, dễ parallelize.
- *Hook lifecycle*: context manager pattern. Register/unregister an toàn ngay cả khi exception.
- *Broadcasting semantics*: theo numpy/PyTorch convention. Document rõ, test mọi shape combination.

---

### Layer C — Runtime

**Mục tiêu.** Chạy pipeline hiệu quả. User chính: người deploy trong production, người chạy large-scale experiment, người build embodied agent cần real-time latent rollout.

**Scope.**

- *Batch execution*: efficient batching qua model adapter và method, handle variable-length trajectory.
- *Caching*: cache encoded latent, fitted method state, intermediate trajectory result. Pluggable backend.
- *Async execution*: pipeline với async/await, overlap I/O và compute.
- *Streaming*: process trajectory dài bằng streaming, không load hết vào memory.
- *Profiling & introspection*: built-in profiler, latency breakdown, memory usage.
- *Distributed execution*: chỉ khi có need thật, không implement sớm.
- *Real-time mode*: chỉ sau khi có Rust core.

**Implement approach.**

Bắt đầu eager batch executor với explicit batching. Profile trên use case A và B thật. Identify hot path. Optimize có evidence.

Tiếp theo: caching layer in-memory, key bằng content hash. Sau đó disk backend.

Sau đó: async pipeline. Lý do cần sớm: nhiều use case có I/O (load model từ HF Hub, fetch remote data) interleave với compute.

Streaming và profiling parallel. Real-time mode và Rust core là phase sau.

**Tech stack.**

- `asyncio` cho async execution.
- `diskcache` hoặc tự build trên `sqlite` cho disk cache. Tránh Redis ở 0.x.
- `pyinstrument` hoặc `py-spy` cho profiling integration.
- `pyarrow` cho serialize latent dataset lớn. Cross-language friendly cho Rust port sau.

**Technical decisions.**

- *Cache invalidation*: key bao gồm input data hash + model version + method config hash + framework version. Quyết format từ đầu.
- *Sync vs async API*: async primary, sync là thin wrapper qua `asyncio.run`. Không để diverge.
- *Executor*: per-pipeline, không singleton. Option share executor giữa pipeline khi cần.

---

## 3. Plugin architecture

### Nguyên tắc

Plugin surface tối thiểu ban đầu: `ModelAdapter`, `Method`, `Pipeline`. Mọi thứ khác internal. Mở rộng surface chỉ khi có evidence từ ít nhất hai use case thực tế.

Plugin từ ngôn ngữ khác được support từ thiết kế. Interface định nghĩa qua data structure (pydantic model, dict, array) thay vì Python-specific object. Rust plugin sau này plug vào được mà không cần redesign.

Stateful và stateless đều first-class. Stateful method có `fit`, `save`, `load`. Stateless chỉ có `__call__`.

### Discovery mechanism

Registry + config, không phải vector DB hay semantic search. Hai layer:

**Registry (decorator/entry point).** Plugin tự register qua decorator hoặc Python entry points. Framework discover tự động. User thấy danh sách plugin available, không cần biết internal structure.

**Config-driven instantiation.** Plugin được khai báo và instantiate qua config (pydantic model hoặc yaml). Swap method mà không sửa code. Tự build wrapper thay vì dùng hydra-core — hydra nặng và opinionated.

### Reproducibility

Mỗi plugin save config + version vào output. Load lại sau biết chính xác method nào, version nào đã chạy. Cache key bao gồm: input data hash, model version, method config hash, framework version.

### "Hello world" plugin

Có một plugin trivial trong repo làm template. New contributor copy, modify. Đây là cái biến framework từ "code của bạn" thành "code của community".

---

## 4. Model adapters

Mỗi model family là một file adapter. Tất cả implement cùng `ModelAdapter` interface. Thứ tự implement:

**Giai đoạn 1:** VAE (own training, hiểu hết internals — stress test cơ bản) và VLA (OpenVLA hoặc tương đương — stress test với pretrained large model).

**Giai đoạn 6:** World model (LeWM hoặc tương đương), diffusion model, LLM hidden state. Mỗi cái là stress test thêm cho `LatentSpace` abstraction.

**Lưu ý đặc biệt cho 3DGS adapter.** Adapter cho một model 3D Gaussian Splatting có thể expose `LatentSpace` với geometry là "set of 3D Gaussians" — mỗi latent state là một tập {μ_i, Σ_i, α_i, sh_i}. `encode` map observation → Gaussian set; `decode` có thể là Gaussian rasterizer deterministic. `latent_space` metadata mô tả Gaussian parameterization.

**Lưu ý đặc biệt cho JEPA/LeWM adapter.** LeWM là world model decoder-free theo kiểu JEPA: adapter expose predicted/observed embeddings và transition semantics qua `ModelAdapter`, không gán geometry 3DGS và không bịa ra `decode`.

---

## 5. Thứ tự implement

Mỗi giai đoạn phải đạt production quality trước khi sang giai đoạn sau. Không shortcut.

**Giai đoạn 1 — Core primitives & first integrations.**

Định nghĩa `LatentSpace`, `Trajectory`, `ModelAdapter`, `Method`, `Pipeline`. Chưa cần plugin infrastructure, hardcoded import là ổn.

Implement adapter cho VAE (own training) và VLA (pretrained). Mục đích: stress-test `LatentSpace` abstraction trên hai world rất khác nhau.

Implement PCA + UMAP hardcoded, end-to-end qua pipeline.

Tiêu chí: load VLA, lấy trajectory từ một task, visualize bằng UMAP, interactive trong notebook.

**Giai đoạn 2 — Layer B foundation.**

Implement linear interpolation + steering vector hardcoded.

End-to-end story: load VLA, identify failure case, edit latent tại moment fail, decode lại, quan sát behavior thay đổi. Đây là showcase use case đầu tiên — validate rằng framework có giá trị thật.

Tiêu chí: story chạy được, documented, reproducible từ config.

**Giai đoạn 3 — Plugin extraction.**

Refactor code đã có để extract plugin interface từ working code. Không design từ trí tưởng tượng.

Implement registry mechanism (decorator + entry point) và config loader.

Tiêu chí: convert 4 method đã có (PCA, UMAP, lerp, steering) thành plugin, không thay đổi behavior, tất cả test pass.

**Giai đoạn 4 — Plugin verification & layer expansion.**

Thêm method thứ ba cho layer A (sparse autoencoder) và layer B (activation patching). Mỗi cái có triết lý khác hẳn — đây là test thật của plugin interface.

Nếu interface phải sửa, sửa luôn, migrate plugin cũ.

Tiêu chí: thêm một method mới chỉ cần viết một file plugin, không sửa core.

**Giai đoạn 5 — Layer C foundation.**

Build execution runtime: batching, caching (in-memory + disk), async pipeline, profiling.

Tiêu chí: cache hit speedup đo được, async throughput tốt hơn sync đo được, profiler chỉ ra bottleneck rõ ràng.

**Giai đoạn 6 — Ecosystem expansion.**

Thêm adapter cho world model (JEPA/LeWM), diffusion, LLM hidden state, và một 3DGS adapter riêng. Mỗi cái stress test thêm cho `LatentSpace`; LeWM giữ decoder-free, còn Gaussian-set latent và rasterizer thuộc 3DGS.

Thêm method theo demand từ user thật.

Tiêu chí: ít nhất 5 adapter, ít nhất 5 method mỗi layer, tất cả là plugin.

**Giai đoạn 7 — Cross-language plugin support.**

Định nghĩa wire protocol cho plugin viết bằng ngôn ngữ khác (Rust trước). Plugin Rust expose qua PyO3 hoặc subprocess với protocol định sẵn.

Tiêu chí: một method Rust plug vào framework Python, performance đo được tốt hơn Python equivalent.

**Giai đoạn 8 — Rust core (optional, evidence-based).**

Chỉ khi profiling cho thấy Python core là bottleneck thật. Port hot path sang Rust với Python binding. Không phải toàn bộ framework — chỉ hot path.

Đây là điểm framework sẵn sàng cho real-time embodied use case.

**Giai đoạn 9 — 1.0.**

Khi interface không còn muốn refactor, freeze API, release 1.0. Cam kết backward compatibility từ điểm này.

---

## 6. Tech stack tổng kết

**Core (giai đoạn 1–5).**

- Python 3.12+.
- `numpy`, `torch`, `einops` cho numerical.
- `pydantic` cho config và data contract.
- `typing.Protocol` + `runtime_checkable` cho plugin interface.
- `pytest` + `hypothesis` cho test. Property-based test đặc biệt có giá trị cho transform layer.
- `mkdocs-material` + `mkdocstrings` cho documentation.
- `ruff` + `pyright` cho lint và type check.

**Plugin & config (giai đoạn 3+).**

- Python entry points qua `importlib.metadata` cho plugin discovery.
- Tự build config wrapper trên pydantic. Không dùng hydra-core — quá nặng và opinionated.

**Execution (giai đoạn 5+).**

- `asyncio` cho async.
- `diskcache` cho disk cache.
- `pyarrow` cho serialize. Cross-language friendly.
- `pyinstrument` cho profiling.

**Visualization (xuyên suốt).**

- `plotly` cho interactive.
- `matplotlib` cho static.
- `anywidget` cho notebook widget custom.

**3D / Gaussian (giai đoạn 6+, khi tích hợp 3DGS).**

- `gsplat` hoặc tự implement Gaussian rasterizer cho adapter decode.
- `open3d` cho point cloud operation nếu cần.

**Cross-language & Rust (giai đoạn 7–8).**

- `pyo3` + `maturin` cho Rust binding.
- `arrow` (Rust crate) cho cross-language data exchange.
- Wire protocol: Arrow IPC hoặc gRPC. Quyết ở giai đoạn 7 dựa trên benchmark.

---

## 7. Technical decisions đã chốt

| Quyết định | Lựa chọn |
|---|---|
| Framing | Framework layer ngang cho latent space (A + B + C) |
| Ngôn ngữ chính | Python trước, Rust sau theo evidence |
| Primitive cốt lõi | `LatentSpace` + `Trajectory` |
| Plugin discovery | Registry (decorator/entry point) + config |
| Plugin language | Cross-language từ thiết kế interface |
| Plugin state | Cả stateful và stateless là first-class |
| Plugin surface | Tối thiểu ban đầu (ModelAdapter, Method, Pipeline) |
| Interface evolution | Incremental dưới 1.0, extract từ working code |
| Quality bar | Production từ đầu, chỉ thứ tự là incremental |
| Tensor backend trong public API | numpy, không leak PyTorch |
| Trajectory mutability | Immutable |
| Sync vs async | Async primary, sync wrapper |
| Cache backend | Pluggable, default in-memory + disk |
| LatentSpace geometry | Flexible — support flat vector, sequence, Gaussian set |

---

## 8. High-level architecture diagram

```
┌─────────────────────────────────────────────────────────┐
│                        Pipeline                          │
│          Config-driven: adapter + methods + runtime      │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ A — Intros-  │ │ B — Manipu-  │ │ C — Runtime  │
│   pection    │ │   lation     │ │              │
│              │ │              │ │ Batching     │
│ PCA, UMAP    │ │ Lerp, slerp  │ │ Caching      │
│ SAE, probe   │ │ Steering     │ │ Async        │
│ TCAV, viz    │ │ Patching     │ │ Profiling    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       └────────────────┼────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Core Primitives                             │
│                                                          │
│   LatentSpace  ·  Trajectory  ·  Method interface        │
│   ModelAdapter interface  ·  Pipeline definition         │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼────────────────────────┐
         ▼               ▼               ▼         ▼
┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐
│     VAE      │ │     VLA      │ │   3DGS   │ │ JEPA/LeWM│
│   Adapter    │ │   Adapter    │ │ (Gaussian)│ │ / Diffusion│
└──────────────┘ └──────────────┘ └──────────┘ └──────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│              Plugin Layer                                │
│   Registry + Config · Python plugins · Rust plugins      │
│   Arrow IPC for cross-language wire protocol             │
└─────────────────────────────────────────────────────────┘
```

---

*Tài liệu này là architecture reference của Latent-Anything. Cập nhật khi có quyết định kỹ thuật mới hoặc khi interface thay đổi.*
