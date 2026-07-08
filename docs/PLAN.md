# Global Project Plan

## Overview
Xây dựng `src` của Latent-Anything theo quy trình incremental trong [docs/INCREMENTAL.md](INCREMENTAL.md): mỗi sprint là **một increment-round** (một việc lớn — thêm đúng một instance cụ thể chạy end-to-end), mỗi task là **một concern nhỏ**. Interface được *extract từ working code*, freeze theo Rule of Three; ba ADR ngày 2026-06-16 được coi là giả thuyết để code validate/đảo. Nền lý thuyết (tầng 1–14) coi như đã đủ để khởi động Giai đoạn 1.

## Milestones
* [x] **Milestone 0:** Theory foundation — tầng 1–14 research + notebook (Sprint 1–2).
* [x] **Milestone 1 — Giai đoạn 1:** Core primitives (`LatentSpace`, `Trajectory`, Layer A `Method`) + first adapters and geometry dispatch (Sprint 3–9). **Complete — 2 ADRs validated; `ModelAdapter` remains intentionally pending.**
* [x] **Milestone 2 — Giai đoạn 2:** Layer B foundation (lerp → steering → activation patching) + showcase edit-latent story (Sprint 10–13). **Sprint 13 completed — VAE-based end-to-end showcase proven.** The showcase composes existing primitives (VAE, PCA, ActivationPatch, Lerp) into a coherent edit story with 68.2% distance improvement. Real VLA showcase remains future ecosystem work when a VLA adapter exists in the codebase.
* [x] **Milestone 3 — Adapter/geometry closure:** Close the remaining `ModelAdapter` ADR evidence gap with no-explicit-latent and deterministic-renderer modes; add the first structured Gaussian-set geometry (Sprint 14–16). **Complete — all three 2026-06-16 ADRs fully validated with all modes confirmed by running code.**
* [x] **Milestone 4 — Plugin extraction:** Introduce registry + config-driven instantiation and convert existing built-ins without changing behavior (Sprint 17–19). **Complete — all 10 built-in classes registered with registry-first pattern; behavior parity proven by 22 parity tests; infrastructure remains entry-point-free pending external plugin demand.**
* [x] **Milestone 5 — Pipeline foundation:** Add concrete `Pipeline` composition rounds, then freeze only after enough distinct execution stories exist (Sprint 20–21). **Complete — two pipeline instances (Analysis + Manipulation) with shared `_PipelineBase` sketch. Freeze waits for Pipeline #3 (runtime/streaming).**
* [~] **Milestone 6 — Layer C runtime foundation:** Add batching, cache, async execution, and profiling in small evidence-backed increments (Sprint 22–24). **Sprint 22 completed — BatchExecutor #1 added as eager/sync runtime instance; cache and async remain future increments.**

## Active Sprints
* [Sprint 23](sprint-plans/sprint-23.md) - *Status: Planned* — Add cache layer, starting with in-memory cache.

## Completed Sprints
* [Sprint 22](sprint-plans/sprint-22.md) - *Status: Completed* — Add `BatchExecutor` Runtime #1 for deterministic first-axis numpy batching. Supports adapter `encode`/`decode` and Layer A `transform` paths, preserves output order/shape, includes synthetic direct-vs-batched timing snapshot, and stays eager/sync with no cache or async. 23 new tests, 575 total. (Round 19).
* [Sprint 21](sprint-plans/sprint-21.md) - *Status: Completed* — Add concrete Pipeline #2 (`ManipulationPipeline`) for Layer B manipulation (adapter-mediated data-space + latent-only trajectory stories). Sketch `_PipelineBase` shared with `AnalysisPipeline`. Config-backed construction via `ManipulationPipelineSpec`. 28 new tests, 551 total. Pipeline stays concrete — no DAG/executor abstraction. (Round 18).
* [Sprint 20](sprint-plans/sprint-20.md) - *Status: Completed* — Add concrete Pipeline #1 (`AnalysisPipeline`) for adapter → encode → Layer A method → typed `PipelineResult`. Config-backed construction via `PipelineSpec` + `build_pipeline_from_config`. 21 pipeline tests, 523 total. Pipeline stays concrete — no DAG/executor abstraction. (Round 17).
* [Sprint 19](sprint-plans/sprint-19.md) - *Status: Completed* — Convert built-in adapters/methods to registry-first built-ins with separate `_plugin_builtins.py` module, proving behavior parity via 22 parity tests + 15 demo smoke tests. Infrastructure-only round — no new adapter/method/geometry instances. 502 total tests. (Round 16).
* [Sprint 18](sprint-plans/sprint-18.md) - *Status: Completed* — Registry-backed config instantiation using pydantic v2. ObjectSpec model with kind/name/params, build_from_config resolver, nested spec resolution for adapter-in-method (ActivationPatch), clear validation errors, and 36 config tests. 465 total tests. (Round 15).
* [Sprint 17](sprint-plans/sprint-17.md) - *Status: Completed* — In-process registry for adapters and methods. Registry class with register/lookup/list APIs, kind constants, convenience helpers, and all 10 built-in classes registered (4 adapters + 3 method_a + 3 method_b). 48 registry tests. 429 total tests. (Round 14).
* [Sprint 16](sprint-plans/sprint-16.md) - *Status: Completed* — GaussianRendererAdapter (ModelAdapter #4, mode iii: deterministic renderer). Closes the last evidence gap for the 3-mode ModelAdapter ADR — all three 2026-06-16 ADRs now fully validated. 52 GaussianRenderer tests, 381 total. (Round 13).
* [Sprint 15](sprint-plans/sprint-15.md) - *Status: Completed* — Add structured `gaussian_set` latent geometry as geometry case #3. Geometry-keyed and geometry-dispatch ADRs exercised by set-like structured shape. Inline `if/elif` dispatch survives 3 geometries (Rule of Three §4a: instance #3 → keep hardcoded). 78 LatentSpace tests pass. (Round 12).
* [Sprint 14](sprint-plans/sprint-14.md) - *Status: Completed* — HiddenStateAdapter (ModelAdapter #3, mode ii: no-explicit-latent); freeze `ModelAdapter` + `DecodableAdapter` Protocols; remove `_ModelAdapterBase`. ModelAdapter 3-mode ADR gains mode ii evidence, with full validation still pending the deterministic renderer mode iii later completed in Sprint 16. (Round 11).
* [Sprint 13](sprint-plans/sprint-13.md) - *Status: Completed* — End-to-end VAE-based showcase: adapter → latent inspection (PCA Layer A) → latent edit (ActivationPatch Layer B) → decode → before/after metric (68.2% improvement) → trajectory panel (Lerp). Reproducible from config lightweight; no new abstraction added. (Round 10 — composition/validation round).
* [Sprint 12](sprint-plans/sprint-12.md) - *Status: Completed* — Layer B: activation patching (B-Method #3, model-mediated) → **freeze BMethod Protocol**, migrate Lerp + SteeringVector (Round 9).
* [Sprint 9](sprint-plans/sprint-9.md) - *Status: Completed* — Geometry case #2 (unit-norm/spherical) → validate ADR `LatentSpace` + ADR geometry-dispatch (Round 6). **Two ADRs validated.** Last sprint of Giai đoạn 1.
* [Sprint 1](sprint-plans/sprint-1.md) - *Status: Completed* — Hoàn tất tầng 11 + 2 mục tầng 12.
* [Sprint 2](sprint-plans/sprint-2.md) - *Status: Completed* — Thêm tầng "Mô hình dựng 3D thực tiễn".
* [Sprint 3](sprint-plans/sprint-3.md) - *Status: Completed* — Scaffold package `latent_anything` + tooling/CI (Round 0).
* [Sprint 4](sprint-plans/sprint-4.md) - *Status: Completed* — Increment đầu: `LatentSpace`+`Trajectory`+PCA hardcoded, end-to-end (Round 1).
* [Sprint 5](sprint-plans/sprint-5.md) - *Status: Completed* — Layer A: UMAP (Method #2) + phác `_MethodBase` unstable internal (Round 2).
* [Sprint 6](sprint-plans/sprint-6.md) - *Status: Completed* — Layer A: SAE (Method #3) + freeze `Method` Protocol (Round 3).
* [Sprint 7](sprint-plans/sprint-7.md) - *Status: Completed* — Adapter VAE (ModelAdapter #1, explicit learned latent), end-to-end (Round 4).
* [Sprint 8](sprint-plans/sprint-8.md) - *Status: Completed* — Adapter RandomProjection (ModelAdapter #2, stateless/fixed-weight), phác `_ModelAdapterBase` unstable (Round 5).
* [Sprint 11](sprint-plans/sprint-11.md) - *Status: Completed* — Layer B: steering vector (B-Method #2, stateful), sketch `_BMethodBase` internal UNSTABLE, end-to-end (Round 8).
* [Sprint 10](sprint-plans/sprint-10.md) - *Status: Completed* — Layer B: lerp (B-Method #1, stateless), trajectory blending, end-to-end (Round 7). **Giai đoạn 2 begins.**

## Backlog / Future Work
*Mỗi dòng là một sprint tương lai = một increment-round trong [INCREMENTAL.md §6](INCREMENTAL.md). Các sprint tương lai là plan có chủ đích, nhưng vẫn provisional: nếu code trong sprint trước phản bác giả định, sprint sau phải được sửa thay vì bám máy móc.*

**Pipeline foundation:** *(Complete — two pipeline instances validate shared shape sketch.)*

**Layer C runtime foundation:**
* [Sprint 23](sprint-plans/sprint-23.md) - *Status: Planned* — Add cache layer, starting with in-memory cache.
* [Sprint 24](sprint-plans/sprint-24.md) - *Status: Planned* — Add async execution and profiling hooks.

**Later ecosystem expansion, not yet sprint-filed:**
* Add label-aware probing / TCAV work from theory tầng 5.
* Add trajectory similarity, rollout, and transition-model methods from tầng 6.
* Add MPC/CEM planning methods from tầng 7.
* Add discrete/tokenized world-model adapters from tầng 9.
