# Global Project Plan

## Overview
Xây dựng `src` của Latent-Anything theo quy trình incremental trong [docs/INCREMENTAL.md](INCREMENTAL.md): mỗi sprint là **một increment-round** (một việc lớn — thêm đúng một instance cụ thể chạy end-to-end), mỗi task là **một concern nhỏ**. Interface được *extract từ working code*, freeze theo Rule of Three; ba ADR ngày 2026-06-16 được coi là giả thuyết để code validate/đảo. Nền lý thuyết (tầng 1–14) coi như đã đủ để khởi động Giai đoạn 1.

## Milestones
* [x] **Milestone 0:** Theory foundation — tầng 1–14 research + notebook (Sprint 1–2).
* [x] **Milestone 1 — Giai đoạn 1:** Core primitives (`LatentSpace`, `Trajectory`, `ModelAdapter`, `Method`, `Pipeline`) + Layer A trio + adapter VAE/VLA, end-to-end qua pipeline (Sprint 3–9). **✅ Complete — 2 ADRs validated.**
* [~] **Milestone 2 — Giai đoạn 2:** Layer B foundation (lerp → steering → activation patching) + showcase edit-latent trên VLA (Sprint 10–13). **Sprint 10 active — Lerp (B-Method #1).**

## Active Sprints
* [Sprint 10](sprint-plans/sprint-10.md) - *Status: Active* — Layer B: lerp (B-Method #1, stateless), trajectory blending, end-to-end (Round 7). **Giai đoạn 2 begins.**

## Completed Sprints
* [Sprint 9](sprint-plans/sprint-9.md) - *Status: Completed* — Geometry case #2 (unit-norm/spherical) → validate ADR `LatentSpace` + ADR geometry-dispatch (Round 6). **Two ADRs validated.** Last sprint of Giai đoạn 1.
* [Sprint 1](sprint-plans/sprint-1.md) - *Status: Completed* — Hoàn tất tầng 11 + 2 mục tầng 12.
* [Sprint 2](sprint-plans/sprint-2.md) - *Status: Completed* — Thêm tầng "Mô hình dựng 3D thực tiễn".
* [Sprint 3](sprint-plans/sprint-3.md) - *Status: Completed* — Scaffold package `latent_anything` + tooling/CI (Round 0).
* [Sprint 4](sprint-plans/sprint-4.md) - *Status: Completed* — Increment đầu: `LatentSpace`+`Trajectory`+PCA hardcoded, end-to-end (Round 1).
* [Sprint 5](sprint-plans/sprint-5.md) - *Status: Completed* — Layer A: UMAP (Method #2) + phác `_MethodBase` unstable internal (Round 2).
* [Sprint 6](sprint-plans/sprint-6.md) - *Status: Completed* — Layer A: SAE (Method #3) + freeze `Method` Protocol (Round 3).
* [Sprint 7](sprint-plans/sprint-7.md) - *Status: Completed* — Adapter VAE (ModelAdapter #1, explicit learned latent), end-to-end (Round 4).
* [Sprint 8](sprint-plans/sprint-8.md) - *Status: Completed* — Adapter RandomProjection (ModelAdapter #2, stateless/fixed-weight), phác `_ModelAdapterBase` unstable (Round 5).

## Backlog / Future Work
*Mỗi dòng là một sprint tương lai = một increment-round trong [INCREMENTAL.md §6](INCREMENTAL.md). Chỉ chi tiết hóa thành sprint file khi tới lượt — tránh thiết kế trước cái code chưa lộ ra.*

**Giai đoạn 2 (tiếp theo Sprint 10):**
* Sprint 11 — Layer B: steering vector (B-Method #2, stateful) (Round 8).
* Sprint 12 — Layer B: activation patching (B-Method #3 khác triết lý) → **freeze Layer-B interface**, migrate (Round 9).
* Sprint 13 — Showcase end-to-end: load VLA → tìm failure → edit latent → decode → quan sát behavior, reproducible từ config (Round 10).
