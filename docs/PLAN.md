# Global Project Plan

## Overview
Xây dựng `src` của Latent-Anything theo quy trình incremental trong [docs/INCREMENTAL.md](INCREMENTAL.md): mỗi sprint là **một increment-round** (một việc lớn — thêm đúng một instance cụ thể chạy end-to-end), mỗi task là **một concern nhỏ**. Interface được *extract từ working code*, freeze theo Rule of Three; ba ADR ngày 2026-06-16 được coi là giả thuyết để code validate/đảo. Nền lý thuyết (tầng 1–14) coi như đã đủ để khởi động Giai đoạn 1.

## Milestones
* [x] **Milestone 0:** Theory foundation — tầng 1–14 research + notebook (Sprint 1–2).
* [ ] **Milestone 1 — Giai đoạn 1:** Core primitives (`LatentSpace`, `Trajectory`, `ModelAdapter`, `Method`, `Pipeline`) + Layer A trio + adapter VAE/VLA, end-to-end qua pipeline (Sprint 3–9).
* [ ] **Milestone 2 — Giai đoạn 2:** Layer B foundation (lerp → steering → activation patching) + showcase edit-latent trên VLA (Sprint 10–13).

## Active Sprints
* [Sprint 3](sprint-plans/sprint-3.md) - *Status: Not Started* — Scaffold package `latent_anything` + tooling/CI (Round 0).
* [Sprint 4](sprint-plans/sprint-4.md) - *Status: Completed* — Increment đầu: `LatentSpace`+`Trajectory`+PCA hardcoded, end-to-end (Round 1).

## Completed Sprints
* [Sprint 1](sprint-plans/sprint-1.md) - *Status: Completed* — Hoàn tất tầng 11 + 2 mục tầng 12.
* [Sprint 2](sprint-plans/sprint-2.md) - *Status: Completed* — Thêm tầng "Mô hình dựng 3D thực tiễn".

## Backlog / Future Work
*Mỗi dòng là một sprint tương lai = một increment-round trong [INCREMENTAL.md §6](INCREMENTAL.md). Chỉ chi tiết hóa thành sprint file khi tới lượt — tránh thiết kế trước cái code chưa lộ ra.*

**Giai đoạn 1 (tiếp theo Sprint 3–4):**
* Sprint 5 — Layer A: thêm UMAP (Method #2), phác `Method` shape tạm *unstable* (Round 2).
* Sprint 6 — Layer A: thêm SAE (Method #3 khác triết lý) → **freeze `Method` interface**, migrate PCA+UMAP (Round 3).
* Sprint 7 — Adapter VAE (own training, explicit learned latent — ModelAdapter #1) (Round 4).
* Sprint 8 — Adapter VLA (pretrained — ModelAdapter #2), phác `ModelAdapter` shape tạm (Round 5).
* Sprint 9 — Geometry case #2 (unit-norm/spherical) → validate/đảo ADR `LatentSpace` + ADR geometry-dispatch (Round 6).

**Giai đoạn 2:**
* Sprint 10 — Layer B: lerp (B-Method #1, stateless) (Round 7).
* Sprint 11 — Layer B: steering vector (B-Method #2, stateful) (Round 8).
* Sprint 12 — Layer B: activation patching (B-Method #3 khác triết lý) → **freeze Layer-B interface**, migrate (Round 9).
* Sprint 13 — Showcase end-to-end: load VLA → tìm failure → edit latent → decode → quan sát behavior, reproducible từ config (Round 10).
