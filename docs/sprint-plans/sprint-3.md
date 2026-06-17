# Sprint 3 Plan

## Sprint Goal
Khởi tạo package `latent_anything` với tooling đầy đủ (uv, ruff, pyright strict, pytest, CI) chạy xanh end-to-end — **chưa có abstraction/interface nào**. Đây là Round 0 trong [INCREMENTAL.md §6](../INCREMENTAL.md).

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [x] Task 1: Khởi tạo package `latent_anything` bằng `uv` (pyproject PEP 621, Python ≥3.12), chốt vị trí package (root-level vs sub-project) theo [python rule](../../.agents/rules/python.md).
- [x] Task 2: Cấu hình `ruff` (check + format) trong pyproject — rule set + line length.
- [x] Task 3: Cấu hình `pyright` strict mode trong pyproject/pyrightconfig.
- [x] Task 4: Thêm `pytest` + tạo `tests/` mirroring `src/`, cấu hình test discovery.
- [x] Task 5: Thêm `hypothesis` vào dev dependencies (chuẩn bị cho property-based test của core primitive).
- [x] Task 6: Tạo `latent_anything/__init__.py` (chỉ version + docstring) và một smoke test trivial chạy xanh.
- [x] Task 7: Thêm CI workflow (GitHub Actions) chạy `ruff check` + `ruff format --check` + `pyright` + `pytest`.
- [x] Task 8: Cập nhật `README.md` (Project Structure + Quick Start) phản ánh package mới.
- [x] Task 9: Thêm entry vào `CHANGELOG.md` `[Unreleased]` cho việc khởi tạo package.
- [x] Task 10: Điều chỉnh Python version range: min 3.12, max 3.14; local dev dùng 3.13. Cập nhật `requires-python`, `ruff target-version`, `pyright pythonVersion`, CI matrix, và rules doc.
- [x] Task 11: Sửa `deploy-latent-anything-theory.yml` — chỉ trigger khi push tag (pattern `theory-v*`) hoặc `workflow_dispatch` thủ công. Loại bỏ trigger từ push thường trên `main` để tránh deploy mỗi khi commit `src/`/`tests/`.

## Notes / Blockers
* **Decision-point Task 1:** vị trí package (root `src/latent_anything/` vs một sub-project `uv` riêng như `latent-anything-theory/`). Chốt trong khi thực thi, ghi lại bằng skill `log-decision` nếu lệch khỏi convention.
* Sprint này **không** định nghĩa Protocol/ABC/interface — đúng Rule of Three §4a (0 instance ⇒ không trừu tượng hóa).
* Public API arrays = `numpy`; `torch` chỉ internal — không leak ([python rule](../../.agents/rules/python.md)).
* Mỗi task một commit riêng theo [git rule](../../.agents/rules/git.md); stage file theo tên, không `git add .`.
