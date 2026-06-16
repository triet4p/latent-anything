# Sprint 3 Plan

## Sprint Goal
Khởi tạo package `latent_anything` với tooling đầy đủ (uv, ruff, pyright strict, pytest, CI) chạy xanh end-to-end — **chưa có abstraction/interface nào**. Đây là Round 0 trong [INCREMENTAL.md §6](../INCREMENTAL.md).

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: Khởi tạo package `latent_anything` bằng `uv` (pyproject PEP 621, Python ≥3.11), chốt vị trí package (root-level vs sub-project) theo [python rule](../../.agents/rules/python.md).
- [ ] Task 2: Cấu hình `ruff` (check + format) trong pyproject — rule set + line length.
- [ ] Task 3: Cấu hình `pyright` strict mode trong pyproject/pyrightconfig.
- [ ] Task 4: Thêm `pytest` + tạo `tests/` mirroring `src/`, cấu hình test discovery.
- [ ] Task 5: Thêm `hypothesis` vào dev dependencies (chuẩn bị cho property-based test của core primitive).
- [ ] Task 6: Tạo `latent_anything/__init__.py` (chỉ version + docstring) và một smoke test trivial chạy xanh.
- [ ] Task 7: Thêm CI workflow (GitHub Actions) chạy `ruff check` + `ruff format --check` + `pyright` + `pytest`.
- [ ] Task 8: Cập nhật `README.md` (Project Structure + Quick Start) phản ánh package mới.
- [ ] Task 9: Thêm entry vào `CHANGELOG.md` `[Unreleased]` cho việc khởi tạo package.

## Notes / Blockers
* **Decision-point Task 1:** vị trí package (root `src/latent_anything/` vs một sub-project `uv` riêng như `latent-anything-theory/`). Chốt trong khi thực thi, ghi lại bằng skill `log-decision` nếu lệch khỏi convention.
* Sprint này **không** định nghĩa Protocol/ABC/interface — đúng Rule of Three §4a (0 instance ⇒ không trừu tượng hóa).
* Public API arrays = `numpy`; `torch` chỉ internal — không leak ([python rule](../../.agents/rules/python.md)).
* Mỗi task một commit riêng theo [git rule](../../.agents/rules/git.md); stage file theo tên, không `git add .`.
