# Sprint 4 Plan

## Sprint Goal
Increment đầu tiên (Round 1): `LatentSpace` (euclidean flat) + `Trajectory` (immutable) + PCA method, **hardcoded**, chạy end-to-end từ latent tổng hợp đến visualize 2D. Kết thúc: **giữ hardcoded** theo Rule of Three (instance #1 — chưa extract `Method` interface).

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

- [ ] Task 1: `LatentSpace` concrete class — chỉ euclidean flat vector (`dim`, `geometry="euclidean"`, metadata model nguồn). Public surface là `numpy`.
- [ ] Task 2: `Trajectory` concrete class — immutable, giữ sequence latent state (numpy); trajectory một-điểm hợp lệ; ops tối thiểu (`len`, indexing/slice trả về `Trajectory` mới, `to_numpy`).
- [ ] Task 3: PCA method concrete class — stateful (`fit`/`transform`), dùng `scikit-learn` internal, in/out `numpy`.
- [ ] Task 4: Đường đi end-to-end (script hoặc notebook): latent tổng hợp → `Trajectory` → PCA `fit` → projection 2D.
- [ ] Task 5: Visualization 2D của projection (`matplotlib` static, hoặc `plotly` interactive).
- [ ] Task 6: Tests — pytest cho `LatentSpace`; `hypothesis` cho tính immutable/slice của `Trajectory`; roundtrip/shape của PCA.
- [ ] Task 7: `ruff check` + `ruff format` + `pyright` strict sạch.
- [ ] Task 8: Áp Rule of Three §4a — ghi quyết định "giữ hardcoded, chưa tạo `Method` interface" + artifact summary (skill `implement-atomic-task`).
- [ ] Task 9: Đối chiếu ADR §4c — ghi ADR `LatentSpace` vẫn `pending` (mới chạm euclidean) + entry `CHANGELOG.md`.

## Notes / Blockers
* Phụ thuộc Sprint 3 (package + tooling phải xong trước).
* Đây là vertical slice đầu chạm `LatentSpace` + `Trajectory` + `Method` cùng lúc, nhưng **không** tách interface — ba primitive ở dạng concrete class. Interface chỉ extract khi instance #3 khác triết lý xuất hiện (Sprint 6).
* Không kéo PyTorch vào public signature; nếu PCA cần torch thì chỉ internal (ở đây sklearn là đủ).
* Mỗi task một commit riêng; `Trajectory` immutable là quyết định đã chốt (ARCHITECTURE §7) — không làm mutable.
