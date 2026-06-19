# Sprint 13 Plan

## Sprint Goal
Increment thứ mười (Round 10): tạo **showcase end-to-end đầu tiên** chứng minh Latent-Anything có giá trị ở mức framework, không chỉ ở mức từng primitive riêng lẻ. Story của sprint: **load/train adapter hiện có → encode vào latent → dùng Layer A để nhìn structure/failure region → dùng Layer B để edit latent → decode → đo và quan sát behavior thay đổi**. Kết thúc: có một showcase script/artifact reproducible từ config nhẹ, **không** introduce public interface mới.

## Vì sao Sprint 13 không dùng VLA thật?

`docs/INCREMENTAL.md` và `docs/PLAN.md` cũ ghi aspiration là "load VLA → edit latent → decode". Nhưng trạng thái repo hiện tại khác:

| Khía cạnh | VAE | RandomProjection | VLA thật |
|---|---|---|---|
| Có trong repo? | **Có** | **Có** | **Chưa có** |
| `encode`/`decode` meaningful? | **Có** — learned latent + learned decoder | Có nhưng decode chỉ xấp xỉ tuyến tính | Chưa khả dụng trong codebase |
| Phù hợp cho showcase edit→decode? | **Tốt nhất hiện tại** | Dùng như control/phụ nếu cần, không phải story chính | Là target dài hạn, không phải baseline khả thi của sprint này |

Vì vậy Sprint 13 phải trung thực với code hiện có:

- **Adapter chính:** `VAE`
- **B-Method chính:** `ActivationPatch`
- **Layer A support:** PCA projection của latent / failure region
- **Optional control:** RandomProjection hoặc trajectory interpolation, chỉ nếu giúp story rõ hơn

Sprint này **không** cố "giả vờ đã có VLA adapter". Thay vào đó, nó prove rằng với primitives hiện có, framework đã kể được một story end-to-end có giá trị; real VLA showcase sẽ là bước ecosystem tiếp theo khi adapter thật tồn tại.

## Đây là sprint composition/validation, không phải sprint freeze interface

Sprint 13 khác Sprint 10–12 ở chỗ nó **không thêm method/adapter instance mới**. Mục tiêu là **compose** những instance đã có:

- Adapter: `VAE`
- Layer A method: `PCA` (hoặc `UMAP` nếu cần cho plot phụ)
- Layer B method: `ActivationPatch` (primary), có thể dùng `Lerp` ở panel phụ để làm trajectory baseline

Nghĩa là success criterion của sprint này không phải "Protocol mới đã freeze", mà là:

1. Story chạy end-to-end với primitive hiện có.
2. Story có metric trước/sau rõ ràng, không chỉ eyeballing.
3. Story reproducible từ config nhẹ / fixed seed.
4. Không introduce abstraction mới chỉ để phục vụ demo.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

> **Sprint 13 status: COMPLETED** (2026-06-19). All 13 tasks implemented. First composition story of the framework — 68.2% distance improvement on synthetic VAE showcase.

- [x] Task 1: Chốt **showcase contract** ở mức artifact, không ở mức public API. Tạo một config nhẹ cho showcase (ví dụ JSON/TOML/pydantic-local-only) chứa seed, data generation params, VAE params, split source/target/test, output paths. Mục tiêu duy nhất: reproducibility cho Sprint 13; **không** promote thành framework-wide config system hay `Pipeline` API mới.
- [x] Task 2: Implement script `scripts/end_to_end_showcase_demo.py` làm orchestration entry point cho toàn bộ story. Script phải đọc config ở Task 1, generate/load synthetic data, train/load VAE, rồi chạy full path encode → inspect → edit → decode. Đây là showcase-level runner, không phải abstraction reusable mới trong `src/`.
- [x] Task 3: Implement **failure-case selection + baseline metrics** trong showcase script. Chọn một tập held-out source samples (failure region) và compute baseline metrics trước edit, ví dụ: reconstruction error, khoảng cách đến target cluster trong data space, hoặc latent-space distance đến target centroid. Mục tiêu là có tiêu chí "before vs after" rõ ràng thay vì chỉ nhìn plot.
- [x] Task 4: Integrate **Layer A introspection** vào story: project latent train/test points bằng PCA, highlight source region, target region, và failure samples sẽ bị edit. Nếu UMAP cần thiết để plot đẹp hơn thì chỉ thêm như view phụ; PCA vẫn là baseline deterministic, dễ kiểm định hơn cho showcase.
- [x] Task 5: Integrate **Layer B edit path** dùng `ActivationPatch` làm path chính: fit patch từ source/target split, apply lên held-out failure samples, decode lại, và đo metric sau edit. Success criterion tối thiểu: edited outputs dịch chuyển theo hướng target rõ ràng hơn baseline ở metric đã chọn.
- [x] Task 6: Add **trajectory-level panel** cho story bằng cách reuse primitive sẵn có thay vì viết logic mới. Dùng `Lerp` để tạo latent trajectory baseline, rồi apply `ActivationPatch.apply_trajectory()` để decode chuỗi before/after. Mục tiêu: cho thấy framework xử lý được cả point-level edit và trajectory-level edit trong cùng narrative.
- [x] Task 7: Build visualization/output artifact hoàn chỉnh cho showcase: ít nhất một figure tổng hợp lưu vào `artifacts/` (ví dụ 2×2 hoặc 1×4 panels) và một console summary nêu seed, metric before/after, đường dẫn artifact. Artifact phải đủ rõ để được dùng như "first real story" của project.
- [x] Task 8: Add tests cho **showcase helpers / config loading / metric semantics**. Không cần snapshot test toàn bộ figure, nhưng cần test những phần có contract rõ: config parse ổn định, failure selection shape đúng, metric before/after có chiều hướng mong đợi trên synthetic setup, output arrays không mutate input. Nếu tách helper ra file riêng thì test file mirror tương ứng dưới `tests/`.
- [x] Task 9: Tooling gate — `ruff check` + `ruff format` + `pyright` strict clean. Toàn bộ test suite pass, cộng thêm test mới của showcase. Nếu script dùng config model local, vẫn phải giữ public API sạch numpy-only và không leak torch ra public surface.
- [x] Task 10: Rule of Three §4a — ghi artifact summary: "Sprint 13 là **composition/validation round**, không thêm adapter/method instance mới và không freeze interface mới. Showcase chỉ compose các primitive đã validated (`LatentSpace` geometry dispatch, `BMethod`, VAE/RandomProjection shared adapter shape) thành một story end-to-end." Đây là điểm quan trọng: **không vì cần demo mà lén tạo abstraction mới**.
- [x] Task 11: ADR check §4c — không có ADR nào đổi trạng thái ở Sprint 13 nếu implementation chỉ compose existing parts. `ModelAdapter` 3-mode ADR vẫn `pending`: showcase thêm evidence rằng Layer B consumer path hoạt động tốt với mode (i), nhưng vẫn **không** xác nhận mode (ii)/(iii). Nếu trong quá trình làm story phải đổi giả định nào về adapter/method shape, phải append ADR mới thay vì silent drift.
- [x] Task 12: Update `CHANGELOG.md` `[Unreleased]` — thêm end-to-end showcase script, reproducible config artifact, metric/report artifact, và user-facing note rằng framework đã có first composition story ở Layer A+B với adapter hiện có.
- [x] Task 13: Update `docs/PLAN.md` — mark Sprint 13 completed, move nó ra khỏi backlog, và mô tả ngắn rằng sprint này dùng **VAE-based showcase** làm current executable milestone while real VLA showcase remains future ecosystem work.

## Rule-of-Three checkpoint (to verify at end)
| Check | Status |
|---|---|
| New method instance? | **No** — Sprint 13 chỉ compose Lerp / ActivationPatch / PCA đã có |
| New adapter instance? | **No** — dùng `VAE` làm adapter chính; `RandomProjection` chỉ có thể là control phụ |
| Rule branch | **Composition round** → không extract/freeze interface mới |
| Public API change? | **Không nên có** — nếu cần reproducibility/config, giữ ở mức local artifact cho showcase |
| ADR impact | `ModelAdapter` 3-mode ADR vẫn `pending`; geometry ADRs và `BMethod` ADR giữ nguyên validated state |

## Showcase Design Notes

Story mong muốn của Sprint 13:

1. Generate/train một latent world đơn giản nhưng meaningful bằng `VAE`.
2. Encode data vào latent, dùng PCA để thấy hai region/source-target rõ ràng.
3. Chọn một failure slice held-out từ source region.
4. Fit `ActivationPatch` để học hướng dịch chuyển source → target.
5. Apply patch lên failure slice và decode ra data space.
6. So sánh before/after bằng metric + visualization.
7. Reuse `Lerp` để tạo trajectory baseline, rồi patch trajectory để thấy framework handle cả point và sequence.

Điểm mấu chốt: đây phải là **framework showcase**, không phải "thêm một demo method nữa". Nghĩa là script phải làm nổi bật:

- Adapter boundary (`encode` / `decode`)
- `LatentSpace` như handle chung
- Layer A để *thấy* problem
- Layer B để *sửa* problem
- Artifact reproducible để người khác rerun được

## Notes / Blockers

* **Real VLA adapter chưa tồn tại trong repo.** Sprint 13 không được assume VLA đã có. Nếu muốn bám literal roadmap "load VLA", phải tách thành sprint khác hoặc pre-sprint infrastructure task — không nhét lén vào sprint showcase này.
* **Không tạo `Pipeline` abstraction chỉ vì cần một script đẹp.** Repo hiện chưa có `Pipeline` class hay config system frozen; showcase config ở sprint này chỉ là local mechanism để reproducibility, không phải public contract.
* **Prefer reuse over copy-paste.** `end_to_end_activation_patch_demo.py`, `end_to_end_lerp_demo.py`, và `end_to_end_vae_demo.py` đã có logic hữu ích. Sprint 13 nên consolidate/refactor vừa đủ để kể story chung, không nhân bản thêm một đống demo gần giống nhau.
* **Metric phải đi trước aesthetics.** Plot đẹp là tốt, nhưng sprint này fail nếu chỉ có hình mà không có before/after metric hoặc acceptance criterion rõ ràng.
* **Artifacts phải được lưu rõ ràng.** Ít nhất một figure và một text summary/console summary với seed + metric + output path. Đây là deliverable để chứng minh project value theo `IDEA.md` §9.
* Mỗi task một commit theo Conventional Commits (`feat(showcase):`, `test(showcase):`, `docs(plan):`, `chore:`) khi triển khai.
