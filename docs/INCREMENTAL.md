# Latent-Anything — Incremental Development & Generalization Rule

> Quy trình phát triển `src` theo kiểu incremental, và rule tổng quát hóa interface sau mỗi vòng.
> Phiên bản: 0.1 — 2026-06-16
>
> Đây là tài liệu **quy trình**, refinement của 9 Giai đoạn trong [ARCHITECTURE §5](ARCHITECTURE.md), không phải một roadmap song song. Khi tài liệu này và ARCHITECTURE mâu thuẫn về *thứ tự lớn*, ARCHITECTURE thắng; tài liệu này chỉ chi tiết hóa *cách* đi bên trong mỗi giai đoạn.

---

## 1. Nguyên tắc nền

Hai cam kết đã chốt từ trước, tài liệu này biến chúng thành quy trình thực thi được:

- *"Interface đúng được **discovered**, không được **designed**"* — [IDEA §7](IDEA.md).
- *"Interface evolution: incremental dưới 1.0, **extract từ working code**"* — [ARCHITECTURE §7](ARCHITECTURE.md).

Hệ quả trực tiếp: **không build interface trừu tượng trước khi có code cụ thể chạy được.** Mỗi primitive (`LatentSpace`, `Trajectory`, `ModelAdapter`, `Method`, `Pipeline`) bắt đầu đời sống của nó dưới dạng class cụ thể, hardcoded, hẹp — rồi mới được nâng lên interface khi đủ bằng chứng.

---

## 2. Đơn vị "increment"

Một **increment** (vòng) là đơn vị công việc nhỏ nhất thỏa cả ba:

1. Thêm **đúng một** instance cụ thể mới (một method, một adapter, một geometry case, hoặc một operation của `Trajectory`).
2. Chạy được **end-to-end** qua đường đi đang có (không để code chết, không stub treo).
3. Kết thúc bằng một lần áp dụng [Generalization Rule §4](#4-generalization-rule-—-rule-of-three) và ghi lại kết quả.

Increment ≠ Giai đoạn. Một Giai đoạn (ARCHITECTURE §5) gồm nhiều increment. Một increment **không bao giờ** vừa thêm instance vừa freeze interface trong cùng một bước trừ khi nó chính là instance kích hoạt Rule of Three (xem dưới).

---

## 3. ADR là giả thuyết, không phải spec khởi đầu

Ba ADR ngày 2026-06-16 trong [decisions.md](../.agents/memory/decisions.md) — geometry-keyed `LatentSpace`, `ModelAdapter` 3 mode, geometry-dispatch cho distance/interpolation — tự dán nhãn *"theory-supported position to validate in Giai đoạn 1"*. Tài liệu này khẳng định cách đọc đúng:

> Ba ADR đó là **giả thuyết về hình dạng cuối**, không phải interface phải build đủ ngay từ increment đầu.

Cụ thể:

- Increment đầu **được phép — và nên** — implement hẹp hơn ADR nhiều. Ví dụ: `LatentSpace` đầu tiên chỉ cần geometry `euclidean` cho flat vector; chưa cần enum đầy đủ, chưa cần Gaussian set.
- Mỗi ADR mang trạng thái ngầm: `pending` (chưa có code chạm tới) → `validated` (code xác nhận) hoặc `revised` (code phản bác, đã có ADR đảo).
- Coi ADR như spec-build-sẵn = quay lại "design from imagination", đúng cái IDEA §7 cấm. Tránh.

---

## 4. Generalization Rule — Rule of Three

Đây là rule trung tâm bạn hỏi. Áp dụng **cuối mỗi increment**, hai chiều: khi nào *nâng* abstraction, và làm gì khi code *phản bác* một ADR.

### 4a. Chiều nâng abstraction (theo số instance)

Đếm số instance cụ thể của abstraction mà increment vừa rồi chạm tới (một `Method`, một `ModelAdapter`, một geometry, một `Trajectory` op):

| Số instance | Hành động bắt buộc |
|---|---|
| **1** | Giữ **hardcoded**. Không trừu tượng hóa. Không tạo Protocol/ABC "cho tương lai". Lặp lại code là chấp nhận được ở đây. |
| **2** | Được phép phác **một shared shape tạm thời**, đánh dấu *unstable* (docstring ghi rõ "chưa freeze"). **Không** công bố ra public surface. Nếu hai instance vẫn vừa khít một class cụ thể, đừng tách interface vội. |
| **≥3, khác triết lý** | **Extract và freeze interface.** Đây là điểm verify mà [ARCHITECTURE §2](ARCHITECTURE.md) đã chỉ định sẵn (PCA→UMAP→**SAE**; lerp→steering→**activation patching**). Instance thứ 3 phải khác triết lý thật (stateless vs stateful vs train-trong-forward-pass), không phải bản sao gần giống. |

"Khác triết lý" là điều kiện cứng: ba method linear-deterministic gần giống nhau **không** kích hoạt freeze — chúng chưa stress-test được interface.

### 4b. Quy tắc migrate đồng thời

Khi freeze hoặc sửa interface, **migrate tất cả instance/call-site cũ trong cùng một commit.** Không để hình dạng cũ tồn tại song song. Đây là *"incremental không phải cẩu thả"* — [IDEA §7](IDEA.md): dưới 1.0 là license để break, không phải license để bỏ qua consistency.

### 4c. Chiều đối chiếu ADR

Mỗi increment trả lời: *code vừa viết xác nhận hay phản bác một ADR đang `pending`?*

- **Xác nhận** → ghi một dòng vào ADR đó chuyển trạng thái sang `validated`, kèm trỏ tới code/test chứng minh.
- **Phản bác** → **viết một ADR mới đảo/sửa** (append vào decisions.md, không xóa ADR cũ — log là append-only), nêu rõ code nào ép phải đảo, rồi migrate theo §4b.
- Tuyệt đối **không** im lặng làm khác ADR. Mâu thuẫn phải được ghi nhận thành quyết định mới có chủ đích.

### 4d. Kỷ luật public surface

Chỉ promote một interface ra public surface (`ModelAdapter`, `Method`, `Pipeline`) khi **≥2 use case thật** cần tới nó — [IDEA §7](IDEA.md) / [ARCHITECTURE §3](ARCHITECTURE.md). Mọi thứ khác giữ internal, tự do refactor.

---

## 5. Checklist cuối mỗi increment

Chạy đủ 5 mục này trước khi coi increment là xong:

1. [ ] Instance mới chạy **end-to-end**, có test (pytest; `hypothesis` cho core primitive).
2. [ ] `ruff check` + `ruff format` + `pyright` (strict) sạch — [python rule](../.agents/rules/python.md).
3. [ ] Áp [Rule of Three §4a](#4a-chiều-nâng-abstraction-theo-số-instance): hoặc giữ hardcoded, hoặc phác shape tạm, hoặc freeze+migrate. Ghi rõ đã chọn nhánh nào.
4. [ ] Đối chiếu ADR [§4c](#4c-chiều-đối-chiếu-adr): cập nhật `validated` hoặc viết ADR đảo nếu phản bác.
5. [ ] Ghi artifact summary ngắn (theo skill `implement-atomic-task`) nêu: instance gì, extract gì, ADR nào dịch chuyển.

---

## 6. Plan tổng incremental — Giai đoạn 1 → 2

Diễn giải [ARCHITECTURE §5](ARCHITECTURE.md) thành chuỗi increment cụ thể. Mỗi dòng là một vòng; instance kích hoạt freeze được **in đậm**.

**Giai đoạn 1 — Core primitives & first integrations**

| Vòng | Nội dung | Instance # | Mục tiêu Rule-of-Three |
|---|---|---|---|
| 0 | Scaffold: `uv init`, package skeleton, ruff/pyright/pytest, CI. Chưa có abstraction. | — | — |
| 1 | `LatentSpace` (euclidean flat) + `Trajectory` một-điểm + PCA, hardcoded, end-to-end → visualize. | Method #1, LatentSpace #1 | Giữ hardcoded |
| 2 | Thêm UMAP (nonlinear, stochastic, stateful). | Method #2 | Phác `Method` shape tạm, *unstable* |
| 3 | Thêm **SAE** (neural, train khác hẳn). | **Method #3 khác triết lý** | **Freeze `Method` interface, migrate PCA+UMAP** |
| 4 | Adapter **VAE** (own training, explicit learned latent — ADR `ModelAdapter` mode i). | ModelAdapter #1 | Giữ hardcoded |
| 5 | Adapter **VLA** (pretrained large). | ModelAdapter #2 | Phác `ModelAdapter` shape tạm |
| 6 | Geometry case thứ 2 (unit-norm/spherical) buộc geometry hint mang theo *metric* → validate ADR `LatentSpace` + ADR geometry-dispatch (slerp). | LatentSpace #2 | Xác nhận/đảo 2 ADR |

**Giai đoạn 2 — Layer B foundation**

| Vòng | Nội dung | Instance # | Mục tiêu |
|---|---|---|---|
| 7 | Lerp (stateless pure transform) trên `Trajectory`. | B-Method #1 | Giữ hardcoded |
| 8 | Steering vector (stateful, fit từ contrast pair). | B-Method #2 | Phác shape tạm |
| 9 | **Activation patching** (intervene trong forward pass). | **B-Method #3 khác triết lý** | **Freeze, migrate** |
| 10 | Showcase end-to-end: load VLA → tìm failure → edit latent → decode → quan sát đổi behavior. Reproducible từ config. | — | Validate giá trị thật của framework |

Geometry-dispatch ADR (slerp/Mahalanobis/log-exp/SO(3)) chỉ được freeze khi **method thứ 3 phụ thuộc metric** xuất hiện (vòng 6–9 vùng lân cận), đúng tinh thần ADR gốc "validate when the third Layer-B method lands".

Các Giai đoạn 3–9 giữ nguyên như ARCHITECTURE §5; tài liệu này sẽ được mở rộng khi Giai đoạn 1–2 hoàn tất và lộ ra pattern mới.

---

*Tài liệu này cập nhật khi quy trình incremental thay đổi hoặc khi một increment lộ ra rule tổng quát hóa cần điều chỉnh.*
