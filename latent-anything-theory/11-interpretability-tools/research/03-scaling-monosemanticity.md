# Scaling Monosemanticity (Anthropic, 2024)

> **TL;DR.** "Scaling Monosemanticity" (Templeton et al., 2024) đưa phương pháp [SAE](02-towards-monosemanticity.md) từ transformer 1 lớp lên **Claude 3 Sonnet** (model production cỡ trung), trích **hàng triệu feature** từ residual stream lớp giữa — gồm feature về người nổi tiếng, thành phố, lỗ hổng bảo mật, và các feature *an toàn* (lừa dối, thiên kiến, sycophancy). Chứng minh feature **đa ngôn ngữ + đa phương thức** và **lái được nhân quả** (clamp feature "Cầu Cổng Vàng" cao → "Golden Gate Claude"). Trọng tâm bài: *cách đánh giá chất lượng feature ở quy mô lớn* (specificity, influence, feature neighborhoods). Caveat: chỉ cover một phần feature của model; SAE đắt; "feature" vẫn phụ thuộc dung lượng và đánh giá còn thủ công.

[Towards Monosemanticity](02-towards-monosemanticity.md) chứng minh SAE *hoạt động* ở quy mô toy. Câu hỏi sống còn: nó có *scale* lên model thật không? Bài này trả lời "có" trên Claude 3 Sonnet, và quan trọng hơn cho framework — nó đặt ra *quy trình đánh giá feature* mà Layer A cần để dùng SAE một cách đáng tin.

---

## 1. Trực giác: SAE tuân scaling law

Phát hiện cốt lõi: train SAE lớn hơn (nhiều feature hơn, nhiều compute hơn) trên activation của một model lớn vẫn cho feature *sạch và diễn giải được* — chất lượng tăng theo quy mô như mọi thứ trong deep learning. Anthropic train ba SAE với **~1 triệu, 4 triệu, và 34 triệu feature** trên residual stream lớp giữa của Sonnet. Càng nhiều feature, càng bắt được khái niệm hiếm/mịn (feature splitting ở quy mô lớn).

Điều này biến SAE từ "thí nghiệm trên toy" thành *công cụ thực tế cho model production* — tiền đề để Layer A của Latent-Anything áp SAE lên model lớn người dùng load vào.

---

## 2. Các loại feature tìm được

| Nhóm feature | Ví dụ |
|---|---|
| Thực thể | người nổi tiếng, thành phố, địa danh (Cầu Cổng Vàng) |
| Kỹ thuật | lỗ hổng bảo mật trong code, hàm/cú pháp |
| Trừu tượng | sự lừa dối, sycophancy (nịnh), thiên kiến, "bí mật" |
| Đa phương thức | một feature active cho cả *text* lẫn *ảnh* của cùng khái niệm |

Hai tính chất nổi bật:

- **Đa ngôn ngữ**: feature cho một khái niệm active bất kể ngôn ngữ (tiếng Anh, Trung, ...). Gợi ý model có biểu diễn *khái niệm-trung tâm*, không phải token-trung tâm.
- **Đa phương thức**: feature kích hoạt cho cả mô tả văn bản lẫn hình ảnh — biểu diễn vượt modality.

Đặc biệt giá trị cho an toàn: feature về lừa dối/thiên kiến cho phép *phát hiện và can thiệp* hành vi rủi ro — đúng loại introspection Layer A nhắm tới.

---

## 3. Đánh giá chất lượng feature (trọng tâm)

Ở quy mô triệu feature, không thể kiểm tra thủ công từng cái; bài chuẩn hóa bộ tiêu chí:

1. **Specificity** — feature có chỉ active đúng khái niệm của nó không (xét các ví dụ kích hoạt mạnh nhất, kiểm độ thuần).
2. **Influence (nhân quả)** — *clamp* activation của feature lên cao/thấp rồi đo thay đổi output. Đây là bằng chứng feature *làm gì đó*, không chỉ tương quan. Ví dụ kinh điển: clamp feature "Cầu Cổng Vàng" → model đồng nhất nó với cây cầu trong mọi câu trả lời ("Golden Gate Claude").
3. **Feature neighborhoods (hình học)** — feature gần nhau (cosine) thường *liên quan ngữ nghĩa* (vd các feature về tâm lý nằm cụm với nhau). Cấu trúc hình học của feature space *có nghĩa*, mở ra phân tích geometry.
4. **Completeness/coverage** — SAE bắt được bao nhiêu phần hành vi model; còn bao nhiêu "dark matter" chưa giải thích (reconstruction error còn lại).

Bộ tiêu chí này chính là *spec đánh giá* cho bất kỳ phương pháp introspection nào — framework nên áp dụng nguyên xi.

---

## 4. Giới hạn / Khi nào thất bại

**Chỉ một phần feature.** 34M feature vẫn không phủ hết; nhiều hành vi nằm trong "dark matter" reconstruction error chưa giải thích. Không phải interpretability hoàn chỉnh.

**Cực kỳ tốn kém.** Train SAE 34M feature trên activation model production cần hạ tầng lớn; ngoài tầm hầu hết nhóm nhỏ.

**Feature phụ thuộc dung lượng.** Như feature splitting đã chỉ, "feature" không tuyệt đối — đổi kích thước SAE đổi tập feature; khó nói "đây là feature thật của model".

**Đánh giá còn thủ công/LLM-aided.** Specificity và gán nhãn vẫn dựa con người hoặc model phụ; rủi ro bias, chưa hoàn toàn tự động/khách quan.

**Một lớp một lúc.** Chủ yếu phân tích residual stream một lớp; *circuit* nối feature qua nhiều lớp (cách feature tương tác) vẫn là biên giới (cross-layer transcoders sau này).

---

## 5. Liên hệ với Latent-Anything

Bài này cho framework hai thứ: bằng chứng SAE *scale*, và một *bộ tiêu chí đánh giá feature* chuẩn. Layer A nên phơi bày đúng các metric này.

```python
class FeatureReport(Protocol):
    def specificity(self, feature_id: int) -> float: ...          # độ thuần
    def steer(self, feature_id: int, strength: float) -> None: ... # clamp -> đo influence nhân quả
    def neighbors(self, feature_id: int, k: int) -> list[int]: ... # feature geometry
    def coverage(self) -> float: ...                               # 1 - reconstruction error tỉ lệ
```

- **Layer A — Introspection**: đây *là* bản thiết kế Layer A ở quy mô lớn — trích feature, đo specificity/influence/coverage, duyệt feature neighborhoods. Feature an toàn (lừa dối, bias) là use case introspection giá trị nhất.
- **Layer B — Manipulation**: feature steering (clamp) là manipulation nhân quả sạch nhất — "Golden Gate Claude" là minh chứng [steering](../../05-probing-intervention/research/09-steering-vectors.md) ở mức feature mono-semantic, mạnh hơn steering vector thô.
- **Layer C — Runtime**: train + chạy SAE 34M feature trên activation production là workload runtime cực nặng; Layer C lo thu hoạch activation, train phân tán, và phục vµ feature lookup hiệu quả.

Hai mục SAE khép trục mech-interp. Tầng chuyển sang công cụ phân tích còn lại: **probing classifiers survey** (khi nào dùng probe gì), rồi trực quan hóa chiều cao (**UMAP**, **PaCMAP**).

---

## Liên quan

- [Towards Monosemanticity](02-towards-monosemanticity.md) — phương pháp SAE mà bài này scale lên Claude 3 Sonnet.
- [Mechanistic Interpretability — Tổng quan](01-mechanistic-interpretability-overview.md) — bối cảnh chương trình.
- [Steering Vectors](../../05-probing-intervention/research/09-steering-vectors.md) — feature steering (Golden Gate Claude) là dạng mạnh hơn.
- [Sparse Autoencoder — SAE](../../05-probing-intervention/research/07-sparse-autoencoder.md) — kiến trúc cơ sở.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — feature neighborhoods là phân tích hình học feature space.

## Tham khảo

- A. Templeton, T. Conerly, J. Marcus, et al., *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (Anthropic, Transformer Circuits, 2024).
- T. Bricken et al., *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning* (Anthropic, 2023).
- N. Elhage et al., *Toy Models of Superposition* (Anthropic, 2022, arXiv:2209.10652).
