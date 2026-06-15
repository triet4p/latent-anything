# Towards Monosemanticity (Anthropic, 2023)

> **TL;DR.** Bài "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning" (Bricken et al., 2023) chứng minh trên một **transformer 1 lớp** rằng [sparse autoencoder](../../05-probing-intervention/research/07-sparse-autoencoder.md) huấn luyện trên activation MLP tách được [superposition](../../05-probing-intervention/research/06-superposition-hypothesis.md) thành hàng nghìn **feature mono-semantic** (mỗi feature một khái niệm: chữ Ả-Rập, DNA, base64...). Phương pháp: dictionary **overcomplete** (lớn hơn nhiều số chiều), loss = reconstruction + **L1 sparsity**, decoder chuẩn hóa norm đơn vị. Phát hiện then chốt: feature *dễ diễn giải hơn neuron* nhiều, **feature splitting** khi tăng dung lượng, và feature *universal* qua các lần train. Caveat: mới ở 1 lớp toy; dead feature, feature splitting, và đánh đổi sparsity–reconstruction là vấn đề mở.

[Mech interp overview](01-mechanistic-interpretability-overview.md) nêu SAE là lời giải cho superposition. Mục này đọc *kỹ phương pháp* của bài đặt nền cho cả chương trình SAE — nó biến "feature = direction" từ giả thuyết thành công cụ chạy được, và là khuôn mẫu mà Layer A của Latent-Anything triển khai.

---

## 1. Trực giác: đổi cơ sở để gỡ chồng chập

Neuron polysemantic vì model nhồi nhiều feature hơn số chiều ([superposition](../../05-probing-intervention/research/06-superposition-hypothesis.md)). Ý tưởng: nếu feature thật là các *hướng thưa* trong activation space, hãy học một **dictionary** các hướng đó sao cho mỗi activation = tổ hợp **thưa** của một vài hướng. Đổi từ "cơ sở neuron" (cố định theo trục) sang "cơ sở feature" (học được, overcomplete) phơi bày các đơn vị mono-semantic mà neuron che giấu.

Đây đúng là [dictionary learning](../../05-probing-intervention/research/08-dictionary-learning.md), với ràng buộc sparsity, học bằng autoencoder.

---

## 2. Phương pháp (đọc kỹ)

### Kiến trúc SAE

Cho activation $x \in \mathbb{R}^d$ (ở đây là activation lớp MLP của transformer 1 lớp), SAE tái tạo:

$$
f(x) = \mathrm{ReLU}\big(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}}\big), \qquad \hat x = b_{\text{dec}} + \sum_{i} f_i(x)\, d_i,
$$

trong đó $f(x) \in \mathbb{R}^{m}$ là **feature activations** (thưa, không âm nhờ ReLU), $m \gg d$ là kích thước dictionary (**overcomplete**), $d_i$ là cột dictionary (hướng feature $i$), và $b_{\text{dec}}$ là bias trừ trước/cộng lại (centering). Mỗi $d_i$ được **chuẩn hóa norm đơn vị** để $f_i$ đo đúng "lượng" feature, tránh model lách bằng cách phóng to $d_i$.

### Hàm mất mát

$$
\mathcal{L} = \underbrace{\lVert x - \hat x\rVert_2^2}_{\text{reconstruction}} \;+\; \lambda \underbrace{\sum_i f_i(x)\,\lVert d_i\rVert_2}_{\text{L1 sparsity}}.
$$

Số hạng đầu ép $\hat x$ giống $x$; số hạng L1 ép *ít feature active* mỗi lúc (sparsity). Hệ số $\lambda$ cân bằng hai mục tiêu — đây là hyperparameter quan trọng nhất: $\lambda$ lớn → thưa hơn, dễ diễn giải hơn, nhưng reconstruction tệ hơn (mất thông tin).

### Đánh giá feature

Bài đo chất lượng feature qua: **specificity** (feature chỉ active đúng khái niệm của nó), **automated interpretability** (mô tả feature rồi kiểm tra dự đoán activation), trực quan hóa các ví dụ kích hoạt mạnh nhất, và **ablation/steering** (tắt/bật feature, đo ảnh hưởng output → bằng chứng *nhân quả*, không chỉ tương quan).

---

## 3. Phát hiện then chốt

| Phát hiện | Ý nghĩa |
|---|---|
| **Feature ≫ neuron về diễn giải** | feature SAE mono-semantic; người đánh giá thấy đa số map sạch sang một khái niệm — hơn hẳn đọc neuron polysemantic. |
| **Feature splitting** | tăng $m$ (dung lượng dictionary) làm một feature "thô" *tách* thành nhiều feature mịn hơn (vd "chữ Ả-Rập" → các biến thể). Số feature là một *thang phân giải*, không cố định. |
| **Universality** | feature tương tự xuất hiện qua các lần train độc lập — gợi ý chúng là cấu trúc *thật* của model, không phải artifact. |
| **Nhân quả** | ablation/steering feature thay đổi output đúng dự đoán → feature *làm gì đó*, không chỉ tương quan. |

Đây là bằng chứng thực nghiệm mạnh đầu tiên rằng superposition *gỡ được*, biến mech interp từ lý thuyết thành quy trình.

---

## 4. Giới hạn / Khi nào thất bại

**Mới ở quy mô toy.** Transformer 1 lớp; câu hỏi liệu phương pháp scale lên model lớn là chủ đề của [Scaling Monosemanticity](03-scaling-monosemanticity.md).

**Dead features.** Nhiều feature không bao giờ active (chết) — lãng phí dung lượng, cần resampling (giống [codebook collapse](../../09-discrete-latent/research/04-codebook-collapse.md) của VQ).

**Feature splitting nhai mặt.** Là tính năng (phân giải điều chỉnh được) nhưng cũng là vấn đề: "feature" không phải đơn vị tuyệt đối, làm khó so sánh và đếm.

**Đánh đổi sparsity–reconstruction.** $\lambda$ lớn → diễn giải tốt nhưng SAE bỏ sót thông tin (reconstruction error cao); không có điểm tối ưu phổ quát.

**Diễn giải vẫn chủ quan.** Gán nhãn khái niệm cho feature dựa vào con người/LLM; rủi ro confirmation bias, cần đánh giá nhân quả nghiêm ngặt.

---

## 5. Liên hệ với Latent-Anything

Bài này là *thuật toán tham chiếu* cho một `Method` cốt lõi của Layer A: phân rã latent thành feature mono-semantic.

```python
class SparseAutoencoder(Protocol):
    dict_size: int                  # m, overcomplete (m >> d)
    l1_coeff: float                 # lambda
    def encode(self, x: np.ndarray) -> np.ndarray: ...   # -> sparse feature activations (m,)
    def decode(self, f: np.ndarray) -> np.ndarray: ...   # -> reconstruction
    def feature_directions(self) -> np.ndarray: ...      # (m, d), unit-norm
```

- **Layer A — Introspection**: SAE là công cụ introspection hạng nhất — load activation của model bất kỳ, train SAE, phơi bày feature mono-semantic + ví dụ kích hoạt. Các metric (specificity, dead-feature rate, sparsity) là chẩn đoán chuẩn Layer A cần phơi bày.
- **Layer B — Manipulation**: feature SAE là *hướng có ngữ nghĩa sạch* để [steering](../../05-probing-intervention/research/09-steering-vectors.md) — bật/tắt một feature là can thiệp diễn giải được, sạch hơn steering vector thô.
- **Layer C — Runtime**: train SAE trên hàng tỉ activation là workload nặng; Layer C lo thu hoạch/cache activation và train dictionary hiệu quả — đúng loại pipeline scale mà framework hướng tới.

Bài đặt nền; mục kế tiếp — [Scaling Monosemanticity](03-scaling-monosemanticity.md) — scale chính phương pháp này lên Claude 3 Sonnet và bàn cách *đánh giá chất lượng feature* ở quy mô lớn.

---

## Liên quan

- [Sparse Autoencoder — SAE](../../05-probing-intervention/research/07-sparse-autoencoder.md) — kiến trúc/loss SAE; mục này là ứng dụng đặt nền.
- [Dictionary Learning](../../05-probing-intervention/research/08-dictionary-learning.md) — khung tổng quát; SAE là instance có sparsity.
- [Superposition Hypothesis](../../05-probing-intervention/research/06-superposition-hypothesis.md) — vấn đề SAE giải.
- [Mechanistic Interpretability — Tổng quan](01-mechanistic-interpretability-overview.md) — bối cảnh chương trình SAE.
- [Codebook Collapse](../../09-discrete-latent/research/04-codebook-collapse.md) — dead feature ~ dead code, cần resampling.

## Tham khảo

- T. Bricken, A. Templeton, J. Batson, et al., *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning* (Anthropic, Transformer Circuits, 2023).
- N. Elhage et al., *Toy Models of Superposition* (Anthropic, 2022, arXiv:2209.10652).
- B. A. Olshausen, D. J. Field, *Sparse coding with an overcomplete basis set* (Vision Research, 1997) — nguồn gốc sparse coding.
