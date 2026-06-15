# Mechanistic Interpretability — Tổng quan

> **TL;DR.** Mechanistic interpretability (mech interp) cố *reverse-engineer* mạng nơ-ron thành thuật toán hiểu được. Ba tuyên bố nền (Olah et al., "Zoom In"): **feature** là đơn vị cơ bản, feature **tương ứng với direction** trong activation space, và feature nối nhau bằng trọng số tạo thành **circuit** — đều nghiên cứu được chặt chẽ. Trở ngại trung tâm: [superposition](../../05-probing-intervention/research/06-superposition-hypothesis.md) — model nhồi *nhiều feature hơn số chiều*, nên neuron polysemantic; lời giải hàng đầu là [sparse autoencoder](../../05-probing-intervention/research/07-sparse-autoencoder.md) tách feature mono-semantic. Đây là nền lý thuyết cho Layer A của Latent-Anything. Caveat: phần lớn bằng chứng ở quy mô nhỏ/toy; "circuit" hoàn chỉnh cho model lớn vẫn hiếm.

Tầng 5 đã dựng từng *công cụ* (probing, activation patching, SAE, steering). Mục này là *bản đồ* gắn chúng lại: mech interp là chương trình nghiên cứu tổng thể mà các công cụ đó phục vụ, và là khung khái niệm trực tiếp định hình Layer A (introspection). Đọc song song tầng 5.

---

## 1. Ba tuyên bố nền (Zoom In)

Olah et al. (Distill, 2020) đặt ba giả thuyết làm nền cho cả lĩnh vực:

1. **Features** — đơn vị cơ bản của mạng là *feature* (một khái niệm/đặc trưng model học), không phải neuron hay layer.
2. **Features là directions** — mỗi feature tương ứng một **hướng** trong activation space; activation = tổ hợp tuyến tính các hướng feature. Đây là nền của [linear structure](../../03-geometry-structure/research/01-linear-structure.md) và mọi phép [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md).
3. **Circuits** — feature nối với nhau qua trọng số tạo thành *circuit* (mạch tính toán); circuit nghiên cứu được chặt chẽ như feature.

Tuyên bố then chốt cho framework là (2): nếu khái niệm là *hướng*, thì introspection (đo hướng có encode gì) và manipulation (cộng/trừ hướng) đều có cơ sở. Đó chính là giả định ngầm dưới [probing](../../05-probing-intervention/research/01-linear-probing.md), [TCAV](../../05-probing-intervention/research/03-tcav.md), và [steering vectors](../../05-probing-intervention/research/09-steering-vectors.md).

---

## 2. Superposition: trở ngại trung tâm

Nếu mỗi feature là một hướng *trực giao*, ta chỉ có $d$ feature trong không gian $d$ chiều. Nhưng model học **nhiều feature hơn số chiều** khi feature *thưa* (hiếm khi active cùng lúc): đó là [superposition](../../05-probing-intervention/research/06-superposition-hypothesis.md) — nhồi $k > d$ hướng gần-trực-giao, chấp nhận nhiễu nhỏ. Hệ quả nặng nề:

- **Neuron polysemantic**: một neuron active cho nhiều khái niệm không liên quan → không đọc được trực tiếp.
- **PCA không đủ**: các hướng feature không trùng trục phương sai lớn; cần phương pháp khác để tách.

Superposition là lý do mech interp khó: thông tin *có* ở đó (linearly decodable) nhưng *chồng chập*, không phơi bày theo neuron.

---

## 3. Lời giải: dictionary learning / SAE

Cách tách superposition được ưa chuộng nhất: học một **overcomplete dictionary** các hướng feature sao cho mỗi activation = tổ hợp *thưa* của chúng. [Sparse autoencoder](../../05-probing-intervention/research/07-sparse-autoencoder.md) là hiện thân: encode activation thành một vector thưa chiều cao (nhiều feature hơn neuron), mỗi feature kỳ vọng **mono-semantic**. Đây là một instance của [dictionary learning](../../05-probing-intervention/research/08-dictionary-learning.md) tổng quát.

Chương trình SAE là trục chính của mech interp hiện đại: [Towards Monosemanticity](02-towards-monosemanticity.md) chứng minh trên một layer nhỏ, còn [Scaling Monosemanticity](03-scaling-monosemanticity.md) scale lên Claude 3 Sonnet. Đây là nơi tầng 11 đào sâu nhất.

---

## 4. Circuits và transformer

Mạch (circuit) là *thuật toán con* thực hiện bởi một tập feature + trọng số nối. Hai mốc:

- **A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021): phân tích attention thành **QK circuit** (head chú ý vào đâu) và **OV circuit** (head copy/biến đổi gì), cho ngôn ngữ chính xác để mô tả mạch.
- **Induction heads** (Olsson et al., 2022): một mạch hai-head nổi tiếng thực hiện in-context learning kiểu "[A][B]...[A]→[B]" (tìm lần xuất hiện trước của token hiện tại rồi copy token theo sau). Đây là ví dụ "circuit hoàn chỉnh" đẹp nhất.

Công cụ tìm circuit: [activation patching](../../05-probing-intervention/research/05-activation-patching.md) (can thiệp để định vị nơi thông tin được xử lý), [logit lens](../../05-probing-intervention/research/10-logit-lens-tuned-lens.md) (đọc sớm để theo dòng thông tin), và ablation. TransformerLens (Neel Nanda) là thư viện tooling phổ biến.

---

## 5. Giới hạn / Khi nào thất bại

**Bằng chứng phần lớn ở quy mô nhỏ.** Circuit đầy đủ chủ yếu tìm thấy trên toy model/layer ít; mạch hoàn chỉnh cho model lớn vẫn hiếm và tốn công.

**SAE không hoàn hảo.** Feature SAE có thể *chia nhỏ* (feature splitting), không ổn định giữa các seed, và "mono-semantic" là mức độ chứ không tuyệt đối; reconstruction error đánh đổi với sparsity.

**Giả định tuyến tính.** "Feature = direction" mạnh nhưng không phổ quát; có feature phi tuyến/đa chiều (manifold) mà mô hình hướng tuyến tính bỏ sót.

**Diễn giải mang tính kể chuyện.** Gán nhãn ngữ nghĩa cho feature/circuit dễ rơi vào *confirmation bias*; cần đánh giá nhân quả (intervene, đo causal effect) chứ không chỉ tương quan.

**Quy mô con người.** Một model lớn có hàng triệu feature; hiểu *toàn bộ* là bất khả thi thủ công — cần tự động hóa (auto-interpretation).

---

## 6. Liên hệ với Latent-Anything

Mech interp *là* nền lý thuyết của **Layer A (introspection)**. Nó định nghĩa câu hỏi Layer A trả lời ("latent encode feature gì, ở hướng nào, qua mạch nào") và bộ công cụ chuẩn.

```python
class InterpToolkit(Protocol):
    def probe(self, latent: np.ndarray, labels: np.ndarray) -> float: ...        # feature có tuyến tính?
    def sae_features(self, latent: np.ndarray) -> np.ndarray: ...                # tách superposition
    def patch(self, run_a: np.ndarray, run_b: np.ndarray, loc: int) -> np.ndarray: ...  # causal localize
    def feature_directions(self) -> np.ndarray: ...                              # (n_features, d)
```

- **Layer A — Introspection**: ba tuyên bố Zoom In là *spec* của Layer A — phơi bày feature directions, đo chúng (probe/TCAV), tách superposition (SAE), và truy mạch (patching). Một model load vào framework nên cho Layer A truy cập activation để chạy đúng các phân tích này.
- **Layer B — Manipulation**: "feature = direction" biến manipulation thành cộng/chiếu vector ([subspace projection](../../04-latent-computation/research/04-subspace-projection.md), steering); circuit cho biết *can thiệp ở đâu* để có hiệu quả mong muốn.
- **Layer C — Runtime**: chạy SAE/patching ở quy mô là workload nặng (nhiều forward, activation lớn) — Layer C cache activation và batch các can thiệp.

Mục này khung hóa cả tầng 11. Hai mục kế tiếp đào sâu trục SAE — [Towards Monosemanticity](02-towards-monosemanticity.md) và [Scaling Monosemanticity](03-scaling-monosemanticity.md) — rồi tầng chuyển sang probing survey và các công cụ trực quan hóa (UMAP, PaCMAP).

---

## Liên quan

- [Superposition Hypothesis](../../05-probing-intervention/research/06-superposition-hypothesis.md) — trở ngại trung tâm mech interp giải.
- [Sparse Autoencoder — SAE](../../05-probing-intervention/research/07-sparse-autoencoder.md) — lời giải hàng đầu cho superposition.
- [Dictionary Learning](../../05-probing-intervention/research/08-dictionary-learning.md) — khung tổng quát mà SAE là một instance.
- [Activation Patching](../../05-probing-intervention/research/05-activation-patching.md) — công cụ tìm circuit nhân quả.
- [Cấu trúc tuyến tính trong latent](../../03-geometry-structure/research/01-linear-structure.md) — "feature = direction" là nền.

## Tham khảo

- C. Olah, N. Cammarata, L. Schubert, et al., *Zoom In: An Introduction to Circuits* (Distill, 2020).
- N. Elhage, N. Nanda, C. Olsson, et al., *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021).
- C. Olsson et al., *In-context Learning and Induction Heads* (Anthropic, 2022).
- N. Elhage et al., *Toy Models of Superposition* (Anthropic, 2022, arXiv:2209.10652).
