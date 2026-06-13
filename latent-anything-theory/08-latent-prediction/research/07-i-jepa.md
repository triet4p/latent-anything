# I-JEPA — Image-based Joint-Embedding Predictive Architecture

> **TL;DR.** I-JEPA dùng một context block lớn nhưng bị khoét bỏ các target regions để dự đoán representation của bốn target blocks tương đối lớn trong cùng ảnh. Target được lấy bằng cách encode **ảnh đầy đủ trước rồi mới chọn block**, predictor nhận positional mask tokens, loss là khoảng cách $L_2$ trong latent space, và target encoder được cập nhật bằng EMA. Điểm mạnh nằm ở multi-block masking tạo bài toán semantic; caveat là quality phụ thuộc mạnh vào geometry của mask, EMA target vẫn có thể collapse, và model chỉ học completion trong một ảnh chứ chưa phải temporal world model.

[JEPA](06-jepa.md) là họ kiến trúc tổng quát. I-JEPA là hiện thực cụ thể đầu tiên cho ảnh tĩnh: $x$ là một tập visible patches, $y$ là các vùng khác trong cùng ảnh, còn conditioning $z$ là positional mask tokens cho biết vị trí cần dự đoán. Không có pixel decoder trong training objective.

---

## **1. Pipeline tổng thể**

Cho ảnh $y$ được chia thành $N$ patches không chồng lấp. I-JEPA có ba module:

- context encoder $f_\theta$: ViT chỉ xử lý visible context patches;
- target encoder $f_{\bar\theta}$: ViT xử lý toàn bộ ảnh, không nhận gradient;
- predictor $g_\phi$: một ViT hẹp nhận context representations và positional mask tokens.

Target encoder tạo patch-level representations:

$$
s_y=f_{\bar\theta}(y)
=
\{s_{y_1},\ldots,s_{y_N}\}.
$$

Trong đó $s_{y_j}\in\mathbb{R}^d$ là target representation tại patch $j$. Toàn ảnh được encode trước khi target masks được áp dụng.

Với target block $B_i\subset\{1,\ldots,N\}$:

$$
s_y(i)=\{s_{y_j}\}_{j\in B_i}.
$$

Trong đó $s_y(i)$ là tập target tokens thuộc block thứ $i$. Target block được chọn **trên output của target encoder**, không phải bằng cách mask input target encoder.

Context mask $B_x$ xác định visible patches:

$$
s_x=f_\theta(y_{B_x})
=
\{s_{x_j}\}_{j\in B_x}.
$$

Trong đó $y_{B_x}$ là các input patches còn nhìn thấy sau khi loại vùng target khỏi context.

Predictor chạy riêng cho từng target block:

$$
\hat s_y(i)
=
g_\phi\!\left(
s_x,
\{m_j+p_j\}_{j\in B_i}
\right).
$$

Trong đó $m_j$ là shared learnable mask token, $p_j$ là positional embedding của patch $j$, và $\hat s_y(i)$ là predicted representation cho block $i$. Position nói predictor **dự đoán ở đâu**, không tiết lộ nội dung target.

---

## **2. Tại sao target phải encode ảnh đầy đủ trước**

Hai pipeline nhìn giống nhau nhưng tạo target khác bản chất:

**Mask trước rồi encode:**

$$
\tilde s_y=f_{\bar\theta}(\operatorname{mask}(y,B_i)).
$$

Trong đó representation tại target location bị ảnh hưởng trực tiếp bởi mask token hoặc thiếu context, nên target có thể chứa artifact của corruption.

**Encode trước rồi chọn target:**

$$
s_y(i)=\operatorname{select}\!\left(f_{\bar\theta}(y),B_i\right).
$$

Trong đó target token được tính khi target encoder nhìn toàn ảnh. Nó là contextualized representation của nội dung thật tại block, không phải representation của một vùng đã bị che.

Khác biệt này quan trọng vì self-attention của target encoder cho mỗi patch truy cập global context. Target có thể encode object part, pose và scene relation. Context encoder phải suy ra representation giàu ngữ nghĩa đó từ phần ảnh còn lại.

---

## **3. Multi-block masking**

Recipe trong paper cho ảnh $224\times224$:

1. sample $M=4$ target blocks;
2. mỗi target chiếm scale trong $(0.15,0.20)$ diện tích ảnh;
3. target aspect ratio trong $(0.75,1.5)$;
4. sample một context block scale $(0.85,1.0)$, aspect ratio 1;
5. xoá mọi target-overlap khỏi context.

Ký hiệu:

$$
B_x
=
B_{\mathrm{context}}
\setminus
\bigcup_{i=1}^{M}B_i.
$$

Trong đó $B_{\mathrm{context}}$ là block lớn ban đầu, $B_i$ là target masks, và $B_x$ là visible context cuối. Phép trừ đảm bảo predictor không nhìn thấy target content.

### Vì sao target cần lớn

Target patch đơn lẻ có thể được dự đoán bằng local texture hoặc edge continuation. Target block lớn thường cắt qua một phần có ý nghĩa của object/scene; để predict representation, model phải dùng shape, pose, object identity và quan hệ xa hơn.

### Vì sao context cần phân tán và đủ thông tin

Một context quadrant nhỏ có thể không chứa evidence về target. Ngược lại, context là toàn bộ complement quanh target cung cấp local boundary shortcut. I-JEPA bắt đầu từ block lớn rồi khoét nhiều target regions, tạo visible patches phân tán nhưng vẫn đủ scene evidence.

Paper ablation trên ViT-B/16, pretrain 300 epochs và ImageNet-1K 1% linear evaluation:

| Masking | Target | Context | Top-1 |
|---|---|---|---:|
| Multi-block | 4 blocks, scale 0.15–0.20 | large block trừ targets, avg. ratio 0.25 | 54.2 |
| Rasterized | 3 quadrants | 1 quadrant | 15.5 |
| Single block | 1 block, scale 0.6 | complement | 20.2 |
| Random patches | random 60% | complement | 17.6 |

Các số không chứng minh multi-block luôn tối ưu cho mọi domain. Chúng cho thấy **mask distribution định nghĩa abstraction được học**; giữ architecture nhưng đổi geometry của task có thể làm chất lượng semantic sụt mạnh.

---

## **4. Objective và cập nhật tham số**

Loss là trung bình squared $L_2$ trên target blocks và patch tokens:

$$
\mathcal{L}_{\mathrm{I\text{-}JEPA}}
=
\frac{1}{M}
\sum_{i=1}^{M}
\sum_{j\in B_i}
\left\|
\hat s_{y_j}-\operatorname{sg}(s_{y_j})
\right\|_2^2.
$$

Trong đó $M$ là số target blocks, $B_i$ là patch indices của block $i$, $\hat s_{y_j}$ là predictor output, và $s_{y_j}$ là EMA-target representation. Gradient cập nhật $\theta,\phi$, không chạy qua $\bar\theta$.

Context encoder và predictor:

$$
(\theta,\phi)
\leftarrow
\operatorname{Opt}\!\left(
\theta,\phi,
\nabla_{\theta,\phi}\mathcal{L}
\right).
$$

Trong đó $\operatorname{Opt}$ là optimizer step.

Target encoder:

$$
\bar\theta
\leftarrow
\mu\bar\theta+(1-\mu)\theta.
$$

Trong đó $\mu$ là [EMA momentum](05-ema-target-encoder.md). Target coordinate system thay đổi chậm hơn context encoder.

Sau pretraining, predictor và target encoder có thể bị bỏ; context encoder là representation backbone dùng cho downstream task.

---

## **5. Representation target vs pixel target**

I-JEPA paper thay target-encoder output bằng pixels trong một ablation cùng họ architecture. ImageNet-1K 1% linear evaluation giảm từ 66.9 xuống 40.7 trong cấu hình được báo cáo.

Sự khác biệt objective:

$$
\mathcal{L}_{\mathrm{pixel}}
=
\sum_{j\in B_i}
\|\hat y_j-y_j\|_2^2,
$$

trong đó predictor phải giải thích màu và texture của patch.

$$
\mathcal{L}_{\mathrm{latent}}
=
\sum_{j\in B_i}
\|\hat s_{y_j}-s_{y_j}\|_2^2,
$$

trong đó target encoder được phép quotient out chi tiết không cần cho representation. Predictor có thể giữ object part và pose mà bỏ precise background/texture.

Kết quả này không có nghĩa latent target luôn tốt hơn pixel target trên mọi downstream task. Pixel reconstruction có thể giữ low-level detail hữu ích cho dense prediction hoặc generation; I-JEPA ưu tiên off-the-shelf semantic representation.

---

## **6. Hiệu quả tính toán**

Context encoder chỉ xử lý visible patches $B_x$, không xử lý target patches. Với self-attention có cost xấp xỉ bậc hai theo số token:

$$
C_{\mathrm{context}}
\propto
|B_x|^2d.
$$

Trong đó $|B_x|$ là số visible tokens và $d$ là hidden dimension. Khi visible ratio nhỏ, encoder backbone tiết kiệm đáng kể so với xử lý toàn ảnh.

Target encoder vẫn chạy toàn ảnh, nhưng không lưu activation cho backward vì nằm trong `no_grad`. Predictor hẹp chạy trên context tokens cộng target mask tokens; nó nhẹ hơn encoder backbone.

Paper báo ViT-H/14 pretraining dưới 1200 GPU-hours và nhanh hơn các baseline lớn được so sánh, nhưng cost thực tế còn phụ thuộc hardware, implementation, batch size và distributed efficiency. Điểm kiến trúc bền hơn con số benchmark là: **backprop chỉ qua sparse context encoder và narrow predictor**.

---

## **7. I-JEPA không dùng augmentation nghĩa là gì**

I-JEPA không cần hai image views được tạo bằng crop/color jitter pipeline kiểu SimCLR/DINO. Nó vẫn dùng stochastic masking, nên không phải "không có corruption" hay "không có inductive bias".

| View-invariance SSL | I-JEPA |
|---|---|
| hai augmented views | một ảnh, nhiều masks |
| invariance do augmentation policy | predictability do spatial masking |
| global embedding agreement | patch-level target prediction |
| thường cần color/crop semantics | cần block size/context geometry semantics |

Bias không biến mất; nó chuyển từ **augmentation semantics** sang **masking semantics**.

---

## **8. Giới hạn / Khi nào thất bại**

### Masking recipe phụ thuộc domain

Scale 0.15–0.20 hợp lý cho object-centric ImageNet không chắc phù hợp medical image, satellite image, document hoặc microscopy. Target lớn có thể chứa nhiều object; target nhỏ có thể chỉ còn texture.

### Context có thể quá dễ hoặc bất khả thi

Nếu context overlap target, model copy. Nếu context gần như toàn complement, local boundary shortcut có thể đủ. Nếu context quá ít, target thực sự không xác định và squared loss học conditional mean latent.

### EMA vẫn là heuristic chống collapse

Target encoder chậm + asymmetry hoạt động tốt thực nghiệm, nhưng complete/dimensional collapse vẫn phải monitor. Prediction loss thấp không chứng minh effective rank cao.

### Ảnh tĩnh chưa học dynamics

I-JEPA học spatial completion, không học motion, action-conditioned transition hay temporal causality. Nó là representation learner, chưa phải world model đầy đủ.

### Không reconstruct nên khó audit

Predicted embedding không nhìn được trực tiếp. Paper dùng một generative decoder riêng để probe thông tin predictor giữ/bỏ; decoder đó không thuộc training objective. Evaluation cần probes, retrieval hoặc diagnostic heads.

### Target encoder có thể bỏ chi tiết downstream cần

Semantic abstraction tốt cho classification nhưng có thể loại texture, precise boundary hoặc color. Downstream dense task có thể cần fine-tune sâu hơn.

### Multi-target overlap và loss weighting

Target blocks có thể overlap, khiến một patch được tính nhiều lần. Block size khác nhau cũng làm số patch contribution khác nhau. Runtime cần định nghĩa sampling và normalization chính xác để tái lập objective.

---

## **9. Liên hệ với Latent-Anything**

I-JEPA là stress test tốt cho structured `LatentSpace`: representation không phải một vector mà là **patch-token grid có positional semantics**.

- **`LatentSpace`**: shape là $(H_p,W_p,d)$ hoặc sequence tokens + 2D positions; geometry cần phân biệt token-level $L_2$ với pooled embedding similarity.
- **`ModelAdapter`**: expose full-image target encoding, sparse context encoding và predictor. Masking phải là explicit input, không phải hidden augmentation.
- **Layer A**: visualize token variance/effective rank theo vị trí, prediction residual heatmap, nearest-neighbour patch semantics và attention từ target tokens về context.
- **Layer B**: thay target mask/location để query model; can thiệp positional tokens để kiểm tra model encode "what" hay chỉ "where".
- **Layer C**: batch variable-length visible token sets, reuse một context encoding cho $M$ target predictions, chạy target under `no_grad`, rồi EMA sau optimizer step.

Một adapter-level flow:

```python
target_tokens = adapter.encode_target(full_image)
context_tokens = adapter.encode_context(full_image, context_mask)

for target_mask in target_masks:
    prediction = adapter.predict_target(
        context_tokens,
        positions=target_mask.positions,
    )
    target = target_tokens.select(target_mask)
    loss += squared_latent_distance(prediction, stop_gradient(target))
```

Tính chất quan trọng cho framework là target được select **sau** `encode_target`; đảo thứ tự này thay đổi objective.

---

## Liên quan

- [JEPA](06-jepa.md) — abstraction tổng quát mà I-JEPA hiện thực cho spatial image completion.
- [EMA Target Encoder](05-ema-target-encoder.md) — cơ chế cập nhật target encoder.
- [Masked Autoencoder (MAE)](01-masked-autoencoder-mae.md) — kiến trúc masking gần nhất nhưng reconstruct pixel.
- [Representation Collapse](02-representation-collapse.md) — diagnostic cần dùng dù loss latent giảm.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) — target encoder quyết định chi tiết nào bị bỏ.
- [Linear Probing](../../05-probing-intervention/research/01-linear-probing.md) — protocol chính để đo semantic accessibility của frozen I-JEPA features.

## Tham khảo

- M. Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (CVPR 2023, arXiv:2301.08243).
- Y. LeCun, *A Path Towards Autonomous Machine Intelligence* (OpenReview position paper, Version 0.9.2, 2022-06-27).
- K. He et al., *Masked Autoencoders Are Scalable Vision Learners* (CVPR 2022, arXiv:2111.06377).
- A. Baevski et al., *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language* (ICML 2022, arXiv:2202.03555).
