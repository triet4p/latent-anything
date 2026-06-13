# V-JEPA - Video Joint-Embedding Predictive Architecture

> **TL;DR.** V-JEPA mở rộng [I-JEPA](07-i-jepa.md) từ patch grid 2D sang token grid không-thời gian 3D. Context encoder chỉ nhìn các video tokens còn lại sau masking; EMA target encoder nhìn toàn clip; predictor nhận context features cùng positional mask tokens để dự đoán target features bằng loss $L_1$. Mask quan trọng nhất không phải "che vài frame": paper lấy các spatial blocks rồi lặp chúng xuyên suốt toàn bộ temporal dimension, tạo khoảng trống không-thời gian khó nội suy bằng frame lân cận. V-JEPA học representation tốt cho cả appearance lẫn motion, nhưng bản 2024 vẫn là masked video representation learner, chưa phải action-conditioned hay causal future world model.

V-JEPA được giới thiệu trong *Revisiting Feature Prediction for Learning Visual Representations from Video* (Bardes et al., 2024). Mục tiêu của paper rất hẹp và có giá trị thực nghiệm rõ: kiểm tra liệu **feature prediction một mình** có đủ để pretrain video encoder hay không, không dùng image encoder pretrained, text, negative pairs, human labels hoặc pixel reconstruction.

---

## **1. Từ ảnh sang token grid không-thời gian**

Cho video clip:

$$
v\in\mathbb{R}^{T\times H\times W\times C}.
$$

Trong đó $T$ là số frame, $H,W$ là spatial resolution và $C$ là số channel. V-JEPA chia clip thành tubelets kích thước:

$$
2\times16\times16,
$$

nghĩa là mỗi token bao phủ 2 frame liên tiếp và một vùng $16\times16$ pixels. Sau tokenization:

$$
v\longrightarrow
\{u_j\}_{j=1}^{L},
\qquad
L=\frac{T}{2}\frac{H}{16}\frac{W}{16}.
$$

Trong đó $u_j$ là spatiotemporal token thứ $j$. Khác với ảnh tĩnh, position của token có ba tọa độ:

$$
p_j=(t_j,h_j,w_j).
$$

Do đó target condition không chỉ trả lời "ở đâu trong ảnh" mà còn "ở đâu trong clip".

Recipe pretraining chính dùng 16 frames với frame skip 4, tương ứng khoảng 3 giây video trung bình. ViT xử lý sequence tubelets trực tiếp; không có CNN motion branch hoặc optical-flow target riêng.

---

## **2. Kiến trúc ba module**

V-JEPA giữ contract của [JEPA](06-jepa.md):

- context encoder $E_\theta$: ViT xử lý visible video tokens;
- target encoder $\bar E_\theta$: EMA ViT xử lý toàn clip;
- predictor $P_\phi$: narrow transformer dự đoán representation tại masked positions.

Gọi $B_y$ là target mask và $B_x$ là complement:

$$
B_x=\{1,\ldots,L\}\setminus B_y.
$$

Context branch drop target tokens ngay tại input:

$$
s_x=E_\theta(v_{B_x}).
$$

Target branch encode toàn clip trước, rồi mới chọn target positions:

$$
s_y
=
\operatorname{select}\!\left(
\bar E_\theta(v),
B_y
\right).
$$

Predictor nhận context sequence cùng learnable mask tokens có spatiotemporal positional embeddings:

$$
\hat s_y
=
P_\phi
\left(
s_x,
\{m+p_j\}_{j\in B_y}
\right).
$$

Trong đó $m$ là learnable mask embedding và $p_j$ cho biết target coordinate. Predictor dùng 12 transformer blocks, embedding dimension 384; nó hẹp hơn backbone ViT-L/H.

### Encode trước, select sau vẫn là invariant cốt lõi

Nếu target encoder nhận masked clip:

$$
\tilde s_y
=
\bar E_\theta(\operatorname{mask}(v,B_y)),
$$

thì target representations mang artifact do thiếu spatial và temporal context. V-JEPA dùng contextualized target:

$$
s_y
=
\operatorname{select}(\bar E_\theta(v),B_y),
$$

để mỗi target token được định nghĩa khi encoder nhìn toàn clip thật.

---

## **3. Objective $L_1$ và chống collapse**

V-JEPA tối ưu:

$$
\mathcal{L}_{\mathrm{V\text{-}JEPA}}
=
\frac{1}{|B_y|}
\sum_{j\in B_y}
\left\|
\hat s_{y_j}
-
\operatorname{sg}(s_{y_j})
\right\|_1.
$$

Trong đó:

- $\hat s_{y_j}$ là predictor output tại target token $j$;
- $s_{y_j}$ là target-encoder feature;
- $\operatorname{sg}$ chặn gradient qua target;
- loss lấy trên masked tokens, không reconstruct RGB.

Đây là khác biệt implementation đáng chú ý so với I-JEPA: I-JEPA dùng squared $L_2$, còn V-JEPA paper chuyển sang $L_1$ vì tác giả thấy ổn định hơn.

Online parameters được optimizer cập nhật:

$$
(\theta,\phi)
\leftarrow
\operatorname{Opt}
\left(
\theta,\phi,
\nabla_{\theta,\phi}\mathcal{L}
\right).
$$

Target parameters được cập nhật sau optimizer step:

$$
\bar\theta
\leftarrow
\mu\bar\theta+(1-\mu)\theta.
$$

Asymmetry giữa online branch, predictor và [EMA target encoder](05-ema-target-encoder.md) ngăn trivial constant solution trong thực nghiệm. Tuy vậy, loss thấp vẫn không đủ chứng minh representation không collapse; variance, covariance spectrum và effective rank vẫn cần được monitor.

---

## **4. Multi-block masking trong video**

Masking là phần biến một objective đơn giản thành bài toán có ích.

### Spatial block được kéo dài qua toàn bộ thời gian

V-JEPA sample spatially continuous blocks rồi lặp mỗi block qua toàn temporal dimension:

$$
B_i
=
S_i\times\{1,\ldots,T'\},
$$

trong đó $S_i$ là spatial block và $T'=T/2$ là số tubelet steps. Target mask là union:

$$
B_y=\bigcup_i B_i.
$$

Nếu chỉ che một vùng tại một thời điểm, frame ngay trước hoặc sau thường tiết lộ gần như cùng pixels vì video có temporal redundancy. Tube mask buộc predictor dùng object identity, motion, scene structure và context xa hơn.

### Hai họ mask

Paper trộn hai loại:

| Loại mask | Số target blocks | Diện tích mỗi block trên một frame | Aspect ratio |
|---|---:|---:|---:|
| Short-range | 8 | 15% | 0.75-1.5 |
| Long-range | 2 | 70% | 0.75-1.5 |

Các blocks có thể overlap. Sau khi lấy union, masking ratio trung bình xấp xỉ 90%.

Tên "short-range" và "long-range" nói về quy mô spatial relation cần suy luận, không phải chỉ horizon thời gian:

- nhiều block nhỏ yêu cầu local-to-mid-range completion;
- ít block rất lớn để lại context thưa, buộc dùng global scene và motion cues.

### Vì sao random patches quá dễ

Video có redundancy cao ở cả không gian và thời gian. Random independent holes để lại hàng xóm sát target ở gần như mọi hướng; model có thể dựa vào texture continuation hoặc copy temporal neighbor. Continuous tubes loại bớt shortcut đó.

Mask distribution vì thế xác định loại representation được học:

$$
p(B_y)
\quad\Longrightarrow\quad
\text{prediction task distribution}
\quad\Longrightarrow\quad
\text{learned invariances}.
$$

---

## **5. V-JEPA học motion mà không predict optical flow**

V-JEPA không có explicit flow loss. Motion information xuất hiện vì một target token phụ thuộc vào biến đổi của scene xuyên thời gian, còn predictor phải giải thích target từ visible spatiotemporal context.

Cho latent state tại token position $(t,h,w)$:

$$
s_{t,h,w}
=
\bar E_\theta(v)_{t,h,w}.
$$

Prediction:

$$
\hat s_{t,h,w}
=
P_\phi
\left(
E_\theta(v_{B_x}),
p_{t,h,w}
\right).
$$

Để giảm loss trên nhiều clip, encoder có lợi khi biểu diễn các yếu tố ổn định và có tính dự báo:

- object identity và part structure;
- pose, trajectory và direction;
- camera motion;
- interaction giữa object;
- scene context.

Điều này không bảo đảm latent factor disentanglement hoặc physical causality. Nó chỉ tạo áp lực để motion-relevant structure trở nên dễ truy cập trong representation.

---

## **6. Feature prediction so với pixel reconstruction**

V-JEPA paper so sánh ViT-L/16 cùng data, 90K iterations, batch size 3072 và multi-block masking:

| Target | K400 frozen | SSv2 frozen | IN1K frozen | K400 fine-tune |
|---|---:|---:|---:|---:|
| Pixels | 68.6 | 66.0 | 73.3 | 85.4 |
| Features | 73.7 | 66.2 | 74.8 | 85.6 |

Feature target cải thiện rõ nhất trong frozen evaluation K400, cải thiện nhỏ trên SSv2 trong ablation này, và gần ngang pixel target khi full fine-tune. Cách đọc hợp lý:

1. latent objective tạo feature sẵn dùng hơn khi backbone bị freeze;
2. fine-tuning có thể sửa pixel-pretrained representation nên gap giảm;
3. không có một con số duy nhất chứng minh latent target thắng mọi task.

Pixel objective:

$$
\mathcal{L}_{\mathrm{pixel}}
=
\sum_{j\in B_y}
\|\hat v_j-v_j\|_2^2
$$

phải phân bổ capacity cho màu, texture và chi tiết chuyển động khó đoán.

Feature objective:

$$
\mathcal{L}_{\mathrm{feature}}
=
\sum_{j\in B_y}
\|\hat s_j-s_j\|_1
$$

cho target encoder quyền bỏ nuisance detail nếu chi tiết đó không ổn định hoặc không hữu ích cho representation.

Meta AI blog báo training/sample-efficiency tăng khoảng 1.5-6 lần trong các so sánh của họ. Đây là range phụ thuộc baseline và protocol, không phải hệ số complexity phổ quát.

---

## **7. Frozen evaluation và attentive probe**

Sau pretraining, encoder bị freeze. Một attentive probe nhẹ học query để pool token sequence rồi classify:

$$
r
=
\operatorname{CrossAttention}
\left(
q,\{s_j\}_{j=1}^{L}
\right),
\qquad
\hat y=Wr.
$$

Trong đó $q$ là learnable query và $W$ là linear head. Protocol này kiểm tra information đã có trong representation thay vì cho toàn backbone thích nghi.

Paper pretrain trên VideoMix2M, khoảng 2 triệu videos từ HowTo100M, Kinetics-400/600/700 và Something-Something-v2 sau khi loại validation overlap. ViT-H/16 lớn nhất trong paper đạt kết quả mạnh trên:

- Kinetics-400, thiên về appearance/action recognition;
- Something-Something-v2, cần fine-grained temporal understanding;
- ImageNet-1K, kiểm tra transfer về ảnh tĩnh.

Việc một video-only backbone transfer sang image classification cho thấy feature target không chỉ giữ motion; nó đồng thời học object và scene appearance.

---

## **8. Không phải mọi "world model" đều giống nhau**

V-JEPA 2024 đôi khi được gọi rộng là physical world model, nhưng cần tách ba capability:

| Capability | V-JEPA 2024 |
|---|---|
| Masked spatiotemporal representation completion | Có |
| Causal future prediction chỉ từ quá khứ | Không phải objective mặc định |
| Action-conditioned transition $s_{t+1}=F(s_t,a_t)$ | Không |
| Pixel/video generation | Không |
| Planning bằng rollout action | Không |

Context có thể chứa tokens ở cả trước và sau target time. Vì vậy:

$$
p(s_{\mathrm{masked}}\mid s_{\mathrm{visible}})
\neq
p(s_{\mathrm{future}}\mid s_{\mathrm{past}},a).
$$

V-JEPA là nền tảng representation predictive tốt cho world modeling, nhưng không tự động cung cấp transition model phục vụ control.

---

## **9. Giới hạn / Khi nào thất bại**

### Mask tube có thể bỏ qua temporal direction

Khi mask lặp nguyên spatial footprint qua toàn clip, task nhấn mạnh object/region completion nhưng không bắt buộc phân biệt forward với backward dynamics. Một representation có thể tốt cho motion classification mà chưa học causal arrow of time.

### Deterministic predictor nén uncertainty

Nhiều hidden outcomes tương thích với cùng context. $L_1$ predictor có xu hướng học conditional median trong feature space:

$$
P_\phi^\star(s_x,p)
\in
\operatorname{median}
\left[
s_y\mid s_x,p
\right].
$$

Objective không có explicit latent variable để biểu diễn nhiều future modes.

### Dataset bias

VideoMix2M trộn human actions và narrated internet videos. Camera style, editing, frame rate và activity distribution ảnh hưởng mạnh tới representation. "Unlabeled" không có nghĩa "không bias".

### Spatial masking recipe không phổ quát

Mask ratio khoảng 90% phù hợp với architecture/data của paper. Video y tế, autonomous driving, egocentric stream hoặc microscopy có temporal scale khác; block size và clip duration cần retune.

### Không giữ mọi low-level detail

Abstraction hữu ích cho classification có thể làm mất color, texture, precise boundary hoặc subtle motion cần cho dense prediction. Full fine-tuning hoặc auxiliary objective có thể cần thiết.

### Compute vẫn lớn

Sparse context giảm online-backbone cost, nhưng target encoder vẫn xử lý full clip trong forward pass và video token sequence dài. Models được train distributed với batch hàng nghìn; recipe không phải single-GPU baseline nhẹ.

### Latent prediction khó quan sát trực tiếp

Feature prediction không có ảnh/video output tự nhiên để audit. Official repo dùng một conditional diffusion decoder riêng để visualize frozen V-JEPA predictions; decoder này không thuộc pretraining objective.

---

## **10. Liên hệ với Latent-Anything**

V-JEPA yêu cầu framework coi latent là tensor có **thời gian, không gian và vị trí**, không phải một pooled vector.

- **`LatentSpace`**: shape $(T',H',W',d)$ cùng temporal stride, spatial patch size và positional metadata.
- **`ModelAdapter`**: tách `encode_context`, `encode_target`, `predict_masked`; target selection phải xảy ra sau full-clip encoding.
- **Layer A**: visualize residual theo $(t,h,w)$, token variance/effective rank theo thời gian, motion-sensitive probe và nearest-neighbour clips.
- **Layer B**: thay target tubes, shift temporal positions, reverse clip hoặc giữ appearance nhưng đổi motion để test latent sensitivity.
- **Layer C**: batch variable visible-token sets, target forward dưới `no_grad`, reuse context features cho nhiều mask queries, EMA sau optimizer step.

Một adapter flow:

```python
target_tokens = adapter.encode_target(full_clip)
context_tokens = adapter.encode_context(full_clip, visible_mask)

prediction = adapter.predict_masked(
    context_tokens,
    positions=target_mask.spatiotemporal_positions,
)
target = target_tokens.select(target_mask)

loss = latent_l1(prediction, stop_gradient(target))
```

Metadata tối thiểu cần bảo toàn:

```python
latent.layout = ("time", "height", "width", "channel")
latent.temporal_stride = 2
latent.patch_size = (16, 16)
latent.mask_semantics = "spatiotemporal-tube"
```

Nếu flatten sequence mà bỏ mapping token-index sang $(t,h,w)$, predictor condition và residual diagnostics không còn đúng nghĩa.

---

## Liên quan

- [JEPA](06-jepa.md) - contract chung giữa encoder, target encoder và predictor.
- [I-JEPA](07-i-jepa.md) - phiên bản ảnh tĩnh mà V-JEPA mở rộng sang spatiotemporal tokens.
- [EMA Target Encoder](05-ema-target-encoder.md) - cơ chế cập nhật target branch.
- [Representation Collapse](02-representation-collapse.md) - rủi ro không được phát hiện chỉ bằng prediction loss.
- [Masked Autoencoder](01-masked-autoencoder-mae.md) - baseline reconstruct pixels với masking.
- [Latent Transition Model](../../06-latent-temporal/research/02-latent-transition-model.md) - phân biệt masked completion với transition theo thời gian.
- [Stochastic Transition](../../06-latent-temporal/research/03-stochastic-transition.md) - cách biểu diễn uncertainty mà deterministic V-JEPA predictor chưa có.
- [Linear Probing](../../05-probing-intervention/research/01-linear-probing.md) - protocol đánh giá information trong frozen features.

## Tham khảo

- A. Bardes et al., [*Revisiting Feature Prediction for Learning Visual Representations from Video*](https://arxiv.org/abs/2404.08471), 2024.
- Meta AI, [*V-JEPA: The next step toward Yann LeCun's vision of advanced machine intelligence*](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/), 2024-02-15.
- Meta AI Research, [*facebookresearch/jepa: Official PyTorch code and models for V-JEPA*](https://github.com/facebookresearch/jepa).
- M. Assran et al., [*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html), CVPR 2023.
- T. Tong et al., [*VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*](https://arxiv.org/abs/2203.12602), NeurIPS 2022.
