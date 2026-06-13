# Tại sao latent prediction thường tốt hơn pixel prediction

> **TL;DR.** Pixel prediction tối ưu một distortion metric cố định trên toàn bộ tín hiệu quan sát, nên loss buộc model giải thích cả semantic structure lẫn texture, lighting, sensor noise và các chi tiết không thể suy ra từ context. Latent prediction thay target $y$ bằng representation $e(y)$; nếu encoder loại đúng nuisance factors, conditional uncertainty giảm, gradient tập trung hơn và frozen features dễ dùng hơn. Lợi thế không đến từ việc latent "nhiều thông tin hơn", mà từ việc **target metric căn chỉnh tốt hơn với downstream semantics**. Đổi lại, learned target có thể collapse, bỏ nhầm chi tiết, hoặc không phù hợp với generation và dense photometric tasks.

[MAE](01-masked-autoencoder-mae.md), [I-JEPA](07-i-jepa.md) và [V-JEPA](08-v-jepa.md) có masked input gần nhau nhưng khác prediction target. Đây là một thay đổi nhỏ về ký hiệu và rất lớn về inductive bias:

$$
\text{pixel target}: y,
\qquad
\text{latent target}: z_y=e(y).
$$

---

## **1. Pixel loss đo mọi sai khác quan sát được**

Cho target observation tách thành:

$$
y=(s,n),
$$

trong đó:

- $s$ là semantic structure có thể dự đoán từ context $x$;
- $n$ là nuisance detail như texture realization, illumination, sensor noise hoặc leaf motion.

Với squared pixel loss:

$$
\mathcal{L}_{\mathrm{pixel}}
=
\mathbb{E}
\left[
\|\hat y(x)-y\|_2^2
\right].
$$

Nếu semantic và nuisance components trực giao theo metric đang dùng:

$$
\mathcal{L}_{\mathrm{pixel}}
=
\mathbb{E}\|\hat s-s\|_2^2
+
\mathbb{E}\|\hat n-n\|_2^2.
$$

Ngay cả khi downstream task chỉ cần $s$, gradient vẫn chứa term cho $n$.

Predictor MSE tối ưu là conditional mean:

$$
\hat y^\star(x)=\mathbb{E}[y\mid x].
$$

Bayes risk tối thiểu:

$$
\mathcal{R}_{\mathrm{pixel}}^\star
=
\mathbb{E}
\left[
\operatorname{Var}(y\mid x)
\right].
$$

Nếu $n$ độc lập với context:

$$
\operatorname{Var}(n\mid x)=\operatorname{Var}(n),
$$

thì một phần pixel loss là irreducible. Model không thể biết realization cụ thể nhưng vẫn bị đánh giá trên nó.

### Multimodality tạo trung bình không giống sample thật

Nếu hidden region có hai outcome pixel hợp lệ $y_1,y_2$ với xác suất bằng nhau:

$$
\hat y^\star
=
\frac{1}{2}(y_1+y_2).
$$

Output có thể mờ hoặc không tương ứng với mode nào. Adversarial, autoregressive hoặc diffusion decoders xử lý multimodality tốt hơn deterministic MSE, nhưng phải trả thêm model capacity và sampling compute.

---

## **2. Latent target thay đổi distortion metric**

Latent prediction tối ưu:

$$
\mathcal{L}_{\mathrm{latent}}
=
\mathbb{E}
\left[
d
\left(
p_\phi(e_\theta(x),c),
\operatorname{sg}(e_{\bar\theta}(y))
\right)
\right].
$$

Trong đó:

- $e_\theta$ là context encoder;
- $e_{\bar\theta}$ là target encoder;
- $p_\phi$ là predictor;
- $c$ là position/time/action condition;
- $d$ là latent-space distance.

Encoder học một equivalence relation:

$$
y_1\sim_e y_2
\iff
e(y_1)\approx e(y_2).
$$

Nếu hai observations khác texture nhưng có cùng object, pose hoặc dynamics, target encoder có thể map chúng gần nhau. Predictor không còn bị phạt mạnh vì bỏ chi tiết đã được quotient out.

Giả sử:

$$
e(y)=h(s),
$$

tức target representation không phụ thuộc nuisance $n$. Khi đó:

$$
\operatorname{Var}(e(y)\mid x)
=
\operatorname{Var}(h(s)\mid x),
$$

không còn term $\operatorname{Var}(n)$. Prediction task có conditional entropy thấp hơn:

$$
H(e(y)\mid x)\leq H(y\mid x)
$$

đối với deterministic encoder $e$. Bất đẳng thức này chỉ nói latent không thể có nhiều uncertainty hơn full observation; usefulness phụ thuộc **thông tin nào bị bỏ**.

---

## **3. "Tốt hơn" nghĩa là metric alignment**

Pixel Euclidean distance xem mọi channel-location error theo weighting cố định:

$$
d_{\mathrm{pixel}}(y_1,y_2)
=
\sum_{i,c}
(y_{1,i,c}-y_{2,i,c})^2.
$$

Hai ảnh cùng object nhưng khác ánh sáng có thể xa. Hai ảnh khác object nhưng cùng màu nền có thể tương đối gần.

Latent distance:

$$
d_{\mathrm{latent}}(y_1,y_2)
=
\|e(y_1)-e(y_2)\|_p
$$

được target encoder định nghĩa. Nếu encoder được học tốt:

$$
d_{\mathrm{latent}}
\approx
d_{\mathrm{task}},
$$

trong đó $d_{\mathrm{task}}$ đo khác biệt có ý nghĩa cho downstream family.

Do đó latent prediction tốt hơn khi ba thứ căn chỉnh:

1. target encoder giữ factors downstream cần;
2. masking/conditioning tạo relation cần học;
3. latent distance phản ánh semantic error hợp lý.

Không có căn chỉnh đó, latent loss chỉ là một metric learned nhưng sai.

---

## **4. Information bottleneck: giữ đủ, bỏ đúng**

Pixel reconstruction khuyến khích representation giữ đủ thông tin để tái tạo observation:

$$
I(z;y)
\quad\text{cao}.
$$

Representation learning cho downstream target $u$ muốn:

$$
I(z;u)
\quad\text{cao},
\qquad
I(z;n)
\quad\text{thấp}.
$$

Đây là sufficiency-minimality trade-off:

- **sufficient**: latent giữ thông tin cần dự đoán task-relevant variable;
- **minimal**: latent không tiêu tốn capacity cho nuisance.

Pixel target thiên về sufficiency cho toàn bộ $y$. Latent target có thể gần minimal sufficient representation hơn, nhưng không tự động đạt được. Nếu encoder bỏ màu trong khi task cần color, abstraction trở thành information loss.

Một cách nhìn theo rate-distortion:

$$
\min I(x;z)
\quad
\text{s.t.}
\quad
\mathbb{E}[d(y,\hat y)]\leq D.
$$

Chuyển từ $d_{\mathrm{pixel}}$ sang $d_{\mathrm{latent}}$ thay đổi distortion constraint. Lợi thế đến từ việc cho phép nén mạnh theo semantic equivalence, không phải phá vỡ giới hạn thông tin.

---

## **5. Gradient budget và optimization**

Với pixel MSE:

$$
\nabla_\theta\mathcal{L}_{\mathrm{pixel}}
=
2J_\theta^\top(\hat y-y),
$$

trong đó $J_\theta$ là Jacobian từ parameters đến output. Residual high-frequency có thể lớn và xuất hiện trên rất nhiều pixels, nên đóng góp đáng kể vào gradient norm.

PixMIM đưa ra bằng chứng thực nghiệm phù hợp với trực giác này: lọc high-frequency components khỏi pixel reconstruction target để giảm tập trung vào texture-rich detail cải thiện nhiều masked-image-modeling baselines. Điều đó không chứng minh mọi high frequency đều vô ích; nó cho thấy raw pixel target có weighting chưa chắc phù hợp representation learning.

Với latent target:

$$
\nabla_\theta\mathcal{L}_{\mathrm{latent}}
=
J_{\theta,\phi}^\top
\nabla_{\hat z}d(\hat z,z_y).
$$

Target encoder đã reweight input factors trước khi residual được tính. Nếu $z_y$ ổn định với texture/noise, các nuisance directions gần như không vào loss.

### Target dimension không phải yếu tố duy nhất

Latent tensor có thể vẫn lớn, thậm chí có nhiều channels hơn RGB. Compute giảm không chỉ vì dimension:

- không cần pixel decoder đủ mạnh để synthesize chi tiết;
- context encoder có thể bỏ masked tokens;
- target branch không backpropagate;
- semantic objective có thể cần ít epochs hơn;
- downstream adaptation có thể dùng frozen backbone.

---

## **6. Bằng chứng thực nghiệm cần đọc đúng protocol**

### I-JEPA

I-JEPA paper thay learned target-encoder representations bằng pixel targets trong ablation cùng họ architecture. ImageNet-1K 1% evaluation giảm từ 66.9 xuống 40.7 trong cấu hình được báo cáo.

Paper cũng cho thấy I-JEPA có linear-probe features mạnh hơn MAE trong nhiều cấu hình và cần ít pretraining epochs/compute hơn. Đây là evidence cho **off-the-shelf semantic accessibility**, không phải bằng chứng rằng pixel reconstruction không thể tạo representation tốt.

MAE đạt kết quả fine-tuning rất mạnh và scaling tốt. Pixel-pretrained features có thể cần end-to-end adaptation để reorganize information cho classification.

### V-JEPA

V-JEPA so sánh ViT-L/16 trên cùng VideoMix2M, 90K iterations và multi-block masks:

| Target | K400 frozen | SSv2 frozen | IN1K frozen | K400 fine-tune |
|---|---:|---:|---:|---:|
| Pixels | 68.6 | 66.0 | 73.3 | 85.4 |
| Features | 73.7 | 66.2 | 74.8 | 85.6 |

Gap lớn nhất nằm ở K400 frozen; full fine-tuning gần ngang nhau. Kết luận:

$$
\text{latent target}
\Rightarrow
\text{representation dễ đọc hơn sớm},
$$

không nhất thiết:

$$
\text{latent target}
\Rightarrow
\text{mọi downstream optimum cao hơn}.
$$

### data2vec 2.0

data2vec 2.0 dùng contextualized teacher targets trên vision, speech và language, đồng thời báo efficiency gains lớn so với modality-specific baselines. Cross-modal applicability là một lợi thế của target abstraction: pixel, waveform và token ID khác nhau, còn contextual feature prediction dùng cùng contract.

---

## **7. Một phổ target thay vì hai phe**

Prediction target không chỉ có "raw pixel" và "learned latent":

| Target | Giữ gì | Bỏ gì | Rủi ro |
|---|---|---|---|
| Raw pixels | toàn bộ photometric detail | gần như không | texture/noise chi phối |
| Filtered pixels | low-frequency structure | một phần texture | mất edge nhỏ |
| Handcrafted features | edge/orientation | color/texture tùy feature | bias cố định |
| Discrete tokenizer | codebook semantics | ngoài-codebook detail | phụ thuộc tokenizer |
| Frozen teacher features | teacher invariances | teacher-discarded factors | teacher bias |
| EMA learned features | target thích nghi cùng model | nuisance model tự bỏ | collapse/co-adaptation |
| Task-specific targets | factors liên quan task | phần còn lại | transfer kém |

Target choice là thiết kế distortion function. Hệ tốt có thể kết hợp nhiều mức:

$$
\mathcal{L}
=
\lambda_{\mathrm{latent}}\mathcal{L}_{\mathrm{latent}}
+
\lambda_{\mathrm{pixel}}\mathcal{L}_{\mathrm{pixel}}
+
\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}}.
$$

Hybrid objective phù hợp khi cần semantic abstraction nhưng vẫn phải giữ geometry hoặc appearance detail.

---

## **8. Khi pixel prediction là lựa chọn đúng**

### Generation và rendering

Nếu output cuối là ảnh/video/audio chính xác, model phải học observation distribution. Latent-only predictor không cung cấp decoder hoặc likelihood trong input space.

### Photometric và dense tasks

Depth boundary, optical flow, super-resolution, restoration, compression và inverse rendering có thể cần high-frequency structure. Một semantic target quá invariant làm mất tín hiệu.

### Target encoder chưa đáng tin

Early training target có thể noisy, collapse hoặc thay đổi nhanh. Pixel target cố định và không collapse; reconstruction loss luôn có ground truth rõ.

### Auditability

Pixel residual nhìn thấy trực tiếp. Latent residual nhỏ khó giải thích: predictor đúng, target collapse, hay metric bỏ qua lỗi đều có thể cho loss thấp.

### Full fine-tuning budget lớn

Nếu downstream có nhiều labels và cho phép update toàn backbone, pixel-pretrained model có thể reorganize low-level information và thu hẹp gap đáng kể.

---

## **9. Khi latent prediction thất bại**

### Complete collapse

Nếu:

$$
e(y)=c
\quad\forall y,
$$

thì predictor output $c$ đạt zero loss nhưng representation vô dụng. Pixel reconstruction không có trivial constant solution tương tự khi decoder condition capacity bị giới hạn.

### Dimensional collapse

Target có thể giữ variance ở vài directions nhưng mất phần lớn rank. Prediction loss vẫn giảm; cần monitor:

$$
\operatorname{Var}(z_k),
\quad
\operatorname{Cov}(z),
\quad
\operatorname{rank}_{\mathrm{eff}}(z).
$$

### Sai invariance

Nếu target encoder học bất biến với factor downstream cần:

$$
e(y_1)\approx e(y_2)
\quad\text{nhưng}\quad
u(y_1)\neq u(y_2),
$$

thì information đã bị xóa trước predictor.

### Semantic multimodality vẫn tồn tại

Latent abstraction bỏ texture modes nhưng không giải quyết nhiều future semantics. Một deterministic predictor vẫn average/median các latent futures không tương thích. Cần latent variable, stochastic transition hoặc energy-based set of predictions.

### Shortcut trong context

Nếu target dễ copy từ context, model không cần học abstraction dù loss nằm trong latent space. Mask geometry và condition quan trọng ngang target type.

---

## **10. Cách đánh giá công bằng**

Không nên so pixel và latent pretraining bằng một metric duy nhất.

### Representation quality

- frozen linear/attentive probe;
- low-shot adaptation;
- retrieval và nearest neighbours;
- transfer qua nhiều semantic và dense tasks.

### Information health

- per-dimension variance;
- covariance eigenspectrum;
- effective rank;
- feature uniformity;
- target/prediction residual theo vị trí.

### Optimization và efficiency

- samples/epochs đến cùng accuracy;
- GPU-hours, throughput, memory;
- encoder, target encoder và decoder FLOPs tách riêng;
- cost của downstream adaptation.

### Fidelity

- pixel/feature reconstruction;
- boundary, texture, color và geometry probes;
- generative likelihood hoặc perceptual quality nếu cần generation.

Một model có linear probe tốt hơn nhưng reconstruction kém hơn không mâu thuẫn; hai metric đo hai distortion priorities khác nhau.

---

## **11. Liên hệ với Latent-Anything**

Framework không nên hard-code giả định rằng latent metric luôn ưu việt. Nó cần làm target semantics và information loss quan sát được.

- **`LatentSpace`**: lưu encoder identity, layer, normalization, token layout và distance metric; cùng tensor shape nhưng khác encoder không cùng coordinate system.
- **`ModelAdapter`**: expose cả observation target và latent target để chạy controlled ablation.
- **Layer A**: compare pixel residual, latent residual, variance spectrum, probe accuracy và nuisance sensitivity.
- **Layer B**: can thiệp texture/color/pose độc lập để đo factor nào target encoder giữ hoặc bỏ.
- **Layer C**: profile encoder/predictor/decoder cost, target `no_grad`, EMA schedule và mixed objectives.

Một evaluation contract:

```python
report = compare_prediction_targets(
    context=batch.context,
    target=batch.target,
    targets={
        "pixel": PixelTarget(normalize_per_patch=True),
        "latent": EncoderTarget(
            adapter=target_adapter,
            layer="last",
            distance="l1",
        ),
    },
    diagnostics=[
        "prediction_loss",
        "effective_rank",
        "linear_probe",
        "nuisance_sensitivity",
        "gpu_hours",
    ],
)
```

Kết quả cần được diễn giải theo use case:

```python
assert report["latent"].linear_probe > report["pixel"].linear_probe
# This does not imply:
assert not report.implies("latent_preserves_all_observation_detail")
```

---

## Liên quan

- [Masked Autoencoder](01-masked-autoencoder-mae.md) - pixel-reconstruction baseline hiệu quả và scalable.
- [JEPA](06-jepa.md) - contract dự đoán representation thay vì input.
- [I-JEPA](07-i-jepa.md) - image ablation giữa target encoder output và pixels.
- [V-JEPA](08-v-jepa.md) - video ablation giữa feature prediction và pixel reconstruction.
- [Representation Collapse](02-representation-collapse.md) - failure mode riêng của learned targets.
- [EMA Target Encoder](05-ema-target-encoder.md) - target co-adaptation và stability.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) - sufficiency-minimality trade-off.
- [Stochastic Transition](../../06-latent-temporal/research/03-stochastic-transition.md) - xử lý multimodal futures trong latent.
- [Linear Probing](../../05-probing-intervention/research/01-linear-probing.md) - đo semantic accessibility khi backbone bị freeze.

## Tham khảo

- M. Assran et al., [*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*](https://arxiv.org/abs/2301.08243), CVPR 2023.
- A. Bardes et al., [*Revisiting Feature Prediction for Learning Visual Representations from Video*](https://arxiv.org/abs/2404.08471), 2024.
- K. He et al., [*Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377), CVPR 2022.
- Y. Liu et al., [*PixMIM: Rethinking Pixel Reconstruction in Masked Image Modeling*](https://arxiv.org/abs/2303.02416), 2023.
- A. Baevski et al., [*Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language*](https://arxiv.org/abs/2212.07525), ICML 2023.
- D. Pathak et al., [*Context Encoders: Feature Learning by Inpainting*](https://arxiv.org/abs/1604.07379), CVPR 2016.
