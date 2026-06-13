# EMA Target Encoder

> **TL;DR.** EMA target encoder là một bản sao không nhận gradient của online encoder, được cập nhật sau mỗi bước bằng $\xi_t=m_t\xi_{t-1}+(1-m_t)\theta_t$. Nó tạo target thay đổi chậm, giảm vòng phản hồi tức thời khi model học từ chính representation của mình và là thành phần trung tâm của BYOL, DINO, MoCo và JEPA. Caveat: EMA chỉ làm target ổn định; nó không tự loại bỏ collapse, tạo thêm state phải checkpoint chính xác, và momentum quá lớn khiến teacher lỗi thời.

[Contrastive learning](04-contrastive-learning.md) dùng negatives để neo geometry của latent space. Các phương pháp bootstrap không có negatives cần một loại neo khác: target ở vế phải của loss không được chạy theo online network ngay trong cùng bước gradient. EMA target encoder tạo ra **một thang thời gian chậm** cho target, biến bài toán "mạng tự khớp chính nó" thành "mạng nhanh dự đoán một ensemble trễ của chính lịch sử nó".

---

## **1. Định nghĩa: hai mạng, hai quy tắc cập nhật**

Gọi online/student encoder có tham số $\theta_t$ và target/teacher encoder có tham số $\xi_t$. Một training step đúng thứ tự gồm:

1. student và teacher chạy forward trên các view tương ứng;
2. target output được stop-gradient;
3. loss chỉ backprop qua student;
4. optimizer cập nhật $\theta_t$;
5. EMA cập nhật teacher từ **student mới**.

Quy tắc EMA là:

$$
\xi_t=m_t\xi_{t-1}+(1-m_t)\theta_t.
$$

Trong đó $m_t\in[0,1)$ là momentum hoặc decay tại bước $t$, $\theta_t$ là student sau optimizer step, và $\xi_t$ là teacher mới. Khi $m_t$ gần 1, teacher thay đổi chậm; khi $m_t=0$, teacher copy student ngay lập tức.

Teacher không có optimizer và không nhận gradient:

$$
\nabla_{\xi}\mathcal{L}=0,\qquad
\theta_t=\operatorname{Opt}\!\left(\theta_{t-1},\nabla_{\theta}\mathcal{L}\right).
$$

Trong đó $\mathcal{L}$ là self-supervised loss và $\operatorname{Opt}$ là một bước optimizer. Stop-gradient xác định **đường gradient**; EMA xác định **đường tiến hoá tham số**. Hai cơ chế khác nhau và thường cùng xuất hiện.

Pseudo-code tối thiểu:

```python
student_pred = predictor(student(view_a))
with no_grad():
    teacher_target = teacher(view_b)

loss = distance(student_pred, teacher_target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

with no_grad():
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.mul_(momentum).add_(student_param, alpha=1 - momentum)
```

Nếu cập nhật teacher trước `optimizer.step()`, teacher nhận student cũ thêm một bước và độ trễ thực tế thay đổi. Nếu quên `no_grad()` hoặc vô tình đưa teacher parameters vào optimizer, kiến trúc không còn là EMA teacher.

---

## **2. EMA là ensemble theo thời gian**

Với momentum hằng $m$ và khởi tạo $\xi_0=\theta_0$, khai triển truy hồi cho:

$$
\xi_t
=
m^t\theta_0
+(1-m)\sum_{k=1}^{t}m^{t-k}\theta_k.
$$

Trong đó trọng số của student tại bước $k$ giảm theo $m^{t-k}$. Teacher là một **exponentially weighted ensemble** của các student checkpoint trước, không phải chỉ là "student chậm hơn" theo nghĩa hình ảnh.

Trọng số của một checkpoint cách hiện tại $\Delta$ bước là:

$$
w(\Delta)=(1-m)m^\Delta.
$$

Trong đó $w(\Delta)$ là phần đóng góp của checkpoint cũ $\Delta$ bước. Tổng trọng số tiệm cận 1; lịch sử rất xa vẫn có đóng góp nhưng giảm theo cấp số nhân.

Hai đại lượng trực giác:

$$
N_{\mathrm{eff}}\approx\frac{1}{1-m},
\qquad
h_{1/2}=\frac{\log(1/2)}{\log m}.
$$

Trong đó $N_{\mathrm{eff}}$ là độ dài cửa sổ hiệu dụng xấp xỉ và $h_{1/2}$ là số bước để trọng số giảm một nửa. Ví dụ, $m=0.99$ tương ứng cửa sổ khoảng 100 bước và half-life khoảng 69 bước; $m=0.999$ tương ứng khoảng 1000 và 693 bước.

### Vì sao ensemble trễ ổn định hơn

Student thay đổi do minibatch noise, augmentation noise và optimizer momentum. Teacher averaging lọc bớt thành phần tần số cao của trajectory tham số. Nếu xem student scalar $\theta_t$ như tín hiệu và teacher $\xi_t$ như output của bộ lọc, transfer function rời rạc là:

$$
H(z)=\frac{1-m}{1-mz^{-1}}.
$$

Trong đó $z^{-1}$ là toán tử trễ một bước. Đây là low-pass filter bậc một: biến động nhanh bị suy giảm, xu hướng chậm được giữ lại. Target vì thế ít rung giữa minibatch, giúp student không đuổi theo nhiễu do chính nó vừa tạo.

---

## **3. BYOL: EMA để ổn định bootstrap representation**

BYOL có online encoder $f_\theta$, projector $g_\theta$, predictor $q_\theta$ và target encoder/projector $(f_\xi,g_\xi)$. Với hai augmented view $v,v'$:

$$
p_\theta=q_\theta(g_\theta(f_\theta(v))),
\qquad
z'_\xi=g_\xi(f_\xi(v')).
$$

Trong đó $p_\theta$ là prediction từ online branch và $z'_\xi$ là target projection từ view kia. Predictor chỉ tồn tại ở online branch, tạo bất đối xứng.

Loss trên vector chuẩn hoá là:

$$
\mathcal{L}_{\theta,\xi}
=
\left\|
\frac{p_\theta}{\lVert p_\theta\rVert_2}
-
\operatorname{sg}\!\left(
\frac{z'_\xi}{\lVert z'_\xi\rVert_2}
\right)
\right\|_2^2.
$$

Trong đó $\operatorname{sg}$ chặn gradient qua target. Loss thường được đối xứng hoá bằng cách đổi vai hai view.

Sau optimizer step của online branch:

$$
\xi\leftarrow m\xi+(1-m)\theta.
$$

Trong đó teacher thừa hưởng dần representation mới nhưng không nhảy theo gradient tức thời của loss. BYOL mô tả target network như một chuỗi representation được bootstrap và cải thiện dần.

Điểm cần tách rõ: [SimSiam](03-stop-gradient-asymmetric.md) cho thấy momentum encoder **không bắt buộc** để tránh collapse trong mọi kiến trúc; stop-gradient + predictor có thể đủ. Trong BYOL, EMA cải thiện độ ổn định và chất lượng target, nhưng không nên được diễn giải đơn giản là lực chống collapse duy nhất.

---

## **4. DINO: EMA teacher, centering và sharpening**

DINO áp dụng self-distillation không nhãn. Student và teacher xuất phân phối trên $K$ output dimensions. Với student logits $s_\theta(x)$:

$$
P_s^{(k)}(x)
=
\frac{\exp\left(s_\theta^{(k)}(x)/\tau_s\right)}
{\sum_{j=1}^{K}\exp\left(s_\theta^{(j)}(x)/\tau_s\right)}.
$$

Trong đó $P_s^{(k)}$ là xác suất student ở dimension $k$, $\tau_s$ là student temperature và $K$ là output dimension.

Teacher logits được trừ center $c$ và dùng temperature riêng:

$$
P_t^{(k)}(x)
=
\frac{\exp\left((s_\xi^{(k)}(x)-c^{(k)})/\tau_t\right)}
{\sum_{j=1}^{K}\exp\left((s_\xi^{(j)}(x)-c^{(j)})/\tau_t\right)}.
$$

Trong đó $c$ là running mean của teacher logits và $\tau_t$ thường nhỏ để **sharpen** target distribution. Centering ngăn một dimension thống trị toàn bộ dataset; sharpening ngăn phân phối trở nên đồng đều vô thông tin.

Cross-view distillation loss là:

$$
\mathcal{L}_{\mathrm{DINO}}
=
-\sum_k P_t^{(k)}(x_b)\log P_s^{(k)}(x_a),
\qquad a\ne b.
$$

Trong đó $x_a,x_b$ là hai crop khác nhau của cùng ảnh. Student học distribution của teacher trên view khác; teacher output được detach.

DINO có **hai EMA state**:

$$
\xi_t=m_t\xi_{t-1}+(1-m_t)\theta_t,
\qquad
c_t=m_c c_{t-1}+(1-m_c)\bar s_{\xi,t}.
$$

Trong đó $\bar s_{\xi,t}$ là teacher-logit mean toàn batch và toàn worker, còn $m_c$ là center momentum. Official implementation dùng teacher momentum bắt đầu khoảng $0.996$ rồi tăng theo cosine schedule về 1; center cũng được cập nhật bằng EMA sau distributed all-reduce.

| Cơ chế | Tác dụng chính | Không tự giải quyết |
|---|---|---|
| EMA teacher | làm target thay đổi chậm | output collapse |
| Stop-gradient | chặn teacher chạy theo loss trực tiếp | target quality |
| Centering | chống một vài dimension chiếm ưu thế | uniform collapse |
| Sharpening | làm target có thông tin, entropy thấp hơn | dimension domination |
| Multi-crop | học consistency local-to-global | ổn định teacher |

DINO ổn định vì các cơ chế này phối hợp, không phải chỉ vì có EMA.

---

## **5. Momentum schedule và độ trễ**

Momentum hằng dễ hiểu nhưng thường không tối ưu suốt training. Giai đoạn đầu, student thay đổi nhanh và teacher cần theo kịp; về cuối, student gần hội tụ và teacher nên averaging dài hơn. Một cosine schedule phổ biến là:

$$
m_t
=
1-(1-m_0)\frac{\cos(\pi t/T)+1}{2}.
$$

Trong đó $m_0$ là momentum ban đầu, $t$ là bước hiện tại và $T$ là tổng số bước. Schedule tăng từ $m_0$ về 1, làm teacher ngày càng ổn định.

Trade-off cốt lõi:

| Momentum | Teacher behavior | Rủi ro |
|---|---|---|
| nhỏ | bám student nhanh | target rung, gần copy trực tiếp |
| vừa | lọc noise nhưng vẫn thích nghi | cần tune theo batch và learning rate |
| rất gần 1 | ensemble dài, target mượt | teacher stale, warm-up chậm |

Momentum không thể chọn độc lập với learning rate, batch size và tổng số steps. Cùng $m=0.999$ nhưng run 10 nghìn bước và 1 triệu bước tạo lịch sử hiệu dụng rất khác so với tiến trình optimization.

---

## **6. Biến thể và phân biệt với các target khác**

| Target mechanism | Cập nhật target | Độ mượt | Ví dụ |
|---|---|---|---|
| Shared encoder + stop-gradient | copy tức thời ở forward | không có temporal averaging | SimSiam |
| Hard target copy | copy mỗi $C$ bước | piecewise constant | DQN-style target network |
| EMA target encoder | cập nhật mỗi bước bằng moving average | mượt | BYOL, DINO, MoCo, I-JEPA |
| Joint gradient | cả hai nhánh tối ưu cùng loss | target di chuyển tức thời | symmetric Siamese dễ collapse |

MoCo dùng EMA encoder để các keys cũ trong queue vẫn nhất quán với keys mới. BYOL/DINO dùng EMA để tạo bootstrap target. JEPA dùng EMA encoder để tạo latent target cho vùng bị mask. Cùng một primitive cập nhật, nhưng invariant cần bảo vệ khác nhau.

---

## **7. Giới hạn / Khi nào thất bại**

### EMA không tự chống collapse

Một teacher và student cùng output hằng số vẫn thoả mãn consistency loss. BYOL cần predictor + stop-gradient; DINO cần centering, sharpening và thiết kế multi-crop. Gắn EMA vào một objective sai không biến nó thành objective đúng.

### Teacher có thể quá stale

Momentum quá cao ở đầu training khiến student học theo representation gần ngẫu nhiên quá lâu. Khi data distribution đổi nhanh hoặc online model có phase transition, teacher trễ có thể cản thích nghi.

### Thêm gấp đôi state huấn luyện

Teacher weights không cần gradient, nhưng vẫn chiếm parameter memory, checkpoint storage và forward compute. Với model lớn, chi phí này đáng kể dù inference cuối chỉ giữ student encoder.

### Checkpoint và resume dễ sai âm thầm

Phải lưu student, teacher, optimizer, scheduler, momentum schedule position và các EMA buffers như DINO center. Resume chỉ từ student rồi khởi tạo lại teacher làm mất ensemble history; run vẫn chạy nhưng target dynamics đổi.

### Parameters không phải toàn bộ state

BatchNorm running statistics, quantizer statistics hoặc custom buffers không nhất thiết được cập nhật đúng khi chỉ zip `parameters()`. Cần policy rõ: copy buffer, EMA buffer, hay forward teacher để buffer tự cập nhật. Official DINO còn phải xử lý SyncBatchNorm và distributed center.

### Distributed update phải đồng nhất

Mọi worker phải bắt đầu từ cùng teacher và áp dụng cùng student parameters sau synchronized optimizer step. Center hoặc statistic EMA phải all-reduce trước khi update; nếu mỗi rank có teacher state khác nhau, target phụ thuộc worker.

---

## **8. Liên hệ với Latent-Anything**

EMA target encoder là **stateful training primitive** nằm giữa ModelAdapter và Layer C runtime.

- **`ModelAdapter`**: cần phân biệt online parameters, target parameters và inference encoder; `encode()` sau training không nên vô tình dùng nhánh sai.
- **Layer A — Introspection**: theo dõi teacher-student parameter distance, target variance, feature effective rank, output entropy và lag theo layer. Loss mượt không đồng nghĩa representation khoẻ.
- **Layer B — Manipulation**: stop-gradient và EMA là hai phép can thiệp khác nhau lên learning dynamics; config cần mô tả cả gradient routing lẫn state update.
- **Layer C — Runtime**: phải đảm bảo thứ tự `forward → backward → optimizer → EMA`, hỗ trợ distributed synchronization, mixed precision an toàn và checkpoint atomic.
- **Reproducibility**: cache/config hash cần chứa momentum schedule, update frequency, buffer policy và target initialization.

Một interface tối thiểu có thể xem EMA updater như state riêng:

```python
target_updater = EMAUpdater(
    initial_momentum=0.996,
    final_momentum=1.0,
    schedule="cosine",
    update_after="optimizer_step",
)
```

Primitive này sẽ được dùng trực tiếp trong **JEPA (mục 6)**: context encoder nhận gradient, target encoder tạo representation đích cho vùng cần dự đoán, và EMA giữ coordinate system của target đủ ổn định để predictor học được.

---

## Liên quan

- [Stop-gradient và Kiến trúc Bất đối xứng](03-stop-gradient-asymmetric.md) — tách gradient routing khỏi target-parameter dynamics.
- [Contrastive Learning](04-contrastive-learning.md) — MoCo dùng EMA cho dictionary consistency; BYOL/DINO dùng nó cho bootstrap target.
- [Representation Collapse](02-representation-collapse.md) — các collapse mode mà EMA một mình không loại bỏ.
- [Latent Transition Model](../../06-latent-temporal/research/02-latent-transition-model.md) — target ổn định đặc biệt quan trọng khi prediction target cũng thay đổi theo model.
- [Value Function trong Latent](../../07-latent-planning/research/03-value-function-in-latent.md) — target networks trong temporal-difference learning có cùng động cơ giảm moving-target feedback.

## Tham khảo

- A. Tarvainen, H. Valpola, *Mean Teachers Are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results* (NeurIPS 2017, arXiv:1703.01780).
- J.-B. Grill et al., *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning* (NeurIPS 2020, arXiv:2006.07733).
- K. He, H. Fan, Y. Wu, S. Xie, R. Girshick, *Momentum Contrast for Unsupervised Visual Representation Learning* (CVPR 2020, arXiv:1911.05722).
- M. Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (ICCV 2021, arXiv:2104.14294).
- M. Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (CVPR 2023, arXiv:2301.08243).
