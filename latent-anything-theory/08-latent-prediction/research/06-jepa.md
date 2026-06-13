# JEPA — Joint-Embedding Predictive Architecture

> **TL;DR.** JEPA encode hai tín hiệu liên quan $x,y$ thành $s_x,s_y$, rồi học predictor $\hat s_y=\operatorname{Pred}(s_x,z)$ sao cho $\hat s_y$ gần $s_y$ — loss nằm trong representation space, không trong pixel/token space. Encoder target có thể bỏ chi tiết khó dự đoán nhưng không cần thiết, còn latent $z$ có thể biểu diễn nhiều kết quả hợp lệ. Caveat: JEPA là một họ kiến trúc chứ không phải một thuật toán hoàn chỉnh; collapse prevention, target construction và uncertainty modeling vẫn phải được thiết kế riêng.

[Masked Autoencoder](01-masked-autoencoder-mae.md) dự đoán phần quan sát bị che trong pixel space. [Contrastive learning](04-contrastive-learning.md) so sánh representation của positive với negatives. JEPA chọn trục thứ ba: **dự đoán representation của phần chưa thấy từ representation của phần đã thấy**, không cần tạo lại quan sát và không bắt buộc dùng negatives.

Ý tưởng này phù hợp trực tiếp với world model. Một agent thường không cần dự đoán chính xác từng pixel của tương lai; nó cần dự đoán trạng thái trừu tượng đủ để biết vật thể ở đâu, chúng có thể làm gì, action nào khả thi và kết quả nào nguy hiểm.

---

## **1. Kiến trúc tổng quát**

Cho hai biến tương thích $x$ và $y$:

$$
s_x=f_\theta(x),
\qquad
s_y=f_{\bar\theta}(y).
$$

Trong đó $f_\theta$ là context encoder, $f_{\bar\theta}$ là target encoder, còn $s_x,s_y$ là representation của hai tín hiệu. Hai encoder không bắt buộc cùng kiến trúc, cùng modality hoặc share parameters.

Predictor nhận context representation và một conditioning/latent variable $z$:

$$
\hat s_y=g_\phi(s_x,z).
$$

Trong đó $g_\phi$ là predictor và $z$ chứa thông tin cần cho prediction nhưng không có trong $x$, chẳng hạn vị trí target block, action, time offset hoặc một nhánh tương lai khả dĩ.

Energy của một bộ $(x,y,z)$ là prediction error trong embedding space:

$$
E_w(x,y,z)
=
D\!\left(s_y,g_\phi(s_x,z)\right).
$$

Trong đó $D$ là dissimilarity như $L_1$, $L_2$ hoặc cosine distance, còn $w$ gom tham số của encoders và predictor. Cặp tương thích có energy thấp khi target representation dự đoán được từ context.

Nếu $z$ không quan sát được, energy giữa $x$ và $y$ có thể lấy minimum:

$$
F_w(x,y)
=
\min_{z\in\mathcal{Z}} E_w(x,y,z),
\qquad
\hat z=\arg\min_{z\in\mathcal{Z}}E_w(x,y,z).
$$

Trong đó $\mathcal{Z}$ là không gian latent uncertainty và $\hat z$ là explanation làm $x,y$ tương thích nhất. Đây là cách JEPA gốc nối joint embedding với latent-variable energy-based model.

### Ba vai trò không được trộn

- **Encoder $f_\theta$**: rút representation của thông tin đang có.
- **Target encoder $f_{\bar\theta}$**: định nghĩa cái gì trong $y$ đáng được dự đoán.
- **Predictor $g_\phi$**: mô hình hoá quan hệ có điều kiện giữa hai representation.

Nếu predictor quá mạnh hoặc $z$ mang quá nhiều thông tin, predictor có thể bỏ qua $s_x$. Nếu target encoder quá bất biến, $s_y$ mất thông tin. Nếu target encoder giữ mọi chi tiết, bài toán quay lại gần pixel reconstruction.

---

## **2. JEPA khác JEA và generative architecture thế nào**

| Kiến trúc | Mục tiêu | Không gian loss | Output cần tạo | Collapse |
|---|---|---|---|---|
| Generative / reconstruction | $\hat y\approx y$ | input space | pixel, token hoặc signal | thường được neo bởi reconstruction, nhưng latent có thể bị bỏ qua |
| Joint Embedding Architecture (JEA) | $f(x)\approx f(y)$ | representation space | không | có, nếu mọi input map về hằng số |
| Contrastive JEA | positive gần, negative xa | representation space | không | chặn bằng negatives |
| JEPA | $\operatorname{Pred}(f(x),z)\approx f(y)$ | representation space | target embedding | có, cần cơ chế riêng |

JEA học **invariance**: hai view tương thích nên có embedding giống nhau. JEPA học **predictability**: representation của $y$ phải suy ra được từ representation của $x$ khi biết conditioning $z$. Predictor cho phép $s_x$ và $s_y$ khác nhau có hệ thống thay vì buộc chúng bằng nhau.

Generative model phải dành capacity cho mọi chi tiết của $y$ mà loss quan sát được. JEPA cho target encoder học quotient space:

$$
y_1\sim y_2
\quad\Longleftrightarrow\quad
f_{\bar\theta}(y_1)=f_{\bar\theta}(y_2).
$$

Trong đó quan hệ tương đương $\sim$ gom các observation khác nhau nhưng có cùng representation. Nếu texture hoặc noise bị encoder loại bỏ, predictor không phải mô hình hoá chúng.

Đổi lại, JEPA thường không thể reconstruct $y$ từ $\hat s_y$. Nó học **compatibility và state abstraction**, không học observation generator.

---

## **3. Tại sao prediction trong representation space có thể tốt hơn**

Giả sử target observation tách thành phần dự đoán được có ý nghĩa $u$, chi tiết nuisance $n$, và bất định không quan sát được $\epsilon$:

$$
y=h(u,n,\epsilon).
$$

Trong đó $h$ là quá trình sinh observation. Pixel loss buộc predictor giải thích cả $u,n,\epsilon$, dù context chỉ đủ thông tin cho $u$.

Target encoder có thể học:

$$
s_y=f_{\bar\theta}(y)\approx r(u),
$$

trong đó $r(u)$ giữ cấu trúc semantic và bỏ $n,\epsilon$ không hữu ích. Predictor khi đó tối ưu:

$$
g_\phi(s_x,z)\approx r(u),
$$

thay vì tạo lại toàn bộ $h(u,n,\epsilon)$. Bài toán có entropy hiệu dụng thấp hơn và tập trung capacity vào dependency giữa context và target.

Đây không phải bảo đảm tự động. Encoder chỉ bỏ "chi tiết không liên quan" nếu objective, masking, architecture và anti-collapse mechanism ép nó làm vậy. Một target encoder tệ có thể bỏ cả thông tin cần thiết hoặc giữ shortcut.

---

## **4. Latent variable $z$ và nhiều tương lai hợp lệ**

Prediction thường đa trị. Từ một frame xe đến ngã rẽ, tương lai có thể rẽ trái hoặc phải. Một deterministic predictor với squared loss dễ trả về trung bình:

$$
\hat s_y^\star
=
\mathbb{E}[s_y\mid s_x].
$$

Trong đó $\hat s_y^\star$ là nghiệm tối ưu của mean squared error. Nếu hai mode nằm xa nhau, trung bình có thể không tương ứng với tương lai hợp lệ nào.

JEPA tổng quát cho phép:

$$
\hat s_y^{(k)}=g_\phi(s_x,z_k),
\qquad z_k\in\mathcal{Z},
$$

trong đó mỗi $z_k$ tạo một prediction mode. Với ví dụ ngã rẽ, $z$ có thể encode "trái" hoặc "phải"; trong world model, $z$ có thể encode action, intent của agent khác hoặc stochastic event.

Nhưng capacity của $z$ phải bị giới hạn. Nếu $z$ đủ lớn để copy toàn bộ $s_y$:

$$
g_\phi(s_x,z)=z,\qquad z=s_y,
$$

thì predictor bỏ qua context và energy luôn bằng 0. Đây là latent bypass collapse. Các cách giới hạn gồm:

- dimension nhỏ;
- discrete code với ít trạng thái;
- sparsity;
- noise hoặc prior regularization;
- information bottleneck trên $I(Z;Y)$;
- action/position variable có semantics được biết trước.

I-JEPA và V-JEPA hiện thực đơn giản thường không dùng latent uncertainty được inference theo nghĩa này; mask/position tokens chủ yếu cho predictor biết **dự đoán ở đâu**. Modeling nhiều tương lai dài hạn vẫn là bài toán mở hơn.

---

## **5. Huấn luyện JEPA và chống collapse**

Với target representation được stop-gradient, objective cơ bản là:

$$
\mathcal{L}_{\mathrm{pred}}
=
\mathbb{E}_{(x,y)}
\left[
D\!\left(
g_\phi(f_\theta(x),z),
\operatorname{sg}(f_{\bar\theta}(y))
\right)
\right].
$$

Trong đó gradient cập nhật context encoder $\theta$ và predictor $\phi$, không cập nhật target trực tiếp qua loss.

Objective này vẫn có nghiệm hằng:

$$
f_\theta(x)=f_{\bar\theta}(y)=c,
\qquad
g_\phi(c,z)=c.
$$

Trong đó mọi input map về cùng $c$, làm prediction error bằng 0. Vì vậy "predict latent thay vì pixel" không tự chống collapse.

Các họ giải pháp:

| Cơ chế | Cách tạo áp lực thông tin | Ví dụ |
|---|---|---|
| Negatives | representation khác phải tách nhau | CPC, contrastive JEPA |
| Variance/covariance regularization | mỗi chiều phải có variance, giảm redundancy | VICReg-style |
| Clustering/entropy | duy trì usage của prototypes | DINO-style |
| Asymmetry + stop-gradient | target không chạy theo cùng gradient | SimSiam, BYOL |
| [EMA target encoder](05-ema-target-encoder.md) | target coordinate system thay đổi chậm | I-JEPA, V-JEPA |
| Masking/task design | context phải đủ thông tin, target phải có semantics | I-JEPA, V-JEPA |

I-JEPA cập nhật target encoder bằng EMA của context encoder và tối ưu $L_1$ trung bình giữa predicted patch representations với target patch representations. Target được encode từ ảnh đầy đủ rồi mới chọn block; context encoder chỉ thấy visible context. Cách tách này tránh để mask token làm giảm chất lượng target.

---

## **6. Conditioning variable khác uncertainty latent**

Ký hiệu $z$ trong JEPA tổng quát bao phủ nhiều loại thông tin, nhưng nên phân biệt:

| Loại conditioning | Có quan sát lúc prediction? | Vai trò |
|---|---|---|
| Vị trí target | có | cho biết vùng cần dự đoán |
| Time offset | có | cho biết horizon |
| Action | có hoặc do planner đề xuất | điều kiện dynamics |
| Modality identifier | có | chọn output space |
| Stochastic latent | không | biểu diễn nhiều outcome hợp lệ |

Position token trong masked JEPA không phải uncertainty: nó xác định câu hỏi. Action trong world model cũng thường là control input, không phải noise. Chỉ phần thông tin không quan sát được nhưng cần để giải thích target mới là latent uncertainty theo nghĩa chặt.

---

## **7. Từ representation learner đến world model**

JEPA trên ảnh tĩnh học spatial completion:

$$
\text{visible context}\rightarrow\text{masked-region representation}.
$$

JEPA trên video học spatiotemporal completion:

$$
\text{visible tubes}\rightarrow\text{masked space-time representation}.
$$

Action-conditioned JEPA world model học:

$$
(s_t,a_t,z_t)\rightarrow \hat s_{t+1}.
$$

Trong đó $s_t$ là latent state, $a_t$ là action và $z_t$ là stochastic latent nếu dynamics đa trị. Đây chính là [latent transition model](../../06-latent-temporal/research/02-latent-transition-model.md), nhưng target state được học đồng thời bằng joint-embedding prediction thay vì được định nghĩa sẵn.

Một planner có thể đánh giá trajectory hoàn toàn trong latent:

$$
\hat s_{t+1}=g_\phi(\hat s_t,a_t,z_t),
\qquad
J=\sum_{t=0}^{T} C(\hat s_t).
$$

Trong đó $C$ là cost/reward model và $J$ là tổng cost của imagined trajectory. Không cần decode từng state ra pixel để planning, miễn latent giữ đúng biến liên quan đến cost và feasibility.

---

## **8. Giới hạn / Khi nào thất bại**

### JEPA không phải một loss duy nhất

Tên kiến trúc chưa xác định distance, target update, anti-collapse regularizer, masking distribution, predictor capacity hay uncertainty inference. Hai model cùng gọi là JEPA có thể có learning dynamics rất khác.

### Semantic abstraction phụ thuộc task

Target lớn và context phân tán có thể khuyến khích semantics trong ảnh; target nhỏ có thể khuyến khích texture. Với robotics, semantic object identity chưa chắc đủ: planning còn cần contact, velocity, affordance và uncertainty.

### Loss latent khó diễn giải

Pixel error có đơn vị trực quan; latent $L_1$ nhỏ chỉ nói predictor khớp target encoder hiện tại. Nếu target encoder bỏ thông tin quan trọng, loss vẫn đẹp. Cần downstream probes và causal tests.

### Moving target và coordinate drift

Context encoder, target encoder và predictor cùng tiến hoá. [EMA](05-ema-target-encoder.md) giảm drift nhưng không loại bỏ nó. Checkpoint hoặc distributed update sai có thể đổi target coordinate system âm thầm.

### Multimodality chưa được giải quyết đầy đủ

Một deterministic predictor thường học conditional mean trong representation space. Latent-variable JEPA đưa ra khung cho nhiều mode, nhưng inference, regularization và tránh latent bypass vẫn khó, đặc biệt qua rollout dài.

### Không có decoder làm debugging khó hơn

Không thể nhìn trực tiếp $\hat s_y$ để biết model dự đoán gì. Cần nearest-neighbour retrieval, probe, attention map, linear decoder dùng riêng cho evaluation hoặc compare với known factors.

### Collapse vẫn là rủi ro

EMA và masking là heuristic mạnh trong các hiện thực thành công, không phải chứng minh tổng quát. Representation có thể complete collapse hoặc chỉ dùng một subspace nhỏ; phải theo dõi effective rank và variance.

---

## **9. Liên hệ với Latent-Anything**

JEPA là trường hợp sử dụng trung tâm cho thesis "latent space là môi trường suy luận chính".

- **`LatentSpace`**: phải mô tả context space, target space, geometry và liệu hai encoder có dùng cùng coordinate system hay không.
- **`ModelAdapter`**: cần expose `encode_context`, `encode_target`, `predict_latent` và optional conditioning schema thay vì giả định luôn có `decode`.
- **`Trajectory`**: temporal JEPA tạo imagined sequence trực tiếp bằng repeated latent prediction; uncertainty latent cần được lưu cùng mỗi transition.
- **Layer A — Introspection**: audit target variance, effective rank, prediction residual theo region/horizon, nearest neighbours và information retained by probes.
- **Layer B — Manipulation**: action, mask, position và stochastic latent đều là conditioning có thể thay đổi để can thiệp prediction.
- **Layer C — Runtime**: quản lý online/EMA target, masking, predictor batching, multi-target loss và rollout không decode.

Một contract khái niệm:

```python
context = adapter.encode_context(observation, visible_mask)
target = adapter.encode_target(observation, target_mask)
prediction = adapter.predict_latent(
    context,
    condition={"target_mask": target_mask, "action": action, "latent": z},
)
residual = latent_metric(prediction, stop_gradient(target))
```

Điểm quan trọng là `decode` trở thành optional output head. Planning và reasoning dùng `prediction`; decode chỉ phục vụ quan sát, supervision bổ sung hoặc debugging.

---

## Liên quan

- [Masked Autoencoder (MAE)](01-masked-autoencoder-mae.md) — baseline reconstruct pixel thay vì target embedding.
- [Representation Collapse](02-representation-collapse.md) — nghiệm tầm thường vẫn tồn tại trong JEPA.
- [Stop-gradient và Kiến trúc Bất đối xứng](03-stop-gradient-asymmetric.md) — gradient routing giữa context và target branch.
- [EMA Target Encoder](05-ema-target-encoder.md) — giữ target representation thay đổi chậm.
- [Latent Transition Model](../../06-latent-temporal/research/02-latent-transition-model.md) — dạng temporal/action-conditioned của latent predictor.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — lặp predictor thành imagined trajectory.
- [Value Equivalence (MuZero)](../../07-latent-planning/research/08-value-equivalence-muzero.md) — ví dụ latent chỉ cần giữ thông tin phục vụ prediction/planning, không cần reconstruct observation.

## Tham khảo

- Y. LeCun, *A Path Towards Autonomous Machine Intelligence* (OpenReview position paper, Version 0.9.2, 2022-06-27).
- M. Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (CVPR 2023, arXiv:2301.08243).
- A. Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (ECCV 2024, arXiv:2404.08471).
- A. van den Oord, Y. Li, O. Vinyals, *Representation Learning with Contrastive Predictive Coding* (arXiv 2018, arXiv:1807.03748).
- A. Baevski et al., *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language* (ICML 2022, arXiv:2202.03555).
