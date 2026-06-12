# RSSM — Recurrent State Space Model

> **TL;DR.** RSSM biểu diễn state bằng cặp $s_t=(h_t,z_t)$, trong đó $h_t$ là recurrent state tất định giữ memory dài hạn, còn $z_t$ là stochastic state nhận thông tin mới và biểu diễn phần bất định. Khi quan sát dữ liệu, posterior $q_\phi(z_t\mid h_t,o_t)$ hiệu chỉnh state; khi imagination, prior $p_\theta(z_t\mid h_t)$ tự rollout mà không cần observation. Caveat chính là split này không tự bảo đảm world model đúng: posterior collapse, prior kém, reconstruction shortcut và compounding error vẫn có thể làm imagined trajectory sai.

[Stochastic Transition](03-stochastic-transition.md) mở rộng transition từ một điểm sang phân phối. RSSM đi thêm một bước: không buộc toàn bộ history, uncertainty và thông tin mới phải đi qua cùng một stochastic vector. State được tách thành một đường recurrent tất định và một biến ẩn stochastic, rồi hai phần phối hợp để dự báo observation, reward và tương lai.

Kiến trúc này được đặt tên trong PlaNet. DreamerV1 dùng RSSM làm world model và thay online planning bằng actor-critic học trên imagined trajectories. DreamerV2 và DreamerV3 tiếp tục giữ khung recurrent-plus-stochastic nhưng thay Gaussian latent bằng categorical latent và bổ sung các kỹ thuật ổn định KL.

---

## **1. Vì sao không chỉ dùng RNN hoặc state-space model stochastic?**

Một recurrent neural network thuần tất định cập nhật hidden state:

$$
h_t=f_\theta(h_{t-1},a_{t-1},e_t),
$$

trong đó $h_t$ là hidden state, $a_{t-1}$ là action trước và $e_t$ là embedding của observation hiện tại. Cách này giữ memory tốt, nhưng mỗi history chỉ ánh xạ tới một state duy nhất nên khó biểu diễn nhiều tương lai hoặc uncertainty có cấu trúc.

Một state-space model thuần stochastic dùng:

$$
z_t\sim p_\theta(z_t\mid z_{t-1},a_{t-1}),
\qquad
o_t\sim p_\theta(o_t\mid z_t).
$$

Trong đó $z_t$ là stochastic state và $o_t$ là observation. Mô hình có thể biểu diễn phân phối tương lai, nhưng toàn bộ memory phải được truyền qua chuỗi sample stochastic; trong thực tế việc tối ưu để một số chiều giữ thông tin gần tất định qua horizon dài có thể khó.

RSSM kết hợp hai đường:

| Kiến trúc | Memory dài hạn | Nhiều tương lai / uncertainty | Điểm yếu chính |
|---|---|---|---|
| **RNN thuần tất định** | mạnh | yếu | planner có thể khai thác prediction quá tự tin |
| **SSM thuần stochastic** | khó tối ưu ổn định | mạnh | thông tin phải đi qua sampling ở mọi bước |
| **RSSM** | recurrent state $h_t$ | stochastic state $z_t$ | training và semantics phức tạp hơn |

PlaNet thực nghiệm thấy cả deterministic path lẫn stochastic path đều quan trọng cho planning. Ý nghĩa của split không phải "phần chắc chắn" và "phần ngẫu nhiên" được tách hoàn hảo theo nghĩa vật lý; đây là inductive bias giúp model có một kênh memory ổn định và một bottleneck xác suất để nhận thông tin mới.

---

## **2. State của RSSM**

Ký hiệu state tổng hợp:

$$
s_t=(h_t,z_t),
$$

trong đó $h_t\in\mathbb{R}^{d_h}$ là deterministic recurrent state và $z_t$ là stochastic latent state. Decoder, reward head, continuation head, actor và critic thường đọc feature được tạo từ cả hai phần.

Một bước RSSM chuẩn gồm bốn phép tính.

### 2.1 Deterministic transition

Recurrent state được cập nhật từ state và action trước:

$$
h_t
=
f_\theta(h_{t-1},z_{t-1},a_{t-1}).
$$

Trong đó $f_\theta$ thường là GRU hoặc recurrent block tương đương. $h_t$ nén history của các stochastic state và action trước đó; observation $o_t$ chưa được dùng trực tiếp ở bước này.

### 2.2 Prior hoặc dynamics predictor

Từ recurrent state, model dự báo phân phối stochastic state:

$$
\hat z_t
\sim
p_\theta(z_t\mid h_t).
$$

Trong đó $p_\theta$ là prior transition, còn $\hat z_t$ là sample dùng khi model tự imagination. Prior không nhìn observation hiện tại nên đây là phân phối phải đủ tốt cho open-loop rollout.

### 2.3 Posterior hoặc representation model

Khi có observation thật, encoder tạo embedding:

$$
e_t=\operatorname{Enc}_\phi(o_t),
$$

trong đó $e_t$ là representation của observation $o_t$. Embedding này cùng $h_t$ tham số hóa posterior:

$$
z_t
\sim
q_\phi(z_t\mid h_t,e_t).
$$

Trong đó $q_\phi$ là approximate posterior hoặc representation model. Posterior được quan sát $o_t$ hiệu chỉnh nên thường mang nhiều thông tin hơn prior.

### 2.4 Prediction heads

State tổng hợp dự báo các đại lượng cần thiết:

$$
\hat o_t\sim p_\theta(o_t\mid h_t,z_t),
\qquad
\hat r_t\sim p_\theta(r_t\mid h_t,z_t),
\qquad
\hat c_t\sim p_\theta(c_t\mid h_t,z_t).
$$

Trong đó $\hat o_t$, $\hat r_t$ và $\hat c_t$ lần lượt là observation, reward và continuation prediction. Continuation $c_t$ biểu diễn xác suất episode còn tiếp tục; một số RSSM chỉ có observation và reward head.

Trong thiết kế PlaNet, observation không đi thẳng vào deterministic path. Thông tin mới phải đi qua posterior sample $z_t$ trước khi ảnh hưởng bước recurrent tiếp theo. Ràng buộc này tránh một deterministic shortcut cho phép decoder tái tạo ảnh mà stochastic state bị bỏ qua.

---

## **3. Hai chế độ chạy: filtering và imagination**

Cùng một RSSM có hai execution mode khác nhau.

### Filtering với observation thật

Tại mỗi bước dữ liệu:

$$
h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1}),
\qquad
z_t\sim q_\phi(z_t\mid h_t,\operatorname{Enc}_\phi(o_t)).
$$

Trong đó posterior dùng observation hiện tại để correction. Chuỗi $(h_t,z_t)$ tạo ra theo cách này thường được gọi là posterior trajectory hoặc observed trajectory.

Filtering xấp xỉ belief update trong POMDP: $h_t$ mang memory hữu hạn chiều, còn distribution của $z_t$ biểu diễn phần state được suy luận từ observation mới.

### Imagination không có observation tương lai

Sau state khởi đầu, model lặp:

$$
h_{t+1}
=
f_\theta(h_t,\hat z_t,a_t),
\qquad
\hat z_{t+1}
\sim
p_\theta(z_{t+1}\mid h_{t+1}).
$$

Trong đó $\hat z_t$ là prior sample và action $a_t$ đến từ planner hoặc policy. Không có encoder và không có observation correction; toàn bộ trajectory phụ thuộc vào dynamics đã học.

| Chế độ | Stochastic state đến từ | Có observation correction? | Dùng cho |
|---|---|---|---|
| **Filtering / posterior** | $q_\phi(z_t\mid h_t,e_t)$ | có | state estimation, world-model training |
| **One-step prediction** | $p_\theta(z_{t+1}\mid h_{t+1})$ | chưa | đánh giá dynamics prior |
| **Open-loop imagination** | prior lặp nhiều bước | không | planning, actor-critic, counterfactual rollout |

Khoảng cách giữa posterior và prior là tín hiệu cốt lõi. Posterior có thể dự báo tốt không có nghĩa prior imagination tốt; decoder đẹp trên posterior trajectory vẫn có thể che một transition model yếu.

---

## **4. Học RSSM bằng variational objective**

Với posterior trajectory, world model tối ưu prediction loss và regularization giữa posterior với prior.

### Prediction loss

Một objective tổng quát là:

$$
\mathcal{L}_{\text{pred}}
=
\sum_{t=1}^{T}
\left[
-\log p_\theta(o_t\mid h_t,z_t)
-\log p_\theta(r_t\mid h_t,z_t)
-\log p_\theta(c_t\mid h_t,z_t)
\right].
$$

Trong đó ba số hạng lần lượt huấn luyện observation, reward và continuation head. Tùy distribution head, negative log-likelihood có thể trở thành MSE, cross-entropy hoặc một transformed regression loss.

### KL regularization

Posterior được kéo về prior:

$$
\mathcal{L}_{\text{KL}}
=
\sum_{t=1}^{T}
D_{\mathrm{KL}}
\left(
q_\phi(z_t\mid h_t,e_t)
\;\|\;
p_\theta(z_t\mid h_t)
\right).
$$

Trong đó KL đo lượng thông tin observation hiện tại đưa vào stochastic state mà prior chưa dự báo được. KL nhỏ khiến prior và posterior nhất quán hơn cho imagination; KL quá mạnh có thể làm posterior bỏ qua observation.

World-model loss cơ bản:

$$
\mathcal{L}_{\text{model}}
=
\mathcal{L}_{\text{pred}}
+
\beta\mathcal{L}_{\text{KL}},
$$

trong đó $\beta$ điều chỉnh trade-off giữa state giàu thông tin và state dễ dự báo. Đây là sequential analogue của [VAE](../../02-representation-learning/research/03-vae.md): reconstruction/prediction giữ thông tin, còn KL tạo prior có thể sample.

### ELBO interpretation

Bỏ reward và continuation để đơn giản, objective tương ứng với negative evidence lower bound:

$$
\log p_\theta(o_{1:T}\mid a_{1:T-1})
\ge
\sum_{t=1}^{T}
\mathbb{E}_{q_\phi}
\left[
\log p_\theta(o_t\mid h_t,z_t)
\right]
-
\sum_{t=1}^{T}
D_{\mathrm{KL}}(q_\phi(z_t\mid h_t,e_t)\|p_\theta(z_t\mid h_t)).
$$

Trong đó vế phải gồm expected observation likelihood và KL theo thời gian. Tối đa hóa cận dưới đồng thời học representation posterior, transition prior và decoder.

### Reparameterization hoặc straight-through

DreamerV1 dùng Gaussian stochastic state:

$$
z_t
=
\mu_\phi(h_t,e_t)
+
\sigma_\phi(h_t,e_t)\odot\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,I).
$$

Trong đó reparameterization cho phép gradient đi qua posterior sample. DreamerV2/V3 dùng nhiều biến categorical và straight-through estimator: forward pass dùng one-hot sample, backward pass truyền gradient qua probability vector xấp xỉ.

---

## **5. Vì sao split deterministic–stochastic giúp long-horizon prediction?**

### Deterministic path giữ memory

$h_t$ có thể mang thông tin qua nhiều bước mà không phải sample lại. Ví dụ object tạm thời ra khỏi camera vẫn có thể được recurrent state ghi nhớ cho tới khi xuất hiện lại.

### Stochastic path tạo information bottleneck

$z_t$ là nơi observation mới đi vào dynamics. KL giới hạn lượng thông tin mới, khuyến khích model tận dụng memory từ $h_t$ và chỉ encode residual mà prior chưa biết.

### Prior học cách bắt chước state đã được observation hiệu chỉnh

Posterior đóng vai trò target giàu thông tin. KL huấn luyện prior dự báo distribution của posterior chỉ từ history và action, nhờ đó model có thể tiếp tục khi observation tương lai bị bỏ đi.

### Uncertainty không làm nhiễu toàn bộ memory

Các yếu tố cần giữ ổn định có thể đi qua $h_t$, còn event khó đoán được biểu diễn qua $z_t$. Đây là lợi thế tối ưu thực dụng, không phải bảo đảm disentanglement: model vẫn có thể đặt thông tin "ngẫu nhiên" vào $h_t$ hoặc information dài hạn vào $z_t$ nếu loss cho phép.

### Ablation của PlaNet

PlaNet so sánh RSSM với GRU thuần tất định và SSM thuần stochastic trên sáu control tasks từ pixel. Kết quả cho thấy deterministic path giúp nhớ thông tin qua nhiều bước, còn stochastic component đặc biệt quan trọng dưới partial observability và dynamics khó dự báo.

Không nên đọc ablation này thành định luật rằng mọi task đều cần RSSM. Với state thật đầy đủ và dynamics gần tất định, residual deterministic model có thể đơn giản và hiệu quả hơn. RSSM có giá trị khi observation là high-dimensional, history quan trọng và open-loop rollout cần một latent prior có cấu trúc.

---

## **6. KL balancing, free nats và stop-gradient**

KL chuẩn có thể giảm theo hai hướng:

- prior tốt hơn, tiến gần posterior;
- posterior ít thông tin hơn, tiến gần prior.

Hai hướng không có giá trị như nhau cho imagination. Nếu posterior tăng entropy hoặc collapse chỉ để giảm KL, prior chưa chắc học dynamics tốt hơn.

### KL balancing trong DreamerV2

KL được tách bằng stop-gradient:

$$
\mathcal{L}_{\text{dyn}}
=
D_{\mathrm{KL}}
\left(
\operatorname{sg}(q_\phi)
\;\|\;
p_\theta
\right),
\qquad
\mathcal{L}_{\text{rep}}
=
D_{\mathrm{KL}}
\left(
q_\phi
\;\|\;
\operatorname{sg}(p_\theta)
\right).
$$

Trong đó $\operatorname{sg}$ là stop-gradient. $\mathcal{L}_{\text{dyn}}$ chỉ cập nhật prior theo posterior cố định, còn $\mathcal{L}_{\text{rep}}$ chỉ regularize posterior theo prior cố định.

Hai loss được cân bằng:

$$
\mathcal{L}_{\text{KL-balanced}}
=
\alpha\mathcal{L}_{\text{dyn}}
+
(1-\alpha)\mathcal{L}_{\text{rep}},
$$

trong đó $\alpha$ lớn hơn ưu tiên học prior dynamics thay vì làm posterior mất thông tin. DreamerV2 báo cáo KL balancing tốt hơn KL chuẩn trên phần lớn Atari tasks trong ablation của paper.

### Free nats / free bits

Để tránh regularization bóp stochastic state trước khi prediction heads học đủ, KL có thể được clip dưới một ngưỡng:

$$
\mathcal{L}_{\text{free}}
=
\max(\tau,\mathcal{L}_{\text{KL}}),
$$

trong đó $\tau$ là lượng thông tin miễn phí, đo bằng nat hoặc bit. Khi KL đã thấp hơn $\tau$, gradient regularization tạm dừng và model tập trung cải thiện prediction.

DreamerV3 dùng dynamics loss và representation loss tách riêng với stop-gradient, rồi áp dụng free bits cho từng loss. Paper dùng ngưỡng 1 nat và thêm một lượng nhỏ uniform probability vào categorical distributions để tránh chúng trở nên gần deterministic, gây KL spike.

### Các failure mode của KL

| Failure mode | Dấu hiệu | Nguyên nhân thường gặp |
|---|---|---|
| **Posterior collapse** | $q\approx p$, latent không mang observation | KL quá mạnh, decoder quá khỏe |
| **Prior lag** | posterior reconstruction tốt, imagination kém | prior học chậm hơn encoder |
| **KL spike** | loss và gradient tăng đột ngột | categorical probability gần 0/1, distribution quá sắc |
| **Information leak** | decoder tốt nhưng stochastic state vô dụng | observation đi qua shortcut ngoài bottleneck |
| **Over-regularization** | chi tiết task-relevant biến mất | representation loss quá mạnh |

Vì vậy chỉ theo dõi tổng ELBO là chưa đủ. Cần log riêng prediction loss, dynamics KL, representation KL, prior entropy, posterior entropy và usage của từng stochastic dimension/category.

---

## **7. RSSM trong Dreamer**

RSSM là world model; Dreamer là cách dùng world model đó để học behavior.

Quy trình Dreamer có ba giai đoạn lặp:

1. Thu thập trajectory thật và lưu vào replay buffer.
2. Học RSSM cùng observation, reward và continuation heads trên sequence batch.
3. Bắt đầu từ posterior states trong batch, rollout prior bằng policy và học actor-critic trên imagined trajectories.

### Latent imagination

Actor tạo action từ model state:

$$
a_t\sim\pi_\psi(a_t\mid h_t,z_t).
$$

Trong đó $\pi_\psi$ là policy có tham số $\psi$. RSSM dùng action này để tạo prior state tiếp theo mà không decode image.

Reward và continuation được dự báo:

$$
\hat r_t=r_\theta(h_t,z_t),
\qquad
\hat c_t=c_\theta(h_t,z_t).
$$

Trong đó $\hat r_t$ và $\hat c_t$ cung cấp tín hiệu return trên imagined trajectory. Critic học value trên cùng latent feature.

DreamerV1 truyền analytic gradient qua differentiable dynamics và value predictions để cập nhật actor. Điểm quan trọng cho RSSM là actor-critic tiêu thụ **prior imagined states**, không phải posterior states luôn được observation sửa lỗi. World model vì thế phải được đánh giá ở chế độ open-loop mà behavior learning thực sự dùng.

World-model parameters thường không được actor-critic cập nhật trực tiếp trong bước behavior learning. Việc tách objective ngăn policy làm biến dạng latent dynamics chỉ để tăng imagined return.

### RSSM không đồng nghĩa reconstruction-only

Observation decoder cung cấp dense training signal, nhưng reward và continuation heads cũng định hình state. Nếu decoder dành capacity cho texture không liên quan, representation có thể tốt về pixel nhưng yếu cho control. Nếu bỏ decoder hoàn toàn, cần objective khác chống collapse và bảo toàn task-relevant information.

**Latent imagination (mục sau)** sẽ đi sâu vào rollout và cost. Ở đây điểm cốt lõi là RSSM tạo một simulator stateful có thể chuyển từ posterior filtering sang prior-only imagination.

---

## **8. Tiến hóa từ PlaNet đến DreamerV3**

| Model | Stochastic state | Cách dùng world model | Kỹ thuật nổi bật |
|---|---|---|---|
| **PlaNet** | Gaussian | CEM planning online | RSSM, latent overshooting |
| **DreamerV1** | Gaussian | actor-critic bằng latent imagination | analytic gradients qua imagined trajectories |
| **DreamerV2** | categorical | actor-critic trên Atari và continuous control | straight-through, KL balancing |
| **DreamerV3** | categorical | một cấu hình trên nhiều domain | dynamics/representation loss, free bits, symlog/two-hot predictions |

### Gaussian RSSM

Gaussian state có reparameterization đơn giản và phù hợp với smooth continuous dynamics. Giới hạn chính là prior Gaussian đơn khó khớp aggregate posterior đa mode.

### Categorical RSSM

DreamerV2 dùng một vector gồm nhiều categorical variables. Một sample được flatten thành sparse one-hot feature; paper dùng 32 categorical variables, mỗi biến có 32 classes. Đây là lựa chọn implementation của DreamerV2/V3, không phải yêu cầu định nghĩa RSSM.

Categorical latent phù hợp với event không trơn như đổi room, item biến mất hoặc contact regime thay đổi. DreamerV2 đưa ra một số giả thuyết cho lợi ích này, nhưng không khẳng định một giải thích duy nhất.

### DreamerV3 stability stack

DreamerV3 giữ RSSM categorical và thêm các kỹ thuật scale ổn định giữa domain:

- symlog transform cho đại lượng có dynamic range lớn;
- two-hot regression cho reward và value;
- dynamics loss và representation loss tách bằng stop-gradient;
- free bits để tránh over-regularization;
- uniform mixing để categorical distribution không quá sắc;
- normalization và percentile-based return scaling trong behavior learning.

Các kỹ thuật này thuộc full DreamerV3 agent, không phải tất cả đều là thành phần bắt buộc của RSSM.

---

## **9. Latent overshooting và multi-step consistency**

One-step KL chỉ huấn luyện prior tại $t+1$ từ posterior state tại $t$. Trong open-loop rollout, prior phải ăn sample do chính nó sinh ra nhiều bước liên tiếp.

PlaNet định nghĩa multi-step prior:

$$
p_\theta(z_t\mid z_{t-d},a_{t-d:t-1})
=
\int
\prod_{k=t-d+1}^{t}
p_\theta(z_k\mid h_k)
\;dz_{t-d+1:t-1},
$$

trong đó $d$ là overshooting distance và các state trung gian được marginalize hoặc xấp xỉ bằng sample. Distribution này dự báo $z_t$ từ một state cách $d$ bước mà không dùng observation trung gian.

Latent overshooting thêm KL:

$$
\mathcal{L}_{\text{over}}
=
\sum_{t}
\sum_{d=2}^{D}
\lambda_d
D_{\mathrm{KL}}
\left(
\operatorname{sg}(q_\phi(z_t\mid h_t,e_t))
\;\|\;
p_\theta(z_t\mid z_{t-d},a_{t-d:t-1})
\right).
$$

Trong đó $D$ là horizon tối đa, $\lambda_d$ là trọng số và posterior target được stop-gradient. Loss ép multi-step prior khớp state đã được observation hiệu chỉnh mà không cần decode thêm image cho mọi overshooting path.

PlaNet báo cáo một số dynamics model hưởng lợi từ latent overshooting, nhưng RSSM cuối cùng của họ không bắt buộc kỹ thuật này để đạt kết quả chính. Vì vậy overshooting là regularizer tùy chọn, không phải phần định nghĩa RSSM.

---

## **10. Đánh giá một RSSM**

### Posterior reconstruction

Đo observation/reward prediction khi $z_t$ đến từ posterior. Đây là kiểm tra encoder và heads, nhưng chưa đánh giá prior imagination đầy đủ.

### Prior–posterior gap

Theo dõi KL, distance giữa prior/posterior mean, category agreement hoặc decoded difference. Gap lớn cho thấy observation liên tục phải sửa những gì dynamics không dự báo được.

### Open-loop prediction

Cung cấp vài context frames, sau đó chuyển sang prior-only rollout. Báo cáo prediction quality theo horizon, reward error, continuation calibration và latent support drift.

### State ablation

So full RSSM với:

- bỏ stochastic state;
- bỏ deterministic state;
- dùng posterior mean thay sample;
- randomize hoặc zero một phần $h_t$;
- giảm history context.

Ablation giúp xác định model thực sự dùng memory và uncertainty hay chỉ dựa vào shortcut.

### Information usage

Với Gaussian, theo dõi active dimensions, posterior variance và KL per dimension. Với categorical, theo dõi entropy, class usage và dead categories. KL thấp không tự động tốt nếu stochastic state đã collapse.

### Behavior consistency

Nếu RSSM phục vụ control, đo correlation giữa imagined return và return thật, action ranking, constraint violations và policy performance khi thay đổi imagination horizon. Decoder đẹp nhưng xếp sai action vẫn là world model kém.

---

## **11. Giới hạn / Khi nào thất bại**

**Posterior tốt, prior yếu.** Observation correction có thể giữ trajectory đúng trong training, trong khi prior drift nhanh khi imagination. Đây là failure mode quan trọng nhất vì actor và planner dùng prior.

**Posterior collapse.** Decoder hoặc recurrent state giải thích dữ liệu mà không cần $z_t$. KL gần zero nhưng stochastic path không mang uncertainty hay thông tin mới.

**Deterministic shortcut.** Nếu observation embedding đi thẳng vào $h_t$ hoặc decoder, reconstruction không còn ép stochastic bottleneck hoạt động.

**Recurrent state che uncertainty.** $h_t$ là một điểm tất định. Nếu model nhét nhiều hypothesis chưa được resolve vào cùng hidden vector, uncertainty semantics của $z_t$ trở nên khó diễn giải.

**Unimodal Gaussian không đủ.** DreamerV1 RSSM vẫn chịu giới hạn của Gaussian đơn trước transition đa mode. Categorical latent giúp một số domain nhưng distribution factorized vẫn không biểu diễn mọi dependency.

**Categorical state mất geometry liên tục.** One-hot representation thuận lợi cho event rời rạc nhưng interpolation, metric và covariance không còn tự nhiên như Gaussian latent.

**KL objective không đồng nghĩa calibrated uncertainty.** Prior/posterior khớp nhau có thể cùng sai hoặc quá tự tin. RSSM không tự mô hình hóa epistemic uncertainty ngoài vùng dữ liệu.

**Compounding error.** Recurrent memory không loại bỏ distribution shift giữa posterior training và prior rollout. Error trong cả $h_t$ và $z_t$ có thể tích lũy.

**Reconstruction giữ nuisance.** Pixel decoder có thể ép state dành capacity cho texture, lighting hoặc background không liên quan tới reward.

**Task-oriented heads bỏ thông tin.** Ngược lại, representation chỉ tối ưu reward/value hiện tại có thể không còn phù hợp cho task mới, visualization hoặc transfer.

**Long sequence optimization vẫn khó.** Truncated backpropagation, gradient clipping, state reset và batch chunking ảnh hưởng memory học được. RSSM có recurrent path không có nghĩa model tận dụng được dependency tùy ý dài.

**Action distribution shift.** Policy học trong imagination có thể tìm action mà world model dự báo quá lạc quan. RSSM không tự giải quyết model exploitation.

**Structured world state khó đóng gói.** Một vector $z_t$ và GRU state có thể không giữ identity, permutation và cardinality của object/Gaussian set qua thời gian.

---

## **12. Liên hệ với Latent-Anything**

RSSM đặt ra một state schema giàu hơn một tensor latent:

```python
state = RSSMState(
    deterministic=h_t,
    stochastic=z_t,
    source="posterior",
)
```

Một RSSM adapter cần tách ít nhất:

- `deterministic`: recurrent memory $h_t$;
- `stochastic`: sample $z_t$;
- `stochastic_params`: Gaussian hoặc categorical parameters;
- `source`: `posterior` hay `prior`;
- `observation_embedding`: nếu cần audit representation model;
- recurrent cache, reset mask và continuation semantics;
- RNG state cho stochastic rollout.

Interface nên phân biệt ba operation:

```python
posterior_state = rssm.observe(previous_state, action, observation)
prior_prediction = rssm.predict(previous_state, action)
imagined = rssm.imagine(initial_state, actions)
```

`observe` thực hiện filtering bằng posterior; `predict` trả prior distribution; `imagine` rollout prior-only. Nếu cùng gọi chung là `step`, caller dễ vô tình dùng posterior trong benchmark rollout hoặc dùng prior khi đang cần correction từ sensor.

### Layer A — Introspection

Layer A có thể:

- so prior và posterior theo thời gian;
- visualize KL, entropy và active latent units;
- tìm đoạn observation correction lớn;
- kiểm tra memory bằng hidden-state ablation;
- đo posterior-to-prior switch drift;
- phát hiện categorical collapse hoặc Gaussian inactive dimensions.

### Layer B — Manipulation và imagination

Layer B dùng RSSM state để rollout action sequence, edit stochastic state, giữ hoặc reset deterministic memory và tạo counterfactual trajectory. Manipulation phải khai báo tác động lên phần nào: đổi $z_t$ nhưng giữ $h_t$ có thể tạo cặp state không từng xuất hiện trong training.

### Layer C — Runtime

Layer C cần batch recurrent state theo particle/action candidate, giữ reset mask cho episode boundary, quản lý RNG và cache decoder-independent rollout. Với Dreamer-style behavior learning, hàng nghìn imagined states được sinh mà không decode pixel; runtime không nên materialize observation prediction trừ khi diagnostic yêu cầu.

### Structured Gaussian world state

Với [Gaussian Parameters là Latent Variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md), RSSM có thể dùng:

- $h_t$ làm global scene memory;
- $z_t$ làm stochastic global code;
- một Gaussian set làm structured spatial state;
- posterior update để thêm thông tin từ camera mới.

Tuy nhiên, một global GRU không tự giải quyết correspondence giữa Gaussian primitives. Adapter cần khai báo liệu deterministic memory nằm ở scene level, object level hay per-primitive, và stochastic sample có geometry nào.

Mục tiếp theo, **Kalman filter và variants**, cung cấp trường hợp tuyến tính-Gaussian có belief update giải tích. RSSM có thể được xem như nonlinear learned filter, nhưng posterior của nó là amortized inference chứ không phải Bayes update chính xác.

---

## Liên quan

- [Markov Property và State Space](01-markov-property-state-space.md) — RSSM xấp xỉ belief state khi observation riêng lẻ không Markov.
- [Latent Transition Model](02-latent-transition-model.md) — cung cấp deterministic rollout, training horizon và diagnostics mà RSSM mở rộng bằng recurrent memory.
- [Stochastic Transition](03-stochastic-transition.md) — giải thích prior/posterior distribution, reparameterization và uncertainty semantics của $z_t$.
- [VAE](../../02-representation-learning/research/03-vae.md) — nền tảng ELBO, KL và stochastic bottleneck cho variational sequence model.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) — KL giới hạn lượng thông tin observation mới đi vào state.
- [Density Estimation trong Latent](../../04-latent-computation/research/06-density-estimation.md) — bổ sung support diagnostics khi prior imagination rời vùng posterior states.
- [Gaussian Parameters là Latent Variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) — ví dụ structured state cần RSSM schema vượt ra ngoài vector phẳng.

## Tham khảo

- J. Chung, K. Kastner, L. Dinh, K. Goel, A. Courville, Y. Bengio, *A Recurrent Latent Variable Model for Sequential Data* (NeurIPS 2015, arXiv:1506.02216).
- M. Karl, M. Soelch, J. Bayer, P. van der Smagt, *Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data* (ICLR 2017, arXiv:1605.06432).
- R. G. Krishnan, U. Shalit, D. Sontag, *Structured Inference Networks for Nonlinear State Space Models* (AAAI 2017, arXiv:1609.09869).
- L. Buesing, T. Weber, S. Racaniere, S. M. A. Eslami, D. Rezende, D. P. Reichert, F. Viola, F. Besse, K. Gregor, D. Hassabis, D. Wierstra, *Learning and Querying Fast Generative Models for Reinforcement Learning* (arXiv 2018, arXiv:1802.03006).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (ICML 2019, arXiv:1811.04551).
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- D. Hafner, T. Lillicrap, M. Norouzi, J. Ba, *Mastering Atari with Discrete World Models* (ICLR 2021, arXiv:2010.02193).
- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, *Mastering Diverse Domains through World Models* (arXiv 2023, arXiv:2301.04104).
