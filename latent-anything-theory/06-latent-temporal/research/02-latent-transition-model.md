# Latent Transition Model

> **TL;DR.** Latent transition model học quy luật tiến hóa trực tiếp trong state ẩn, thường bắt đầu bằng dạng tất định $\hat z_{t+1}=f_\theta(z_t,a_t)$ rồi lặp hàm này để rollout nhiều bước mà không cần decode ở mỗi bước. Mô hình chỉ hữu ích khi encoder tạo ra state đủ cho dynamics và objective buộc transition đúng không chỉ một bước mà cả trên rollout, reward hoặc đại lượng task-relevant. Caveat chính là một hàm tất định sẽ average các tương lai đa khả năng, còn sai số nhỏ ở one-step vẫn có thể tích lũy thành trajectory sai hoàn toàn.

Trong [Markov Property và State Space](01-markov-property-state-space.md), state được định nghĩa như bản tóm tắt đủ của history để dự báo tương lai. Latent transition model là thành phần hiện thực hóa giả định đó: nhận latent state hiện tại cùng action, rồi dự báo latent state tiếp theo.

Điểm mới so với autoencoder tĩnh là latent không còn chỉ cần tái tạo observation hiện tại. Coordinate system của nó phải thuận lợi cho **dynamics**: các state gần nhau nên có evolution tương thích, action phải tạo ra thay đổi có hệ thống, và việc lặp transition phải giữ trajectory trong vùng latent hợp lệ.

---

## **1. Từ encoder tĩnh đến world model tối giản**

Một latent dynamics system tối giản có ba thành phần:

$$
z_t = e_\phi(o_t),
\qquad
\hat z_{t+1}=f_\theta(z_t,a_t),
\qquad
\hat o_t=g_\psi(z_t).
$$

Trong đó $e_\phi$ là encoder từ observation $o_t$ sang latent state $z_t$; $f_\theta$ là transition model có tham số $\theta$; $a_t$ là action tại bước $t$; và $g_\psi$ là decoder hoặc observation head. Ba phương trình lần lượt thực hiện state inference, state prediction và observation reconstruction.

Nếu môi trường partially observed, encoder thường phải nhận history hoặc recurrent state thay vì chỉ một frame:

$$
z_t=e_\phi(h_t),
\qquad
h_t=(o_0,a_0,\ldots,o_t).
$$

Trong đó $h_t$ là observation-action history. Công thức nhấn mạnh rằng transition Markov trong latent chỉ hợp lý khi $z_t$ đã nén đủ thông tin của history liên quan tới tương lai.

Transition tất định cơ bản là:

$$
\hat z_{t+1}=f_\theta(z_t,a_t).
$$

Kết quả $\hat z_{t+1}$ là một điểm duy nhất trong latent space. Đây là baseline nên học trước vì dễ train, dễ debug, và tách rõ lỗi representation khỏi lỗi uncertainty. Khi cùng $(z_t,a_t)$ có thể dẫn tới nhiều tương lai hợp lệ, cần chuyển sang **stochastic transition** ở mục tiếp theo.

### Vì sao dự báo trong latent thay vì observation?

| Không gian dự báo | Model phải học | Lợi thế | Rủi ro |
|---|---|---|---|
| **Observation space** | cả dynamics lẫn pixel, texture, lighting, sensor noise | output kiểm tra trực tiếp được | tốn compute, dễ dành capacity cho chi tiết không liên quan |
| **Latent space** | evolution của representation nén | rollout nhanh, dễ gắn reward/value head | latent có thể bỏ mất dynamics hoặc học metric không phù hợp |
| **Structured latent** | motion và interaction của object/token/primitive | dễ can thiệp và giải thích hơn | correspondence, permutation và cardinality phức tạp |

World Models của Ha và Schmidhuber minh họa decomposition thực dụng: một vision model nén frame, một recurrent model học temporal dynamics trong latent, còn controller hoạt động trên representation đó. PlaNet đi xa hơn bằng cách planning trực tiếp trong latent state-space model thay vì sinh từng frame để đánh giá mọi action sequence.

---

## **2. Action-conditioned dynamics**

Transition dùng cho control phải phân biệt rõ dynamics tự nhiên của hệ và tác động của action:

$$
z_{t+1}=f_\theta(z_t,a_t).
$$

Nếu bỏ $a_t$, model chỉ học dynamics trung bình dưới behavior policy trong dữ liệu. Nó có thể dự báo trajectory đã quan sát nhưng không trả lời đúng câu hỏi phản thực tế: "state này sẽ thay đổi thế nào nếu chọn action khác?"

Một parameterization dễ tối ưu là residual dynamics:

$$
\hat z_{t+1}=z_t+\Delta_\theta(z_t,a_t).
$$

Trong đó $\Delta_\theta$ dự báo độ thay đổi của latent thay vì toàn bộ state kế tiếp. Dạng residual phù hợp khi timestep nhỏ và state biến đổi tương đối trơn; identity mapping trở thành baseline tự nhiên, còn network tập trung học phần motion.

Với hệ có timestep $\Delta t$, có thể đọc residual như một discretization:

$$
\hat z_{t+1}=z_t+\Delta t\,F_\theta(z_t,a_t).
$$

Trong đó $F_\theta$ xấp xỉ vector field trong latent space. Công thức này nối transition rời rạc với cách nhìn dynamics liên tục, nhưng không tự bảo đảm rằng vector field học được ổn định hay có ý nghĩa vật lý.

### Locally linear dynamics

Embed to Control (E2C) áp đặt dynamics cục bộ tuyến tính:

$$
\hat z_{t+1}
=
A_\theta(z_t)z_t+B_\theta(z_t)a_t+c_\theta(z_t).
$$

Trong đó $A_\theta(z_t)$, $B_\theta(z_t)$ và $c_\theta(z_t)$ được dự đoán quanh state hiện tại. Mô hình vẫn phi tuyến trên toàn không gian vì các ma trận thay đổi theo $z_t$, nhưng mỗi lân cận có dạng tuyến tính thuận lợi cho optimal control.

Điểm mạnh của locally linear transition là có inductive bias rõ và dễ dùng với controller cổ điển. Điểm yếu là linearization chỉ đáng tin trong vùng cục bộ; action lớn, contact dynamics hay rollout xa có thể rời vùng mà các ma trận được fit.

---

## **3. Rollout trong latent**

Từ state đầu $z_t$ và một action sequence $a_{t:t+H-1}$, rollout tất định được định nghĩa đệ quy:

$$
\hat z_{t}^{(0)}=z_t,
\qquad
\hat z_{t}^{(k+1)}
=
f_\theta\!\left(\hat z_t^{(k)},a_{t+k}\right),
\quad k=0,\ldots,H-1.
$$

Trong đó $\hat z_t^{(k)}$ là state dự báo sau $k$ bước và $H$ là planning horizon. Sau bước đầu tiên, input của transition là output do chính model sinh ra, không còn là latent encode từ observation thật.

Nếu có decoder, observation tương lai được lấy khi cần:

$$
\hat o_{t+k}=g_\psi\!\left(\hat z_t^{(k)}\right).
$$

Trong đó decoder chỉ dùng để visualize, compute reconstruction objective, hoặc cung cấp output cho task cần observation. Planning thuần latent có thể bỏ decode và dùng reward/value head:

$$
\hat r_{t+k}=r_\omega\!\left(\hat z_t^{(k)},a_{t+k}\right),
\qquad
\hat V_{t+k}=V_\eta\!\left(\hat z_t^{(k)}\right).
$$

Trong đó $r_\omega$ dự báo reward cục bộ và $V_\eta$ ước lượng return sau horizon. TD-MPC sử dụng đúng tinh thần task-oriented này: latent model phục vụ trajectory optimization ngắn hạn, còn value function ước lượng phần dài hạn thay vì buộc model reconstruct mọi chi tiết observation.

### Teacher forcing và open-loop rollout

Trong training one-step, transition thường nhận state từ encoder:

$$
\hat z_{t+1}=f_\theta(e_\phi(o_t),a_t).
$$

Trong open-loop rollout, model nhận prediction của chính nó:

$$
\hat z_{t+k+1}=f_\theta(\hat z_{t+k},a_{t+k}).
$$

Sự khác biệt này tạo distribution shift. Transition có thể rất chính xác quanh latent thật nhưng chưa từng học cách tự sửa khi prediction lệch nhẹ khỏi data manifold. Vì vậy one-step validation loss không đủ để đánh giá khả năng imagination.

---

## **4. Objective để học transition**

### 4.1 Latent one-step consistency

Objective trực tiếp nhất so prediction với latent của observation kế tiếp:

$$
\mathcal{L}_{\text{dyn}}^{(1)}
=
d\!\left(
f_\theta(e_\phi(o_t),a_t),
e_\phi(o_{t+1})
\right).
$$

Trong đó $d$ là distance trong latent, thường bắt đầu bằng squared Euclidean distance. Loss ép transition prediction khớp với state target do encoder sinh ra.

Objective này có hai vấn đề. Thứ nhất, latent coordinate có thể drift vì encoder và transition cùng thay đổi trong training. Thứ hai, nghiệm collapse $e_\phi(o)=\text{constant}$ làm loss dynamics bằng không nếu không có reconstruction, contrastive, reward, variance hoặc objective chống collapse khác.

### 4.2 Reconstruction và decoded prediction

Autoencoder reconstruction giữ cho latent không vứt bỏ toàn bộ observation:

$$
\mathcal{L}_{\text{rec}}
=
\ell_o\!\left(g_\psi(e_\phi(o_t)),o_t\right).
$$

Trong đó $\ell_o$ là observation loss như MSE, likelihood hoặc perceptual loss. Loss này bảo đảm decoder khôi phục được observation hiện tại, nhưng không bảo đảm representation giữ đúng biến cần cho tương lai.

Có thể decode state dự báo và so trực tiếp với observation kế tiếp:

$$
\mathcal{L}_{\text{pred}}
=
\ell_o\!\left(
g_\psi(f_\theta(e_\phi(o_t),a_t)),
o_{t+1}
\right).
$$

Trong đó gradient đi qua encoder, transition và decoder. Deep Variational Bayes Filters nhấn mạnh lợi ích của việc backpropagate qua transition: representation được học cùng state-space assumptions thay vì encode frame tĩnh trước rồi mới fit dynamics.

### 4.3 Multi-step latent consistency

Để transition học behavior khi tự rollout, tối ưu nhiều horizon:

$$
\mathcal{L}_{\text{dyn}}^{(H)}
=
\sum_{k=1}^{H}
\lambda_k\,
d\!\left(
\hat z_t^{(k)},
e_\phi(o_{t+k})
\right).
$$

Trong đó $\hat z_t^{(k)}$ là rollout prediction sau $k$ bước, $\lambda_k$ là trọng số theo horizon, và target $e_\phi(o_{t+k})$ đến từ observation thật. Loss này phạt trực tiếp compounding error thay vì chỉ lỗi cục bộ.

PlaNet đưa ra latent overshooting trong mô hình xác suất: prior rollout nhiều bước phải khớp với posterior state tương lai. Với baseline tất định, phương trình trên là analogue dễ hiểu nhất: unroll transition và kiểm tra consistency ở nhiều khoảng cách thời gian.

### 4.4 Task-oriented objectives

Nếu mục tiêu là control, reconstruction pixel có thể giữ quá nhiều nuisance. Có thể học latent cùng reward, value hoặc inverse-dynamics heads:

$$
\mathcal{L}
=
\alpha\mathcal{L}_{\text{dyn}}
+\beta\mathcal{L}_{\text{reward}}
+\gamma\mathcal{L}_{\text{value}}
+\delta\mathcal{L}_{\text{aux}}.
$$

Trong đó các hệ số $\alpha,\beta,\gamma,\delta$ cân bằng transition consistency, reward prediction, value learning và auxiliary objectives. Ý nghĩa là latent chỉ cần bảo toàn thông tin phục vụ task, không nhất thiết reconstruct hoàn hảo mọi pixel.

Không có một tổ hợp loss đúng cho mọi use case. Reconstruction-heavy model thuận lợi cho visualization và general-purpose simulation; task-oriented model thường compact hơn nhưng có thể mất thông tin khi đổi downstream task.

---

## **5. Các họ transition model tất định**

| Họ model | Dạng điển hình | Phù hợp khi | Giới hạn chính |
|---|---|---|---|
| **Linear** | $Az_t+Ba_t+c$ | hệ gần tuyến tính, cần interpretability | thiếu capacity cho dynamics phức tạp |
| **Locally linear** | $A(z_t)z_t+B(z_t)a_t+c(z_t)$ | local control, smooth dynamics | linearization hỏng khi rollout xa |
| **Residual MLP** | $z_t+\Delta_\theta(z_t,a_t)$ | vector latent, timestep nhỏ | không mô hình hóa interaction có cấu trúc rõ |
| **Recurrent transition** | $h_{t+1}=F(h_t,z_t,a_t)$ | partial observability, dependency dài | hidden state khó diễn giải và audit |
| **Set/graph transition** | message passing giữa latent entities | object-centric hoặc Gaussian set | correspondence và cardinality khó |

### Vector latent

Với $z_t\in\mathbb{R}^d$, MLP hoặc gated recurrent network là baseline tự nhiên. Implementation đơn giản, batching tốt, nhưng mọi interaction bị entangle trong một vector duy nhất.

### Structured latent

Với [Gaussian parameters là latent variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md), state là một tập:

$$
\mathcal{Z}_t=\{z_{t,i}\}_{i=1}^{N_t}.
$$

Trong đó mỗi $z_{t,i}$ có thể mang position, covariance, opacity, appearance và motion features. Transition cần cập nhật từng primitive dựa trên action và interaction:

$$
\hat z_{t+1,i}
=
f_\theta\!\left(
z_{t,i},
a_t,
\operatorname{Agg}\{m_\theta(z_{t,i},z_{t,j})\}_{j\ne i}
\right).
$$

Trong đó $m_\theta$ là message giữa hai phần tử và $\operatorname{Agg}$ là phép tổng hợp bất biến theo permutation. Dạng này giữ locality tốt hơn vector latent, nhưng phải giải quyết object correspondence, phần tử xuất hiện/biến mất và set size thay đổi.

---

## **6. Đánh giá một latent transition model**

### One-step error

Đo $d(\hat z_{t+1},z_{t+1})$ hoặc decoded prediction error là kiểm tra cơ bản. Nó hữu ích để debug lỗi implementation nhưng không phản ánh đầy đủ rollout.

### Multi-step rollout error

Đo error theo horizon:

$$
E(k)
=
\mathbb{E}\left[
d\!\left(\hat z_t^{(k)},e_\phi(o_{t+k})\right)
\right].
$$

Trong đó $E(k)$ cho thấy tốc độ lỗi tích lũy. Một model có $E(1)$ thấp nhưng $E(k)$ tăng rất nhanh không phù hợp cho planning horizon dài.

### Decoded trajectory quality

Decode rollout để quan sát geometry, identity, contact và object persistence. Pixel metric đơn thuần có thể phạt các tương lai hợp lệ nhưng khác sample thật, nên cần đọc cùng latent/task metrics.

### Reward và planning consistency

Nếu model phục vụ control, kiểm tra reward ranking giữa candidate action sequences và hiệu quả khi dùng model để chọn action. Transition có prediction đẹp nhưng xếp sai action vẫn là model kém cho planning.

### Action coverage

Đánh giá theo vùng $(z,a)$, không chỉ trung bình toàn dataset. Sai số thấp dưới behavior policy có thể che failure nghiêm trọng ở action hiếm mà planner sẽ khai thác.

### Stability và manifold drift

Theo dõi norm, density hoặc distance tới encoded data manifold trong rollout. Nếu state dần rời khỏi vùng encoder từng tạo ra, decoder và heads đang extrapolate ngoài phân phối.

---

## **7. Giới hạn / Khi nào thất bại**

**Tương lai đa khả năng bị average.** Nếu cùng state và action có thể dẫn tới nhiều outcome, deterministic MSE predictor học conditional mean. Trong observation space điều này tạo frame mờ; trong latent nó có thể tạo state nằm giữa các mode và không tương ứng với tương lai hợp lệ nào.

**Compounding error.** Sai số nhỏ ở mỗi bước trở thành input cho bước sau. Transition được train quanh latent thật nhưng rollout dần đi vào vùng chưa thấy, khiến error tăng phi tuyến theo horizon.

**Latent target không cố định.** Khi encoder và transition train đồng thời, coordinate system thay đổi liên tục. Transition có thể đuổi theo moving target hoặc hai module phối hợp tạo representation dễ fit nhưng nghèo thông tin.

**Metric latent tùy ý.** Euclidean distance chỉ có nghĩa nếu geometry của representation hỗ trợ nó. Hai latent xa nhau theo norm vẫn có thể decode gần giống, hoặc ngược lại. Loss dynamics cần được kiểm chứng bằng decoded và task-level behavior.

**Partial observability bị hiểu nhầm thành stochasticity.** Một frame thiếu velocity hoặc occluded object làm transition tất định thất bại, nhưng nguyên nhân có thể là state chưa đủ chứ không phải môi trường ngẫu nhiên. Thêm memory đúng đôi khi quan trọng hơn thêm noise.

**Action distribution shift.** Model học từ behavior policy nhưng planner tối ưu action dựa trên model. Planner có xu hướng tìm vùng action mà model quá lạc quan hoặc chưa có dữ liệu, một dạng model exploitation.

**Shortcut prediction.** Dataset có background tĩnh hoặc action autocorrelation mạnh có thể khiến model đạt loss thấp bằng cách copy state hay đoán action phổ biến, không học interaction thật.

**Task-oriented latent không tổng quát.** Latent tối ưu cho reward hiện tại có thể bỏ thông tin cần cho task mới, introspection hoặc reconstruction. Khả năng swap method của framework vì thế cần đi kèm declaration về information contract.

**Structured state làm transition khó hơn.** Set latent có permutation symmetry, cardinality thay đổi và data association. Sai correspondence có thể tạo error lớn dù scene tổng thể vẫn hợp lý.

---

## **8. Liên hệ với Latent-Anything**

Latent transition model là operation đầu tiên biến `Trajectory` từ container thụ động thành object có thể rollout:

```python
next_state = transition.step(state, action)
future = transition.rollout(state, actions)
```

Một transition plugin cần khai báo ít nhất:

- input state schema và latent geometry;
- action schema, bounds và timestep;
- deterministic hay stochastic output;
- Markov order hoặc recurrent context cần giữ;
- training horizon và rollout horizon đã được validate;
- heads đi kèm: decoder, reward, termination, value;
- supported batch, device và serialization behavior.

`Trajectory` nên lưu phân biệt hai nguồn state:

- **posterior/encoded state** được suy ra từ observation thật;
- **prior/predicted state** được transition rollout.

Phân biệt này quan trọng cho Layer A: có thể vẽ rollout drift, so distribution encoded/predicted, tìm state aliasing và đo error theo horizon. Layer B dùng transition để imagination, trajectory editing và planning; Layer C chịu trách nhiệm batching nhiều action sequence, cache state dùng chung và profile cost theo horizon.

Với structured Gaussian state, transition interface không nên giả định shape cố định `[batch, dim]`. Nó cần support set state, masks, correspondence metadata và operation add/remove/split/merge. Đây là stress test trực tiếp cho lựa chọn `LatentSpace` flexible trong kiến trúc dự án.

Mục tiếp theo, **Stochastic transition**, mở rộng output từ một điểm $\hat z_{t+1}$ thành phân phối $p(z_{t+1}\mid z_t,a_t)$. Sau đó **RSSM** sẽ kết hợp deterministic recurrent memory và stochastic state trong cùng một model.

---

## Liên quan

- [Markov Property và State Space](01-markov-property-state-space.md) — định nghĩa điều kiện để $z_t$ đủ làm input cho transition mà không cần toàn bộ history.
- [Autoencoder](../../02-representation-learning/research/02-autoencoder.md) — cung cấp encoder/decoder cơ bản, nhưng reconstruction tĩnh chưa bảo đảm latent thuận lợi cho dynamics.
- [Gaussian Parameters là Latent Variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) — ví dụ structured state đòi hỏi set/graph transition thay vì MLP trên vector phẳng.
- [Dynamic 3DGS](../../03b-3d-representation/research/11-dynamic-3dgs.md) — trường hợp cụ thể của dynamics trên position, deformation và appearance của Gaussian theo thời gian.
- [Causal Intervention vs Observational Study](../../05-probing-intervention/research/04-causal-intervention-vs-observational.md) — action-conditioned transition cần đúng dưới intervention, không chỉ fit correlation của behavior policy.
- [Density Estimation trong Latent](../../04-latent-computation/research/06-density-estimation.md) — density hoặc support score giúp phát hiện rollout đã drift khỏi vùng latent in-distribution.

## Tham khảo

- M. Watter, J. T. Springenberg, J. Boedecker, M. Riedmiller, *Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images* (NeurIPS 2015, arXiv:1506.07365).
- M. Karl, M. Soelch, J. Bayer, P. van der Smagt, *Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data* (ICLR 2017, arXiv:1605.06432).
- D. Ha, J. Schmidhuber, *World Models* (arXiv 2018, arXiv:1803.10122).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (ICML 2019, arXiv:1811.04551).
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- N. Hansen, X. Wang, H. Su, *Temporal Difference Learning for Model Predictive Control* (ICML 2022, arXiv:2203.04955).
