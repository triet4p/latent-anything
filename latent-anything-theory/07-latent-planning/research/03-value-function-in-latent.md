# Value Function trong Latent

> **TL;DR.** Value function $\hat v_\xi(z_t)\to\mathbb{R}$ ước lượng *expected return* từ một latent state, thỏa Bellman consistency $\hat v(z_t)=\mathbb{E}[\hat r_t+\gamma\hat v(z_{t+1})]$ trong latent space. Vai trò then chốt trong planning: nó **bootstrap phần return sau horizon**, cho phép rollout ngắn (ít compounding error) mà vẫn "nhìn" được reward xa. Caveat: value học bằng bootstrap + function approximation + off-policy data là bộ ba "deadly triad" dễ phân kỳ và overestimate, và giống reward head, planner sẽ khai thác value sai-lạc-quan.

[Mục 2](02-reward-model-in-latent.md) cho ta số hạng reward $\hat r_\theta$ trong imagined return. Nhưng tổng reward chỉ chạy đến horizon $H$ — phần sau thì sao? Đó là việc của value function: thay vì rollout vô hạn (đắt và đầy compounding error), ta cắt rollout tại $H$ rồi *bootstrap* phần đuôi bằng $\hat v_\xi(\hat z_H)$. Mục này là thành phần thứ hai và là thứ làm planning trong latent vừa rẻ vừa không thiển cận.

---

## **1. Trực giác / Định nghĩa**

Value của một state là tổng reward chiết khấu kỳ vọng nếu xuất phát từ state đó và đi theo policy $\pi$:

$$
v^\pi(z_t)=\mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k\,r_{t+k}\;\middle|\;z_t\right].
$$

Trong đó $\gamma\in[0,1)$ là discount và kỳ vọng lấy trên trajectory sinh bởi $\pi$ và dynamics. Trực giác: reward đo "ngay bây giờ tốt bao nhiêu", còn value đo "đứng ở đây thì *tương lai* tốt bao nhiêu". Cũng có dạng action-value $q^\pi(z_t,a_t)$ — value khi thực hiện $a_t$ rồi mới theo $\pi$.

Trong latent world model, value là một head $\hat v_\xi(z)$ (MLP) đọc thẳng từ latent state, song song reward/continuation head. Nó cho phép planner biết một imagined state "có triển vọng" hay không mà không phải rollout đến tận cùng.

---

## **2. Bellman consistency trong latent space**

Định nghĩa tổng vô hạn ở trên không tính được trực tiếp; value được đặc trưng đệ quy bằng **Bellman equation**:

$$
v^\pi(z_t)=\mathbb{E}_\pi\!\big[\,r_t+\gamma\,v^\pi(z_{t+1})\,\big].
$$

Trong đó value của state hiện tại bằng reward tức thời cộng value chiết khấu của state kế. Đây là điều kiện *self-consistency*: value đúng phải khớp với chính nó qua một bước transition. Critic được train để ép khớp đẳng thức này trên imagined latent trajectory — Dreamer gọi đây là tối ưu Bellman consistency của imagined trajectories.

Học bằng **TD(0)**: tối thiểu bình phương TD error với một target tách rời:

$$
\mathcal{L}(\xi)=\mathbb{E}\Big[\big(\hat v_\xi(z_t)-y_t\big)^2\Big],\qquad y_t=\hat r_t+\gamma\,\hat v_{\bar\xi}(z_{t+1}).
$$

Trong đó $y_t$ là TD target và $\bar\xi$ là tham số của **target network** (bản sao critic cập nhật chậm). Tách $\bar\xi$ khỏi $\xi$ là cần thiết: nếu target di chuyển cùng lúc với prediction, bootstrap dễ tự khuếch đại lỗi (xem §5).

---

## **3. $\lambda$-return: nối rollout ngắn với bootstrap**

Bootstrap một bước (TD(0)) ít variance nhưng dính bias của critic; rollout dài (Monte Carlo) ít bias nhưng nhiều variance và compounding error. **$\lambda$-return** trộn mọi $n$-step return để cân bằng — đây là return Dreamer dùng trên imagined trajectory:

$$
V^\lambda_t=\hat r_t+\gamma\hat c_t\Big[(1-\lambda)\,\hat v_\xi(\hat z_{t+1})+\lambda\,V^\lambda_{t+1}\Big],\qquad V^\lambda_H=\hat v_\xi(\hat z_H).
$$

Trong đó $\lambda\in[0,1]$ nội suy giữa TD(0) ($\lambda=0$, dựa value nhiều) và Monte Carlo ($\lambda=1$, dựa rollout nhiều), $\hat c_t$ là continuation. Đệ quy này chạy ngược từ cuối horizon: $V^\lambda_H$ khởi tạo bằng bootstrap value, rồi cuộn về đầu. Critic sau đó hồi quy về $V^\lambda_t$ (thường với target network và stop-gradient ở target).

Điểm cốt lõi cho planning: $\lambda$-return cho phép horizon ngắn (rẻ, ít drift) mà **không thiển cận**, vì $\hat v_\xi(\hat z_H)$ thay mặt cho toàn bộ reward sau $H$. Đây chính là cách value "mua lại" tầm nhìn xa mà rollout ngắn đánh mất.

### Mã hóa value: symlog two-hot

Giống reward head ([mục 2 §3](02-reward-model-in-latent.md)), DreamerV3 dùng critic distributional với symlog two-hot thay vì MSE, để value ổn định qua các dải return magnitude rất khác nhau mà không cần tuning theo task.

---

## **4. Biến thể**

| | $v(z)$ — state value | $q(z,a)$ — action value |
|---|---|---|
| Cần model để chọn action? | có (phải biết $z'$ của mỗi $a$) | không (so $q$ trực tiếp) |
| Dùng trong | Dreamer critic, MPC bootstrap | DQN-style, TD-MPC |
| Action space lớn/liên tục | thuận lợi (không enumerate $a$) | cần actor hoặc sampling |

| Cách học target | Bias | Variance | Ghi chú |
|---|---|---|---|
| Monte Carlo ($\lambda=1$) | thấp | cao | cần trajectory đầy đủ, dính compounding error |
| TD(0) ($\lambda=0$) | cao (critic bias) | thấp | bootstrap một bước |
| $\lambda$-return | điều chỉnh được | điều chỉnh được | mặc định của Dreamer |

Ngoài ra: **target network / EMA critic** (cập nhật chậm để ổn định), **double critic** (hai critic, lấy min để chống overestimation — kiểu TD3/TD-MPC2), và **value-equivalent value** (MuZero: value học mà không cần reconstruct observation, sẽ gặp ở mục riêng).

---

## **5. Giới hạn / Khi nào thất bại**

**Deadly triad.** Kết hợp đồng thời ba thứ — *bootstrapping*, *function approximation*, và *off-policy data* — có thể làm value **phân kỳ**. Lỗi xấp xỉ bị nhét vào target bootstrap, tạo vòng lặp khuếch đại đẩy value estimate ra vô cực. Latent world model dính đủ ba: critic là neural net (approximation), học bằng TD (bootstrap), trên replay/imagined data lệch policy hiện tại (off-policy). Mitigation chuẩn: target network, EMA, two-hot distributional critic, regularization.

**Overestimation bias.** Target dạng $\max$ (hoặc planner chọn action max value) làm value bị thiên *lên* do Jensen trên nhiễu ước lượng: $\mathbb{E}[\max_a \hat q]\ge\max_a\mathbb{E}[\hat q]$. Double/clipped-double critic giảm bias này.

**Value exploitation.** Planner tối ưu $\hat J$ có chứa $\hat v_\xi(\hat z_H)$; nếu critic over-confident ở vùng OOD, planner bị kéo tới đúng đó — y hệt reward hacking nhưng qua value. Horizon ngắn *chuyển* rủi ro từ dynamics sang critic chứ không xóa nó.

**Off-support extrapolation.** Critic chỉ tin cậy gần support của data; imagined state cuối horizon có thể đã rời support, làm bootstrap vô nghĩa.

**Posterior–prior & policy shift.** Value học dưới một policy/state distribution, dùng dưới policy khác khi planner cải thiện — value cũ có thể lệch.

---

## **6. Liên hệ với Latent-Anything**

Value head là thành phần thứ ba (cùng reward, continuation) của `ModelAdapter`, và là thứ cho phép planning *ngắn mà sâu*:

```python
class ModelAdapter(Protocol):
    def value(self, z: np.ndarray) -> np.ndarray: ...   # v̂(z): bootstrap return sau horizon
```

- **Layer A — Introspection**: đo value calibration (predicted vs realized return trên holdout), phát hiện divergence/overestimation, vẽ value landscape để thấy planner bị bootstrap kéo về đâu. Theo dõi TD error theo thời gian là tín hiệu sớm của deadly triad.
- **Layer B — Manipulation**: với value head, một candidate trajectory bị cắt ngắn vẫn chấm điểm được qua $\lambda$-return — nền cho MPC/CEM dùng terminal value bootstrap thay vì rollout dài.
- **Layer C — Runtime**: chọn horizon $H$ và $\lambda$ là trade-off compute–bias mà runtime phải lộ ra; value head cho phép giảm $H$ (giảm compounding error, mục [Rollout](../../06-latent-temporal/research/07-rollout-latent-imagination.md)) đổi lấy phụ thuộc critic.

Hai mục đầu của tầng 7 (reward, value) hợp thành toàn bộ objective $\hat J$. Mục tiếp theo, **Model Predictive Control (MPC)**, là planner cụ thể đầu tiên tối ưu $\hat J$ đó: rollout nhiều action sequence, chọn tốt nhất, thực thi bước đầu, rồi lặp.

---

## Liên quan

- [Reward Model trong Latent](02-reward-model-in-latent.md) — số hạng reward; value bootstrap phần đuôi mà reward không vươn tới.
- [Model-based vs Model-free RL](01-model-based-vs-model-free-rl.md) — value là cách model-based và model-free gặp nhau (actor-critic trên imagined data).
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — $\lambda$-return và return trên imagined trajectory.
- [Markov Property và State Space](../../06-latent-temporal/research/01-markov-property-state-space.md) — điều kiện để Bellman recursion hợp lệ.
- [Density Estimation](../../04-latent-computation/research/06-density-estimation.md) — phát hiện state cuối horizon rời support, chống value exploitation.

## Tham khảo

- R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press 2018).
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, *Mastering Diverse Domains through World Models* (DreamerV3, 2023, arXiv:2301.04104).
- H. van Hasselt, Y. Doron, F. Strub, M. Hessel, N. Sonnerat, J. Modayil, *Deep Reinforcement Learning and the Deadly Triad* (2018, arXiv:1812.02648).
- H. van Hasselt, A. Guez, D. Silver, *Deep Reinforcement Learning with Double Q-learning* (AAAI 2016, arXiv:1509.06461).
- N. Hansen, H. Su, X. Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control* (ICLR 2024, arXiv:2310.16828).
- J. Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero, Nature 2020, arXiv:1911.08265).
