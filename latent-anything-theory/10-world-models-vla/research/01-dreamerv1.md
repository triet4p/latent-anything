# DreamerV1 (Hafner et al., 2019)

> **TL;DR.** DreamerV1 ("Dream to Control") là world model latent *hoàn chỉnh* đầu tiên cho RL: học một [RSSM](../../06-latent-temporal/research/04-rssm-recurrent-state-space-model.md) từ replay buffer, rồi học hành vi **thuần trong tưởng tượng** bằng actor-critic chạy trên các rollout latent. Điểm cốt lõi: actor được tối ưu bằng **analytic value gradient** — lan truyền gradient của λ-return *xuyên qua* mô hình động lực khả vi, thay vì policy gradient nhiễu. Caveat: phụ thuộc reconstruction pixel để học latent (sẽ là điểm V2/V3 và JEPA tấn công), và dynamics sai thì hành vi tưởng tượng lệch.

Tầng 6 và 7 đã dựng từng mảnh: [RSSM](../../06-latent-temporal/research/04-rssm-recurrent-state-space-model.md) cho transition, [value function trong latent](../../07-latent-planning/research/03-value-function-in-latent.md), [policy gradient trên imagined trajectory](../../07-latent-planning/research/07-policy-gradient-imagined-dreamer.md). DreamerV1 (Hafner et al., 2019, ICLR 2020) là nơi các mảnh đó ráp thành một agent chạy được — và là khuôn mẫu mà cả dòng Dreamer (V2, V3) lẫn nhiều world model sau kế thừa.

---

## 1. Trực giác: học trong giấc mơ

Model-free RL học từ tương tác thật, tốn sample. PlaNet (tiền thân của Dreamer) học một world model rồi *plan* bằng [CEM](../../07-latent-planning/research/05-cross-entropy-method.md) — nhưng planning mỗi bước rất đắt. DreamerV1 thay planning bằng một **policy học-được**: thay vì tìm action sequence lúc chạy, nó huấn luyện một actor network bằng cách cho nó "mơ" — rollout trong latent của world model, không đụng môi trường thật — rồi cập nhật actor để tối đa hóa return tưởng tượng.

Vì rollout xảy ra trong [latent compact](../../06-latent-temporal/research/07-rollout-latent-imagination.md) (vector chiều thấp, không phải pixel), chi phí mỗi bước tưởng tượng là $O(d)$ thay vì $O(H\cdot W\cdot C)$ — cho phép mơ hàng nghìn quỹ đạo rẻ để học hành vi.

---

## 2. Cơ chế: ba thành phần lồng nhau

### (a) Học world model (RSSM)

Từ replay buffer các chuỗi thật, học một [RSSM](../../06-latent-temporal/research/04-rssm-recurrent-state-space-model.md) gồm deterministic state $h_t$ (GRU) và stochastic state $s_t$ (Gaussian). Ba head dự đoán: transition prior $p(s_t\mid h_t)$, observation $p(x_t\mid h_t,s_t)$ (reconstruction), và reward $p(r_t\mid h_t,s_t)$. Loss là ELBO: reconstruction + reward + KL giữa posterior $q(s_t\mid h_t,x_t)$ và prior.

### (b) Tưởng tượng quỹ đạo

Từ mỗi state trong buffer làm điểm xuất phát, rollout $H$ bước *thuần trong latent*: $s_{t+1}\sim p(s_{t+1}\mid h_{t+1})$ với action lấy từ actor hiện tại. Không decode pixel — chỉ cần latent state và reward dự đoán.

### (c) Actor-critic với analytic value gradient

Trên quỹ đạo tưởng tượng, ước lượng value bằng **λ-return** (trung bình có trọng số của các n-step return, cân bằng bias–variance):

$$
V^\lambda_t = (1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1} V^n_t + \lambda^{H-1} V^H_t,
$$

trong đó $V^n_t$ là n-step return bootstrap bằng critic $v_\psi$, và $\lambda\in[0,1]$ điều khiển độ dài hiệu dụng của bootstrap. Critic $v_\psi$ hồi quy về $V^\lambda_t$; actor $\pi_\theta$ tối đa hóa $V^\lambda_t$.

Điểm then chốt — và là đột phá so với policy gradient thường: vì transition Gaussian dùng [reparameterization](../../02-representation-learning/research/03-vae.md) và reward/value đều khả vi, ta tính được **gradient giải tích** $\nabla_\theta V^\lambda_t$ bằng cách backprop *xuyên qua* chuỗi động lực tưởng tượng. Đây là một dạng [policy gradient trên imagined trajectory](../../07-latent-planning/research/07-policy-gradient-imagined-dreamer.md) với phương sai thấp hơn nhiều REINFORCE, vì tận dụng được độ khả vi của world model.

---

## 3. Vì sao DreamerV1 quan trọng

| | PlaNet (planning) | DreamerV1 (learned policy) |
|---|---|---|
| Chọn action | CEM/MPC mỗi bước (đắt) | một forward của actor (rẻ) |
| Tín hiệu học hành vi | không có policy | analytic value gradient qua dynamics |
| Tận dụng độ khả vi | không | **có** — backprop qua imagination |
| Horizon | giới hạn bởi planning budget | λ-return + critic vượt horizon |

DreamerV1 chứng minh rằng *học hành vi hoàn toàn trong latent imagination* khả thi và data-efficient trên continuous control từ pixel (DeepMind Control Suite), vượt cả model-free (D4PG) lẫn PlaNet. Nó định nghĩa template "world model + imagination + actor-critic" cho mọi phiên bản sau.

---

## 4. Giới hạn / Khi nào thất bại

**Phụ thuộc reconstruction pixel.** RSSM học latent qua dựng lại observation — tốn capacity cho chi tiết vô nghĩa, đúng phê phán của [latent prediction](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md) và [value equivalence](../../07-latent-planning/research/08-value-equivalence-muzero.md). Đây là điều **DreamerV2** (discrete latent) và các JEPA-style nhắm tới.

**Model bias lan vào policy.** Actor tối ưu theo *world model*, không phải thế giới thật; nếu dynamics sai ở vùng nào, actor học khai thác lỗi đó (model exploitation) — [compounding error](../../07-latent-planning/research/10-latent-imagination-horizon.md) giới hạn horizon.

**Continuous action giả định reparameterizable.** Analytic gradient cần action khả vi (Gaussian policy); với action rời rạc phải dùng straight-through hoặc REINFORCE, kém mượt — một lý do V2 cần xử lý riêng cho Atari.

**Nhạy KL và hyperparameter.** Cân bằng reconstruction vs KL, chọn $\lambda$, horizon $H$ đều nhạy; chưa có cơ chế tự cân bằng (sẽ là KL balancing của V2/V3).

---

## 5. Liên hệ với Latent-Anything

DreamerV1 là *bản thiết kế tham chiếu* cho một world model adapter đầy đủ: nó dùng đúng các primitive mà framework phơi bày — transition, reward, value, rollout, trên một latent state.

```python
class DreamerAdapter(Protocol):
    def observe(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray: ...   # -> latent state (h,s)
    def imagine(self, state: np.ndarray, policy) -> np.ndarray: ...             # rollout latent
    def reward(self, state: np.ndarray) -> float: ...
    def value(self, state: np.ndarray) -> float: ...
```

- **Layer A — Introspection**: latent state $(h,s)$ là đối tượng để soi — vẽ trajectory tưởng tượng, so reconstruction với thật để audit world model, đo KL prior/posterior như tín hiệu "model có ngạc nhiên không".
- **Layer B — Manipulation**: imagination là một phép rollout có điều khiển; can thiệp vào latent state rồi mơ tiếp là một thí nghiệm phản thực trực tiếp.
- **Layer C — Runtime**: rollout latent rẻ ($O(d)$) là đúng loại workload Layer C tối ưu — batch hàng nghìn giấc mơ song song.

DreamerV1 mở nhóm world model quy mô lớn. Hai mục kế tiếp là tiến hóa trực tiếp của nó: **DreamerV2** (discrete latent, ổn định hơn) và **DreamerV3** (scale đa domain, KL balancing, symlog).

---

## Liên quan

- [RSSM — Dreamer](../../06-latent-temporal/research/04-rssm-recurrent-state-space-model.md) — world model cốt lõi của DreamerV1.
- [Policy Gradient Imagined (Dreamer)](../../07-latent-planning/research/07-policy-gradient-imagined-dreamer.md) — chi tiết analytic value gradient.
- [Value Function trong Latent](../../07-latent-planning/research/03-value-function-in-latent.md) — critic regress λ-return.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — cơ chế "mơ" trong latent.
- [Latent Imagination Horizon](../../07-latent-planning/research/10-latent-imagination-horizon.md) — trade-off horizon vs compounding error.

## Tham khảo

- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- D. Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet, ICML 2019, arXiv:1811.04551) — tiền thân dùng planning.
- R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction* (2018) — λ-return, actor-critic.
