# Policy Gradient trên Imagined Trajectory (Dreamer)

> **TL;DR.** Thay vì tối ưu lại action mỗi bước như MPC/CEM/MPPI, Dreamer **amortize** planning thành một actor $\pi_\psi(a\mid z)$ và critic $\hat v_\xi(z)$, huấn luyện hoàn toàn trên *imagined latent trajectory*. Vì transition/reward/value/action đều khả vi, ta backprop **analytic gradient** của $\lambda$-return qua dynamics (pathwise) để cập nhật actor; với action rời rạc thì dùng REINFORCE. Caveat: pathwise cần model khả vi và dễ exploding/vanishing gradient qua horizon dài, lại thừa hưởng bias từ model — nên DreamerV3 trộn pathwise (variance thấp, biased) với REINFORCE (unbiased, variance cao) cộng entropy.

Sáu mục trước cho objective $\hat J$ và các planner *decision-time* (shooting → CEM → MPPI) giải nó *online*, mỗi bước phải search lại — đắt ở runtime. Dreamer đặt câu hỏi khác: nếu model khả vi, sao không *học một policy* bằng cách backprop gradient của return qua chính dynamics, để runtime chỉ cần một forward pass? Đây là bước chuyển từ planning-as-search sang planning-as-learning, và là kiến trúc của dòng Dreamer.

---

## **1. Trực giác / Định nghĩa**

MPC trả về một action bằng cách tối ưu *tại chỗ*; mỗi state mới lại làm lại từ đầu. Tốn kém và không tích lũy kinh nghiệm giữa các state. **Amortized planning** đảo ngược: bỏ công *một lần* để học một hàm $\pi_\psi$ ánh xạ state → action tốt, rồi tái dùng nó miễn phí ở mọi state.

Dreamer học $\pi_\psi$ (actor) và $\hat v_\xi$ (critic) **trong latent space**, trên imagined trajectory do world model sinh ra:

- Lấy nhiều *posterior* state từ replay làm điểm xuất phát.
- Rollout $\pi_\psi$ trong latent ([imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md)), thu $\hat r,\hat v,\hat c$ ở mỗi bước.
- Cập nhật critic về [$\lambda$-return](03-value-function-in-latent.md), và cập nhật actor để *tối đa* return đó.

Điểm đặc biệt: vì mọi thành phần khả vi, actor có thể học bằng cách **đẩy gradient của return ngược qua dynamics** — không cần optimizer ngoài như CEM.

---

## **2. Mục tiêu của actor**

Actor tối đa imagined $\lambda$-return dọc horizon $H$:

$$
\max_\psi\;\mathbb{E}_{\pi_\psi}\!\left[\sum_{t=0}^{H-1} V^\lambda_t\right],\qquad a_t\sim\pi_\psi(\cdot\mid \hat z_t),\quad \hat z_{t+1}=\hat f_\theta(\hat z_t,a_t).
$$

Trong đó $V^\lambda_t$ là $\lambda$-return (mục 3), trộn reward tưởng tượng với value bootstrap. Câu hỏi cốt lõi: tính $\nabla_\psi$ của kỳ vọng này thế nào khi $a_t$ là biến ngẫu nhiên phụ thuộc $\psi$? Có hai estimator, và khác biệt giữa chúng định hình toàn bộ thiết kế.

---

## **3. Hai estimator gradient**

### (a) Dynamics backprop — pathwise / reparameterized

Reparameterize action stochastic thành hàm tất định của noise (giống [VAE](../../02-representation-learning/research/03-vae.md)):

$$
a_t=\tanh\!\big(\mu_\psi(\hat z_t)+\sigma_\psi(\hat z_t)\odot\varepsilon_t\big),\qquad \varepsilon_t\sim\mathcal{N}(0,I).
$$

Trong đó randomness nằm ở $\varepsilon_t$ độc lập $\psi$, nên $a_t$ khả vi theo $\psi$. Khi đó gradient chảy *qua* chuỗi dynamics:

$$
\nabla_\psi V^\lambda_t=\sum_{k\ge t}\frac{\partial V^\lambda_t}{\partial \hat z_k}\frac{\partial \hat z_k}{\partial a_{k-1}}\frac{\partial a_{k-1}}{\partial\psi}+\dots
$$

Trong đó mỗi bước nhân thêm Jacobian của transition $\partial\hat z_{k}/\partial\hat z_{k-1}$ — đây là backpropagation-through-time qua world model. Dreamer "propagate analytic value gradients back through the latent dynamics". Estimator này **variance thấp** nhưng **biased** (phụ thuộc độ chính xác gradient của model), và **bắt buộc model khả vi**. Tốt cho continuous control.

### (b) Score function — REINFORCE

Không cần gradient của dynamics, chỉ cần log-prob của action:

$$
\nabla_\psi\,\mathbb{E}[R]=\mathbb{E}\big[\nabla_\psi\log\pi_\psi(a_t\mid \hat z_t)\,(R_t-b(\hat z_t))\big].
$$

Trong đó $R_t$ là return (ví dụ $V^\lambda_t$) và $b$ là baseline (thường $\hat v_\xi$) để giảm variance. Estimator này **unbiased** nhưng **variance cao**, không cần model khả vi — dùng được cho action **rời rạc** (qua straight-through hoặc thuần score function). Tốt cho Atari.

### Hybrid của DreamerV3

DreamerV3 cộng cả hai cộng entropy:

$$
\mathcal{L}(\psi)=-\,\mathbb{E}\Big[\underbrace{\rho\,\log\pi_\psi(a_t\mid\hat z_t)\operatorname{sg}(A_t)}_{\text{REINFORCE}}+\underbrace{(1-\rho)\,V^\lambda_t}_{\text{dynamics backprop}}+\underbrace{\eta\,\mathrm{H}[\pi_\psi(\cdot\mid\hat z_t)]}_{\text{entropy}}\Big].
$$

Trong đó $A_t$ là advantage, $\operatorname{sg}$ là stop-gradient, $\rho$ trộn hai estimator ($\rho=1$ cho Atari rời rạc, $\rho=0$ cho continuous control), và $\eta$ là hệ số entropy khuyến khích exploration. Trực giác: dynamics backprop (biased, low-variance) học nhanh lúc đầu, REINFORCE (unbiased, high-variance) hội tụ tới nghiệm tốt hơn.

| | Pathwise (reparam) | Score function (REINFORCE) |
|---|---|---|
| Bias | biased (qua model) | unbiased |
| Variance | thấp | cao (cần baseline) |
| Cần model khả vi? | có | không |
| Action | liên tục | rời rạc/liên tục |
| Dreamer dùng cho | continuous ($\rho=0$) | Atari ($\rho=1$) |

---

## **4. Amortized vs decision-time planning**

| | Decision-time (MPC/CEM/MPPI) | Amortized (Dreamer actor) |
|---|---|---|
| Khi nào tính | tối ưu lại mỗi bước, online | học một lần, dùng lại |
| Runtime cost | $O(N\cdot I\cdot H)$ model calls | $O(1)$ forward pass |
| Training cost | thấp | cao (học actor/critic) |
| Thích nghi state mới | tự nhiên (replan) | cần generalize |
| Tận dụng kinh nghiệm chung | không | có (policy chia sẻ) |

Hai hướng không loại trừ nhau: TD-MPC dùng learned policy để *guide* MPPI (warm-start/proposal), kết hợp tốc độ runtime của actor với khả năng tinh chỉnh online của search. Đây là phổ amortization, không phải nhị phân.

---

## **5. Giới hạn / Khi nào thất bại**

**Cần model khả vi (pathwise).** Nếu dynamics không khả vi (discrete, contact-rich, hoặc adapter blackbox), pathwise không dùng được — phải REINFORCE (variance cao) hoặc search.

**Exploding/vanishing gradient.** Backprop qua $H$ bước nhân chồng Jacobian transition; spectral radius $>1$ → gradient nổ, $<1$ → tiêu biến ([recursion $\|e_{t+1}\|\le L\|e_t\|$](../../06-latent-temporal/research/07-rollout-latent-imagination.md) áp cho gradient). Đây là lý do dùng horizon vừa phải + value bootstrap thay vì backprop tới tận cùng.

**Model-gradient bias.** Pathwise tin vào *gradient* của model, không chỉ giá trị. Model có thể dự đoán đúng value nhưng gradient sai hướng, dẫn actor đi sai — một dạng [model exploitation](02-reward-model-in-latent.md) tinh vi hơn.

**REINFORCE variance.** Không baseline tốt thì gradient quá nhiễu để học; baseline/critic lại kế thừa [deadly triad](03-value-function-in-latent.md).

**Non-stationary imagined distribution.** Khi actor cải thiện, phân phối imagined start/visited state đổi; critic và model phải theo kịp, dễ dao động.

**Entropy tuning.** $\eta$ quá nhỏ → collapse sớm, mất exploration; quá lớn → policy không quyết đoán. Phụ thuộc domain.

**Vẫn là imagination.** Mọi gradient đến từ model; nếu model sai một cách hệ thống, actor học một behavior tối ưu cho *giấc mơ* chứ không cho thế giới thật.

---

## **6. Liên hệ với Latent-Anything**

Actor amortize chính là Layer B planning "đóng gói" thành một policy, và tận dụng cờ `differentiable` của adapter:

```python
class ModelAdapter(Protocol):
    differentiable: bool          # pathwise actor gradient khả thi?

def actor_loss(z0, model, actor, critic, H, rho, eta):
    z, logps, vals, rews = z0, [], [], []
    for _ in range(H):                                  # imagined rollout
        a = actor.rsample(z)                            # reparameterized action
        logps.append(actor.log_prob(a)); rews.append(model.reward(z))
        z = model.predict(z, a); vals.append(critic(z))  # differentiable transition
    lam_ret = lambda_return(rews, vals)                 # mục 3
    adv = lam_ret - stop_grad(vals)
    return -(rho * sum(logps) * stop_grad(adv)          # REINFORCE
             + (1 - rho) * lam_ret.sum()                # dynamics backprop
             + eta * actor.entropy())                   # exploration
```

- **Layer A — Introspection**: theo dõi gradient norm qua horizon (phát hiện exploding/vanishing), so imagined return của actor với realized return, đo entropy của policy.
- **Layer B — Manipulation**: actor là "planner đã biên dịch"; nó có thể warm-start search (TD-MPC) hoặc đứng độc lập. Cùng interface với MPC optimizer.
- **Layer C — Runtime**: dựa `differentiable` mà chọn pathwise/REINFORCE/hybrid; quản training-time vs runtime budget; chunk/checkpoint autodiff graph cho horizon dài.

Hết Dreamer, ta đã có cả planning-as-search (MPC family) lẫn planning-as-learning (actor-critic). Cả hai vẫn giả định model dự đoán *được* observation. Mục tiếp theo, **Value equivalence (MuZero)**, phá giả định đó: latent không cần reconstruct observation, chỉ cần dự đoán đúng value/policy/reward để phục vụ planning.

---

## Liên quan

- [Value Function trong Latent](03-value-function-in-latent.md) — critic và $\lambda$-return mà actor tối đa.
- [Reward Model trong Latent](02-reward-model-in-latent.md) — reward khả vi trong chuỗi backprop; cảnh báo gradient exploitation.
- [Model Predictive Control (MPC)](04-model-predictive-control.md) — đối cực decision-time; actor có thể guide MPPI.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — imagined trajectory, actor amortizes planning, compounding gradient.
- [VAE](../../02-representation-learning/research/03-vae.md) — reparameterization trick cho pathwise gradient.
- [Model-based vs Model-free RL](01-model-based-vs-model-free-rl.md) — actor-critic trên imagined data là nơi hai trường phái gặp nhau.

## Tham khảo

- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- D. Hafner, T. Lillicrap, M. Norouzi, J. Ba, *Mastering Atari with Discrete World Models* (DreamerV2, ICLR 2021, arXiv:2010.02193).
- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, *Mastering Diverse Domains through World Models* (DreamerV3, 2023, arXiv:2301.04104).
- N. Heess, G. Wayne, D. Silver, T. Lillicrap, Y. Tassa, T. Erez, *Learning Continuous Control Policies by Stochastic Value Gradients* (SVG, NeurIPS 2015, arXiv:1510.09142).
- R. J. Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (REINFORCE, Machine Learning 1992).
- D. P. Kingma, M. Welling, *Auto-Encoding Variational Bayes* (reparameterization, ICLR 2014, arXiv:1312.6114).
- N. Hansen, X. Wang, H. Su, *Temporal Difference Learning for Model Predictive Control* (TD-MPC, ICML 2022, arXiv:2203.04955).
