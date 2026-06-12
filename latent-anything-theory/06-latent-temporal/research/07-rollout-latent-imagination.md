# Rollout và Latent Imagination

> **TL;DR.** Rollout lặp transition model từ một state khởi đầu để sinh trajectory tương lai; latent imagination làm việc này trong representation space mà không cần decode observation ở mỗi bước. Lợi ích là có thể đánh giá nhiều action sequences với chi phí phụ thuộc latent dimension thay vì pixel dimension. Caveat quyết định là distribution shift: model ăn chính prediction của nó, nên bias, uncertainty và policy exploitation tích lũy theo horizon.

[Latent Trajectory](06-latent-trajectory.md) định nghĩa đối tượng đường đi theo thời gian. Rollout là operation **sinh** trajectory đó từ dynamics model, initial state và một chuỗi action hoặc policy.

PlaNet dùng latent rollout để search action sequence online. Dreamer dùng imagined trajectories để học actor và critic. PETS giữ nhiều particles qua probabilistic ensemble. MBPO chủ động dùng rollout ngắn để giới hạn model bias. Các hệ này khác thuật toán, nhưng cùng dựa trên câu hỏi:

> Nếu thực hiện action này từ state hiện tại, model tin rằng tương lai nào sẽ xảy ra và return là bao nhiêu?

---

## **1. Định nghĩa rollout**

Với transition tất định:

$$
\hat z_{t+1}
=
f_\theta(\hat z_t,a_t),
\qquad
\hat z_0=z_{\mathrm{start}}.
$$

Trong đó $f_\theta$ là learned dynamics, $a_t$ là action tại bước $t$, còn $\hat z_t$ là imagined state. Dấu mũ nhấn mạnh trajectory đến từ model, không phải state đã được observation tương lai hiệu chỉnh.

Rollout horizon $H$ tạo:

$$
\hat{\mathcal{Z}}_{0:H}
=
(\hat z_0,\hat z_1,\ldots,\hat z_H).
$$

Trong đó trajectory có $H+1$ states và thường $H$ actions. Horizon là số transition được mô phỏng, không nhất thiết là số giây; duration còn phụ thuộc control interval.

Với stochastic transition:

$$
\hat z_{t+1}
\sim
p_\theta(z_{t+1}\mid \hat z_t,a_t).
$$

Trong đó mỗi rollout là một sample trajectory. Phân phối trên trajectories được tạo bởi tích các conditional transitions:

$$
p_\theta(\hat z_{1:H}\mid z_0,a_{0:H-1})
=
\prod_{t=0}^{H-1}
p_\theta(\hat z_{t+1}\mid \hat z_t,a_t).
$$

Trong đó Markov factorization giả định current latent state đủ cho bước tiếp theo. Dù mỗi transition đơn giản, trajectory distribution có thể trở nên đa mode qua nhiều bước.

---

## **2. Open-loop, closed-loop và receding horizon**

### Open-loop action rollout

Chuỗi action được cố định trước:

$$
a_{0:H-1}
=
(a_0,a_1,\ldots,a_{H-1}).
$$

Trong đó model đánh giá một candidate plan mà action sau không phản ứng với imagined state. Cách này dễ batch nhưng nhạy với disturbance và model error.

### Policy-conditioned rollout

Action được tạo từ imagined state:

$$
a_t
\sim
\pi_\psi(a_t\mid \hat z_t).
$$

Trong đó $\pi_\psi$ là policy. Rollout đóng vòng **bên trong model**, nhưng vẫn open-loop đối với environment thật vì không có observation correction.

### Receding-horizon control

Model Predictive Control lập kế hoạch $H$ bước, thực hiện một hoặc vài action đầu, nhận observation mới rồi plan lại:

$$
a_t
=
\operatorname{first}
\left[
\arg\max_{a_{t:t+H-1}}
\hat J(a_{t:t+H-1}\mid z_t)
\right].
$$

Trong đó $\hat J$ là predicted return của model và chỉ action đầu được thực hiện. Replanning liên tục giới hạn thời gian trajectory thật phải đi theo một plan cũ.

| Chế độ | Action tương lai | Có observation thật giữa rollout? | Dùng cho |
|---|---|---|---|
| **Open-loop sequence** | cố định | không | candidate evaluation |
| **Policy imagination** | phụ thuộc imagined state | không | actor/critic learning |
| **Receding horizon** | re-optimize sau mỗi observation | có, giữa các planning cycles | online control |

---

## **3. Latent imagination**

Pixel-space simulator dự báo observation:

$$
\hat o_{t+1}
=
g_\theta(\hat o_t,a_t),
$$

trong đó $\hat o_t\in\mathbb{R}^{H_{\mathrm{img}}\times W_{\mathrm{img}}\times C}$ có kích thước lớn. Model phải dành compute cho texture và chi tiết không phải lúc nào cũng liên quan action selection.

Latent imagination encode context một lần rồi rollout state nhỏ:

$$
z_t
=
\operatorname{Enc}_\phi(o_{\le t}),
\qquad
\hat z_{t+k+1}
=
f_\theta(\hat z_{t+k},a_{t+k}).
$$

Trong đó encoder chỉ dùng observations đã có, còn future rollout chạy trong latent space. Reward, continuation và value heads đọc trực tiếp latent state:

$$
\hat r_t
=
r_\theta(\hat z_t),
\qquad
\hat c_t
=
c_\theta(\hat z_t),
\qquad
\hat v_t
=
v_\xi(\hat z_t).
$$

Trong đó $\hat r_t$ là predicted reward, $\hat c_t$ là continuation probability và $\hat v_t$ là value estimate. Decoder pixel không cần chạy trong inner planning loop nếu downstream objective chỉ cần các heads này.

### Khi nào vẫn cần decode?

Decoder vẫn hữu ích cho:

- audit qualitative rollout;
- human-in-the-loop planning;
- reconstruction objective;
- constraint chỉ định trong observation space;
- phát hiện latent exploitation mà reward head không thấy.

Không decode giúp nhanh hơn, nhưng cũng loại một kênh kiểm tra model đang tưởng tượng điều gì.

---

## **4. RSSM imagination**

Trong [RSSM](04-rssm-recurrent-state-space-model.md), state gồm deterministic memory $h_t$ và stochastic state $z_t$.

Từ posterior context state:

$$
(h_t,z_t)
\sim
q_\phi(h_t,z_t\mid o_{\le t},a_{<t}).
$$

Trong đó context observations tạo state đã được evidence hiệu chỉnh.

Sau đó imagination chỉ dùng dynamics prior:

$$
h_{t+1}
=
f_\theta(h_t,z_t,a_t),
\qquad
z_{t+1}
\sim
p_\theta(z_{t+1}\mid h_{t+1}).
$$

Trong đó observation encoder không xuất hiện ở future steps. Đây là posterior-to-prior switch; đánh giá rollout phải ghi rõ switch time.

Dreamer lấy nhiều posterior states từ replay sequences làm start states, rollout policy trong latent space, rồi học behavior từ predicted reward, continuation và value.

---

## **5. Return trên imagined trajectory**

Discounted finite-horizon return:

$$
\hat G_t^{(H)}
=
\sum_{k=0}^{H-1}
\gamma^k
\left(
\prod_{j=0}^{k-1}\hat c_{t+j}
\right)
\hat r_{t+k}
+
\gamma^H
\left(
\prod_{j=0}^{H-1}\hat c_{t+j}
\right)
\hat v_{t+H}.
$$

Trong đó $\gamma$ là discount, $\hat c$ ngắt contribution sau terminal, $\hat r$ là model reward và $\hat v$ bootstrap phần sau horizon. Empty product tại $k=0$ bằng 1.

Continuation-aware discount thường viết:

$$
\hat\gamma_t
=
\gamma\hat c_t.
$$

Trong đó terminal probability được gộp vào discount. Nếu bỏ continuation, rollout có thể tiếp tục nhận reward sau episode end.

### $\lambda$-return

Dreamer dùng return trộn nhiều n-step horizons:

$$
V_\lambda(s_t)
=
\hat r_t
+
\hat\gamma_t
\left[
(1-\lambda)\hat v_{t+1}
+
\lambda V_\lambda(s_{t+1})
\right].
$$

Trong đó $\lambda\in[0,1]$ cân bằng bootstrap sớm với imagined rewards dài hơn. $\lambda$ nhỏ dựa value nhiều hơn; $\lambda$ lớn dựa rollout nhiều hơn và chịu model bias lâu hơn.

---

## **6. Planning bằng imagined rollouts**

### Random shooting

Sample $N$ action sequences, rollout tất cả rồi chọn candidate có return cao nhất:

$$
a^\star_{0:H-1}
=
\arg\max_{a^{(n)}_{0:H-1}}
\hat J\!\left(a^{(n)}_{0:H-1}\right).
$$

Trong đó $n=1,\ldots,N$. Phương pháp dễ vectorize nhưng sample-inefficient trong action dimension cao.

### Cross-Entropy Method

CEM duy trì distribution $q_\eta(a_{0:H-1})$:

1. sample candidate sequences;
2. rollout và score;
3. chọn elite fraction;
4. refit $\eta$ theo elites;
5. lặp vài iterations.

Với Gaussian action distribution:

$$
q_\eta(a_{0:H-1})
=
\mathcal{N}(\mu,\Sigma).
$$

Trong đó $\mu$ và $\Sigma$ được cập nhật từ elite actions. PlaNet dùng online planning bằng CEM trong latent space.

### Actor amortizes planning

Dreamer thay search online lặp lại bằng actor:

$$
a_t
\sim
\pi_\psi(a_t\mid \hat z_t).
$$

Trong đó actor học từ nhiều imagined trajectories để tạo action nhanh ở runtime. Compute planning được chuyển phần lớn sang training.

### Value-guided search

MuZero học dynamics, reward và value đủ cho tree search mà không bắt buộc reconstruct observation. TD-MPC2 dùng learned latent dynamics và value để tối ưu local action sequence. Đây là value-equivalent imagination: state chỉ cần giữ thông tin phục vụ reward/value/policy, không cần là generative scene model đầy đủ.

---

## **7. Chi phí tính toán**

Giả sử:

- $N$ candidate trajectories;
- horizon $H$;
- batch transition cost $C_f$;
- decoder cost $C_d$.

Latent-only planning có chi phí xấp xỉ:

$$
C_{\text{latent}}
\approx
N H (C_f+C_{\text{head}}).
$$

Trong đó $C_{\text{head}}$ là reward/value/continuation heads. Nếu transition thao tác trên latent dimension $d$, một toy linear model có cost khoảng $O(NHd^2)$ hoặc $O(NHd)$ với structured/diagonal dynamics.

Nếu decode mọi imagined step:

$$
C_{\text{decoded}}
\approx
NH(C_f+C_d+C_{\text{head}}).
$$

Trong đó $C_d$ thường phụ thuộc output resolution và decoder architecture. Roadmap ghi trực giác $O(kd)$ so với $O(kH_{\mathrm{img}}W_{\mathrm{img}}C)$ cho mỗi rollout path khi latent update gần tuyến tính theo $d$ và pixel generation scale theo số output values; đây là simplification, không phải complexity universal cho mọi neural architecture.

### Memory

Backprop qua imagination horizon cần giữ activations:

$$
M
\propto
N H d
$$

trong đó constant phụ thuộc network depth, stochastic parameters và autodiff graph. Gradient checkpointing, shorter horizons và actor/value detach thay đổi trade-off compute-memory.

### Parallelism

Candidates và particles thường batch được. Horizon vẫn có dependency tuần tự:

$$
\hat z_{t+2}
=
f_\theta(
f_\theta(\hat z_t,a_t),
a_{t+1}
).
$$

Trong đó bước sau cần state bước trước. Parallelism chủ yếu nằm trên batch candidates, particles và ensemble members, không nằm hoàn toàn trên time.

---

## **8. Compounding error**

One-step training đánh giá model trên states từ data:

$$
\mathcal{L}_{1}
=
\mathbb{E}_{z_t\sim d_{\text{data}}}
\left[
\ell(f_\theta(z_t,a_t),z_{t+1})
\right].
$$

Trong đó input $z_t$ đến từ data/posterior distribution.

Rollout lại đưa prediction vào model:

$$
\hat z_{t+k}
\sim
d_\theta^{(k)},
$$

trong đó $d_\theta^{(k)}$ là state distribution do model và policy tự sinh sau $k$ bước. Nó có thể lệch khỏi $d_{\text{data}}$.

Với deterministic dynamics thật $F$ và learned dynamics $\hat F$, error recursion:

$$
e_{t+1}
=
\hat F(\hat z_t,a_t)
-
F(z_t,a_t).
$$

Thêm bớt $\hat F(z_t,a_t)$:

$$
\|e_{t+1}\|
\le
\underbrace{
\|\hat F(\hat z_t,a_t)-\hat F(z_t,a_t)\|
}_{\text{error propagation}}
+
\underbrace{
\|\hat F(z_t,a_t)-F(z_t,a_t)\|
}_{\text{one-step model error}}.
$$

Trong đó số hạng đầu cho biết error cũ bị dynamics khuếch đại ra sao, số hạng sau là local prediction error mới.

Nếu $\hat F$ là $L$-Lipschitz:

$$
\|e_{t+1}\|
\le
L\|e_t\|+\epsilon_t.
$$

Trong đó $\epsilon_t$ là one-step error tại true state. Khi $L>1$, error có thể tăng nhanh theo horizon; khi $L<1$, dynamics contractive có thể làm error ổn định hơn.

### Teacher forcing che rollout error

Trong teacher forcing, model luôn nhận true/posterior state ở bước kế:

$$
\hat z_{t+1}=f_\theta(z_t,a_t).
$$

Trong free rollout, model nhận prediction:

$$
\hat z_{t+1}=f_\theta(\hat z_t,a_t).
$$

Trong đó chỉ free rollout kiểm tra distribution mà planning thực sự sử dụng.

---

## **9. Multi-step training**

Multi-step prediction loss:

$$
\mathcal{L}_{H}
=
\sum_{k=1}^{H}
w_k
\ell(\hat z_{t+k},z_{t+k}).
$$

Trong đó $\hat z_{t+k}$ được rollout tự hồi quy và $w_k$ là horizon weights. Loss trực tiếp phạt drift nhưng làm optimization khó hơn và có thể ép latent coordinate matching không cần thiết.

### Latent overshooting

Trong RSSM, multi-step prior được kéo về future posterior:

$$
\mathcal{L}_{\text{over}}
=
\sum_{k=2}^{H}
w_k
D_{\mathrm{KL}}
\left(
\operatorname{sg}(q_\phi(z_{t+k}\mid o_{\le t+k}))
\;\|\;
p_\theta(z_{t+k}\mid z_t,a_{t:t+k-1})
\right).
$$

Trong đó posterior target dùng future observations, prior phải dự báo nhiều bước từ context cũ, và $\operatorname{sg}$ ngăn target di chuyển để giảm loss.

### Scheduled sampling và data aggregation

Sequence model có thể trộn true state với predicted state trong training. Tuy nhiên scheduled sampling thay đổi objective và không tự bảo đảm consistency. Một hướng khác là thu thập data dưới policy hiện tại để giảm mismatch giữa behavior distribution và replay data.

### Short model rollouts

MBPO dùng short branched rollouts từ real states. Horizon ngắn hạn chế model bias trong khi vẫn tạo synthetic transitions. Ý tưởng không phải rollout dài luôn tốt hơn; optimal horizon phụ thuộc model accuracy và policy distribution.

---

## **10. Stochastic imagination và particles**

Một mean rollout:

$$
\hat z_{t+1}
=
\mathbb{E}[z_{t+1}\mid \hat z_t,a_t]
$$

trong đó uncertainty bị collapse mỗi bước. Với nonlinear reward hoặc constraints, reward của mean state không bằng expected reward:

$$
r(\mathbb{E}[z])
\neq
\mathbb{E}[r(z)].
$$

Trong đó bất đẳng thức là hệ quả chung của nonlinearity; planning trên mean có thể bỏ qua tail risk và mode identity.

Particle rollout giữ $P$ samples:

$$
\hat z_{t+1}^{(p)}
\sim
p_\theta(
z_{t+1}\mid
\hat z_t^{(p)},a_t
),
\qquad
p=1,\ldots,P.
$$

Trong đó mỗi particle có trajectory riêng. Expected return được xấp xỉ:

$$
\mathbb{E}[\hat G]
\approx
\frac{1}{P}
\sum_{p=1}^{P}
\hat G^{(p)}.
$$

Trong đó sample average hội tụ khi particle count tăng, nhưng rare events cần nhiều particles.

### Ensemble uncertainty

Probabilistic ensemble dùng model index $m$:

$$
\hat z_{t+1}^{(p,m)}
\sim
p_{\theta_m}
\left(
z_{t+1}\mid
\hat z_t^{(p,m)},a_t
\right).
$$

Trong đó variation bên trong model biểu diễn aleatoric uncertainty, disagreement giữa models xấp xỉ epistemic uncertainty. PETS phát triển trajectory sampling qua probabilistic ensembles cho model-based control.

### Risk-sensitive objectives

Thay vì expected return:

$$
\max_a \mathbb{E}[\hat G(a)],
$$

trong đó action tối ưu theo mean có thể chấp nhận tail risk.

Có thể dùng lower confidence objective:

$$
\max_a
\left[
\mathbb{E}[\hat G(a)]
-
\beta\sqrt{\operatorname{Var}[\hat G(a)]}
\right].
$$

Trong đó $\beta>0$ phạt uncertainty. Đây là heuristic risk objective; variance không phân biệt downside và upside.

---

## **11. Termination, absorbing states và masks**

Rollout phải dừng hoặc mask sau terminal:

$$
m_{t+1}
=
m_t c_t,
\qquad
m_0=1.
$$

Trong đó $m_t$ là alive mask và $c_t\in\{0,1\}$ hoặc continuation probability. Reward/value sau terminal được nhân mask.

Với learned continuation:

$$
\hat c_t
=
\Pr(\text{episode continues}\mid \hat z_t).
$$

Trong đó uncertainty termination ảnh hưởng effective horizon. Model đánh giá continuation quá cao sẽ tạo reward tưởng tượng sau terminal; quá thấp sẽ truncate cơ hội thật.

Absorbing-state formulation chuyển mọi terminal state vào state cố định:

$$
z_{t+1}=z_{\mathrm{absorb}}
\quad\text{khi }c_t=0.
$$

Trong đó transitions sau terminal không thay đổi state và reward thường bằng zero. Cách này làm batching đơn giản nhưng semantics absorbing state phải được train rõ.

---

## **12. Model exploitation**

Planner hoặc actor không chọn action ngẫu nhiên; nó chủ động tìm action có predicted return cao:

$$
a^\star
=
\arg\max_a
\hat J_\theta(a).
$$

Trong đó optimization có thể tìm vùng model sai lạc nhưng lạc quan. Đây là optimizer's curse trong learned simulator: càng search mạnh, càng có khả năng khai thác prediction error.

Dấu hiệu:

- imagined return tăng nhưng real return giảm;
- candidate rời density support của replay states;
- ensemble disagreement tăng theo action magnitude;
- reward head dự báo cao trong khi decoded trajectory phi lý;
- longer planning budget làm performance tệ hơn.

Mitigation:

- uncertainty penalty hoặc pessimistic ensemble aggregation;
- action/state support constraints;
- short horizon + value bootstrap;
- receding-horizon replanning;
- collect data ở regions planner muốn dùng;
- holdout multi-step validation dưới current policy;
- separate model learning khỏi actor gradients khi cần.

Không có mitigation nào thay thế data coverage. Uncertainty model cũng có thể quá tự tin ngoài distribution.

---

## **13. Horizon selection**

Horizon dài:

- nhìn thấy delayed consequences;
- giảm phụ thuộc value bootstrap;
- tăng compounding error, compute và memory;
- tăng khả năng exploitation.

Horizon ngắn:

- đáng tin hơn và rẻ hơn;
- cần value function tốt;
- có thể bỏ qua delayed reward/constraint.

Effective horizon với discount $\gamma$ thường được cảm nhận qua trọng số $\gamma^k$, nhưng termination, continuation và task delay cũng quan trọng.

Một adaptive rule có thể dừng khi:

$$
U(\hat z_{t+k})
>
\tau,
$$

trong đó $U$ là uncertainty/support score và $\tau$ là threshold. Phần sau được bootstrap bằng conservative value hoặc bỏ khỏi objective.

Horizon nên được báo cáo cùng:

- control interval;
- posterior context length;
- number of particles/ensembles;
- replanning frequency;
- bootstrap value;
- termination handling.

Chỉ ghi “rollout 15 steps” không đủ để tái lập semantics.

---

## **14. Đánh giá rollout**

### One-step metrics

Đo local transition quality trên posterior/data states. Cần nhưng không đủ.

### Open-loop error theo horizon

Context bằng observations thật, rồi chuyển prior-only:

$$
E(k)
=
\mathbb{E}
\left[
d(\hat z_{t+k},z_{t+k})
\right].
$$

Trong đó $E(k)$ cho error curve theo horizon. Metric $d$ cần có latent semantics hoặc đo qua decoded/task heads.

### Reward và continuation accuracy

Đo:

- reward error/calibration theo horizon;
- terminal timing;
- false continuation sau terminal;
- missed terminal events;
- return rank correlation giữa candidate actions.

Action ranking thường quan trọng hơn pixel fidelity cho planning.

### Posterior switch test

So ba modes:

1. posterior every step;
2. one-step prior rồi posterior correction;
3. prior-only $H$ steps.

Khoảng cách giữa chúng cho biết reconstruction head, one-step dynamics và free rollout đóng góp riêng thế nào.

### Policy-conditioned validation

Model error phải đo dưới actions mà current planner/actor chọn, không chỉ replay actions. Holdout trajectory từ old policy có thể bỏ qua exploitation region mới.

### Behavioral consistency

So predicted với real:

- return;
- action ranking;
- constraint violation;
- event probability;
- success/failure;
- uncertainty coverage.

Rollout đẹp nhưng xếp sai action là model kém cho control.

---

## **15. Các họ phương pháp**

| Hệ | World-model use | Rollout representation | Behavior optimization |
|---|---|---|---|
| **Dyna** | synthetic experience | model transitions | mix learning and planning |
| **World Models** | controller trong “dream” | recurrent latent model | controller optimization |
| **PlaNet** | online planning | RSSM Gaussian latent | CEM |
| **PETS** | online planning | probabilistic ensemble + particles | CEM |
| **Dreamer** | behavior learning | RSSM imagined trajectories | actor-critic |
| **MBPO** | synthetic short branches | learned dynamics | off-policy RL |
| **MuZero** | tree search | value-equivalent latent dynamics | MCTS + policy/value |
| **TD-MPC2** | local trajectory optimization | task-oriented latent model | MPC + value |

Không phải mọi “latent imagination” đều generative. PlaNet/Dreamer học observation model; MuZero chỉ cần latent dynamics hữu ích cho reward, policy và value.

---

## **16. Giới hạn / Khi nào thất bại**

**Compounding error.** One-step error nhỏ có thể tăng nhanh khi prediction trở thành input.

**Posterior–prior mismatch.** Training reconstruction trên posterior states không bảo đảm prior-only trajectory tốt.

**Policy distribution shift.** Planner chọn actions khác replay data và kéo model tới vùng chưa học.

**Model exploitation.** Search/actor tối ưu sai số lạc quan thay vì real return.

**Mean rollout che multimodality.** Expected state có thể nằm giữa các futures không khả thi.

**Particle impoverishment.** Quá ít particles bỏ miss rare branches; resampling hoặc deterministic collapse làm mất diversity.

**Ensemble không đủ đa dạng.** Models chung data/architecture có thể đồng thuận sai ngoài distribution.

**Termination sai.** Reward sau terminal hoặc missed termination làm return bias mạnh.

**Value bootstrap sai.** Horizon ngắn chuyển lỗi từ dynamics sang critic; critic extrapolation có thể nguy hiểm như model error.

**Long-horizon gradients.** Vanishing/exploding gradient và high memory cost ảnh hưởng actor learning qua dynamics.

**Reward model shortcut.** Latent giữ tín hiệu đủ để dự báo reward training nhưng bỏ constraint hoặc factor cần cho task mới.

**Decoder-free audit khó.** Không decode làm planning nhanh nhưng giảm khả năng phát hiện imagined state phi lý.

**Action aliasing.** Learned latent transition có thể không phân biệt actions cho hậu quả giống ngắn hạn nhưng khác dài hạn.

**Temporal resolution sai.** Control interval quá lớn bỏ event; quá nhỏ tăng horizon và compute cho cùng thời lượng.

**Reset/cache leak.** Recurrent state không reset đúng tạo trajectory nối giữa episodes.

**Counterfactual không identifiable.** Offline data không bao phủ action alternatives nên model không thể suy ra đáng tin mọi “nếu làm khác”.

---

## **17. Liên hệ với Latent-Anything**

Rollout nên là operation trả một [Trajectory](06-latent-trajectory.md) giàu provenance:

```python
imagined = model.rollout(
    initial_state=posterior_state,
    actions=action_sequences,
    horizon=15,
    particles=32,
    mode="prior",
)
```

Kết quả cần lưu:

- initial state và source;
- action/policy source;
- horizon, control interval, alive mask;
- prior/posterior switch time;
- stochastic samples, seeds và particle identity;
- model/ensemble member;
- reward, continuation, value;
- uncertainty/support diagnostics;
- decoded observations nếu được yêu cầu;
- gradient/detach semantics.

### Layer A — Introspection

Layer A có thể:

- plot error/uncertainty theo horizon;
- so posterior, one-step prior và open-loop prior;
- xem particle branching và ensemble disagreement;
- phát hiện support drift;
- audit termination và reward accumulation;
- so imagined return với realized return;
- tìm horizon mà action ranking bắt đầu đảo.

### Layer B — Manipulation

Layer B có thể:

- thay action suffix;
- branch từ intermediate state;
- clamp/edit latent factor;
- resample stochastic futures;
- truncate theo uncertainty;
- splice trajectory với model-consistent bridge;
- optimize action sequence bằng random shooting/CEM/gradient.

Mỗi intervention phải giữ branch provenance. Hai particles sau khi edit không còn là samples từ original model distribution.

### Layer C — Runtime

Layer C cần:

- batch dimensions rõ: candidate, particle, ensemble, time, state;
- recurrent cache và reset mask đúng;
- deterministic RNG replay khi debug;
- lazy decode;
- chunk/checkpoint autodiff graph;
- early termination và adaptive horizon;
- top-k candidate retention thay vì giữ mọi activation;
- không trộn posterior state vào prior-only benchmark.

### API tách biệt

```python
prior = model.predict(state, action)
trajectory = model.rollout(state, actions, mode="prior")
branches = model.branch(state, policy, particles=64)
scores = objective.evaluate(branches)
```

`predict` là một bước distribution, `rollout` là fixed action sequence, `branch` sinh policy-conditioned stochastic futures, còn `evaluate` định nghĩa reward/risk. Tách các lớp này tránh hard-code planning objective vào dynamics adapter.

### Complexity contract

Adapter nên báo:

- transition FLOPs/latency theo batch;
- latent state bytes;
- decode cost;
- max efficient horizon;
- supported particle/ensemble batching;
- differentiability;
- device and precision constraints.

Nhờ đó runtime có thể quyết định latent-only, selective decode hoặc hybrid rollout dựa trên budget thật thay vì giả định mọi latent model đều rẻ.

Mục tiếp theo, **Trajectory similarity metrics**, sẽ đi sâu vào cách so các rollouts có length và speed khác nhau. Tầng 7 dùng các primitives ở đây để xây planning, MPC, CEM và model-based control.

---

## Liên quan

- [Markov Property và State Space](01-markov-property-state-space.md) — điều kiện để transition recursion có state đủ.
- [Latent Transition Model](02-latent-transition-model.md) — định nghĩa một bước dynamics và multi-step training cơ bản.
- [Stochastic Transition](03-stochastic-transition.md) — cung cấp distribution, particles và uncertainty cho rollout.
- [RSSM — Recurrent State Space Model](04-rssm-recurrent-state-space-model.md) — prior-only imagination từ posterior context.
- [Kalman Filter và các biến thể](05-kalman-filter-variants.md) — trường hợp prediction/correction giải tích và uncertainty propagation.
- [Latent Trajectory](06-latent-trajectory.md) — schema, timestamps và operations cho kết quả rollout.
- [Density Estimation](../../04-latent-computation/research/06-density-estimation.md) — phát hiện imagined states rời support.
- [VAE](../../02-representation-learning/research/03-vae.md) — nền tảng stochastic latent sampling và reparameterization.

## Tham khảo

- R. S. Sutton, *Dyna, an Integrated Architecture for Learning, Planning, and Reacting* (ACM SIGART Bulletin 1991).
- D. Ha, J. Schmidhuber, *World Models* (NeurIPS Workshop 2018, arXiv:1803.10122).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (ICML 2019, arXiv:1811.04551).
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- D. Hafner, T. Lillicrap, M. Norouzi, J. Ba, *Mastering Atari with Discrete World Models* (ICLR 2021, arXiv:2010.02193).
- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, *Mastering Diverse Domains through World Models* (2023, arXiv:2301.04104).
- K. Chua, R. Calandra, R. McAllister, S. Levine, *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* (NeurIPS 2018, arXiv:1805.12114).
- M. Janner, J. Fu, M. Zhang, S. Levine, *When to Trust Your Model: Model-Based Policy Optimization* (NeurIPS 2019, arXiv:1906.08253).
- A. Nagabandi, G. Kahn, R. S. Fearing, S. Levine, *Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning* (ICRA 2018, arXiv:1708.02596).
- J. Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (Nature 2020, arXiv:1911.08265).
- N. Hansen, H. Su, X. Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control* (ICLR 2024, arXiv:2310.16828).
