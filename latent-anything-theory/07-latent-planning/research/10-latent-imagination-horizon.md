# Latent Imagination Horizon

> **TL;DR.** Horizon $H$ — số bước imagine trước khi dừng — là núm chung mà *mọi* planner của tầng 7 phải đặt. Trade-off: $H$ dài thấy reward trễ và bớt phụ thuộc value, nhưng compounding model error tăng theo horizon; $H$ ngắn rẻ và đáng tin nhưng thiển cận. Tổng sai số là một đường cong **U** theo $H$ với một $H^\star$ tối ưu. Ba cách giải lặp lại suốt tầng: **value bootstrap** (tóm reward sau horizon → robust với $H$), **short branched rollouts** (MBPO: nhiều rollout ngắn từ real state), và **adaptive truncation** (dừng khi model hết tin cậy). Caveat: $H^\star$ phụ thuộc task/model, không có giá trị phổ quát; bootstrap chỉ *chuyển* lỗi từ dynamics sang critic.

Chín mục trước đều chạm cùng một câu hỏi vận hành: rollout/look-ahead bao xa? [MPC](04-model-predictive-control.md) chọn horizon planning, [Dreamer](07-policy-gradient-imagined-dreamer.md) chọn imagination horizon để backprop, [MCTS](09-mcts-in-latent.md) chọn độ sâu cây. Mục cuối này tổng hợp trade-off đó thành một nguyên lý duy nhất, khép lại tầng 7.

---

## **1. Trực giác / Định nghĩa**

Imagination horizon $H$ là số bước transition model được lặp trước khi đánh giá kết thúc bằng value bootstrap hoặc dừng. Nó cân hai lực ngược chiều:

- **Horizon dài**: nhìn thấy hậu quả trễ (delayed reward), giảm phụ thuộc value bootstrap (ít critic bias), nhưng **compounding error** tích lũy — model ăn chính prediction của nó nên drift xa real state.
- **Horizon ngắn**: rollout ở gần real state nên đáng tin và rẻ, nhưng **thiển cận** — bỏ qua reward xa trừ khi có value tốt bù.

Đây không phải "dài hơn thì tốt hơn": tồn tại một $H^\star$ cân bằng hai lực. DreamerV2 đặt $H=15$ — đủ xa để có tín hiệu, đủ gần để tránh compounding error làm hỏng policy.

---

## **2. Cơ chế: đường cong U của sai số**

Tách sai số của ước lượng return theo horizon thành hai thành phần. Với reward thật $r_k$, model reward lệch $\hat r_k=r_k+b_k$ (bias $b_k$ *tăng* theo $k$ do compounding), và ước lượng truncated không bootstrap $\hat G_H=\sum_{k<H}\gamma^k\hat r_k$:

$$
\hat G_H-G=\underbrace{-\sum_{k\ge H}\gamma^k r_k}_{\text{truncation (thiển cận)}}+\underbrace{\sum_{k<H}\gamma^k b_k}_{\text{compounding model error}}.
$$

Trong đó số hạng đầu là phần reward bị cắt sau horizon (giảm theo $H$ vì $\gamma^H$ co lại), số hạng sau là lỗi model tích lũy (tăng theo $H$). Một bên giảm, một bên tăng → $|\hat G_H-G|$ là đường cong **U** với cực tiểu tại $H^\star$.

### MBPO: chặn cải thiện theo horizon

MBPO hình thức hóa lực này: cải thiện đạt được trên real dynamics khi train bằng model bị chặn bởi hai số hạng chi phối — **generalization error** của model và **distribution shift** của policy — và cả hai *xấu đi theo rollout length*. Kết luận: dùng model ở **horizon ngắn** từ real state để giữ rollout trong vùng gần-thật, giảm mạnh compounding bias.

---

## **3. Ba cách giải (lặp lại suốt tầng 7)**

### Value bootstrap — tháo gỡ horizon

Thay số hạng truncation bằng một value học được: $\hat G_H=\sum_{k<H}\gamma^k\hat r_k+\gamma^H\hat v_\xi(\hat z_H)$. Học value để ước lượng reward sau horizon làm Dreamer **robust với horizon length** — horizon ngắn vẫn không thiển cận. [$\lambda$-return](03-value-function-in-latent.md) là dạng mượt của ý này. Đánh đổi: lỗi chuyển từ dynamics (compounding) sang critic (bias/[deadly triad](03-value-function-in-latent.md)).

### Short branched rollouts (MBPO)

Thay vài rollout dài từ initial state bằng **nhiều rollout ngắn** phân nhánh từ real states trong replay. Điều này **tách rollout length khỏi task horizon**: model chỉ cần đúng vài bước, mỗi nhánh xuất phát từ state on-support, nên compounding bị chặn trong khi vẫn sinh đủ synthetic data.

### Adaptive / uncertainty truncation

Dừng rollout khi state rời support hoặc model uncertainty vượt ngưỡng $\tau$ (xem [density estimation](../../04-latent-computation/research/06-density-estimation.md)), rồi bootstrap value tại đó. Cắt đúng lúc model hết tin cậy — không cần dò $H^\star$ thủ công cho mỗi task.

### Truncated BPTT

Với gradient (Dreamer), backprop qua horizon dài bị [exploding/vanishing](07-policy-gradient-imagined-dreamer.md) ($\sim\rho^H$). Truncate horizon + value bootstrap + gradient clipping giữ gradient ổn định mà vẫn tính delayed reward.

---

## **4. Bảng so sánh**

| | Horizon ngắn | Horizon dài |
|---|---|---|
| Delayed reward | bỏ lỡ (cần value tốt) | nhìn thấy |
| Phụ thuộc value bootstrap | cao | thấp |
| Compounding model error | thấp | cao |
| Compute / memory | thấp | cao |
| BPTT stability | ổn định | dễ explode/vanish |
| Model exploitation | ít | nhiều (search/optimize sâu hơn) |

Công thức chung của tầng 7: **horizon ngắn–vừa + terminal value + replanning/uncertainty truncation** — né cả hai đầu của đường cong U cùng lúc.

---

## **5. Giới hạn / Khi nào thất bại**

**$H^\star$ không phổ quát.** Optimal horizon phụ thuộc độ chính xác model, độ trễ reward của task, và discount $\gamma$. Không có một con số đúng cho mọi bài; phải đo (đường cong error theo horizon).

**Bootstrap chỉ chuyển lỗi.** Value bootstrap làm robust với $H$ nhưng đổ lỗi sang critic; critic sai (overestimation/divergence) cũng hại như model drift. Horizon ngắn không cứu được critic tồi.

**Uncertainty miscalibrated.** Adaptive truncation dựa uncertainty/support score; nếu model quá tự tin ngoài distribution, nó không dừng đúng lúc.

**Horizon không đủ để mô tả semantics.** Ghi "rollout 15 bước" là chưa đủ — còn cần control interval, posterior context length, số particle/ensemble, tần số replanning, bootstrap value, cách xử lý termination (xem [Rollout §13](../../06-latent-temporal/research/07-rollout-latent-imagination.md)).

**Search sâu = exploitation mạnh hơn.** Tăng horizon/độ sâu để thấy xa cũng cho optimizer nhiều cơ hội khai thác model/value sai-lạc-quan hơn.

---

## **6. Liên hệ với Latent-Anything**

Horizon là một **runtime parameter first-class** mà adapter và Layer C phải lộ ra và điều chỉnh được:

```python
@dataclass
class PlanConfig:
    horizon: int                 # H: số bước imagine
    bootstrap_value: bool        # bù phần sau horizon bằng v̂?
    truncate_on_uncertainty: float | None   # ngưỡng τ cho adaptive truncation
    control_interval: float      # mỗi bước imagine = bao nhiêu thời gian thật
    replan_every: int            # receding-horizon period
```

- **Layer A — Introspection**: đo *đường cong sai số theo horizon* (imagined vs realized return ở mỗi $k$) để tìm $H^\star$ thực nghiệm; theo dõi uncertainty growth để đặt $\tau$; phát hiện horizon mà action ranking bắt đầu đảo.
- **Layer B — Manipulation**: horizon là tham số của mọi phép planning (MPC/CEM/MPPI/Dreamer/MCTS); Layer B chọn nó cùng optimizer.
- **Layer C — Runtime**: cân horizon ↔ latency/memory theo budget; hỗ trợ value bootstrap, branched rollouts, adaptive truncation, truncated BPTT như các chiến lược thay thế được.

Đây là mục khép lại **tầng 7 — Planning trong latent space**. Từ [model-based vs model-free](01-model-based-vs-model-free-rl.md), qua reward/value heads, các planner (MPC, CEM, MPPI, Dreamer, MuZero, MCTS), đến horizon — toàn bộ đều dựng trên một ý: *dùng latent world model để plan, không chỉ để represent*. Tầng 8 sẽ đẩy xa hơn — predict trong latent mà không cần decode.

---

## Liên quan

- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — compounding error, horizon selection, semantics đầy đủ của một rollout.
- [Value Function trong Latent](03-value-function-in-latent.md) — value bootstrap tháo gỡ horizon; $\lambda$-return.
- [Model Predictive Control (MPC)](04-model-predictive-control.md) — horizon planning + terminal value + receding-horizon.
- [Policy Gradient trên Imagined Trajectory (Dreamer)](07-policy-gradient-imagined-dreamer.md) — truncated BPTT, imagination horizon, robust nhờ critic.
- [MCTS trong Latent](09-mcts-in-latent.md) — độ sâu cây là horizon của tree search.
- [Model-based vs Model-free RL](01-model-based-vs-model-free-rl.md) — mở đầu tầng; horizon khép lại nó.
- [Density Estimation](../../04-latent-computation/research/06-density-estimation.md) — uncertainty/support score cho adaptive truncation.

## Tham khảo

- M. Janner, J. Fu, M. Zhang, S. Levine, *When to Trust Your Model: Model-Based Policy Optimization* (MBPO, NeurIPS 2019, arXiv:1906.08253).
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- D. Hafner, T. Lillicrap, M. Norouzi, J. Ba, *Mastering Atari with Discrete World Models* (DreamerV2, ICLR 2021, arXiv:2010.02193).
- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, *Mastering Diverse Domains through World Models* (DreamerV3, 2023, arXiv:2301.04104).
- N. Hansen, X. Wang, H. Su, *Temporal Difference Learning for Model Predictive Control* (TD-MPC, ICML 2022, arXiv:2203.04955).
- K. Asadi, D. Misra, M. L. Littman, *Lipschitz Continuity in Model-based Reinforcement Learning* (ICML 2018, arXiv:1804.07193).
