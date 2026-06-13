# Model Predictive Control (MPC)

> **TL;DR.** MPC (receding-horizon control) giải một bài toán tối ưu hữu hạn horizon tại *mỗi* bước — tìm chuỗi action tối đa imagined return $\hat J=\sum_{t}\gamma^t\hat r(\hat z_t)+\gamma^H\hat v(\hat z_H)$ — rồi chỉ thực thi action đầu, quan sát state mới, và lặp lại. Hai ý cốt lõi: **terminal value** $\hat v(\hat z_H)$ thay cho phần return sau horizon (cho phép horizon ngắn), và **replanning** đóng vòng phản hồi nên model error/nhiễu không tích lũy như open-loop. Caveat: phải giải tối ưu *mỗi bước* (ràng buộc real-time), và planner vẫn khai thác được model/value sai trong horizon.

[Mục 2](02-reward-model-in-latent.md) và [mục 3](03-value-function-in-latent.md) đã cho đủ thành phần của objective $\hat J$: reward head, value head, continuation, transition. MPC là *planner cụ thể đầu tiên* tối ưu objective đó. Nó trả lời câu hỏi vận hành: tại state hiện tại, chọn action nào? Câu trả lời của MPC không phải "học một policy" mà "tối ưu lại tại chỗ, mỗi bước, dùng model".

---

## **1. Trực giác / Định nghĩa**

MPC đến từ control lý thuyết và đồng nghĩa với *receding-horizon control*. Vòng lặp:

1. Ở state hiện tại $z_t$, **tối ưu** một chuỗi action $a_{t:t+H-1}$ trên model qua horizon $H$.
2. **Thực thi** chỉ action đầu $a_t^\star$ trên environment thật.
3. **Quan sát** state mới $z_{t+1}$.
4. **Dời horizon** một bước và lặp lại từ (1).

Ẩn dụ: lái xe nhìn xa $H$ mét, vạch lộ trình cả đoạn, nhưng chỉ đi một bước rồi nhìn lại và vạch lại. Ta không bao giờ cam kết toàn bộ kế hoạch — chỉ dùng nó để chọn bước kế tiếp cho tốt.

---

## **2. Bài toán tối ưu**

Dạng control cổ điển cực tiểu hóa stage cost cộng terminal cost, có ràng buộc:

$$
\min_{a_{t:t+H-1}}\;\sum_{k=0}^{H-1}\ell(\hat z_{t+k},a_{t+k})+V_f(\hat z_{t+H})\quad\text{s.t.}\quad \hat z_{t+k+1}=\hat f(\hat z_{t+k},a_{t+k}),\;\; a\in\mathcal{A}.
$$

Trong đó $\ell$ là stage cost, $V_f$ là **terminal cost**, $\hat f$ là dynamics model và $\mathcal{A}$ là ràng buộc action. Trong khung RL latent, đổi dấu (cost → reward) và viết lại bằng các head đã có:

$$
a_{t:t+H-1}^\star=\arg\max_{a_{t:t+H-1}}\;\underbrace{\sum_{k=0}^{H-1}\gamma^k\,\hat r_\theta(\hat z_{t+k})}_{\text{reward trong horizon}}+\underbrace{\gamma^H\,\hat v_\xi(\hat z_{t+H})}_{\text{terminal value bootstrap}},\qquad \hat z_{t+k+1}=\hat f_\theta(\hat z_{t+k},a_{t+k}).
$$

Trong đó tổng reward chạy trong horizon còn value head bootstrap phần đuôi. Sau khi giải, chỉ $a_t^\star$ (phần tử đầu của nghiệm) được thực thi:

$$
a_t=\operatorname{first}(a_{t:t+H-1}^\star).
$$

Trong đó $\operatorname{first}$ lấy action đầu tiên; phần còn lại của chuỗi bị bỏ và sẽ được tính lại ở bước sau với observation mới.

---

## **3. Terminal value: cầu nối với tầng value**

Trong MPC cổ điển, terminal cost $V_f$ thường được yêu cầu là một **control Lyapunov function** để chứng minh ổn định: horizon hữu hạn với terminal cost phù hợp cho cùng tính ổn định như horizon vô hạn. Vấn đề thực tế: với hệ phi tuyến/robot trong môi trường động, $V_f$ đúng thường *không tính được* hoặc quá bảo thủ.

Latent RL giải nút này bằng cách **học** terminal value: $V_f=\hat v_\xi$ chính là value head ở [mục 3](03-value-function-in-latent.md). Đây là ý tưởng trung tâm của TD-MPC — kết hợp MPC horizon ngắn với một terminal value học bằng TD-learning, trên latent dynamics model. Lợi ích kép:

- Horizon $H$ có thể **ngắn** (rẻ, ít [compounding error](../../06-latent-temporal/research/07-rollout-latent-imagination.md)) vì value head lo phần xa.
- Không cần dẫn xuất giải tích $V_f$ — nó được học từ data, thích nghi với task.

Đổi lại, ta thừa hưởng mọi rủi ro của value head (deadly triad, overestimation): terminal value sai sẽ trực tiếp làm MPC chọn sai.

---

## **4. Vì sao replanning quan trọng: closed-loop vs open-loop**

Một chuỗi action tối ưu nếu thực thi trọn vẹn là **open-loop** — không phản ứng với điều xảy ra thật. MPC chỉ dùng bước đầu rồi replanning, biến nó thành **closed-loop feedback**:

| | Open-loop (thực thi cả chuỗi) | Receding-horizon (MPC) |
|---|---|---|
| Phản hồi observation | không | có, mỗi bước |
| Chịu nhiễu / model error | tích lũy không sửa | sửa mỗi lần replanning |
| Compute | một lần | mỗi bước |
| Dùng cho | đánh giá candidate | điều khiển online |

Trực giác: model luôn sai một chút và environment có nhiễu; nếu cam kết cả chuỗi, sai số dồn theo horizon ([recursion $\|e_{t+1}\|\le L\|e_t\|+\epsilon$](../../06-latent-temporal/research/07-rollout-latent-imagination.md)). Replanning "reset" điểm xuất phát về state thật mỗi bước, nên chỉ cần model đúng *một bước tới* là đủ điều khiển tốt. PlaNet và TD-MPC đều chọn action bằng MPC để thích nghi kế hoạch theo observation mới.

---

## **5. Giải bài toán tối ưu: shooting và optimizer**

Bài toán $\arg\max\hat J$ thường phi lồi (dynamics phi tuyến). Hai họ phương pháp:

- **Gradient-based**: nếu $\hat f,\hat r,\hat v$ khả vi, backprop $\nabla_a\hat J$ và leo dốc. Nhanh khi landscape trơn, nhưng kẹt local optimum và cần model khả vi.
- **Sampling-based (shooting)**: sample nhiều chuỗi action, rollout model, chấm điểm, chọn/又 refit. Không cần gradient, dễ batch/vectorize, robust với landscape gồ ghề. **Random shooting** là bản đơn giản nhất; **CEM (mục tiếp theo)** và **MPPI (mục sau)** là các optimizer population-based tinh vi hơn — TD-MPC dùng MPPI làm planner bên trong vòng MPC.

Điểm chung: tất cả chỉ gọi model (rẻ trong latent), nên đánh giá hàng nghìn candidate khả thi trong budget mỗi bước.

---

## **6. Giới hạn / Khi nào thất bại**

**Ràng buộc real-time.** Phải giải một bài tối ưu *mỗi bước*. Với control tần số cao (robot), ngân sách tính toán mỗi bước rất nhỏ; horizon dài hoặc số candidate lớn có thể vượt deadline. Đây là khác biệt lớn so với policy amortized (Dreamer học actor để khỏi search online).

**Model error trong horizon.** Replanning chỉ sửa giữa các bước; *trong* một lần plan, model vẫn rollout chính prediction của nó. Horizon dài + model dở vẫn cho plan tệ.

**Terminal value/cost sai.** Cổ điển: $V_f$ khó dẫn xuất. Học: $\hat v_\xi$ kế thừa overestimation/divergence; terminal value over-confident kéo MPC tới state xấu.

**Model & value exploitation.** Optimizer chủ động tìm action tối đa $\hat J$, nên dễ khai thác vùng model/value lạc quan sai (optimizer's curse). Mitigation: uncertainty penalty, ràng buộc trong support, horizon ngắn.

**Local optima.** Gradient-based dễ kẹt; sampling-based cần đủ candidate để phủ. Action space liên tục chiều cao làm shooting kém hiệu quả nếu không có distribution tốt (lý do cần CEM/MPPI).

**Không bảo đảm ổn định khi học.** Lý thuyết ổn định MPC dựa terminal ingredients đúng; với model/terminal value học được, các bảo đảm đó thường mất.

---

## **7. Liên hệ với Latent-Anything**

MPC là *decision-time planner* mẫu mực mà Layer B/C cần hỗ trợ. Nó ráp đúng các head đã định nghĩa:

```python
def mpc_action(z_t, model, objective, horizon=5, n_candidates=512):
    seqs = sample_action_sequences(n_candidates, horizon)      # shooting
    Z = model.rollout(z_t, seqs, horizon)                      # latent rollout, batched
    J = objective.evaluate(Z, seqs)                            # Σ γ^k r̂ + γ^H v̂
    best = seqs[int(J.argmax())]
    return best[0]                                             # execute first action only
```

- **Layer A — Introspection**: so imagined return của plan với realized return sau khi thực thi (gap = model+value error); theo dõi mức độ "replanning sửa được bao nhiêu" như chỉ số tin cậy model.
- **Layer B — Manipulation**: MPC là một phép biến đổi trên trajectory — sinh candidate, đánh giá, chọn. Các optimizer (random shooting → CEM → MPPI) là các plugin thay thế được trong cùng interface.
- **Layer C — Runtime**: vòng receding-horizon là một control loop runtime phải quản: ngân sách compute mỗi bước, batch candidate/particle, cache recurrent state đúng, và lộ ra trade-off horizon/candidate ↔ latency.

Mục này định khung "rollout-nhiều-chọn-tốt-nhất". Ba mục kế tiếp — **Cross-Entropy Method**, **MPPI**, rồi **Dreamer (policy gradient)** — là các cách *giải* bước tối ưu bên trong MPC ngày một hiệu quả hơn, từ sampling thuần đến amortized policy.

---

## Liên quan

- [Value Function trong Latent](03-value-function-in-latent.md) — terminal value bootstrap, ingredient cho horizon ngắn.
- [Reward Model trong Latent](02-reward-model-in-latent.md) — stage reward trong objective MPC.
- [Model-based vs Model-free RL](01-model-based-vs-model-free-rl.md) — MPC là decision-time planning, một nhánh của model-based.
- **Cross-Entropy Method (mục tiếp theo)** — optimizer population-based giải bước $\arg\max$ của MPC.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — receding horizon, open-loop vs closed-loop, compounding error.

## Tham khảo

- D. Q. Mayne, J. B. Rawlings, C. V. Rao, P. O. M. Scokaert, *Constrained Model Predictive Control: Stability and Optimality* (Automatica 2000).
- J. B. Rawlings, D. Q. Mayne, M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design* (2nd ed., Nob Hill 2017).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (PlaNet, ICML 2019, arXiv:1811.04551).
- N. Hansen, X. Wang, H. Su, *Temporal Difference Learning for Model Predictive Control* (TD-MPC, ICML 2022, arXiv:2203.04955).
- N. Hansen, H. Su, X. Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control* (ICLR 2024, arXiv:2310.16828).
- G. Williams, A. Aldrich, E. A. Theodorou, *Model Predictive Path Integral Control: From Theory to Parallel Computation* (J. Guidance, Control, and Dynamics 2017).
