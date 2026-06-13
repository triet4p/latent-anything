# Model-based vs Model-free RL

> **TL;DR.** Model-free RL học trực tiếp policy hoặc value từ reward, không bao giờ dự đoán "điều gì xảy ra tiếp theo"; model-based RL học một dynamics model $\hat p(s_{t+1}\mid s_t,a_t)$ rồi plan/imagine trên đó. Trade-off cốt lõi: model-based tiết kiệm sample (vài lần ít interaction thật hơn) nhưng tốn compute và chịu model bias; model-free đơn giản, asymptotic mạnh, nhưng ngốn data. Latent world model là "sweet spot" vì model dự đoán trong representation space nhỏ thay vì pixel space, giảm cả compute lẫn compounding error đủ để giữ phần lợi sample-efficiency.

Tầng 6 đã xây dựng các primitive của một dynamics model trong latent space: [Markov state](../../06-latent-temporal/research/01-markov-property-state-space.md), [transition](../../06-latent-temporal/research/02-latent-transition-model.md), [stochastic transition](../../06-latent-temporal/research/03-stochastic-transition.md), [RSSM](../../06-latent-temporal/research/04-rssm-recurrent-state-space-model.md), [trajectory](../../06-latent-temporal/research/06-latent-trajectory.md) và [rollout](../../06-latent-temporal/research/07-rollout-latent-imagination.md). Tầng 7 trả lời câu hỏi: **có model rồi thì dùng để làm gì?** Câu trả lời là *planning* — dùng model để chọn action, không chỉ để dự đoán. Mục mở đầu này định vị toàn bộ tầng: tại sao bỏ công học model thay vì học policy trực tiếp, và tại sao latent là nơi đáng để học model đó.

---

## **1. Hai cách tiếp cận: học cái gì?**

Bài toán control được mô tả bằng Markov Decision Process $(\mathcal{S},\mathcal{A},p,r,\gamma)$, trong đó $p(s_{t+1}\mid s_t,a_t)$ là transition, $r(s_t,a_t)$ là reward và $\gamma\in[0,1)$ là discount. Mục tiêu là tìm policy $\pi$ tối đa expected return:

$$
J(\pi)=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\right].
$$

Trong đó kỳ vọng lấy trên trajectory sinh bởi $\pi$ và $p$. Câu hỏi phân nhánh hai trường phái là: **để tối đa $J$, agent có cần biết $p$ và $r$ không?**

- **Model-free RL** trả lời *không*. Nó học thẳng các đại lượng phục vụ quyết định — value $Q^\pi(s,a)$ hoặc policy $\pi_\theta(a\mid s)$ — từ reward quan sát được, coi environment như hộp đen. Q-learning, DQN, PPO, SAC đều thuộc nhóm này.
- **Model-based RL** trả lời *có*. Nó học một xấp xỉ $\hat p_\theta(s_{t+1}\mid s_t,a_t)$ (và thường cả $\hat r_\theta$), rồi dùng model đó để plan: rollout, search, hoặc sinh synthetic experience để train policy.

Khác biệt không nằm ở "có dùng neural network hay không" mà ở **đối tượng được học**: model-free học hàm quyết định, model-based học hàm dự đoán thế giới rồi suy ra quyết định.

---

## **2. Cơ chế model-free: học value/policy trực tiếp**

Value-based model-free dựa trên Bellman optimality, không cần $p$ tường minh — chỉ cần sample transition $(s,a,r,s')$:

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big].
$$

Trong đó $\alpha$ là learning rate, số hạng trong ngoặc là **TD error**. Điểm mấu chốt: $s'$ là state thật mà environment trả về, nên transition $p$ được "dùng" ngầm qua sample chứ không bao giờ được mô hình hóa. Đây là lý do model-free đơn giản và robust — không có model nào để sai.

Policy-based model-free tối ưu trực tiếp $\theta$ bằng policy gradient:

$$
\nabla_\theta J(\pi_\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right].
$$

Trong đó $\hat A_t$ là advantage estimate. Gradient này cũng chỉ cần sample trajectory thật, không cần biết $p$. Cái giá: mỗi update tiêu thụ data thật, và data thật trong robotics hay là thứ đắt nhất.

---

## **3. Cơ chế model-based: học model rồi plan**

Model-based tách bài toán làm hai pha. **Pha học model:** fit $\hat p_\theta$ bằng maximum likelihood trên data đã thu thập:

$$
\theta^\star=\arg\max_\theta\;\mathbb{E}_{(s,a,s')\sim\mathcal{D}}\big[\log\hat p_\theta(s'\mid s,a)\big].
$$

Trong đó $\mathcal{D}$ là replay buffer các transition thật. Đây là supervised learning thuần túy — ổn định hơn nhiều so với RL.

**Pha dùng model:** có ba cách khai thác $\hat p_\theta$, định hình toàn bộ tầng 7:

| Cách dùng model | Cơ chế | Đại diện |
|---|---|---|
| **Background planning (Dyna)** | sinh synthetic transition từ model để train policy model-free | Dyna-Q, MBPO |
| **Decision-time planning** | tại mỗi state, rollout/search nhiều action sequence rồi chọn | MPC, CEM, MPPI, MCTS |
| **Differentiable planning** | backprop return qua model khả vi để cập nhật actor | Dreamer, SVG |

Số hạng quyết định ở mọi cách là **imagined return** — return ước lượng bằng model thay vì environment thật:

$$
\hat J(a_{0:H-1}\mid s_0)=\sum_{t=0}^{H-1}\gamma^t \hat r_\theta(\hat s_t,a_t)+\gamma^H \hat v(\hat s_H),\qquad \hat s_{t+1}=\hat f_\theta(\hat s_t,a_t).
$$

Trong đó $\hat s_t$ là imagined state, $\hat v$ là value bootstrap cho phần sau horizon $H$. Vì $\hat J$ chỉ gọi model chứ không gọi environment, agent có thể đánh giá hàng nghìn plan mà không tốn một bước thật nào — đó là nguồn gốc của sample efficiency.

---

## **4. Trade-off trung tâm: sample efficiency vs computational cost**

Đây là trục so sánh quan trọng nhất và là lý do tầng 7 tồn tại.

| | Model-free | Model-based |
|---|---|---|
| **Đối tượng học** | $Q$ hoặc $\pi$ | $\hat p$, $\hat r$ (rồi suy ra $\pi$) |
| **Sample efficiency** | thấp — cần nhiều interaction thật | cao — tái dùng model để imagine |
| **Compute mỗi step** | thấp | cao — phải rollout/search model |
| **Nguồn lỗi chính** | variance của gradient/value | model bias + compounding error |
| **Asymptotic performance** | thường cao nếu đủ data | bị chặn bởi độ chính xác model |
| **Ổn định training** | nhạy hyperparameter | pha học model ổn định, pha plan dễ exploit |

**Vì sao model-based tiết kiệm sample:** mỗi transition thật được dùng nhiều lần — một lần để cập nhật model, rồi vô số lần gián tiếp qua các imagined rollout sinh từ model đó. Model nén kinh nghiệm thành một hàm tái sử dụng được. Sutton gọi đây là tinh thần của kiến trúc Dyna: learning, planning và reacting chia sẻ cùng một value update, chỉ khác experience đến từ thật hay từ model.

**Vì sao model-based tốn compute và rủi ro:** học một high-capacity dynamics model trên high-dimensional observation cần nhiều sample để generalize, có thể triệt tiêu chính lợi ích sample-efficiency. Và planning lặp model sai dẫn đến **compounding error** — lỗi một bước được dynamics khuếch đại theo horizon (xem [Rollout](../../06-latent-temporal/research/07-rollout-latent-imagination.md) §8) — cộng **model exploitation**: planner chủ động tìm vùng model lạc quan sai. Một model dở còn tệ hơn không có model.

Trực giác để nhớ: **model-free trả giá bằng data, model-based trả giá bằng compute và độ tin của model.** Việc chọn phía nào phụ thuộc cái gì đắt hơn trong bài toán cụ thể — trong robotics, data thật đắt, nên model-based hấp dẫn.

---

## **5. Tại sao latent world model là sweet spot**

Cả hai cách dùng model ở trên đều giả định model dự đoán trong *observation space*. Với observation là pixel $o_t\in\mathbb{R}^{H\times W\times C}$, model-based gặp hai vấn đề chí mạng: (1) học generative pixel model rất tốn capacity và sample; (2) rollout pixel cực đắt và compounding error tích tụ trên hàng triệu chiều noise/texture không liên quan đến quyết định.

**Latent world model** giải cả hai bằng cách chuyển model vào representation space nhỏ:

$$
z_t=\operatorname{Enc}_\phi(o_{\le t}),\qquad \hat z_{t+1}=\hat f_\theta(\hat z_t,a_t),\qquad \hat r_t=\hat r_\theta(\hat z_t),\quad \hat v_t=\hat v_\xi(\hat z_t).
$$

Trong đó observation chỉ được encode một lần ở context, còn rollout chạy hoàn toàn trên $z\in\mathbb{R}^d$ với $d\ll HWC$. Reward và value đọc trực tiếp từ latent, nên **decoder không cần chạy trong vòng planning**. Điều này dịch chuyển đường cong trade-off:

- **Compute giảm**: chi phí rollout $\sim O(NHd)$ thay vì $O(NH\cdot HWC)$ cho $N$ candidate, horizon $H$ — đánh giá nhiều future trong latent nhỏ song song được trong bộ nhớ.
- **Sample efficiency giữ nguyên hoặc tăng**: predict latent ép model học cấu trúc semantic thay vì pixel detail, ổn định long-horizon prediction hơn. PlaNet cho thấy RSSM latent ổn định dự đoán dài hạn và cho phép online planning bằng CEM trong latent; Dreamer học behavior *hoàn toàn từ imagined latent trajectory*, đạt sample-efficiency và robust trên nhiều continuous-control benchmark.
- **Model bias bớt nguy hiểm**: model không phải đúng từng pixel, chỉ cần đúng phần latent ảnh hưởng reward/value (ý tưởng value-equivalence, sẽ gặp lại ở MuZero).

Vì vậy "sweet spot" không phải vì latent xóa bỏ trade-off, mà vì nó **dời điểm cân bằng**: giữ lợi sample-efficiency của model-based trong khi cắt phần compute và compounding error vốn làm model-based pixel-space bất khả thi. Đây chính là luận điểm để Latent-Anything coi latent là first-class object cho planning, không chỉ cho representation.

---

## **6. Phổ liên tục, không phải nhị phân**

Ranh giới model-based/model-free không sắc nét; thực tế là một phổ:

- **Dyna** dùng model chỉ để sinh thêm experience cho một thuật toán model-free — lai giữa hai phía.
- **MBPO** dùng model rollout *ngắn* phân nhánh từ real state, rồi train off-policy RL trên hỗn hợp — chủ động giới hạn model bias bằng horizon ngắn.
- **MuZero** học latent dynamics nhưng *không* reconstruct observation, chỉ giữ phần đủ cho value/policy/reward — model-based nhưng theo nghĩa value-equivalent.
- **Implicit model-based**: vài thuật toán value-based có thể xem như ngầm học model qua value.

Điểm cần nhớ: câu hỏi đúng không phải "model-based hay model-free" mà "**học model ở mức nào, dùng nó cho bao xa, và chấp nhận model sai tới đâu**". Toàn bộ tầng 7 là các câu trả lời khác nhau cho câu hỏi đó.

---

## **7. Giới hạn / Khi nào thất bại**

**Model-based thất bại khi:**

- **Model khó học hơn policy.** Nếu dynamics phức tạp/hỗn loạn nhưng policy tối ưu đơn giản, học model là đường vòng tốn kém.
- **Compounding error vượt ngưỡng.** Long-horizon rollout của model không đủ chính xác cho ra plan vô nghĩa; phải dựa value bootstrap và horizon ngắn.
- **Model exploitation.** Planner tối ưu sai số lạc quan thay vì return thật — càng search mạnh càng dễ bị.
- **Compute không đáng.** Nếu data thật rẻ (simulator nhanh, song song lớn), lợi sample-efficiency không bù nổi chi phí planning; model-free đơn giản lại thắng.

**Model-free thất bại khi:**

- **Data thật đắt hoặc nguy hiểm.** Robotics thật, y tế — không thể thu hàng triệu interaction.
- **Cần zero-shot adapt sang task/reward mới.** Model-free phải học lại gần như từ đầu; một model + reward mới thì re-plan được ngay.

**Latent cụ thể thất bại khi:** encoder bỏ mất thông tin cần cho task mới (reward shortcut), hoặc latent transition không phân biệt action có hậu quả dài hạn khác nhau (action aliasing). Latent rẻ hơn nhưng cũng khó audit hơn vì không decode mỗi bước.

---

## **8. Liên hệ với Latent-Anything**

Mục này đặt khung cho cả tầng 7 và định hình Layer C (Runtime). Một `ModelAdapter` cần khai báo nó hỗ trợ kiểu planning nào:

```python
class ModelAdapter(Protocol):
    def predict(self, z: np.ndarray, a: np.ndarray) -> np.ndarray: ...      # một bước latent transition
    def reward(self, z: np.ndarray) -> np.ndarray: ...                       # reward head, không cần decode
    def value(self, z: np.ndarray) -> np.ndarray: ...                        # value bootstrap
    differentiable: bool                                                     # backprop được qua transition?
```

- **Layer A — Introspection**: so sánh imagined return với realized return để phát hiện model bias; plot error theo horizon để chọn planning budget hợp lý.
- **Layer B — Manipulation**: vì model rẻ trong latent, Layer B có thể thử nhiều action sequence/branch để chỉnh trajectory — chính là planning (MPC/CEM/MPPI, các mục tiếp theo).
- **Layer C — Runtime**: dựa trên cờ `differentiable` và complexity contract của adapter mà chọn background planning, decision-time planning hay differentiable planning; latent-only rollout cho phép runtime đánh giá nhiều candidate trong budget cố định.

Các mục tiếp theo của tầng 7 sẽ lần lượt chi tiết hóa các thành phần của imagined return: **Reward model trong latent** ($\hat r_\theta$), **Value function trong latent** ($\hat v_\xi$), rồi các planner cụ thể (**MPC**, **CEM**, **MPPI**, **Dreamer**, **MuZero**, **MCTS**) và bài toán chọn **horizon**.

---

## Liên quan

- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — cơ chế sinh imagined trajectory mà planning tối ưu trên đó.
- [Latent Transition Model](../../06-latent-temporal/research/02-latent-transition-model.md) — định nghĩa $\hat f_\theta$ một bước, đối tượng được học trong model-based.
- [RSSM — Recurrent State Space Model](../../06-latent-temporal/research/04-rssm-recurrent-state-space-model.md) — latent dynamics cụ thể của PlaNet/Dreamer.
- [Markov Property và State Space](../../06-latent-temporal/research/01-markov-property-state-space.md) — điều kiện để MDP và Bellman update hợp lệ.
- [Density Estimation](../../04-latent-computation/research/06-density-estimation.md) — phát hiện imagined state rời support, mitigation cho model exploitation.

## Tham khảo

- R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press 2018).
- R. S. Sutton, *Dyna, an Integrated Architecture for Learning, Planning, and Reacting* (ACM SIGART Bulletin 1991).
- T. M. Moerland, J. Broekens, A. Plaat, C. M. Jonker, *Model-based Reinforcement Learning: A Survey* (Foundations and Trends in ML 2023, arXiv:2006.16712).
- A. Plaat, W. Kosters, M. Preuss, *Deep Model-Based Reinforcement Learning for High-Dimensional Problems, a Survey* (2020, arXiv:2008.05598).
- D. Ha, J. Schmidhuber, *World Models* (NeurIPS 2018, arXiv:1803.10122).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (ICML 2019, arXiv:1811.04551).
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- M. Janner, J. Fu, M. Zhang, S. Levine, *When to Trust Your Model: Model-Based Policy Optimization* (NeurIPS 2019, arXiv:1906.08253).
