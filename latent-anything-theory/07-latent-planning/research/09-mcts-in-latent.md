# MCTS trong Latent

> **TL;DR.** Monte Carlo Tree Search xây một cây tìm kiếm qua bốn pha — selection, expansion, evaluation, backup — dùng UCT/PUCT để cân bằng khai thác (Q cao) với khám phá (ít lượt thăm). MuZero chạy MCTS *trong latent*: node là abstract state, edge là action, dynamics network mở rộng node và reward/value/policy network đánh giá — toàn bộ search trên learned model, không cần environment. Phân phối visit-count cải thiện trở thành policy target. Caveat: branching factor bùng nổ (action liên tục/lớn cần sampling/progressive widening), compute mỗi quyết định lớn, và search sâu *khuếch đại* model exploitation.

[Value equivalence (mục 8)](08-value-equivalence-muzero.md) cho ta abstract latent + ba network (representation, dynamics, prediction) nhưng để ngỏ: planner nào dùng chúng? Các planner trước (MPC/CEM/MPPI) là *flat sampling* — sample chuỗi action phẳng. MCTS khác về chất: nó xây **cây** lookahead, dồn ngân sách vào nhánh hứa hẹn và đào sâu chọn lọc. Đây là planner của AlphaZero và MuZero, vừa để *hành động* vừa để *tạo policy target*.

---

## **1. Trực giác / Định nghĩa**

MCTS là planning *anytime*: lặp nhiều "simulation", mỗi lần đi xuống một đường trong cây hiện có, mở rộng một node mới, ước lượng nó, rồi cập nhật thống kê ngược lên. Càng nhiều simulation, cây càng tốt — dừng lúc nào cũng có câu trả lời.

Bốn pha mỗi simulation:

1. **Selection**: từ root, đi xuống theo *tree policy* (UCT/PUCT) tới một node chưa mở rộng hết.
2. **Expansion**: thêm một child mới vào cây.
3. **Evaluation (simulation)**: ước lượng giá trị node mới — bằng random rollout tới cuối episode (MCTS cổ điển) **hoặc** bằng value network (AlphaZero/MuZero).
4. **Backup**: dùng giá trị đó cập nhật $Q$ và visit count $N$ của mọi node trên đường đi.

Sau $K$ simulation, action ở root được chọn theo **visit count** (action được thăm nhiều nhất), không phải Q thô — visit count là ước lượng ổn định hơn của "đâu là action tốt".

---

## **2. Selection: UCT và PUCT**

Pha selection là tim của MCTS. **UCT** (UCB applied to Trees) chọn child tối đa:

$$
a^\star=\arg\max_a\;\underbrace{Q(s,a)}_{\text{exploit}}+\;\underbrace{c\sqrt{\frac{\ln N(s)}{N(s,a)}}}_{\text{explore}}.
$$

Trong đó $Q(s,a)$ là value trung bình ước lượng, $N(s,a)$ là số lượt thăm edge, $N(s)$ của parent, và $c$ là hằng số exploration. Số hạng đầu ưu tiên action đang tốt; số hạng sau ưu tiên action ít được thử (uncertainty cao). Cân bằng này là lý do MCTS không kẹt sớm vào một nhánh.

**PUCT** (AlphaZero/MuZero) thay exploration bằng *policy prior* $P(s,a)$ từ prediction network:

$$
a^\star=\arg\max_a\;Q(s,a)+c\,P(s,a)\,\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}.
$$

Trong đó $P(s,a)$ là prior policy hướng search vào action hứa hẹn ngay từ đầu, $Q$ khởi tạo từ value network ở leaf. Prior tốt làm search hiệu quả hơn nhiều: thay vì khám phá mù, cây ưu tiên đúng vùng đáng đào — đây là chỗ amortized policy ([Dreamer, mục 7](07-policy-gradient-imagined-dreamer.md)) và search gặp nhau.

---

## **3. MCTS trong latent (MuZero)**

MuZero chạy MCTS mà node là **abstract latent state**, không phải state thật:

- Root: $s^0=h_\theta(o_{\le t})$ (representation network).
- Expansion một edge $a$: $s',\hat r=g_\theta(s,a)$ (dynamics network) — sinh latent con và reward.
- Evaluation leaf: $\hat p,\hat v=f_\theta(s')$ (prediction network) — policy prior cho PUCT và value để backup. **Không random rollout, không environment** — value network thay thế simulation.
- Backup: lan $\hat v$ và reward dọc đường, cập nhật $Q,N$.

Cả cây sống hoàn toàn trong learned latent dynamics. Output có hai dùng: (1) chọn action thực thi theo visit count; (2) phân phối visit-count $\pi$ làm **policy target** để train prediction network — search *cải thiện* policy của network, network lại làm prior *tốt hơn* cho search lần sau (vòng self-improvement của AlphaZero/MuZero).

---

## **4. Vì sao tree search, khi đã có flat sampling?**

| | Flat sampling (CEM/MPPI) | Tree search (MCTS) |
|---|---|---|
| Cấu trúc | chuỗi action phẳng | cây, chia sẻ prefix |
| Phân bổ ngân sách | đều trên candidate | thích nghi: đào sâu nhánh tốt |
| Đa mode | mean dễ rơi thung lũng ([mục 6](06-mppi.md)) | nhánh hóa, cam kết một mode |
| Action space | continuous tự nhiên | discrete tự nhiên |
| Lookahead sâu | tốn (phẳng) | chọn lọc, tái dùng subtree |

MCTS mạnh khi action **rời rạc/tổ hợp**, cần lookahead sâu và đa mode — board games là ví dụ kinh điển. Nó giải đúng vấn đề averaging của MPPI: thay vì trung bình hai mode tốt thành điểm xấu ở giữa, cây tách nhánh và đánh giá từng mode riêng.

### Action liên tục

Cây cần branching factor hữu hạn; action liên tục thì vô hạn. Hai cách: **progressive widening** (giới hạn số child theo $N^\alpha$, mở thêm dần) và **Sampled MuZero** (sample một tập action hữu hạn từ policy prior rồi search trên đó). Đây là cầu nối để tree search dùng được cho continuous control như các planner khác.

---

## **5. Giới hạn / Khi nào thất bại**

**Branching factor bùng nổ.** Cây tổ hợp theo (số action × depth); action lớn hoặc liên tục đòi sampling/widening, và lookahead sâu vẫn đắt.

**Compute mỗi quyết định.** Mỗi action thật cần hàng trăm/nghìn simulation, mỗi simulation gọi dynamics + prediction network nhiều lần. Nặng cho real-time tần số cao (khác amortized actor một-forward-pass).

**Search sâu khuếch đại model exploitation.** Càng nhiều simulation càng tối ưu mạnh $\hat J$ của learned model — nếu reward/value/dynamics sai-lạc-quan ở vùng nào, search sẽ tìm ra và khai thác. Optimizer's curse ở mức cây.

**Phụ thuộc prior & value.** PUCT hiệu quả nhờ prior tốt; prior tệ làm search lãng phí. Value network sai làm backup sai toàn cây.

**Abstract latent khó audit.** Node là latent value-equivalent, không decode được ([mục 8](08-value-equivalence-muzero.md)) — khó biết cây đang "tưởng tượng" gì.

**Khó song song trong một cây.** Selection tuần tự (mỗi simulation phụ thuộc thống kê vừa cập nhật); cần virtual loss/parallel tricks, không batch thuần như flat sampling.

**Stochastic/partial observability.** MCTS cổ điển giả định deterministic/MDP; môi trường ngẫu nhiên cần chance node, làm cây phức tạp hơn.

---

## **6. Liên hệ với Latent-Anything**

MCTS là một **decision-time planner plugin** nữa cho cùng interface, nhưng cần thêm prior policy và một cấu trúc cây ở runtime:

```python
def mcts_action(z0, model, n_sim=128, c=1.25):
    root = Node(z0)
    for _ in range(n_sim):
        node, path = select(root, c)               # PUCT descent
        s_child, r = model.dynamics(node.z, node.untried_action())  # g: latent expansion
        p_prior, v = model.prediction(s_child)     # f: prior + value (no rollout)
        child = node.expand(s_child, r, p_prior)
        backup(path + [child], r + GAMMA * v)      # propagate value & visit counts
    return root.most_visited_action()              # act by visit count
```

- **Layer A — Introspection**: trực quan hóa cây (visit counts, Q theo độ sâu), so visit-count policy với prior để đo "search cải thiện bao nhiêu", phát hiện nhánh bị over-exploit (model exploitation).
- **Layer B — Manipulation**: cây là một biến đổi trajectory có cấu trúc; có thể can thiệp prior/giá trị để hướng search, hoặc tái dùng subtree khi replanning.
- **Layer C — Runtime**: quản tree, ngân sách simulation, batching đánh giá leaf, progressive widening cho continuous action; lộ trade-off n_sim ↔ latency.

MCTS khép lại nhóm planner của tầng 7: flat sampling (MPC/CEM/MPPI), amortized actor (Dreamer), value-equivalent latent (MuZero), và tree search (MCTS). Mọi planner đều chạm cùng một câu hỏi vận hành — **rollout/look-ahead bao xa?** Mục cuối, **Latent imagination horizon**, tổng hợp trade-off horizon dài (nhiều tín hiệu) vs compound error (model drift) xuyên suốt toàn tầng.

---

## Liên quan

- [Value Equivalence (MuZero)](08-value-equivalence-muzero.md) — abstract latent làm node cây; visit-count policy làm target.
- [Value Function trong Latent](03-value-function-in-latent.md) — value network đánh giá leaf, thay random rollout.
- [Reward Model trong Latent](02-reward-model-in-latent.md) — reward dọc edge trong backup.
- [Policy Gradient trên Imagined Trajectory (Dreamer)](07-policy-gradient-imagined-dreamer.md) — amortized policy có thể làm prior PUCT.
- [MPPI](06-mppi.md) — flat sampling; MCTS giải vấn đề averaging đa mode bằng branching.
- [Model Predictive Control (MPC)](04-model-predictive-control.md) — decision-time planning, cùng vòng receding-horizon.

## Tham khảo

- R. Coulom, *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search* (Computers and Games 2006).
- L. Kocsis, C. Szepesvári, *Bandit Based Monte-Carlo Planning* (UCT, ECML 2006).
- C. Browne et al., *A Survey of Monte Carlo Tree Search Methods* (IEEE TCIAIG 2012).
- D. Silver et al., *Mastering the Game of Go without Human Knowledge* (AlphaGo Zero, Nature 2017).
- D. Silver et al., *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play* (AlphaZero, Science 2018, arXiv:1712.01815).
- J. Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero, Nature 2020, arXiv:1911.08265).
- T. Hubert, J. Schrittwieser, I. Antonoglou, M. Barekatain, S. Schmitt, D. Silver, *Learning and Planning in Complex Action Spaces* (Sampled MuZero, ICML 2021, arXiv:2104.06303).
