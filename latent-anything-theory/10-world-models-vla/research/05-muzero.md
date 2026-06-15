# MuZero (Schrittwieser et al., 2019)

> **TL;DR.** MuZero là agent planning **decoder-free** kinh điển: ba mạng — *representation* $h$ (observation → latent $s_0$), *dynamics* $g$ ($s_k, a_k \to s_{k+1}, r_{k+1}$), *prediction* $f$ ($s_k \to$ policy $p_k$, value $v_k$) — được dùng làm simulator cho [MCTS](../../07-latent-planning/research/09-mcts-in-latent.md) chạy *trong latent*. Mấu chốt: model **không** được train để dựng lại observation; tín hiệu học chỉ đến từ dự đoán đúng **reward, value, policy** ([value equivalence](../../07-latent-planning/research/08-value-equivalence-muzero.md)). Đạt SOTA Atari và sánh AlphaZero ở Go/Chess/Shogi mà *không biết luật*. Caveat: MCTS đắt; latent abstract khó introspect; cần nhiều search mỗi bước.

[Value equivalence](../../07-latent-planning/research/08-value-equivalence-muzero.md) (tầng 7) nêu *nguyên lý*; [MCTS trong latent](../../07-latent-planning/research/09-mcts-in-latent.md) nêu *cơ chế search*. Mục này ráp chúng thành thuật toán MuZero hoàn chỉnh — agent định nghĩa cả một dòng decoder-free (kế thừa bởi [TD-MPC2](04-td-mpc2.md), [tokenized world model](../../09-discrete-latent/research/07-tokenized-world-model.md)) và là một anchor cho thiết kế adapter planning-based.

---

## 1. Trực giác: model chỉ cần đủ để plan

AlphaZero plan bằng simulator *thật* (luật game). MuZero hỏi: nếu không biết luật, học một model *vừa đủ để plan* thì sao? Câu trả lời triệt để — model không cần dự đoán observation tương lai, chỉ cần ba thứ mà planning thực sự dùng: **reward** (để đánh giá nước đi), **value** (để bootstrap cuối nhánh search), và **policy** (để ưu tiên nhánh nào mở rộng). Latent của MuZero do đó là một *trạng thái trừu tượng* không có nghĩa pixel — nó chỉ là "bộ nhớ đủ để search đúng".

Đây là khác biệt nền tảng với [Dreamer](01-dreamerv1.md) (reconstruction) và là lý do MuZero không lãng phí capacity cho chi tiết vô nghĩa — đúng tinh thần [latent prediction](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md).

---

## 2. Cơ chế: ba hàm + MCTS

### Ba hàm học-được

$$
s_0 = h(o_{1:t}), \qquad (s_{k+1}, r_{k+1}) = g(s_k, a_k), \qquad (p_k, v_k) = f(s_k).
$$

- $h$ (**representation**): nén lịch sử observation thành latent gốc $s_0$.
- $g$ (**dynamics**): từ latent + action ra latent kế và reward dự đoán — đây là [latent transition model](../../06-latent-temporal/research/02-latent-transition-model.md) decoder-free.
- $f$ (**prediction**): từ latent ra phân phối policy $p_k$ và value $v_k$.

### Planning bằng MCTS trong latent

Tại mỗi nước thật, MuZero chạy MCTS *hoàn toàn trong latent*: bắt đầu từ $s_0 = h(o)$, mở rộng cây bằng $g$ (transition tưởng tượng), dùng $f$ cho prior policy và value ở lá, chọn nhánh theo [UCB/PUCT](../../07-latent-planning/research/09-mcts-in-latent.md). Sau $N$ lần mô phỏng, **visit-count** ở gốc cho một *policy cải thiện* $\pi$ và một value gốc $\nu$. Agent chọn action bằng cách sample từ $\pi$.

### Huấn luyện: ba mục tiêu, không reconstruction

Unroll model $K$ bước từ một state thật, khớp ba target dọc theo trajectory thật:

$$
\mathcal{L} = \sum_{k=0}^{K}\Big[ \ell^r(r^{\text{true}}_{t+k}, r_{t+k}) + \ell^v(z_{t+k}, v_{t+k}) + \ell^p(\pi_{t+k}, p_{t+k}) \Big],
$$

trong đó $\ell^r$ khớp reward thật, $\ell^v$ khớp value target $z$ (n-step return bootstrap bằng giá trị MCTS), $\ell^p$ khớp policy MCTS $\pi$ (cross-entropy). **Không có số hạng reconstruction** — toàn bộ ràng buộc lên latent đến từ ba dự đoán này. Đó là value equivalence ở dạng đầy đủ: latent tự do miễn là $g, f$ dự đoán đúng reward/value/policy.

---

## 3. Vì sao quan trọng

| | AlphaZero | MuZero |
|---|---|---|
| Model động lực | luật game cho sẵn | **học** ($g$ trong latent) |
| Học latent | — | reward + value + policy (no recon) |
| Áp dụng | game có luật | game **+ Atari** (luật ẩn) |
| Planning | MCTS trên state thật | **MCTS trong latent** |

MuZero là bằng chứng dứt khoát rằng *planning với một model học-được, decoder-free* hoạt động ở đỉnh cao: SOTA Atari đồng thời sánh AlphaZero ở Go/Chess/Shogi với cùng một thuật toán. Nó đặt nền cho mọi world model "predict-để-plan" sau đó — [TD-MPC2](04-td-mpc2.md) mang ý tưởng sang continuous control + MPPI, các UniZero/EfficientZero mở rộng sang sample-efficiency.

---

## 4. Giới hạn / Khi nào thất bại

**MCTS đắt.** Mỗi nước cần hàng trăm mô phỏng cây; tốn tính toán, khó cho điều khiển realtime tần số cao (lý do TD-MPC2 chọn MPPI cho continuous control).

**Latent abstract, khó introspect.** Không có decoder nên không "xem" được model tưởng tượng gì; Layer A phải probe qua value/policy/reward thay vì ảnh — mất công cụ audit trực quan.

**Phụ thuộc chất lượng value/policy target.** Toàn bộ tín hiệu học từ ba dự đoán; nếu value bootstrap lệch hoặc search nông, target nhiễu → model học sai. Cần search đủ sâu và reanalyze.

**Action rời rạc tự nhiên hơn.** MCTS gốc cho action space rời rạc/hữu hạn; continuous action cần biến thể (Sampled MuZero).

**Không dùng được model cho mục tiêu ngoài planning.** Vì latent chỉ học để serve reward/value/policy, nó *không* hỗ trợ các tác vụ cần observation (vd sinh ảnh) — đối lập với world model reconstruction-based.

---

## 5. Liên hệ với Latent-Anything

MuZero là *bản thiết kế tham chiếu* cho adapter planning-based decoder-free — cùng họ [TD-MPC2](04-td-mpc2.md), đối cực [Dreamer](01-dreamerv1.md). Ba hàm của nó map thẳng vào các primitive framework phơi bày:

```python
class MuZeroAdapter(Protocol):
    def represent(self, obs_history: np.ndarray) -> np.ndarray: ...      # h: o -> s_0
    def dynamics(self, s: np.ndarray, a: int) -> tuple[np.ndarray, float]: ...  # g: (s,a)->(s',r)
    def predict(self, s: np.ndarray) -> tuple[np.ndarray, float]: ...    # f: s -> (policy, value)
    def plan(self, s: np.ndarray, n_sims: int) -> np.ndarray: ...        # MCTS -> improved policy
```

- **Layer A — Introspection**: decoder-free nên introspection chuyển sang probe — kiểm tra latent có encode value/policy tuyến tính không, vẽ search tree, đo entropy của policy MCTS. Latent MuZero là test-case cho "latent thuần planning".
- **Layer B — Manipulation**: MCTS là một method Layer B (search + chọn); can thiệp vào latent rồi plan để xem policy đổi thế nào là thí nghiệm phản thực sạch.
- **Layer C — Runtime**: MCTS là workload search song song hóa được (batch các node expansion); Layer C tối ưu được như một engine planning chung cho nhiều adapter.

MuZero là gốc lý thuyết của nhánh decoder-free planning. Các mục còn lại của tầng rẽ sang **VLA** (vision-language-action) — OpenVLA, π0 — nơi latent không chỉ để plan mà còn để *hành động* trong thế giới vật lý.

---

## Liên quan

- [Value Equivalence (MuZero)](../../07-latent-planning/research/08-value-equivalence-muzero.md) — nguyên lý decoder-free; mục này là thuật toán đầy đủ.
- [MCTS trong Latent](../../07-latent-planning/research/09-mcts-in-latent.md) — bộ planner MuZero dùng trên model học-được.
- [TD-MPC2](04-td-mpc2.md) — kế thừa value equivalence cho continuous control với MPPI.
- [Value Function trong Latent](../../07-latent-planning/research/03-value-function-in-latent.md) — value target bootstrap trong training.
- [Tokenized World Model](../../09-discrete-latent/research/07-tokenized-world-model.md) — cùng tinh thần predict-không-decode ở dạng token.

## Tham khảo

- J. Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero, Nature 2020, arXiv:1911.08265).
- D. Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* (AlphaZero, Science 2018).
- T. Hubert et al., *Learning and Planning in Complex Action Spaces* (Sampled MuZero, ICML 2021, arXiv:2104.06303).
