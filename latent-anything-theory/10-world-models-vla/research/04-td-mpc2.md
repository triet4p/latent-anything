# TD-MPC2 (Hansen et al., 2023)

> **TL;DR.** TD-MPC2 là world model **decoder-free (implicit)** cho continuous control: latent *không* học bằng reconstruction mà bằng [TD-learning](../../07-latent-planning/research/03-value-function-in-latent.md) + dự đoán reward — tinh thần [value equivalence](../../07-latent-planning/research/08-value-equivalence-muzero.md). Lúc chạy, nó **plan cục bộ bằng [MPPI](../../07-latent-planning/research/06-mppi.md)** trong latent, dùng terminal value học-được để vượt horizon ngắn, có một policy prior dẫn hướng sampling. Cộng các kỹ thuật ổn định (SimNorm, LayerNorm+Mish, discrete reward/value regression, max-entropy prior), *một* bộ hyperparameter thắng 104 task; một agent 317M làm 80 task đa embodiment. Caveat: planning mỗi bước đắt hơn policy thuần; latent implicit khó introspect bằng reconstruction.

Dòng [Dreamer](01-dreamerv1.md) học latent bằng dựng lại observation. TD-MPC2 (Hansen et al., 2023) đi hướng đối lập và bổ sung: bỏ decoder, học latent *chỉ để phục vụ control* (predict reward + value), rồi **kết hợp planning (MPC) với value học-được (TD)**. Đây là cầu nối trực tiếp tới embodied use case và là một anchor quan trọng cho thiết kế `ModelAdapter` của framework.

---

## 1. Trực giác: học latent cho control, plan bằng nó

Hai ý tưởng lớn ghép lại:

1. **Implicit world model (value equivalence)**: latent không cần dựng lại pixel; nó chỉ cần đủ để dự đoán *reward* và *value* đúng. Đây chính là luận điểm [MuZero](../../07-latent-planning/research/08-value-equivalence-muzero.md) — latent tồn tại để serve planning, không để render. Bỏ reconstruction loại bỏ áp lực mô hình hóa chi tiết vô nghĩa.

2. **MPC + TD value**: pure [MPC](../../07-latent-planning/research/04-model-predictive-control.md) chỉ nhìn xa $H$ bước; pure value-based RL nhìn vô hạn nhưng không tận dụng model. TD-MPC ghép cả hai: rollout $H$ bước trong latent rồi *bootstrap bằng terminal value* $Q$ học bằng TD. Ngắn hạn dùng model chính xác, dài hạn dùng value — vượt giới hạn horizon.

---

## 2. Cơ chế: các thành phần và planning

Năm mạng (tất cả trên latent $z$):

| Thành phần | Vai trò |
|---|---|
| Encoder $h$ | observation $\to$ latent $z$ |
| Dynamics $d$ | $(z, a) \to z'$ (latent transition, [decoder-free](../../06-latent-temporal/research/02-latent-transition-model.md)) |
| Reward $R$ | $(z, a) \to r$ |
| Value $Q$ | $(z, a) \to$ giá trị, học bằng **TD** |
| Policy prior $\pi$ | đề xuất action để dẫn hướng planning |

**Học model**: không có reconstruction. Loss gồm dự đoán reward, dự đoán latent kế (consistency với encoder của observation thật), và TD loss cho $Q$ — toàn bộ trong latent. Reward và value dùng **discrete regression (two-hot)** như [DreamerV3](03-dreamerv3.md) để ổn định qua mọi scale.

**Planning (lúc chạy)**: tại mỗi state, chạy [MPPI](../../07-latent-planning/research/06-mppi.md) — sample nhiều action sequence, rollout trong latent bằng $d$, tính return $\sum \gamma^t R + \gamma^H Q(z_H, \cdot)$ (bootstrap terminal value), cập nhật phân phối action theo importance weighting, lặp; thực thi bước đầu (receding horizon). Policy prior $\pi$ cung cấp một phần mẫu để planning không phải tìm từ con số 0.

**Ổn định để "một config" chạy 104 task**:
- **SimNorm**: chuẩn hóa latent state bằng cách chia thành nhóm rồi softmax mỗi nhóm — ép $z$ về một tập **simplex**, giới hạn biên độ (họ hàng với latent rời rạc, chống state bùng nổ).
- **LayerNorm + Mish**, max-entropy policy prior, discrete reward/value → chống exploding gradient, robust qua reward scale.

---

## 3. Vì sao quan trọng

| | Dreamer (V1–V3) | TD-MPC2 |
|---|---|---|
| Học latent | reconstruction (ELBO) | **decoder-free**: reward + TD value |
| Chọn action | policy học bằng imagination | **MPPI planning** + policy prior |
| Dài hạn | λ-return trong imagination | terminal **TD value** bootstrap MPC |
| Chuẩn hóa latent | KL/symlog | **SimNorm** (simplex) |
| Điểm mạnh | sample efficiency, đa domain | continuous control robotics, **massively multitask** |

TD-MPC2 chứng minh: world model implicit + planning + TD scale tốt (317M params, 80 task, nhiều embodiment/action space) với *một* bộ hyperparameter — đặc biệt mạnh cho continuous control robotics, đúng [embodied use case](https://github.com/triet4p/latent-anything/blob/main/docs/THEORY.md) mà framework nhắm tới. Nó là cầu giữa [planning tầng 7](../../07-latent-planning/research/04-model-predictive-control.md) và world model quy mô lớn.

---

## 4. Giới hạn / Khi nào thất bại

**Planning đắt lúc chạy.** MPPI mỗi bước cần nhiều rollout — chậm hơn một forward của policy thuần (Dreamer). Với điều khiển tần số cao, đây là rào cản; policy prior giảm nhưng không xóa chi phí.

**Latent implicit khó introspect bằng reconstruction.** Vì decoder-free, không thể "decode để xem model tưởng tượng gì" như Dreamer — Layer A mất một công cụ audit trực quan (phải dựa probe value/reward thay vì ảnh).

**Phụ thuộc chất lượng value.** Bootstrap terminal value lệch thì kế hoạch dài lệch; TD-learning trong latent có thể overestimate (cần double-Q, target network).

**Continuous control-centric.** Thiết kế tối ưu cho continuous action + MPPI; action rời rạc hoặc không gian rất lớn kém tự nhiên hơn.

**SimNorm là một ràng buộc.** Ép latent lên simplex ổn định nhưng giới hạn dạng biểu diễn; không phải lúc nào cũng là inductive bias đúng.

---

## 5. Liên hệ với Latent-Anything

TD-MPC2 là *bản thiết kế tham chiếu* cho adapter một world model **planning-based, decoder-free** — đối cực với Dreamer reconstruction-based. Hai mẫu hình này định hình hai loại `ModelAdapter`:

```python
class TDMPCAdapter(Protocol):
    def encode(self, obs: np.ndarray) -> np.ndarray: ...          # h: obs -> z (SimNorm)
    def dynamics(self, z: np.ndarray, a: np.ndarray) -> np.ndarray: ...  # d: (z,a) -> z'
    def reward(self, z: np.ndarray, a: np.ndarray) -> float: ...
    def value(self, z: np.ndarray, a: np.ndarray) -> float: ...   # TD-learned Q
    def plan(self, z: np.ndarray) -> np.ndarray: ...              # MPPI in latent
```

- **Layer A — Introspection**: vì decoder-free, introspection chuyển từ "decode ảnh" sang *probe value/reward* — vẽ landscape $Q(z,\cdot)$, kiểm tra latent có separate theo value không. SimNorm làm latent có cấu trúc simplex, dễ phân tích như phân phối.
- **Layer B — Manipulation**: MPPI planning *là* một thao tác Layer B trên latent — chính là rollout + chọn của tầng 7. Framework có thể phơi bày "plan" như một method chuẩn áp lên bất kỳ latent transition model nào.
- **Layer C — Runtime**: planning lặp nhiều rollout là workload nặng nhưng song song hóa được — Layer C batch các candidate action sequence, đúng loại tối ưu runtime mà framework hướng tới.

TD-MPC2 đại diện nhánh planning của world model. Mục kế tiếp — **MuZero** — là gốc lý thuyết của ý tưởng decoder-free (value equivalence) mà TD-MPC2 kế thừa, đặt trong bối cảnh planning bằng MCTS.

---

## Liên quan

- [Value Equivalence (MuZero)](../../07-latent-planning/research/08-value-equivalence-muzero.md) — gốc của latent decoder-free; TD-MPC2 dùng tinh thần này cho continuous control.
- [MPPI — Path Integral](../../07-latent-planning/research/06-mppi.md) — bộ planner TD-MPC2 dùng lúc chạy.
- [Model Predictive Control (MPC)](../../07-latent-planning/research/04-model-predictive-control.md) — khung receding-horizon TD-MPC2 đứng trên.
- [Value Function trong Latent](../../07-latent-planning/research/03-value-function-in-latent.md) — terminal value bootstrap vượt horizon.
- [DreamerV3](03-dreamerv3.md) — đối cực reconstruction-based; cùng dùng discrete reward/value regression.

## Tham khảo

- N. Hansen, H. Su, X. Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control* (ICLR 2024, arXiv:2310.16828).
- N. Hansen, X. Wang, H. Su, *Temporal Difference Learning for Model Predictive Control* (TD-MPC, ICML 2022, arXiv:2203.04955).
- J. Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero, Nature 2020, arXiv:1911.08265).
