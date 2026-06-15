# DreamerV3 (Hafner et al., 2023)

> **TL;DR.** DreamerV3 ("Mastering Diverse Domains through World Models") giữ kiến trúc [DreamerV2](02-dreamerv2.md) (RSSM + categorical latent + actor-critic) nhưng thêm một bộ **kỹ thuật chuẩn hóa bất biến theo scale** để *cùng một bộ hyperparameter* thắng trên 150+ domain rất khác nhau: **symlog** nén biên độ reward/value/observation, **two-hot encoding** cho value regression ổn định qua mọi scale, **free bits + KL balancing** tránh over-regularize, và **return normalization** theo phân vị cho entropy ổn định. Kết quả: agent đầu tiên đào được kim cương Minecraft từ đầu, không cần dữ liệu người hay curriculum. Caveat: vẫn là RSSM reconstruction-based; "một config" không có nghĩa tối ưu cho mọi domain.

[DreamerV1](01-dreamerv1.md) và [DreamerV2](02-dreamerv2.md) mạnh nhưng *nhạy hyperparameter theo domain*: reward scale, KL weight, entropy bonus đều phải dò lại cho mỗi môi trường. DreamerV3 biến Dreamer thành một thuật toán *tổng quát* — chìa khóa không phải kiến trúc mới mà là một loạt biến đổi làm việc học **bất biến với scale của tín hiệu**.

---

## 1. Trực giác: vì sao "một config cho mọi domain" khó

Các domain RL khác nhau ở scale tín hiệu cực rộng: reward có thể là $\pm 1$ (Atari clip) hay hàng nghìn (điểm game), thưa hay dày; observation là pixel hay vector trạng thái. Một learning rule cố định dễ vỡ: value loss bùng nổ khi reward lớn, KL bóp nghẹt biểu diễn khi domain đơn giản, entropy bonus sai khi return scale lệch. DreamerV3 giải quyết bằng cách **chuẩn hóa mọi đại lượng về một thang ổn định trước khi học**, để cùng một bộ siêu tham số hoạt động khắp nơi.

---

## 2. Cơ chế: bốn kỹ thuật chuẩn hóa

### (a) Symlog transform

Để xử lý đại lượng có biên độ rất rộng (reward, value, cả observation vector), DreamerV3 dùng **symlog** (symmetric log):

$$
\mathrm{symlog}(x) = \mathrm{sign}(x)\,\ln(1+|x|), \qquad \mathrm{symexp}(y) = \mathrm{sign}(y)\,(e^{|y|}-1).
$$

Trong đó $\mathrm{symlog}$ nén giá trị lớn (gần log) nhưng giữ tuyến tính quanh 0, đối xứng qua gốc, và $\mathrm{symexp}$ là nghịch đảo để khôi phục. Mạng dự đoán trong không gian symlog rồi symexp ra ngoài — nhờ vậy cùng một mạng xử lý được reward $\pm1$ lẫn $\pm10^4$ mà không đổi scale học.

### (b) Two-hot encoding cho value

Thay vì hồi quy value bằng MSE (nhạy scale), DreamerV3 **rời rạc hóa** trục value thành các bin (trong không gian symexp) và biểu diễn một giá trị thực bằng **two-hot**: đặt khối lượng lên hai bin lân cận sao cho kỳ vọng bằng đúng giá trị. Critic học bằng **cross-entropy** trên phân phối bin — ổn định hơn MSE nhiều khi target thay đổi scale, và biểu diễn được phân phối value đa mode. (Đây là họ hàng của distributional RL.)

### (c) Free bits + KL balancing

DreamerV3 giữ [KL balancing](02-dreamerv2.md) của V2 và thêm **free bits**: chặn dưới số hạng KL ở một ngưỡng (≈1 nat) — nếu KL đã dưới ngưỡng thì *không phạt thêm*. Trên domain đơn giản, điều này ngăn model over-regularize (bóp posterior về prior vô ích, mất thông tin); trên domain phức tạp, KL vẫn hoạt động bình thường. Kết hợp hai cơ chế cho biểu diễn giàu mà ổn định khắp các mức độ khó.

### (d) Return normalization theo phân vị

Hệ số entropy regularization của actor nhạy với scale của return. DreamerV3 **chuẩn hóa return** bằng một khoảng động đo từ phân vị (ví dụ 5%–95%) của return gần đây, chỉ co lại khi return *lớn* (không phóng đại khi return nhỏ). Nhờ vậy một hệ số entropy cố định cho hành vi khám phá hợp lý trên mọi reward scale.

---

## 3. Vì sao quan trọng

| | DreamerV2 | DreamerV3 |
|---|---|---|
| Hyperparameter | tuning theo domain | **một config** cho 150+ domain |
| Value learning | (regression) | **two-hot + cross-entropy** (symexp bins) |
| Scale reward/value | nhạy | **symlog** bất biến |
| KL | balancing | balancing **+ free bits** |
| Entropy bonus | scale-sensitive | **return normalization** theo phân vị |
| Cột mốc | Atari-55 mức người | **kim cương Minecraft** từ đầu, không curriculum |

DreamerV3 biến world model latent thành *thuật toán đa dụng*: một bộ siêu tham số, scale theo kích thước model một cách đơn điệu (model lớn hơn → tốt hơn). Đây là bằng chứng rằng pipeline "world model + imagination" không chỉ mạnh mà còn *robust* đủ để triển khai rộng — tiền đề cho các world model nền tảng quy mô lớn sau này.

---

## 4. Giới hạn / Khi nào thất bại

**Vẫn reconstruction-based.** RSSM của V3 vẫn học latent qua dựng lại observation (trong không gian symlog), nên chưa thoát phê phán [latent vs pixel](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md).

**"Một config" ≠ tối ưu mỗi domain.** Robustness đánh đổi với đỉnh hiệu năng; một phương pháp chuyên biệt vẫn có thể vượt DreamerV3 trên domain của nó.

**Sample efficiency tuyệt đối.** Dù data-efficient, các cột mốc khó (Minecraft) vẫn cần nhiều môi trường-bước; không phải mọi domain đều khả thi với ngân sách nhỏ.

**Phức tạp kỹ thuật.** symlog, two-hot, free bits, percentile scaling là nhiều bộ phận chuyển động; hiểu và tái lập đúng cần cẩn thận, dù mỗi mảnh đơn giản.

**Vẫn compounding error.** Horizon tưởng tượng giới hạn bởi [model drift](../../07-latent-planning/research/10-latent-imagination-horizon.md) như mọi Dreamer.

---

## 5. Liên hệ với Latent-Anything

DreamerV3 không thêm primitive latent mới — đóng góp của nó là *các phép chuẩn hóa* mà framework nên xem như **tiện ích Layer B/C tái dùng được** cho mọi model, không riêng Dreamer:

```python
def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

def symexp(y: np.ndarray) -> np.ndarray:
    return np.sign(y) * np.expm1(np.abs(y))
```

- **Layer A — Introspection**: symlog là một phép biến đổi *trực quan hóa* hữu ích — vẽ latent/return có biên độ rộng trong không gian symlog làm rõ cấu trúc mà tuyến tính che mất. Two-hot value head cho introspection cả *phân phối* value, không chỉ điểm.
- **Layer B — Manipulation**: symlog/symexp là cặp toán tử chuẩn hóa khả nghịch mà Layer B có thể áp trước khi làm số học latent xuyên model có scale khác nhau — đúng vấn đề "cùng coordinate system" của [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md).
- **Layer C — Runtime**: các chuẩn hóa bất biến scale làm pipeline *robust* khi load model lạ — runtime không phải tuning lại cho mỗi adapter, đúng tinh thần plugin-first.

DreamerV3 khép nhóm Dreamer. Mục kế tiếp rẽ sang một dòng world model khác cho robotics — **TD-MPC2** — dùng temporal-difference learning trong latent thay vì reconstruction, gần hơn với [value equivalence](../../07-latent-planning/research/08-value-equivalence-muzero.md).

---

## Liên quan

- [DreamerV2](02-dreamerv2.md) — kiến trúc cơ sở; V3 thêm chuẩn hóa bất biến scale.
- [DreamerV1](01-dreamerv1.md) — gốc của khung imagination actor-critic.
- [Value Function trong Latent](../../07-latent-planning/research/03-value-function-in-latent.md) — two-hot là cách học value ổn định.
- [Latent Arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) — symlog như chuẩn hóa scale trước khi tính trong latent.
- [Latent Imagination Horizon](../../07-latent-planning/research/10-latent-imagination-horizon.md) — giới hạn horizon chung của Dreamer.

## Tham khảo

- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, *Mastering Diverse Domains through World Models* (DreamerV3, 2023, arXiv:2301.04104).
- D. Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2, ICLR 2021, arXiv:2010.02193).
- M. G. Bellemare, W. Dabney, R. Munos, *A Distributional Perspective on Reinforcement Learning* (ICML 2017, arXiv:1707.06887) — nền cho two-hot/distributional value.
