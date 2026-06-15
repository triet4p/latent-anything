# DreamerV2 (Hafner et al., 2020)

> **TL;DR.** DreamerV2 ("Mastering Atari with Discrete World Models") giữ nguyên khung [DreamerV1](01-dreamerv1.md) nhưng thay stochastic latent Gaussian bằng **categorical (rời rạc)**: state ngẫu nhiên là một tập biến categorical (32 biến × 32 lớp), sample với [straight-through estimator](../../09-discrete-latent/research/01-vector-quantization.md). Cộng với **KL balancing** (cho prior đuổi theo posterior nhanh hơn chiều ngược lại), thay đổi này ổn định hóa training và là agent world-model đầu tiên đạt mức người trên 55 game Atari. Caveat: vẫn dựa reconstruction pixel; actor dùng REINFORCE cho action rời rạc nên nhiễu hơn analytic gradient.

[DreamerV1](01-dreamerv1.md) hoàn chỉnh nhưng latent Gaussian đơn mode khó nắm động lực đa mode của game (nhiều tương lai khả dĩ), và cân bằng KL nhạy. DreamerV2 vá đúng hai chỗ đó bằng *discrete latent* — kéo dòng Dreamer về cùng họ với [tokenized world model](../../09-discrete-latent/research/07-tokenized-world-model.md) của tầng 9.

---

## 1. Trực giác: vì sao latent rời rạc

Game Atari có động lực **đa mode**: từ một state, nhiều chuyện khác nhau có thể xảy ra (kẻ địch xuất hiện hay không). Một stochastic state [Gaussian](../../06-latent-temporal/research/03-stochastic-transition.md) đơn mode buộc phải *trung bình hóa* các tương lai đó — làm mờ, mất thông tin. Một phân phối **categorical** trên các lớp rời rạc biểu diễn nhiều mode tự nhiên: gán xác suất cao cho vài lớp khác nhau, mỗi lớp một "kịch bản". Đây đúng quan sát đã thấy ở [tokenized world model](../../09-discrete-latent/research/07-tokenized-world-model.md) — categorical bắt được tương lai phân nhánh mà Gaussian không.

Phụ: latent rời rạc khớp tự nhiên với [straight-through estimator](../../09-discrete-latent/research/01-vector-quantization.md), không cần mô hình hóa phương sai, và one-hot sparse dễ cho transformer/RNN dự đoán bằng cross-entropy.

---

## 2. Cơ chế: hai thay đổi cốt lõi

### (a) Categorical latent với straight-through

Stochastic state $s_t$ không còn là vector Gaussian mà là **tập biến categorical**: bản gốc dùng $32$ biến, mỗi biến $32$ lớp (một ma trận logit $32\times 32$). Mỗi biến sample một one-hot từ phân phối categorical; gradient chảy ngược bằng straight-through:

$$
s_t = \text{onehot}(\text{sample}) + \big(p_\theta - \mathrm{sg}[p_\theta]\big),
$$

trong đó $p_\theta$ là vector xác suất softmax, $\mathrm{sg}$ là stop-gradient. Forward trả one-hot (rời rạc); backward truyền gradient qua $p_\theta$ liên tục. Đây chính là STE của tầng 9, áp lên *state* của world model thay vì token observation.

### (b) KL balancing

Loss KL trong [ELBO](../../02-representation-learning/research/03-vae.md) ràng buộc posterior $q(s_t\mid h_t,x_t)$ và prior $p(s_t\mid h_t)$. Vấn đề: nếu phạt KL đối xứng, model có thể *làm yếu posterior* (vứt thông tin observation) để giảm KL — hại biểu diễn. **KL balancing** tách KL thành hai chiều với tốc độ khác nhau:

$$
\mathcal{L}_{\text{KL}} = \alpha\,\mathrm{KL}\big(\mathrm{sg}[q]\,\Vert\,p\big) + (1-\alpha)\,\mathrm{KL}\big(q\,\Vert\,\mathrm{sg}[p]\big),
$$

với $\alpha \approx 0.8$. Số hạng đầu (sg trên $q$) huấn luyện *prior đuổi theo posterior* mạnh hơn; số hạng sau (sg trên $p$) kéo posterior về prior nhẹ hơn. Kết quả: prior học dự đoán tốt mà không bóp nghẹt thông tin trong posterior — ổn định và giữ được biểu diễn giàu.

Phần còn lại giữ nguyên DreamerV1: actor-critic trên imagined rollout, critic regress [λ-return](01-dreamerv1.md). Khác biệt: action Atari rời rạc nên actor dùng **REINFORCE** (cộng entropy regularizer) thay analytic gradient; với continuous control thì vẫn dùng straight-through/analytic.

---

## 3. Vì sao quan trọng

| | DreamerV1 | DreamerV2 |
|---|---|---|
| Stochastic latent | Gaussian (đơn mode) | **categorical** (đa mode) |
| Gradient qua sample | reparameterization | **straight-through** |
| Cân bằng KL | đối xứng | **KL balancing** (α≈0.8) |
| Domain chứng minh | continuous control (DMC) | **Atari-55 mức người** + DMC |

DreamerV2 là **agent world-model đầu tiên đạt human-level trên Atari-55**, vượt Rainbow/IQN với cùng ngân sách tính toán trên một GPU. Nó chứng minh discrete latent không chỉ hợp cho tokenizer (tầng 9) mà còn cho *state* của world model — bắc cầu trực tiếp giữa Dreamer và dòng tokenized world model.

---

## 4. Giới hạn / Khi nào thất bại

**Vẫn reconstruction pixel.** RSSM của V2 vẫn học latent qua dựng lại observation — chưa thoát phê phán [latent vs pixel prediction](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md); đây là chỗ JEPA-style và một số world model sau đi xa hơn.

**REINFORCE cho action rời rạc.** Mất lợi thế phương sai thấp của analytic gradient ([DreamerV1](01-dreamerv1.md)); cần entropy regularizer và tuning cẩn thận.

**Nhạy số biến/lớp.** $32\times 32$ là lựa chọn kinh nghiệm; quá ít thì thiếu dung lượng, quá nhiều thì khó học. Cũng có rủi ro một dạng [collapse](../../09-discrete-latent/research/04-codebook-collapse.md) ở mức biến categorical (lớp chết).

**Model exploitation.** Như mọi Dreamer, actor có thể khai thác lỗi world model; horizon tưởng tượng vẫn bị [compounding error](../../07-latent-planning/research/10-latent-imagination-horizon.md) giới hạn.

**Per-domain tuning.** V2 vẫn cần điều chỉnh hyperparameter theo domain — chính điều **DreamerV3** giải quyết bằng các biến đổi bất biến (symlog, KL free bits, robust normalization).

---

## 5. Liên hệ với Latent-Anything

DreamerV2 cho thấy *state* của một world model có thể là latent rời rạc — một loại latent mà Layer A đo được như token. Adapter của nó phơi bày state categorical:

```python
class DreamerV2Adapter(Protocol):
    def observe(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray: ...   # -> categorical state (32x32)
    def imagine(self, state: np.ndarray, policy) -> np.ndarray: ...
    num_categoricals: int   # 32
    num_classes: int        # 32
```

- **Layer A — Introspection**: state categorical cho phép đo *usage* và entropy của từng biến (giống perplexity codebook tầng 9), phát hiện biến/lớp chết, và xem KL prior–posterior như tín hiệu "model bất ngờ".
- **Layer B — Manipulation**: state rời rạc one-hot dễ can thiệp có ngữ nghĩa — bật/tắt một lớp rồi imagine để xem kịch bản phản thực, sạch hơn nhiễu loạn vector Gaussian.
- **Layer C — Runtime**: one-hot sparse nén tốt và lookup nhanh; rollout categorical batch hiệu quả như tokenized world model.

DreamerV2 đưa discrete latent vào world model. Mục kế tiếp — **DreamerV3** — giữ discrete latent nhưng thêm các kỹ thuật chuẩn hóa để *cùng một bộ hyperparameter* thắng trên hàng trăm domain khác nhau.

---

## Liên quan

- [DreamerV1](01-dreamerv1.md) — khung gốc mà V2 cải tiến phần latent.
- [Vector Quantization](../../09-discrete-latent/research/01-vector-quantization.md) — straight-through estimator dùng cho categorical latent.
- [Tokenized World Model](../../09-discrete-latent/research/07-tokenized-world-model.md) — categorical bắt tương lai đa mode; cùng tinh thần discrete state.
- [Stochastic Transition](../../06-latent-temporal/research/03-stochastic-transition.md) — Gaussian đơn mode mà V2 thay bằng categorical.
- [VAE](../../02-representation-learning/research/03-vae.md) — KL balancing tinh chỉnh số hạng KL của ELBO.

## Tham khảo

- D. Hafner, T. Lillicrap, M. Norouzi, J. Ba, *Mastering Atari with Discrete World Models* (DreamerV2, ICLR 2021, arXiv:2010.02193).
- D. Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (DreamerV1, ICLR 2020, arXiv:1912.01603).
- E. Jang, S. Gu, B. Poole, *Categorical Reparameterization with Gumbel-Softmax* (ICLR 2017, arXiv:1611.01144) — sample categorical khả vi.
