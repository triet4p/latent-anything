# Value Equivalence (MuZero)

> **TL;DR.** Value equivalence principle: một model *không cần* reconstruct observation — nó chỉ cần dự đoán đúng value/policy/reward để phục vụ planning. MuZero hiện thực hóa điều này bằng ba hàm (representation $h$, dynamics $g$, prediction $f$) trên một *abstract latent state* được học **không có decoder, không có reconstruction loss**; latent tồn tại *chỉ để* serve planning. Caveat: abstract state không tương ứng state thật và không decode được, nên cực khó audit; nó tối ưu cho một reward/task và transfer kém.

Mọi latent world model tới giờ — PlaNet, [Dreamer](07-policy-gradient-imagined-dreamer.md) — đều học model một phần bằng **reconstruction**: có decoder, có pixel loss, latent bị ép giữ đủ thông tin để dựng lại observation. Nhưng phần lớn pixel (texture, nền, noise) chẳng liên quan gì tới quyết định. Value equivalence là một lập luận triệt để: nếu mục tiêu cuối là planning, hãy dồn toàn bộ capacity của model vào *thứ phục vụ planning* — value, policy, reward — và vứt phần còn lại.

---

## **1. Trực giác / Định nghĩa**

Hai model là **value-equivalent** đối với một tập policy $\Pi$ và một tập hàm value $\mathcal{V}$ nếu chúng tạo ra **cùng Bellman updates**:

$$
\hat{\mathcal{T}}^\pi v=\mathcal{T}^\pi v\quad\forall\,\pi\in\Pi,\ v\in\mathcal{V},
$$

trong đó $\mathcal{T}^\pi v=r^\pi+\gamma P^\pi v$ là Bellman operator của environment thật và $\hat{\mathcal{T}}^\pi$ của model. Nghĩa là: ta không đòi model khớp environment ở mức transition/observation, chỉ đòi nó cho *cùng kết quả khi backup value*. Tài nguyên biểu diễn hữu hạn của agent nên dồn vào dựng model trực tiếp hữu ích cho value-based planning, thay vì model environment chính xác hoàn hảo.

Trực giác: hai bản đồ thành phố khác nhau hoàn toàn về hình vẽ nhưng cho *cùng thời gian di chuyển* giữa mọi cặp điểm thì "value-equivalent" cho bài toán tìm đường — và bản đồ nào bỏ được chi tiết thừa thì rẻ hơn để học và dùng.

---

## **2. Kiến trúc MuZero**

MuZero gồm ba mạng, không cái nào dự đoán observation:

$$
\underbrace{s^0=h_\theta(o_{\le t})}_{\text{representation}},\qquad \underbrace{s^{k+1},\,\hat r^{k+1}=g_\theta(s^k,a^k)}_{\text{dynamics}},\qquad \underbrace{\hat p^k,\,\hat v^k=f_\theta(s^k)}_{\text{prediction}}.
$$

Trong đó $h$ encode chuỗi observation thành abstract latent $s^0$; $g$ nhận latent + action, trả latent kế và reward; $f$ đọc latent ra policy $\hat p$ và value $\hat v$. **Không có decoder $s\to o$**. Abstract state $s^k$ là *học được*, không phải latent environment state hay observation prediction — nó chỉ mang ý nghĩa qua các đại lượng $f,g$ đọc ra.

### Huấn luyện (không reconstruction)

Unroll model $K$ bước từ $s^0$ rồi khớp ba đầu ra với target:

$$
\mathcal{L}=\sum_{k=0}^{K}\Big[\underbrace{\ell^r(\hat r^k,u_{t+k})}_{\text{reward}}+\underbrace{\ell^v(\hat v^k,z_{t+k})}_{\text{value}}+\underbrace{\ell^p(\hat p^k,\pi_{t+k})}_{\text{policy}}\Big].
$$

Trong đó $u$ là reward thật quan sát, $z$ là n-step return (bootstrap value target), và $\pi$ là policy cải thiện từ **MCTS (mục sau)**. Gradient chảy ngược qua chuỗi dynamics $g$ (giống BPTT). Đây chính là value-equivalent model: dynamics được train để dự đoán *future value*, không phải future observation.

---

## **3. Vì sao bỏ reconstruction lại tốt**

| | Reconstruction-based (PlaNet, Dreamer) | Value-equivalent (MuZero) |
|---|---|---|
| Decoder $s\to o$ | có | không |
| Loss định hình latent | pixel reconstruction + reward/value | reward + value + policy |
| Ý nghĩa latent | xấp xỉ state sinh observation | abstract, chỉ định nghĩa qua $f,g$ |
| Capacity dồn vào | tái dựng cả observation | chỉ thứ ảnh hưởng return |
| Audit (decode) | được | không |

Ràng buộc duy nhất lên latent là *functional*: dự đoán đúng value/policy/reward. Model được tự do **vứt mọi thứ ở mức observation không ảnh hưởng return** — đúng phần (texture, noise, chi tiết nền) làm pixel-prediction tốn capacity và sample. Đây là lý do value equivalence giải thích thành công của một loạt phương pháp: Value Iteration Networks, Predictron, Value Prediction Networks, TreeQN, và MuZero.

### Proper value equivalence

Khi $|\Pi|,|\mathcal{V}|$ tăng, tập các value-equivalent model **co lại**, cuối cùng thu về một điểm = perfect model. Câu hỏi mấu chốt (Grimm 2021, *Proper Value Equivalence*): chọn tập policy/function **nhỏ nhất đủ** cho planning. Quá nhỏ → model degenerate; vừa đủ → model rẻ mà vẫn plan đúng.

---

## **4. Giới hạn / Khi nào thất bại**

**Abstract state không decode/audit được.** Không decoder nghĩa là không biết model "tưởng tượng" cái gì. Phân tích cho thấy abstract state MuZero học được có thể không tương ứng state thật mà vẫn plan tốt — mạnh về planning nhưng mờ về diễn giải. Đây là cái giá trực tiếp Layer A phải trả.

**Tối ưu cho một reward/task.** Latent chỉ giữ thứ liên quan reward *hiện tại*; đổi reward/task thì abstract state có thể thiếu thông tin — transfer kém hơn model generative.

**Phụ thuộc chất lượng target.** Value/policy target đến từ MCTS và n-step return; nếu search hoặc bootstrap tệ, model học sai "thế nào là value đúng".

**Tập $\Pi,\mathcal{V}$ quá nhỏ → degenerate.** VE chỉ ràng buộc trên các function/policy được chọn; ngoài đó model tự do, có thể collapse hoặc sai một cách vô hại-trong-training nhưng hại khi gặp policy mới.

**Không có reconstruction để bắt lỗi.** Reconstruction là một kênh phát hiện model đi hoang (decoded trajectory phi lý). Bỏ nó đi làm planning nhanh nhưng mất một lớp an toàn — cùng trade-off như [decoder-free rollout](../../06-latent-temporal/research/07-rollout-latent-imagination.md).

---

## **5. Liên hệ với Latent-Anything**

Value equivalence là một tuyên bố sâu về **latent dùng để làm gì**. Latent-Anything coi latent là first-class object; VE nói latent *không nhất thiết phải generative* — nó có thể thuần túy functional. Điều này định hình `ModelAdapter`:

```python
class ModelAdapter(Protocol):
    def representation(self, obs) -> np.ndarray: ...        # h: o -> s0
    def dynamics(self, s, a) -> tuple[np.ndarray, float]: ... # g: (s,a) -> (s', r̂)
    def prediction(self, s) -> tuple[np.ndarray, float]: ...  # f: s -> (p̂, v̂)
    decodable: bool   # có decoder để audit không? VE model: False
```

- **Layer A — Introspection**: với VE model `decodable=False`, Layer A *không* dựa vào reconstruction; thay vào đó audit gián tiếp — so predicted value/reward với realized, đo consistency của abstract state qua các unroll, phát hiện degenerate representation. Đây là thách thức introspection mà framework phải xử lý rõ ràng.
- **Layer B — Manipulation**: chỉnh sửa abstract latent ảnh hưởng value/policy theo cách *không thể* kiểm bằng decode — provenance và functional checks thay cho visual checks.
- **Layer C — Runtime**: VE model rẻ nhất cho planning thuần (không chạy decoder); runtime nên lộ `decodable` để chọn giữa generative (audit được) và value-equivalent (nhanh).

Value equivalence cũng là cầu sang **Tầng 8** (predict trong latent, không decode) và **Tầng 9** (discrete latent): cả hai đẩy xa hơn ý "latent không cần dựng lại observation". Mục tiếp theo, **MCTS trong latent**, là planner mà MuZero dùng để *vừa* tạo target policy *vừa* hành động — search trên chính abstract dynamics này.

---

## Liên quan

- [Value Function trong Latent](03-value-function-in-latent.md) — value mà VE model phải dự đoán đúng; Bellman update định nghĩa value equivalence.
- [Reward Model trong Latent](02-reward-model-in-latent.md) — reward head, một trong ba đầu ra của prediction/dynamics.
- [Policy Gradient trên Imagined Trajectory (Dreamer)](07-policy-gradient-imagined-dreamer.md) — đối lực reconstruction-based; VE bỏ decoder.
- [Model-based vs Model-free RL](01-model-based-vs-model-free-rl.md) — VE là cách dùng capacity model hiệu quả nhất cho planning.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — value-equivalent imagination, decoder-free audit.

## Tham khảo

- J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, D. Silver, *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero, Nature 2020, arXiv:1911.08265).
- C. Grimm, A. Barreto, S. Singh, D. Silver, *The Value Equivalence Principle for Model-Based Reinforcement Learning* (NeurIPS 2020, arXiv:2011.03506).
- C. Grimm, A. Barreto, G. Farquhar, D. Silver, S. Singh, *Proper Value Equivalence* (NeurIPS 2021, arXiv:2106.10316).
- J. Oh, S. Singh, H. Lee, *Value Prediction Network* (NeurIPS 2017, arXiv:1707.03497).
- D. Silver, H. van Hasselt, M. Hessel, T. Schaul, A. Guez, et al., *The Predictron: End-to-End Learning and Planning* (ICML 2017, arXiv:1612.08810).
