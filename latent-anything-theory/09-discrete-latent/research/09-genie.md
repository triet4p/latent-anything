# Genie (Google DeepMind, 2024)

> **TL;DR.** Genie là world model tương tác (~11B tham số) học **hoàn toàn không giám sát từ video Internet không nhãn** — không hề có nhãn hành động. Nó gồm ba khối: (1) **spatiotemporal video tokenizer** (VQ) biến video thành token, (2) **Latent Action Model (LAM)** học một *bộ hành động rời rạc nhỏ* (khoảng 8) bằng cách suy ra "đã làm gì giữa hai frame" qua VQ, và (3) **dynamics model** tự hồi quy dự đoán token frame kế cho trước token quá khứ + latent action. Đột phá: tính *điều khiển được theo từng frame* nổi lên từ video thuần, không cần nhãn. Caveat: latent action không khớp 1-1 hành động người, độ phân giải/độ dài còn hạn chế, và tốn kém.

[GAIA-1](08-gaia-1.md) điều khiển world model bằng *nhãn* hành động và text. Genie (Bruce et al., 2024) hỏi câu táo bạo hơn: nếu chỉ có video không nhãn — kho YouTube khổng lồ — liệu có thể *học luôn* cả không gian hành động? Câu trả lời là có, và chìa khóa lại chính là [vector quantization](01-vector-quantization.md) — lần này áp lên *hành động*, không phải observation. Đây là cao trào của tầng 9: discrete latent không chỉ token hóa thế giới mà còn token hóa *cách tác động* lên nó.

---

## 1. Trực giác: học hành động mà không có nhãn hành động

Vấn đề cốt lõi của world model điều khiển được: cần biết hành động $a_t$ giữa các frame để học $p(x_{t+1}\mid x_t, a_t)$. Nhưng video Internet *không có* nhãn hành động. Genie lách qua bằng quan sát: dù không biết tên hành động, *sự khác biệt giữa hai frame liên tiếp* mã hóa "đã xảy ra gì". Nếu nén sự khác biệt đó thành một **latent action rời rạc** (qua VQ), ta được một bộ hành động học-được, nhất quán — và vì nó rời rạc với từ vựng nhỏ, người dùng có thể chọn từng giá trị để *điều khiển* thế giới sinh ra.

Đây là một dạng inverse dynamics: từ $(x_t, x_{t+1})$ suy ra latent action $\tilde a_t$, rồi ép dynamics model dùng đúng $\tilde a_t$ đó để dự đoán $x_{t+1}$. VQ làm action space *nhỏ và rời rạc*, buộc nó nắm các *chế độ thay đổi* chính (sang trái, lên, nhảy...) thay vì copy nguyên frame sau.

---

## 2. Cơ chế: ba khối

### Khối 1 — Spatiotemporal video tokenizer

Một VQ-VAE với **spatiotemporal transformer** (ST-transformer: attention tách rời theo không gian và thời gian) nén video thành lưới token rời rạc, có tính tới ngữ cảnh thời gian — token không chỉ mô tả một frame tĩnh mà cả chuyển động cục bộ.

### Khối 2 — Latent Action Model (LAM)

Trái tim của Genie. LAM nhận chuỗi frame và *suy ra* một latent action giữa mỗi cặp frame liên tiếp:

$$
\tilde a_t = \mathrm{VQ}\big(f_{\text{inv}}(x_{\le t}, x_{t+1})\big), \qquad \tilde a_t \in \{1, \dots, |\mathcal A|\},
$$

trong đó $f_{\text{inv}}$ là một encoder inverse-dynamics nhìn frame hiện tại (và quá khứ) cùng frame *tương lai* $x_{t+1}$ để đoán "đã làm gì", và VQ ép kết quả về một trong $|\mathcal A|$ hành động rời rạc (Genie dùng $|\mathcal A| = 8$). LAM được train cùng một decoder dự đoán $x_{t+1}$ từ $x_t$ và $\tilde a_t$: vì latent action phải *đủ* để tái tạo frame sau, nó buộc phải mang thông tin điều khiển. Vì từ vựng nhỏ (bottleneck VQ), nó không thể "gian lận" bằng cách nhồi cả frame sau — chỉ giữ được vài chế độ hành động chính.

Lúc inference, $x_{t+1}$ chưa có; người dùng *cung cấp* latent action (chọn 1 trong 8) — đó là cách điều khiển thế giới từng frame.

### Khối 3 — Dynamics model

Một transformer tự hồi quy (kiểu MaskGIT — dự đoán token theo kiểu masked, không thuần causal) nhận token quá khứ + latent action và dự đoán token frame kế. Đây là [tokenized world model](07-tokenized-world-model.md) lõi, chỉ khác là *điều kiện hóa thêm bằng latent action học-được*.

---

## 3. Vì sao đây là đỉnh của tầng 9

Genie dùng VQ ở *hai* nơi với hai mục đích khác nhau:

| Nơi dùng VQ | Token hóa cái gì | Để làm gì |
|---|---|---|
| Video tokenizer | observation (frame) | đưa thế giới vào dạng sequence model |
| Latent Action Model | *transition* giữa hai frame | học bộ hành động rời rạc điều khiển được |

Lần thứ hai là cái mới: rời rạc hóa *hành động* biến một bài toán cần nhãn (học điều khiển) thành bài toán không giám sát. Bottleneck VQ nhỏ chính là thứ ép latent action trở nên *có nghĩa và nhất quán* — đúng tinh thần [information bottleneck](../../02-representation-learning/research/01-information-bottleneck.md): ép qua cổ hẹp thì chỉ thông tin điều khiển quan trọng nhất sống sót.

Hệ quả lớn: latent action space học-được cho phép **huấn luyện agent bắt chước hành vi từ video chưa thấy** — một bước về phía generalist agent. Đây là cầu trực tiếp tới [latent action của VLA](https://github.com/triet4p/latent-anything/blob/main/docs/THEORY.md) (tầng bổ sung).

---

## 4. Giới hạn / Khi nào thất bại

**Latent action không khớp hành động người.** 8 latent action là các *chế độ thay đổi* do model tự chia, không nhất thiết tương ứng nút bấm hay lệnh ngữ nghĩa; ánh xạ sang điều khiển người cần hiệu chỉnh.

**Độ dài và độ phân giải hạn chế.** Genie (bản 2024) sinh ở độ phân giải thấp, ít frame; rollout dài vẫn drift (compounding error) như mọi [tokenized world model](07-tokenized-world-model.md).

**Bị chặn bởi tokenizer.** Như GAIA-1, chất lượng video bị chặn bởi video tokenizer; chi tiết mất ở lượng tử không thể khôi phục.

**Chi phí lớn.** 11B tham số, train trên lượng video khổng lồ; inference tương tác realtime là thách thức.

**Phụ thuộc tính nhất quán của video.** LAM giả định khác biệt giữa frame *là* do một hành động; với video cắt cảnh, nhiều agent, hay camera nhảy, latent action có thể nhiễu loạn.

---

## 5. Liên hệ với Latent-Anything

Genie là ví dụ mạnh nhất cho luận điểm của framework: latent space là *first-class object* — ở đây cả *state* lẫn *action* đều là latent rời rạc, học-được, thao tác được. Một adapter Genie-style phơi bày hai mặt latent đó:

```python
class GenieStyleAdapter(Protocol):
    def tokenize(self, video: np.ndarray) -> np.ndarray: ...            # frame -> tokens
    def infer_action(self, x_t: np.ndarray, x_next: np.ndarray) -> int: ...  # latent action (LAM)
    def step(self, tokens: np.ndarray, latent_action: int) -> np.ndarray: ... # next tokens
    num_latent_actions: int   # |A|, vd 8
```

- **Layer A — Introspection**: latent action space là đối tượng vàng để soi — clustering hành vi, xem mỗi latent action "làm gì" (decode một bước dưới mỗi action), đo entropy hành động. Đây là probing/interpretability ([tầng 5](../../05-probing-intervention/research/01-linear-probing.md)) áp lên *hành động học-được*.
- **Layer B — Manipulation**: chọn/chuỗi latent action chính là manipulation định hướng cao nhất — "lái" thế giới sinh ra; đây là steering ở mức world model, mở rộng [steering vectors](../../05-probing-intervention/research/09-steering-vectors.md) sang không gian hành động.
- **Layer C — Runtime**: tương tác từng frame đòi hỏi inference latency thấp; Layer C cần lập lịch sinh token tăng dần (incremental) với latent action người dùng nạp realtime.

Genie khép lại tầng 9 đúng nơi roadmap nhắm tới: *"sau tầng 9 → có thể implement adapter cho tokenized world model."* Discrete latent giờ phủ cả observation lẫn action, và pipeline tokenize → reason → control đã đầy đủ — sẵn sàng nối sang các world model & VLA quy mô lớn ở tầng bổ sung.

---

## Liên quan

- [Tokenized World Model](07-tokenized-world-model.md) — dynamics model của Genie là một tokenized WM có điều kiện hóa latent action.
- [GAIA-1](08-gaia-1.md) — cùng họ world model token; GAIA-1 điều khiển bằng nhãn, Genie học action không giám sát.
- [Vector Quantization](01-vector-quantization.md) — VQ là cơ chế tạo cả token observation lẫn latent action rời rạc.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) — bottleneck VQ nhỏ ép latent action mang đúng thông tin điều khiển.
- [Latent Transition Model](../../06-latent-temporal/research/02-latent-transition-model.md) — Genie là transition model có action học-được.

## Tham khảo

- J. Bruce et al., *Genie: Generative Interactive Environments* (ICML 2024, arXiv:2402.15391).
- H. Chang et al., *MaskGIT: Masked Generative Image Transformer* (CVPR 2022, arXiv:2202.04200) — kiểu dự đoán token của dynamics model.
- V. Micheli, E. Alonso, F. Fleuret, *Transformers are Sample-Efficient World Models* (IRIS, ICLR 2023, arXiv:2209.00588) — tokenized WM nền tảng.
- A. van den Oord, O. Vinyals, K. Kavukcuoglu, *Neural Discrete Representation Learning* (VQ-VAE, NeurIPS 2017, arXiv:1711.00937).
