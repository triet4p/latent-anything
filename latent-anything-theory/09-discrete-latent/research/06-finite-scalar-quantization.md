# Finite Scalar Quantization (FSQ)

> **TL;DR.** FSQ bỏ hẳn codebook học-được: chiếu latent xuống *vài chiều* ($d \lesssim 10$), bound mỗi chiều bằng một hàm dạng $\tanh$, rồi **làm tròn từng chiều về $L$ mức cố định**. Codebook là *ngầm định* — tích Descartes của các mức, $|\mathcal{C}| = \prod_i L_i$ — không cần lưu, không cần học. Vì mọi mức luôn được dùng, FSQ **không có [codebook collapse](04-codebook-collapse.md)** và không cần commitment loss, EMA hay reset. Đổi lại chỉ mất ~0.5–3% chất lượng so với VQ. Caveat: số chiều phải nhỏ (codebook nở theo cấp số nhân), và lưới đều không thích nghi với hình dạng dữ liệu như codebook học-được.

[Codebook collapse](04-codebook-collapse.md) lộ ra rằng phần lớn bộ máy của VQ — commitment loss, EMA, restart, L2-norm — chỉ để *giữ cho codebook học-được khỏi chết*. FSQ (Mentzer et al., 2023) đặt câu hỏi triệt để: nếu codebook *không học* mà là một lưới cố định, thì những vấn đề đó biến mất luôn. Tiêu đề bài báo nói thẳng: *"VQ-VAE Made Simple"*.

---

## 1. Trực giác: làm tròn từng chiều thay vì tra bảng

VQ lượng tử hóa cả *vector* bằng cách tìm láng giềng gần nhất trong một bảng học-được. FSQ làm điều đơn giản hơn nhiều: lượng tử hóa từng *scalar* độc lập bằng cách làm tròn về mức gần nhất trên một lưới đều — đúng nghĩa "lượng tử hóa vô hướng" như ADC trong xử lý tín hiệu.

Mấu chốt: nếu mỗi chiều chỉ nhận $L$ giá trị rời rạc và ta có $d$ chiều, thì số tổ hợp là $L^d$ — một "codebook" khổng lồ mà *không cần lưu một entry nào*. Codebook tồn tại ngầm định trong cấu trúc lưới. Vì mọi điểm lưới đều là một mã hợp lệ và encoder bị ép phủ toàn lưới, không có khái niệm "mã chết".

---

## 2. Cơ chế: bound rồi round (qua STE)

Cho một chiều latent $z \in \mathbb{R}$, muốn lượng tử về $L$ mức. FSQ làm hai bước:

$$
f(z) = \Big\lfloor \tfrac{L}{2} \Big\rfloor \tanh(z), \qquad \hat z = \mathrm{round}(f(z)).
$$

Trong đó $\tanh$ **bound** giá trị vào khoảng hữu hạn, $\lfloor L/2\rfloor$ co giãn để biên độ phủ đúng $L$ mức nguyên, và $\mathrm{round}$ làm tròn về số nguyên gần nhất. Kết quả $\hat z$ nằm trong tập $\{-\lfloor L/2\rfloor, \dots, \lfloor L/2\rfloor\}$ — đúng $L$ mức. Áp dụng độc lập cho cả $d$ chiều với số mức $L_i$ tùy chiều, được vector lượng tử $\hat{\mathbf z}$.

`round` không khả vi (đạo hàm 0 gần như mọi nơi), nên — y như [VQ](01-vector-quantization.md) — FSQ dùng **straight-through estimator**: forward dùng $\mathrm{round}$, backward coi nó là identity, $\hat z = f(z) + \mathrm{sg}[\mathrm{round}(f(z)) - f(z)]$. Đây là *nơi duy nhất* gradient bị xấp xỉ; không có commitment loss, không có codebook loss.

### Token từ chỉ số mức (mixed-radix)

Mỗi chiều cho một chỉ số mức nguyên; gộp $d$ chỉ số thành một token duy nhất bằng mã hóa cơ số hỗn hợp:

$$
\text{token} = \sum_{i=1}^{d} a_i \prod_{j<i} L_j,
$$

trong đó $a_i \in \{0,\dots,L_i-1\}$ là chỉ số mức của chiều $i$. Token chạy từ $0$ tới $\prod_i L_i - 1$ — chính là kích thước codebook ngầm. Ví dụ bài báo: mức $[8,5,5,5]$ cho $8\cdot5\cdot5\cdot5 = 1000$ mã.

---

## 3. FSQ vs VQ: đánh đổi gì

| | VQ (codebook học-được) | FSQ (lưới cố định) |
|---|---|---|
| Codebook | lưu tường minh $K\times D$ tham số | **ngầm định**, 0 tham số |
| Loss phụ | commitment + codebook/EMA | **không** — chỉ reconstruction |
| Codebook collapse | có, cần restart/L2-norm | **không thể xảy ra** |
| Bước không khả vi | `argmin` (STE) | `round` (STE) |
| Chiều latent | trung bình–cao ($D$ vài chục–trăm) | **rất nhỏ** ($d \lesssim 10$) |
| Thích nghi hình dữ liệu | có (mã đặt theo cụm) | không (lưới đều) |
| Chất lượng | chuẩn | thấp hơn ~0.5–3% |
| Codebook usage | thường $\ll 100\%$ | gần $100\%$ |

FSQ đổi *sự thích nghi* lấy *sự đơn giản và ổn định*. Trên các tác vụ sinh ảnh/nén với codebook lớn, mức mất mát nhỏ thường đáng giá vì loại bỏ toàn bộ rủi ro collapse và đống hyperparameter đi kèm.

---

## 4. Vì sao FSQ không collapse

Trong VQ, một mã chết khi không encoder output nào gán cho nó. Trong FSQ điều này *không có nghĩa*: mọi điểm lưới đều "tồn tại" bất kể có được chọn hay không, và quan trọng hơn, $\tanh$ bound ép encoder phải dùng toàn miền giá trị — không có chiều nào "trôi ra vô cực" rồi bỏ rơi các mức. Thực nghiệm cho thấy FSQ đạt codebook usage gần 100% mà không cần can thiệp, trong khi VQ phải chật vật với restart và entropy penalty để tới gần đó.

Đánh đổi đi kèm: vì codebook nở theo $\prod_i L_i$ (cấp số nhân theo $d$), $d$ phải nhỏ — FSQ chỉ thực tế cho codebook cỡ vừa (vài nghìn–vài chục nghìn). Codebook rất lớn ($10^6+$) vẫn là sân của VQ/RVQ.

---

## 5. Giới hạn / Khi nào thất bại

**Lưới đều không thích nghi.** Codebook học-được đặt mã dày ở vùng dữ liệu đông; lưới FSQ rải đều, phí mã ở vùng trống và thiếu phân giải ở vùng đông — nguồn gốc của khoảng 0.5–3% chất lượng bị mất.

**Số chiều bị giới hạn cứng.** $|\mathcal C| = \prod L_i$ buộc $d$ nhỏ; không mở rộng codebook tùy ý như tăng $K$ trong VQ. Muốn vừa lớn vừa mịn phải ghép FSQ với residual (Residual-FSQ) — một hướng mới đang được khám phá.

**Vẫn dùng STE.** `round` không khả vi nên gradient vẫn chệch như VQ; FSQ đơn giản hóa codebook, không loại bỏ xấp xỉ STE.

**Cần encoder đủ mạnh.** Vì latent bị ép xuống rất ít chiều, gánh nặng dồn lên encoder/decoder; với encoder yếu, bottleneck $d$ nhỏ có thể quá chặt.

---

## 6. Liên hệ với Latent-Anything

FSQ là một `Quantizer` đặc biệt *không trạng thái học-được* — toàn bộ định nghĩa của nó là vector số mức $[L_1,\dots,L_d]$. Điều này làm nó cực kỳ thân thiện với introspection và runtime:

```python
class FSQuantizer(Protocol):
    levels: list[int]              # [L_1, ..., L_d]
    def quantize(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    # trả (z_hat, token) với token in [0, prod(levels))
    # KHÔNG có thuộc tính codebook học-được
```

- **Layer A — Introspection**: không có codebook để audit usage, nhưng *từng chiều* có ngữ nghĩa riêng và số mức rõ ràng — Layer A có thể soi phân phối mức trên mỗi chiều, gần với phân tích disentanglement hơn là histogram token. Không bao giờ phải lo "bao nhiêu mã chết".
- **Layer B — Manipulation**: lưới đều, có thứ tự nghĩa là token có *cấu trúc metric* — tăng/giảm một mức trên một chiều là một bước nhỏ có hướng trong latent, thao tác sạch hơn token VQ (vốn không có thứ tự nội tại).
- **Layer C — Runtime**: 0 tham số codebook, lookup chỉ là `tanh`+`round`+mã hóa cơ số — không cần tra bảng, không cache codebook; lý tưởng để hạ xuống kernel và port sang ngôn ngữ khác.

FSQ khép lại nửa "cơ chế lượng tử" của tầng 9. Từ mục kế tiếp, ta dùng *kết quả* — chuỗi token rời rạc — làm đầu vào cho model động lực: **tokenized world model**, rồi các hệ thống quy mô lớn GAIA-1 và Genie.

---

## Liên quan

- [Vector Quantization](01-vector-quantization.md) — FSQ thay `argmin` trên codebook học-được bằng `round` trên lưới cố định.
- [Commitment Loss](02-commitment-loss.md) — FSQ loại bỏ hoàn toàn số hạng này.
- [Codebook Collapse](04-codebook-collapse.md) — vấn đề mà FSQ né được bằng thiết kế.
- [Residual VQ](05-residual-vq.md) — hướng tăng độ phân giải đối lập; có thể ghép thành Residual-FSQ.

## Tham khảo

- F. Mentzer, D. Minnen, E. Agustsson, M. Tschannen, *Finite Scalar Quantization: VQ-VAE Made Simple* (ICLR 2024, arXiv:2309.15505).
- A. van den Oord, O. Vinyals, K. Kavukcuoglu, *Neural Discrete Representation Learning* (NeurIPS 2017, arXiv:1711.00937) — VQ gốc để đối chiếu.
- Y. Bengio, N. Léonard, A. Courville, *Estimating or Propagating Gradients Through Stochastic Neurons* (2013, arXiv:1308.3432) — straight-through estimator dùng cho `round`.
