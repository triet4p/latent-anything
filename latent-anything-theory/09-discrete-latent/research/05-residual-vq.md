# Residual Vector Quantization (RVQ)

> **TL;DR.** RVQ xếp chồng $N_q$ tầng [vector quantization](01-vector-quantization.md): tầng đầu lượng tử hóa $z$ thô, mỗi tầng sau lượng tử hóa *phần dư (residual)* mà tầng trước bỏ sót. Vector cuối là **tổng** các mã: $\hat z = \sum_{i=1}^{N_q} e^{(i)}$. Với $N_q$ codebook cỡ $K$, không gian biểu diễn là $K^{N_q}$ tổ hợp nhưng chỉ tốn $N_q \cdot K$ tham số — biểu cảm theo cấp số nhân, tham số tuyến tính. Là xương sống của codec âm thanh thần kinh (SoundStream, EnCodec). Caveat: lỗi tích lũy qua tầng, tầng sau khó train hơn, và phải xử lý collapse ở từng tầng.

[Codebook collapse](04-codebook-collapse.md) cho thấy một codebook đơn không thể vừa nhỏ vừa mịn: muốn giảm [lỗi lượng tử](01-vector-quantization.md) phải tăng $K$ theo cấp số nhân, mà $K$ lớn lại dễ collapse. RVQ phá thế lưỡng nan này bằng một ý tưởng kinh điển từ nén tín hiệu: thay vì một codebook khổng lồ, dùng *nhiều* codebook nhỏ, mỗi cái sửa sai cho cái trước.

---

## 1. Trực giác: sửa sai theo tầng

Hình dung lượng tử hóa $z$ như làm tròn. Một codebook đơn làm tròn $z$ về mã gần nhất $e^{(1)}$, để lại phần dư $r_1 = z - e^{(1)}$. Phần dư này nhỏ hơn $z$ nhưng không bằng 0. RVQ nói: *hãy lượng tử hóa luôn phần dư đó* bằng một codebook thứ hai → được $e^{(2)}$, để lại dư nhỏ hơn nữa $r_2 = r_1 - e^{(2)}$. Lặp lại $N_q$ lần.

Mỗi tầng "phóng to" vào sai số còn lại và mô tả nó mịn hơn. Kết quả là một sơ đồ **coarse-to-fine** tự nhiên: tầng đầu nắm cấu trúc thô, các tầng sau thêm dần chi tiết — giống cách JPEG progressive hiện ảnh từ mờ tới nét, hay cách dãy Taylor thêm dần số hạng bậc cao.

---

## 2. Cơ chế: thuật toán và biểu cảm

Cho vector encoder $z$, đặt $r_0 = z$. Với mỗi tầng $i = 1, \dots, N_q$:

$$
e^{(i)} = \mathrm{VQ}_i(r_{i-1}), \qquad r_i = r_{i-1} - e^{(i)}, \qquad \hat z = \sum_{i=1}^{N_q} e^{(i)}.
$$

Trong đó $\mathrm{VQ}_i$ là phép lượng tử hóa nearest-neighbor với codebook riêng của tầng $i$, $e^{(i)}$ là mã tầng $i$ chọn cho phần dư hiện tại, $r_i$ là phần dư sau tầng $i$, và $\hat z$ là xấp xỉ cuối (tổng tích lũy). Token đầu ra là **bộ $N_q$ chỉ số** $(k_1, \dots, k_{N_q})$ — một stack token cho mỗi vị trí, không phải một token đơn.

**Biểu cảm theo cấp số nhân.** Mỗi tầng có $K$ lựa chọn độc lập, nên tổng số vector $\hat z$ biểu diễn được là $K^{N_q}$. So sánh chi phí:

| | VQ đơn tương đương | RVQ |
|---|---|---|
| Số vector biểu diễn | $K^{N_q}$ | $K^{N_q}$ |
| Số tham số codebook | $K^{N_q}\cdot D$ | $N_q \cdot K \cdot D$ |
| Bit mỗi vị trí | $N_q \log_2 K$ | $N_q \log_2 K$ |

Để đạt $K^{N_q}$ vector, codebook đơn cần $K^{N_q}$ entry (bất khả thi); RVQ chỉ cần $N_q K$ entry. Đây là lý do RVQ thống trị nén âm thanh: SoundStream/EnCodec dùng $N_q = 8$–$32$ codebook cỡ $1024$, đạt độ phân giải mà codebook đơn không thể.

### Phân bổ rate và quantizer dropout

Ngân sách bit thường chia đều: $r_i = r / N_q = \log_2 K$ bit mỗi tầng. Vì các tầng theo thứ tự coarse→fine, **số tầng dùng khi giải mã quyết định bitrate**: dùng 2/4/8/16 tầng đầu cho ra chất lượng (và bitrate) tăng dần. SoundStream khai thác điều này bằng **quantizer dropout** — lúc train, ngẫu nhiên cắt bớt tầng sau — để *một* model phục vụ nhiều bitrate, chọn lúc inference. Đây là tính chất *scalable* rất mạnh: train một lần, triển khai ở nhiều mức nén.

---

## 3. Vì sao tầng sau khó train hơn

Phần dư co lại nhanh qua mỗi tầng: $\lVert r_i\rVert$ thường giảm gần như cấp số nhân. Hệ quả là phân phối đầu vào của các tầng sau ngày càng nhỏ và tập trung quanh 0, khó phủ bằng codebook → tầng sau dễ [collapse](04-codebook-collapse.md) hơn tầng đầu. Thực tế phải dùng EMA + restart *độc lập cho từng tầng*, và đôi khi chuẩn hóa lại residual giữa các tầng.

Một biến thể giảm tải: **Group-RVQ / product quantization** chia vector thành nhóm chiều rồi RVQ từng nhóm, để mỗi codebook chỉ lo một phần không gian.

---

## 4. Giới hạn / Khi nào thất bại

**Lỗi tích lũy và phụ thuộc tuần tự.** Mỗi tầng phụ thuộc đầu ra tầng trước; lượng tử hóa thô ở tầng đầu giới hạn cận trên cho mọi tầng sau (greedy, không tối ưu toàn cục). Bộ mã $(k_1,\dots,k_{N_q})$ tối ưu greedy không nhất thiết là tổ hợp tốt nhất.

**Token nở ra.** Mỗi vị trí giờ là $N_q$ token thay vì một. Khi đưa vào model tự hồi quy ([tokenized world model](01-vector-quantization.md), mục sau), chuỗi dài gấp $N_q$ lần, hoặc cần kiến trúc đặc biệt (ví dụ RQ-Transformer dự đoán cả stack token tại một bước).

**Collapse nhân lên.** Có $N_q$ codebook nghĩa là $N_q$ cơ hội collapse; tầng sau đặc biệt rủi ro vì residual nhỏ.

**Không sửa được hướng sai của tầng trước.** RVQ chỉ *cộng thêm*; nếu tầng đầu chọn mã sai hẳn vùng, các tầng sau chỉ vá phần dư quanh lựa chọn sai đó.

---

## 5. Liên hệ với Latent-Anything

RVQ là một `Quantizer` ghép tầng — vẫn vào vector, ra token, nhưng token giờ là một *stack*. Layer B nên mô hình hóa nó như một toán tử lượng tử nhiều mức:

```python
class ResidualQuantizer(Protocol):
    n_stages: int                 # N_q
    codebooks: list[np.ndarray]   # mỗi phần tử (K, D)
    def quantize(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    # trả (z_hat, indices) với indices shape (..., n_stages)
```

- **Layer A — Introspection**: cấu trúc coarse-to-fine cho phép phân tích *theo mức* — tầng đầu mã hóa cái gì (cấu trúc thô) vs tầng cuối (chi tiết). Đo norm residual $\lVert r_i\rVert$ theo tầng là chẩn đoán mỗi tầng đóng góp bao nhiêu.
- **Layer B — Manipulation**: bitrate-scalable nghĩa là Layer B có thể *cắt bớt tầng* để nén/làm mượt latent (bỏ token chi tiết, giữ token thô) — một dạng manipulation có ngữ nghĩa "độ phân giải".
- **Layer C — Runtime**: quantizer dropout cho phép runtime chọn số tầng theo ngân sách tính toán/băng thông — một núm đánh đổi chất lượng vs chi phí ngay tại inference.

RVQ mở rộng *độ chính xác* của lượng tử hóa bằng cách xếp chồng. Mục kế tiếp đi hướng ngược lại — đơn giản hóa triệt để: **FSQ** bỏ luôn codebook học-được, lượng tử hóa từng chiều theo lưới cố định.

---

## Liên quan

- [Vector Quantization](01-vector-quantization.md) — RVQ là VQ lặp trên phần dư; mỗi tầng là một VQ.
- [Codebook Collapse](04-codebook-collapse.md) — RVQ có $N_q$ codebook nên $N_q$ rủi ro collapse; tầng sau nặng nhất.
- [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md) — bối cảnh codebook học-được, EMA dùng cho từng tầng.
- [Latent Trajectory](../../06-latent-temporal/research/06-latent-trajectory.md) — codec âm thanh RVQ sinh chuỗi token theo thời gian, gần với trajectory token.

## Tham khảo

- N. Zeghidour et al., *SoundStream: An End-to-End Neural Audio Codec* (IEEE/ACM TASLP 2021, arXiv:2107.03312) — RVQ + quantizer dropout.
- A. Défossez et al., *High Fidelity Neural Audio Compression* (EnCodec, 2022, arXiv:2210.13438) — RVQ quy mô lớn cho audio.
- D. Lee et al., *Autoregressive Image Generation using Residual Quantization* (RQ-VAE/RQ-Transformer, CVPR 2022, arXiv:2203.01941) — RVQ cho ảnh + dự đoán stack token.
- D. Yang et al., *HiFi-Codec: Group-residual Vector Quantization for High Fidelity Audio Codec* (2023, arXiv:2305.02765) — Group-RVQ.
