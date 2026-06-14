# Codebook Collapse

> **TL;DR.** Codebook collapse (còn gọi *index collapse*) là khi chỉ một phần nhỏ vector mã từng được `argmin` chọn, phần còn lại thành **mã chết** — không nhận gradient, không được [EMA](03-ema-codebook-update.md) cập nhật, nằm chết vĩnh viễn. Hệ quả: dung lượng thực của từ vựng sụt thảm dù $K$ lớn. Đo bằng **perplexity** $\exp(-\sum_k p_k \log p_k)$ của phân phối usage; cứu bằng **random restart** (reset mã chết về một encoder output gần đây), L2-normalize, giảm chiều mã, tăng batch. Caveat: collapse là vòng luẩn quẩn tự củng cố — phải can thiệp chủ động, không loss nào tự thoát.

[Commitment loss](02-commitment-loss.md) và [EMA](03-ema-codebook-update.md) đều chỉ tác động lên *mã được chọn*. Câu hỏi còn bỏ ngỏ: chuyện gì xảy ra với mã *không bao giờ* được chọn? Câu trả lời là codebook collapse — lỗi phổ biến và nguy hiểm nhất khi huấn luyện mọi model lượng tử hóa. Mục này giải thích cơ chế vòng luẩn quẩn của nó, cách phát hiện, và bộ công cụ cứu chữa.

---

## 1. Trực giác: vòng luẩn quẩn của mã chết

Một mã $e_k$ chỉ được cập nhật khi nó là láng giềng gần nhất của ít nhất một encoder output. Nếu vì lý do nào đó (khởi tạo xa dữ liệu, encoder dịch phân phối) $e_k$ không bao giờ thắng `argmin`, thì:

1. Không encoder output nào gán cho $e_k$ → $e_k$ không nhận gradient (codebook loss) và không được EMA cập nhật.
2. $e_k$ đứng yên trong khi encoder tiếp tục dịch → $e_k$ ngày càng xa vùng dữ liệu.
3. Càng xa, càng không bao giờ thắng `argmin` → quay lại bước 1.

Đây là **vòng phản hồi dương tự củng cố**: chết rồi thì chết luôn. Kết quả là một codebook khai báo $K$ mã nhưng thực dùng $\ll K$ — như một bộ từ vựng 16384 từ mà model chỉ dùng vài trăm.

---

## 2. Nguyên nhân gốc

| Nguyên nhân | Cơ chế |
|---|---|
| **Khởi tạo tồi** | Mã ngẫu nhiên rơi ngoài vùng dữ liệu thua `argmin` ngay từ batch đầu → chết sớm. |
| **Non-stationarity** | Encoder dịch phân phối khi train; mã không được chọn bị bỏ lại sau, không đuổi kịp (xem [EMA decay γ](03-ema-codebook-update.md)). |
| **Bất đối xứng của loss** | Cả commitment lẫn codebook loss đều *mode-seeking*: chỉ kéo mã đã chọn, bỏ mặc mã chưa chọn. |
| **Chiều mã quá cao** | Trong $\mathbb{R}^D$ với $D$ lớn, dữ liệu encoder chiếm thể tích nhỏ; nhiều mã ở "góc" không gian không bao giờ gần dữ liệu. |
| **Decoder quá mạnh** | Decoder có thể dựng lại tốt chỉ từ vài mã → không có áp lực dùng thêm mã. |

---

## 3. Phát hiện: usage histogram và perplexity

Cách đo trực tiếp là đếm usage: với mỗi mã $k$, đếm số token gán cho nó trong một cửa sổ batch gần đây, $N_k$. Mã chết = $N_k$ dưới ngưỡng (thực nghiệm: ngưỡng 1–5 trên cửa sổ ~10 batch hoạt động tốt).

Một chỉ số gộp tiện hơn là **perplexity** của phân phối usage $p_k = N_k / \sum_j N_j$:

$$
\mathrm{PPL} = \exp\!\Big(-\sum_{k=1}^{K} p_k \log p_k\Big).
$$

Trong đó tổng trong ngoặc là entropy Shannon của phân phối usage, và lấy $\exp$ đưa nó về "số mã hiệu dụng". Perplexity nằm trong $[1, K]$: bằng $K$ khi mọi mã dùng đều nhau (lý tưởng), tụt về gần $1$ khi collapse (chỉ một mã thống trị). Theo dõi perplexity trong lúc train là cách báo động collapse sớm nhất — nó tụt trước khi reconstruction xấu đi rõ rệt.

---

## 4. Cứu chữa

### Random restart (reset mã chết) — cách thực dụng nhất

Theo dõi $N_k$; khi một mã ở dưới ngưỡng suốt cửa sổ, **thay nó bằng một encoder output ngẫu nhiên từ batch hiện tại** (hoặc một điểm dữ liệu được chọn). Mã chết lập tức "tái sinh" ngay trong vùng dữ liệu, có cơ hội thắng `argmin` trở lại. Vì lấy từ chính dữ liệu hiện tại, reset không phá vỡ các mã đang sống. Đây là biện pháp được dùng rộng rãi (ví dụ trong Jukebox, nhiều bản VQ-GAN).

### L2-normalize code và latent (cosine lookup)

Ép cả $z_e$ lẫn $e_k$ lên mặt cầu đơn vị biến `argmin` Euclidean thành tối đa hóa cosine, chặn norm của encoder phình và giữ mọi mã trong cùng một vùng đo lường giới hạn — mã khó bị bỏ lại. Đây là cải tiến cốt lõi của ViT-VQGAN; chi tiết toán trong [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md).

### Giảm chiều mã, tăng số mã

Thực nghiệm nhất quán: muốn tăng dung lượng, **tăng $K$ hiệu quả hơn nhiều tăng $D$**. Chiếu latent xuống chiều thấp ($D = 8$–$32$) trước khi lookup giúp mã phủ không gian dày hơn, đẩy usage lên gần tối đa.

### Tăng batch size và khởi tạo K-means

Batch lớn ở đầu train phủ nhiều vùng không gian hơn → kích hoạt nhiều mã hơn, giảm collapse sớm. Khởi tạo codebook bằng K-means trên batch đầu (thay vì ngẫu nhiên) đặt mọi mã trong vùng dữ liệu ngay từ đầu.

### Online clustered codebook và biến thể nâng cao

Các phương pháp gần đây (Zheng & Vedaldi, 2023) gán lại mã chết theo cụm dữ liệu một cách có hệ thống; những hướng khác phân phối một phần gradient cho mã chưa chọn (NS-VQ). Đây là chủ đề nghiên cứu vẫn đang mở.

---

## 5. Giới hạn / Khi nào thất bại

**Không có "viên đạn bạc".** Mỗi cách chữa có cái giá: reset gây gián đoạn (token map nhảy khi mã đột ngột đổi); L2-norm vứt thông tin độ lớn (hại classification — xem [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md)); giảm $D$ giảm sức biểu diễn mỗi mã.

**Reset có thể che triệu chứng.** Liên tục reset giữ perplexity cao *nhân tạo* nhưng nếu nguyên nhân gốc (encoder dịch quá nhanh, $\beta$ sai) không sửa, mã reset rồi lại chết — vòng lặp tốn kém.

**Ngưỡng nhạy.** Ngưỡng usage và độ dài cửa sổ là hyperparameter; quá nhạy thì reset cả mã đang dùng thưa nhưng hữu ích; quá lỏng thì để mã chết lâu.

**FSQ né hẳn vấn đề.** Một hướng triệt để là bỏ codebook học được — **FSQ** (mục sau) lượng tử hóa từng chiều theo lưới cố định, nên về nguyên tắc không có mã chết. Đây là động lực lớn khiến FSQ hấp dẫn.

---

## 6. Liên hệ với Latent-Anything

Codebook collapse biến một giả định ("model có $K$ token để diễn đạt") thành sai lầm âm thầm — đúng loại lỗi mà Layer A sinh ra để phát hiện. Một `ModelAdapter` cho model VQ nên cho phép introspection truy cập usage:

```python
class CodebookHealth(Protocol):
    def usage_counts(self) -> np.ndarray: ...   # N_k qua một cửa sổ
    def perplexity(self) -> float: ...          # số mã hiệu dụng
    def dead_codes(self, threshold: int) -> np.ndarray: ...  # chỉ số mã chết
```

- **Layer A — Introspection**: perplexity và usage histogram là chẩn đoán sức khỏe codebook hạng nhất. Một model VQ load vào framework nên hiển thị ngay "đang dùng bao nhiêu / $K$ mã" — con số này quyết định mọi phân tích token sau đó có ý nghĩa hay không.
- **Layer B — Manipulation**: thao tác token chỉ đáng tin trên các mã *sống*; biết tập mã chết giúp tránh chỉnh token về một mã model thực ra không bao giờ dùng.
- **Layer C — Runtime**: mã chết là dung lượng lãng phí; ở runtime có thể nén codebook (bỏ mã chết, đánh số lại) để giảm bộ nhớ và tăng tốc lookup.

Collapse là giới hạn cuối của VQ một-tầng học-được. Hai mục kế tiếp mở rộng chính phép lượng tử để vượt qua: **Residual VQ** xếp chồng nhiều codebook để tăng độ chính xác, còn **FSQ** bỏ codebook học-được để né collapse hoàn toàn.

---

## Liên quan

- [Vector Quantization](01-vector-quantization.md) — cơ chế `argmin` + STE; collapse sinh ra từ chính việc STE chỉ cập nhật mã được chọn.
- [Commitment Loss](02-commitment-loss.md) — mode-seeking, một nguồn của bất đối xứng gây collapse.
- [EMA Codebook Update](03-ema-codebook-update.md) — chỉ cập nhật mã được chọn; không tự cứu mã chết.
- [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md) — phân tích collapse và các cách chữa chi tiết (L2-norm, affine reparam).

## Tham khảo

- C. Zheng, A. Vedaldi, *Online Clustered Codebook* (ICCV 2023, arXiv:2307.15139) — gán lại mã chết theo cụm, phân tích collapse.
- A. Łańcucki et al., *Robust Training of Vector Quantized Bottleneck Models* (IJCNN 2020, arXiv:2005.08520) — reset codebook bằng thống kê usage.
- P. Dhariwal et al., *Jukebox: A Generative Model for Music* (2020, arXiv:2005.00341) — random restart cho mã chết ở quy mô lớn.
- J. Yu et al., *Vector-quantized Image Modeling with Improved VQGAN* (ICLR 2022, arXiv:2110.04627) — L2-normalized codes chống collapse.
