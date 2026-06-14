# EMA Codebook Update

> **TL;DR.** Thay vì cập nhật vector mã bằng gradient của [codebook loss](02-commitment-loss.md), ta cập nhật chúng bằng **exponential moving average**: mỗi mã giữ hai thống kê trượt — số lần được chọn $N_k$ và tổng các encoder output gán cho nó $m_k$ — rồi đặt $e_k = m_k / N_k$. Đây chính xác là **online K-means**: mã luôn nằm ở tâm cụm dữ liệu gán cho nó. Lợi: hội tụ nhanh, độc lập với optimizer/learning rate, ổn định hơn gradient. Caveat: vẫn không tự cứu mã chết, và cần **Laplace smoothing** để tránh chia cho 0.

[Straight-through estimator](01-vector-quantization.md) không gửi gradient tới vector mã, nên VQ-VAE phải có một cơ chế riêng kéo mã về phía dữ liệu. [Commitment loss](02-commitment-loss.md) lo phía encoder; phía codebook có hai lựa chọn: codebook loss bằng gradient, hoặc — cách được dùng phổ biến hơn vì ổn định — **cập nhật EMA**. Mục này giải thích EMA như một thuật toán phân cụm trực tuyến, không phải một thủ thuật rời rạc.

---

## 1. Trực giác: codebook là một bộ K-means online

Mỗi vector mã $e_k$ nên nằm ở **tâm (centroid)** của đám encoder output rơi vào ô Voronoi của nó — đó đúng là điều K-means làm: gán điểm vào cụm gần nhất, rồi dời tâm cụm về trung bình các điểm. Vấn đề là dữ liệu (đầu ra encoder) thay đổi mỗi batch và ta không thấy toàn bộ tập một lúc. EMA giải bài toán này theo kiểu *trực tuyến*: thay vì tính trung bình trên toàn tập, nó duy trì một **trung bình trượt** cập nhật dần qua từng batch, với trọng số suy giảm theo cấp số nhân cho dữ liệu cũ.

Nói cách khác: codebook loss + SGD là cách *từ từ* kéo mã về tâm cụm qua nhiều bước gradient nhỏ; EMA là cách *nhảy thẳng* mã tới ước lượng tâm cụm hiện tại. Thực tế, hai cách tương đương về mặt toán học khi learning rate của SGD trên codebook loss bằng 1.

---

## 2. Cơ chế: hai thống kê trượt

Ở mỗi bước huấn luyện $t$, với batch các encoder output $\{z_j\}$ và assignment $q_j = \arg\min_k \lVert z_j - e_k\rVert$, ta cập nhật cho từng mã $k$:

$$
N_k^{(t)} = \gamma\, N_k^{(t-1)} + (1-\gamma)\sum_j \mathbb{1}[q_j = k],
$$

$$
m_k^{(t)} = \gamma\, m_k^{(t-1)} + (1-\gamma)\sum_j \mathbb{1}[q_j = k]\, z_j,
$$

$$
e_k^{(t)} = \frac{m_k^{(t)}}{N_k^{(t)}}.
$$

Trong đó $\gamma \in (0,1)$ là **decay rate** (thường $0.99$), $N_k$ là số lượng (trượt) các vector từng gán cho mã $k$, $m_k$ là tổng (trượt) các vector đó, và $\mathbb{1}[\cdot]$ là hàm chỉ thị. Vector mã mới $e_k = m_k / N_k$ chính là *trung bình trượt* của các encoder output thuộc cụm $k$ — một bước K-means online.

Điểm tinh tế: cập nhật này **không đi qua đồ thị tính toán** của autograd. Nó chạy song song, độc lập với optimizer của encoder/decoder. Đó là lý do EMA "không phụ thuộc choice of optimizer" — đổi Adam sang SGD không ảnh hưởng cách codebook học.

### Laplace smoothing

Nếu một mã không được chọn trong nhiều batch, $N_k \to 0$ và phép chia $m_k/N_k$ phát nổ. Để tránh, dùng **Laplace (additive) smoothing** trên cluster size:

$$
\hat{N}_k = \frac{N_k + \varepsilon}{N + K\varepsilon}\cdot N, \qquad N = \sum_{k} N_k,
$$

trong đó $\varepsilon$ là hằng nhỏ (ví dụ $10^{-5}$) và $K$ là kích thước codebook. Công thức này kéo các count rất nhỏ lên một sàn dương, giữ phép chia ổn định mà gần như không đổi các count lớn.

---

## 3. EMA vs codebook loss (gradient)

| | Codebook loss + SGD/Adam | Cập nhật EMA |
|---|---|---|
| Cách dời mã | gradient từng bước nhỏ, phụ thuộc learning rate | nhảy thẳng tới centroid trượt |
| Phụ thuộc optimizer | có (Adam moments, lr) | **không** — chạy độc lập |
| Tốc độ hội tụ | chậm hơn | thường nhanh hơn |
| Bộ nhớ | lưu optimizer state cho codebook | chỉ lưu $N_k, m_k$ |
| Số hạng loss tương ứng | $\lVert \mathrm{sg}[z_e] - e\rVert^2$ trong objective | **bỏ** khỏi objective |
| Tương đương | — | $\equiv$ SGD trên codebook loss với lr $=1$ |

Khi dùng EMA, hàm mục tiêu rút gọn còn **reconstruction + commitment** (codebook loss biến mất vì codebook không học bằng gradient nữa):

$$
\mathcal{L}_{\text{EMA}} = \log p(x \mid z_q) + \beta\,\lVert z_e - \mathrm{sg}[e]\rVert_2^2.
$$

Commitment loss vẫn được giữ — EMA chỉ thay phần codebook. Đây là cấu hình mặc định của VQ-VAE-2 và phần lớn các hệ thống VQ hiện đại.

---

## 4. Decay rate $\gamma$ điều khiển gì

$\gamma$ đặt "trí nhớ" của trung bình trượt: window hiệu dụng khoảng $1/(1-\gamma)$ batch. $\gamma = 0.99$ → nhớ ~100 batch gần nhất.

- $\gamma$ **cao** ($\to 1$): mã đổi rất chậm, mượt, ổn định, nhưng *trễ* — chậm đuổi theo encoder khi phân phối của nó dịch (non-stationarity). Trễ quá làm mã tụt lại sau dữ liệu, góp phần [codebook collapse](01-vector-quantization.md).
- $\gamma$ **thấp**: mã đuổi nhanh nhưng nhiễu, dao động theo từng batch — mất chính lợi thế ổn định của EMA.

Giá trị $0.99$ (đôi khi $0.999$ cho batch nhỏ) là điểm cân bằng kinh nghiệm.

---

## 5. Giới hạn / Khi nào thất bại

**Không tự cứu mã chết.** EMA chỉ cập nhật mã *được chọn* (qua $N_k, m_k$). Mã không bao giờ là `argmin` thì $N_k \to 0$, mã đứng yên (sau smoothing) và chết vĩnh viễn — EMA không giải quyết [codebook collapse](01-vector-quantization.md), cần reset/restart riêng (mục **Codebook collapse**).

**Trễ pha khi encoder dịch nhanh.** $\gamma$ cao gây độ trễ; ở đầu huấn luyện khi encoder thay đổi mạnh, mã EMA có thể luôn "đuổi sau" phân phối thật.

**Nhạy với khởi tạo.** Vì EMA là K-means online, khởi tạo tồi (mã ngẫu nhiên xa dữ liệu) khiến nhiều mã không bao giờ được gán ngay từ đầu → chết sớm. Khởi tạo bằng K-means trên batch đầu giảm rủi ro này.

**Cần Laplace smoothing đúng.** Quên smoothing → NaN khi $N_k = 0$; $\varepsilon$ quá lớn → bóp méo centroid của mã ít dùng.

---

## 6. Liên hệ với Latent-Anything

EMA là chi tiết *training-time* của codebook, nhưng dấu vết của nó — phân phối usage $N_k$ — là tín hiệu introspection cực giá trị. Một `ModelAdapter` cho model VQ nên phơi bày thống kê này:

```python
class VQCodebookStats(Protocol):
    cluster_size: np.ndarray   # N_k: usage count (trượt) cho mỗi mã
    codebook: np.ndarray       # (K, D)
    decay: float               # gamma
```

- **Layer A — Introspection**: $N_k$ chính là histogram usage codebook; vẽ nó ra là cách phát hiện mã chết tức thì. Perplexity $= \exp(-\sum_k p_k \log p_k)$ với $p_k = N_k/\sum N_k$ đo "số mã hiệu dụng".
- **Layer B — Manipulation**: hiểu rằng mã là centroid cụm cho phép thao tác có nghĩa — ví dụ gộp hai mã gần nhau bằng cách trung bình có trọng số theo $N_k$.
- **Layer C — Runtime**: EMA không nằm trong forward pass inference, nên Layer C có thể đóng băng codebook khi serve, biến lookup thành phép tra bảng tĩnh, tối ưu được.

EMA ổn định phía codebook; nhưng cả commitment lẫn EMA đều bó tay với mã không được chọn. Mục kế tiếp mổ xẻ đúng vấn đề đó — **codebook collapse** — và các cách hồi sinh mã chết.

---

## Liên quan

- [Vector Quantization](01-vector-quantization.md) — STE không cập nhật mã; EMA là một trong hai cách lấp.
- [Commitment Loss](02-commitment-loss.md) — phía encoder; EMA thay phía codebook, commitment vẫn giữ.
- [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md) — bối cảnh đầy đủ, EMA như online K-means.
- [Markov Property & State Space](../../06-latent-temporal/research/01-markov-property-state-space.md) — EMA là một bộ lọc trượt, họ hàng với ước lượng trạng thái online.

## Tham khảo

- A. van den Oord, O. Vinyals, K. Kavukcuoglu, *Neural Discrete Representation Learning* (NeurIPS 2017, arXiv:1711.00937) — mục appendix mô tả biến thể EMA.
- A. Razavi, A. van den Oord, O. Vinyals, *Generating Diverse High-Fidelity Images with VQ-VAE-2* (NeurIPS 2019, arXiv:1906.00446) — EMA + Laplace smoothing làm mặc định.
- A. Łańcucki et al., *Robust Training of Vector Quantized Bottleneck Models* (IJCNN 2020, arXiv:2005.08520) — phân tích EMA và reset codebook.
