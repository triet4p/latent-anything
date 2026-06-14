# Vector Quantization (VQ)

> **TL;DR.** Vector quantization là phép biến một vector liên tục $z_e$ thành một **chỉ số rời rạc** bằng cách tìm láng giềng gần nhất trong một bảng mã (codebook) $\{e_k\}$: forward dùng `argmin` khoảng cách (gán cứng), backward dùng **straight-through estimator** $z_q = z_e + \mathrm{sg}[e_k - z_e]$ để gradient vẫn chảy ngược về encoder. Đây là nguyên thủy biến latent space thành *chuỗi token*, mở đường unify với kiến trúc language model. Caveat: gradient qua STE là **chệch (biased)**, và bảng mã dễ rơi vào trạng thái chết hàng loạt (codebook collapse).

Tầng 9 đặt câu hỏi: nếu muốn dùng chính bộ máy của language model — Transformer tự hồi quy trên chuỗi token — để model hóa latent, thì latent phải *rời rạc hóa* trước. Vector quantization là viên gạch đầu tiên: nó là toán tử biến một lưới vector liên tục thành một lưới **token nguyên**, để phần còn lại của pipeline coi observation y hệt một câu văn. Mục này tách VQ ra khỏi bối cảnh [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md) (nơi nó lần đầu xuất hiện) và nhìn nó như một **lớp toán tử độc lập** — đầu vào vector, đầu ra chỉ số — vì đó chính là cách Layer B của Latent-Anything sẽ phơi bày nó.

---

## 1. Trực giác / Định nghĩa

Quantization (lượng tử hóa) nói chung là thay một đại lượng liên tục bằng phần tử gần nhất trong một tập hữu hạn các giá trị mẫu. *Scalar* quantization làm điều này trên từng số (làm tròn về mức gần nhất); *vector* quantization làm điều này trên cả vector cùng lúc — gần nhất theo một metric trong $\mathbb{R}^D$.

Cụ thể, ta giữ một **codebook** $E = \{e_1, \dots, e_K\}$ gồm $K$ vector mã, mỗi vector chiều $D$. Toán tử VQ nhận một vector encoder $z_e \in \mathbb{R}^D$ và trả về:

$$
k^\star = \arg\min_{j \in \{1,\dots,K\}} \lVert z_e - e_j \rVert_2, \qquad z_q = e_{k^\star}.
$$

Trong đó $k^\star$ là **token** (một số nguyên trong $\{1,\dots,K\}$) và $z_q$ là vector mã được chọn để truyền tiếp cho decoder. Với một ảnh, encoder thường xuất một lưới $H' \times W'$ vector, nên VQ biến ảnh thành một *lưới token* $H' \times W'$ — đúng dạng mà một Transformer có thể duyệt như chuỗi. Đây là sự khác biệt cốt lõi với latent liên tục: thay vì một điểm trong $\mathbb{R}^D$, mỗi vị trí giờ là một *ký hiệu* lấy từ bộ từ vựng $K$ phần tử.

Hai siêu tham số định hình mọi thứ: $K$ (kích thước từ vựng, quyết định bao nhiêu bit/token $= \log_2 K$) và $D$ (chiều của mỗi mã). Thực nghiệm cho thấy $K$ lớn + $D$ nhỏ thường tốt hơn $K$ nhỏ + $D$ lớn — lý do nằm ở chỗ codebook collapse (mục riêng).

---

## 2. Cơ chế forward: gán cứng bằng argmin

Bước forward chỉ là một nearest-neighbor lookup. Khai triển bình phương khoảng cách cho thấy có thể tính nhanh bằng đại số ma trận:

$$
\lVert z_e - e_j \rVert_2^2 = \lVert z_e \rVert_2^2 - 2\, z_e^\top e_j + \lVert e_j \rVert_2^2.
$$

Trong đó số hạng $\lVert z_e\rVert^2$ không đổi theo $j$ nên không ảnh hưởng `argmin`; phần $-2 z_e^\top e_j + \lVert e_j\rVert^2$ tính được bằng một phép nhân ma trận $Z E^\top$ cộng norm của codebook — cho toàn bộ lưới token cùng lúc. Kết quả là một phép gán **cứng**: mỗi $z_e$ rơi vào đúng một ô Voronoi của codebook, và toàn không gian $\mathbb{R}^D$ bị phân hoạch thành $K$ vùng.

Hệ quả quan trọng của gán cứng: VQ là một **information bottleneck rời rạc** với *rate* $\log_2 K$ bit mỗi token. Vì lượng thông tin qua cổ chai bị chặn cứng (không phụ thuộc decoder), VQ-VAE không bị [posterior collapse](../../02-representation-learning/research/03-vae.md) như VAE liên tục — decoder không thể "phớt lờ" latent vì latent là kênh thông tin duy nhất và hữu hạn.

---

## 3. Cơ chế backward: straight-through estimator

`argmin` là hàm bậc thang: đạo hàm bằng 0 gần như mọi nơi và không xác định tại biên Voronoi. Nếu để nguyên, gradient không thể chảy từ decoder về encoder và encoder không học được gì. Lời giải là **straight-through estimator (STE)**: ở forward dùng giá trị đã lượng tử $z_q$, nhưng ở backward *giả vờ* phép lượng tử là hàm đồng nhất, sao chép thẳng gradient từ $z_q$ về $z_e$. Thủ thuật cài đặt kinh điển:

$$
z_q = z_e + \mathrm{sg}[\,e_{k^\star} - z_e\,],
$$

trong đó $\mathrm{sg}[\cdot]$ là stop-gradient (forward trả giá trị bên trong, backward trả gradient 0). Forward: $z_q = e_{k^\star}$ (vì $\mathrm{sg}$ là identity ở forward). Backward: $\partial z_q / \partial z_e = 1$ (vì số hạng trong $\mathrm{sg}$ có gradient 0). Nhờ vậy $\nabla_{z_e} \mathcal{L} = \nabla_{z_q}\mathcal{L}$ — gradient của decoder được "dán" thẳng lên encoder.

Phải nhấn mạnh: ước lượng này **chệch (biased)**. Ta thực sự đang dùng gradient tại $z_q$ để cập nhật điểm $z_e$ ở vị trí khác, bỏ qua hình học của ranh giới Voronoi. Thực tế nó vẫn hoạt động tốt, nhưng đây là gốc rễ của nhiều bất ổn — và là lý do tồn tại của hai cơ chế bổ trợ ngay sau đây:

- **Commitment loss** ép $z_e$ không trôi quá xa mã đã chọn (mục **Commitment loss**, kế tiếp trong tầng).
- Vì STE không cập nhật chính các vector mã, ta cần một cơ chế riêng kéo $e_k$ về phía dữ liệu: hoặc một codebook loss bằng gradient, hoặc **cập nhật EMA** (mục **EMA codebook update**).

---

## 4. Vì sao VQ là cầu nối tới language model

Một khi mỗi vị trí latent là một token nguyên trong $\{1,\dots,K\}$, mọi công cụ của mô hình ngôn ngữ áp dụng được nguyên xi:

| | Latent liên tục | Latent rời rạc (sau VQ) |
|---|---|---|
| Đơn vị | vector $\in \mathbb{R}^D$ | token $\in \{1,\dots,K\}$ |
| Model phân phối | density (flow, diffusion) | **categorical / softmax** như LM |
| Sinh tự hồi quy | khó (không có "từ tiếp theo") | $p(\text{token}_{t} \mid \text{token}_{<t})$ — y hệt văn bản |
| Đo độ giống | khoảng cách hình học | so khớp token / cross-entropy |
| Kiến trúc tái dùng | hạn chế | Transformer, PixelCNN, mọi seq model |

Đây chính là nền tảng để [VQGAN](../../02-representation-learning/research/05-vqgan.md) + Transformer sinh ảnh, và là tiền đề của **tokenized world model** (mục sau trong tầng): encode observation → chuỗi token → Transformer model dynamics trên token. VQ là chỗ "rời rạc hóa" biến giấc mơ unify-with-LM thành hiện thực.

---

## 5. Giới hạn / Khi nào thất bại

**Gradient chệch.** STE bỏ qua ranh giới quyết định; với codebook lớn hoặc dữ liệu nhiều mode, hướng cập nhật encoder có thể lệch và làm chậm hội tụ.

**Codebook collapse.** Phần lớn mã có thể không bao giờ được `argmin` chọn → "mã chết", dung lượng thực của từ vựng sụt thảm. Đây là lỗi phổ biến nhất của VQ và có hẳn một mục riêng (**Codebook collapse**).

**Lỗi lượng tử (quantization error).** Mỗi vector bị thay bằng mã gần nhất, sai số $\lVert z_e - e_{k^\star}\rVert$ là cận dưới của reconstruction error. Một tầng VQ đơn không thể nén mịn; muốn giảm sai số mà không tăng $K$ theo cấp số nhân, cần xếp chồng nhiều tầng — đó là **Residual VQ** (mục sau).

**Phụ thuộc metric và scale.** `argmin` theo Euclidean nhạy với độ lớn vector; nếu encoder bơm norm lớn, lookup mất ý nghĩa. Nhiều biến thể chuẩn hóa L2 cả $z_e$ lẫn $e_k$ (lookup theo cosine) để ổn định — xem phần ViT-VQGAN trong [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md).

**Không khả vi thật.** STE chỉ là xấp xỉ; khi cần gradient đúng (ví dụ tối ưu bậc hai), VQ là rào cản. Đây là động lực cho **FSQ** (mục sau) — bỏ codebook học được, lượng tử hóa từng chiều nên cấu trúc đơn giản hơn.

---

## 6. Liên hệ với Latent-Anything

VQ là một **toán tử Layer B** thuần túy: vào vector, ra token + vector mã. Tách khỏi VQ-VAE, nó là một `Method` độc lập mà Layer B có thể áp lên bất kỳ latent space nào:

```python
class Quantizer(Protocol):
    def quantize(self, z_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    # trả (z_q, indices): z_q là vector mã, indices là token nguyên
    codebook: np.ndarray   # shape (K, D)
```

- **Layer A — Introspection**: vì đầu ra là token, Layer A đo được trực tiếp *codebook usage* (histogram token), perplexity của từ vựng, và phát hiện mã chết — những chẩn đoán bất khả thi với latent liên tục.
- **Layer B — Manipulation**: thao tác trên token rời rạc (đổi một token, mask-rồi-điền) là một dạng manipulation định hướng và *diễn giải được* hơn nhiều so với cộng vector trong không gian liên tục.
- **Layer C — Runtime**: token nguyên nén tốt (chỉ $\log_2 K$ bit) và lập chỉ mục nhanh; lookup là một phép nhân ma trận + argmax, dễ batch và đưa xuống kernel tối ưu.

VQ mở tầng 9: từ đây latent là *ký hiệu*, và các mục kế tiếp lần lượt vá những lỗ hổng của nó — commitment loss giữ encoder cam kết, EMA giữ codebook bám dữ liệu, rồi RVQ và FSQ mở rộng/đơn giản hóa chính phép lượng tử.

---

## Liên quan

- [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md) — model đầu tiên dùng VQ; chứa chi tiết loss đầy đủ và bối cảnh generative.
- [VQGAN](../../02-representation-learning/research/05-vqgan.md) — VQ + perceptual + adversarial loss; token VQ làm đầu vào cho Transformer sinh ảnh.
- [VAE](../../02-representation-learning/research/03-vae.md) — đối chiếu latent liên tục vs rời rạc; VQ né posterior collapse nhờ bottleneck cứng.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — chuẩn hóa L2 đưa code lên mặt siêu cầu, ổn định lookup.

## Tham khảo

- A. van den Oord, O. Vinyals, K. Kavukcuoglu, *Neural Discrete Representation Learning* (NeurIPS 2017, arXiv:1711.00937) — VQ-VAE gốc, định nghĩa VQ + STE.
- Y. Bengio, N. Léonard, A. Courville, *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation* (2013, arXiv:1308.3432) — nền tảng straight-through estimator.
- A. Razavi, A. van den Oord, O. Vinyals, *Generating Diverse High-Fidelity Images with VQ-VAE-2* (NeurIPS 2019, arXiv:1906.00446) — VQ phân cấp.
- J. Yu et al., *Vector-quantized Image Modeling with Improved VQGAN* (ICLR 2022, arXiv:2110.04627) — L2-normalized codes, cosine lookup.
