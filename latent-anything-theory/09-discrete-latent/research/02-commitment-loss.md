# Commitment Loss

> **TL;DR.** Vì [straight-through estimator](01-vector-quantization.md) không cập nhật chính vector mã, hàm mất mát VQ phải tách làm hai số hạng đối xứng-một-nửa: **codebook loss** $\lVert \mathrm{sg}[z_e] - e\rVert^2$ kéo mã về phía encoder, và **commitment loss** $\beta\lVert z_e - \mathrm{sg}[e]\rVert^2$ kéo encoder về phía mã đã chọn. Commitment loss ép encoder *cam kết* với assignment của mình, ngăn $z_e$ phình norm vô hạn và nhảy loạn giữa các mã. Caveat: hệ số $\beta$ phải đủ lớn để giữ ổn định nhưng đủ nhỏ để không bóp nghẹt encoder — bài gốc chọn $\beta = 0.25$ và báo cáo kết quả khá bền với $\beta \in [0.1, 2.0]$.

[Vector quantization](01-vector-quantization.md) để lại một lỗ hổng: STE dán gradient của decoder thẳng lên $z_e$ nhưng *không hề chạm tới* vector mã $e$, và cũng không ràng buộc $z_e$ phải ở gần mã nào. Nếu để vậy, hai phía — encoder và codebook — trôi tự do và không cùng tốc độ, khiến lượng tử hóa mất ý nghĩa. Commitment loss (cùng codebook loss) là cặp số hạng vá đúng lỗ hổng đó: chúng buộc encoder và codebook *gặp nhau ở giữa*.

---

## 1. Trực giác: hai phía phải tìm nhau

Hình dung $z_e$ (đầu ra encoder) và $e_{k^\star}$ (mã gần nhất) là hai điểm. Quá trình lượng tử hóa thay $z_e$ bằng $e_{k^\star}$, nên reconstruction tốt đòi hỏi *khoảng cách giữa chúng nhỏ*. Có hai cách thu hẹp khoảng cách đó, và VQ-VAE dùng **cả hai cùng lúc nhưng tách biệt nhờ stop-gradient**:

- Kéo **mã** về phía encoder: đây là việc của *codebook loss* (hoặc thay bằng [cập nhật EMA](01-vector-quantization.md) — mục **EMA codebook update**).
- Kéo **encoder** về phía mã: đây là việc của *commitment loss*.

Tại sao phải tách? Nếu để một số hạng $\lVert z_e - e\rVert^2$ duy nhất cho gradient chảy về *cả hai*, hai phía sẽ đuổi nhau và có thể không bao giờ ổn định (giống hai người cùng bước sang một bên để tránh nhau). Stop-gradient đóng băng một phía trong mỗi số hạng, tách bài toán thành hai lực một chiều có kiểm soát.

---

## 2. Cơ chế: hàm mất mát ba thành phần

Hàm mục tiêu đầy đủ của VQ-VAE (van den Oord et al., 2017):

$$
\mathcal{L} = \underbrace{\log p(x \mid z_q)}_{\text{reconstruction}} \;+\; \underbrace{\lVert \mathrm{sg}[z_e] - e\rVert_2^2}_{\text{codebook loss}} \;+\; \underbrace{\beta\,\lVert z_e - \mathrm{sg}[e]\rVert_2^2}_{\text{commitment loss}}.
$$

Trong đó $z_e$ là đầu ra encoder, $e = e_{k^\star}$ là mã được chọn, $\mathrm{sg}[\cdot]$ là stop-gradient, và $\beta$ là hệ số cân bằng. Đọc từng số hạng:

- **Reconstruction loss** — huấn luyện encoder + decoder để dựng lại $x$ từ $z_q$; gradient tới encoder đi qua STE.
- **Codebook loss** $\lVert \mathrm{sg}[z_e] - e\rVert^2$ — $\mathrm{sg}$ đóng băng $z_e$, nên gradient chỉ chảy vào $e$: *mã* dịch về phía trung bình các encoder output gán cho nó (đúng là một bước K-means).
- **Commitment loss** $\beta\lVert z_e - \mathrm{sg}[e]\rVert^2$ — $\mathrm{sg}$ đóng băng $e$, nên gradient chỉ chảy vào $z_e$: *encoder* bị kéo lại gần mã nó đã chọn.

Điểm bất đối xứng then chốt nằm ở **chỉ commitment loss có hệ số $\beta$**. Lý do: vector mã chỉ chịu tác động của codebook loss, còn $z_e$ chịu tác động của *cả* reconstruction loss *lẫn* commitment loss. Để hai lực tới $z_e$ không lấn át nhau, $\beta$ thường nhỏ ($0.25$) — encoder vẫn ưu tiên dựng lại tốt, commitment chỉ là dây cương giữ nó không trôi xa.

---

## 3. Vì sao thiếu commitment loss thì hỏng

Nếu bỏ commitment loss, encoder không có lý do gì để giữ $z_e$ gần codebook. Hai hệ quả:

| Triệu chứng | Nguyên nhân |
|---|---|
| **Norm của $z_e$ phình to** | Encoder tự do bơm độ lớn; vì STE chỉ truyền hướng gradient của decoder, không có lực cản nào ghì $z_e$ lại trong vùng codebook. |
| **Assignment dao động (oscillation)** | $z_e$ trôi qua lại giữa các ô Voronoi sát nhau, token nhảy loạn giữa các batch → huấn luyện bất ổn, codebook khó hội tụ. |

Commitment loss khử cả hai: nó là một lò xo kéo $z_e$ về mã đã chọn, vừa chặn norm phình, vừa "khóa" assignment để token ổn định qua các bước. Tên gọi *commitment* phản ánh đúng vai trò — buộc encoder **cam kết** với lựa chọn rời rạc của nó thay vì đứng nước đôi giữa hai mã.

---

## 4. Quan hệ với codebook update và $\beta$

Trong thực tế, codebook loss thường bị thay bằng [cập nhật EMA](01-vector-quantization.md) (xem mục **EMA codebook update**) vì ổn định hơn. Khi đó hàm loss chỉ còn **reconstruction + commitment**:

$$
\mathcal{L}_{\text{EMA}} = \log p(x \mid z_q) + \beta\,\lVert z_e - \mathrm{sg}[e]\rVert_2^2,
$$

và codebook được cập nhật riêng bằng trung bình trượt. Điểm cần nhớ: **commitment loss luôn được giữ lại** dù dùng gradient hay EMA cho codebook — nó là số hạng phía-encoder, độc lập với cách codebook học.

Về $\beta$: bài gốc báo cáo kết quả ổn định trong khoảng rộng $[0.1, 2.0]$ và chốt $0.25$. $\beta$ quá nhỏ → encoder trôi, mất ổn định; $\beta$ quá lớn → encoder bị ghì quá chặt vào codebook hiện tại, giảm khả năng khám phá biểu diễn tốt hơn và có thể làm trầm trọng [codebook collapse](01-vector-quantization.md) (mục **Codebook collapse**).

---

## 5. Giới hạn / Khi nào thất bại

**Không tự cứu được mã chết.** Commitment loss chỉ tác động lên *mã đã được chọn*; mã không bao giờ là `argmin` thì không nhận gradient từ cả commitment lẫn codebook loss — đây chính là cơ chế sinh ra codebook collapse, và commitment loss không giải quyết được nó.

**Nhạy với $\beta$ khi đổi domain.** Giá trị $0.25$ tốt cho ảnh tự nhiên không đảm bảo tối ưu cho audio, time-series hay latent của model khác; cần dò lại.

**Không phải lực toàn cục.** Commitment kéo $z_e$ về mã *gần nhất hiện tại*, không về mã *tốt nhất có thể*; nếu assignment ban đầu kém (codebook khởi tạo tồi), commitment có thể khóa encoder vào một cấu hình dưới tối ưu.

**Trùng vai khi đã chuẩn hóa L2.** Khi cả $z_e$ lẫn $e$ bị ép lên mặt cầu đơn vị (cosine lookup), norm của $z_e$ đã bị chặn sẵn, nên một phần tác dụng của commitment loss bị thừa — cần cân chỉnh lại $\beta$.

---

## 6. Liên hệ với Latent-Anything

Commitment loss là một thành phần của *training objective*, không phải toán tử inference — nên nó nằm ở tầng "đặc tả model" mà `ModelAdapter` cần khai báo để Layer A diễn giải đúng. Một adapter VQ nên phơi bày các hệ số loss để introspection biết model được huấn luyện thế nào:

```python
class VQConfig(Protocol):
    codebook_size: int        # K
    code_dim: int             # D
    commitment_beta: float    # beta trong commitment loss
    codebook_update: str      # "gradient" | "ema"
```

- **Layer A — Introspection**: theo dõi norm trung bình của $z_e$ và khoảng cách $\lVert z_e - e_{k^\star}\rVert$ qua thời gian là một chẩn đoán trực tiếp xem commitment có "ăn" hay không — norm phình hoặc khoảng cách lớn báo hiệu $\beta$ quá nhỏ.
- **Layer B — Manipulation**: hiểu rằng encoder đã *cam kết* với mã giúp việc thao tác token an toàn hơn — đổi token một bước sẽ không bị encoder kéo ngược lại bất ngờ.
- **Layer C — Runtime**: assignment ổn định (nhờ commitment) làm cho cache token và lập chỉ mục đáng tin cậy giữa các lần chạy.

Commitment loss khóa phía encoder; mục kế tiếp khóa phía còn lại — **EMA codebook update** — để cả hai phía cùng hội tụ ổn định.

---

## Liên quan

- [Vector Quantization](01-vector-quantization.md) — STE để lại lỗ hổng mà commitment + codebook loss vá; định nghĩa $\mathrm{sg}[\cdot]$.
- [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md) — hàm mất mát ba thành phần đầy đủ trong bối cảnh generative model.
- [VAE](../../02-representation-learning/research/03-vae.md) — đối chiếu vai trò regularizer: KL trong VAE vs commitment trong VQ-VAE.

## Tham khảo

- A. van den Oord, O. Vinyals, K. Kavukcuoglu, *Neural Discrete Representation Learning* (NeurIPS 2017, arXiv:1711.00937) — định nghĩa commitment loss, $\beta = 0.25$.
- A. Razavi, A. van den Oord, O. Vinyals, *Generating Diverse High-Fidelity Images with VQ-VAE-2* (NeurIPS 2019, arXiv:1906.00446) — chi tiết training với EMA + commitment.
