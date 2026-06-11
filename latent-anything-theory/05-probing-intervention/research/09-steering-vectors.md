# Steering Vectors

> **TL;DR.** Steering vector là một **hướng** trích từ activation tương phản — thường là *hiệu trung bình* giữa ví dụ "có" và "không có" một hành vi — rồi **cộng thẳng vào residual stream** lúc inference để đẩy output theo hướng mong muốn: $h \leftarrow h + \alpha v$. Đơn giản đến mức không cần training, không cần dữ liệu task, chỉ một forward qua tập calibration (Zou et al., 2023; Rimsky et al., 2024). Caveat chính: hệ số $\alpha$ cố định gây under/over-steering tùy ngữ cảnh, hiệu ứng *không ổn định* giữa các hành vi (đôi khi phản tác dụng), và $\alpha$ lớn làm sụp đổ chất lượng sinh (lặp từ, off-manifold).

Tầng 5 đến đây đã có đủ công cụ *đọc* latent: [probe](01-linear-probing.md) tìm hướng decode được, [TCAV](03-tcav.md) tìm hướng khái niệm lái class, [SAE](07-sparse-autoencoder.md) tìm dictionary feature mono-semantic. Steering vector là mặt *ghi*: lấy một hướng đó và **cộng vào** latent để thay đổi hành vi. Đây là primitive manipulation đơn giản và mạnh nhất của Layer B — hiện thực hóa cụ thể của [can thiệp nhân quả theo hướng](04-causal-intervention-vs-observational.md), và khép vòng read→write của cả tầng.

---

## **1. Trực giác / Định nghĩa**

Hình dung latent space có một "thanh trượt trung thực": di chuyển theo một hướng cố định làm output trung thực hơn, đi ngược làm dối hơn. Steering vector *là* thanh trượt đó. Tìm nó cực rẻ: lấy nhiều cặp prompt chỉ khác nhau ở một trục ngữ nghĩa (trung thực vs dối, vui vs buồn, có chủ đề X vs không), chạy model, lấy activation ở một layer, rồi tính **hiệu trung bình** giữa nhóm dương và nhóm âm. Hiệu đó trỏ từ "không có khái niệm" sang "có khái niệm" — chính là hướng cần đẩy.

Dùng nó cũng cực rẻ: tại inference, ở mỗi vị trí token (sau prompt), **cộng** $\alpha v$ vào activation residual stream. $\alpha > 0$ tăng khái niệm, $\alpha < 0$ giảm. Không gradient, không fine-tune, không decoder — chỉ một phép cộng vector.

Điểm cốt lõi nối với cả tầng: steering vector *cùng loại đối tượng* với concept direction của probe/CAV/SAE — đều là một hướng tuyến tính trong latent. Khác biệt là **mục đích dùng**: probe/CAV/SAE *đọc* (phân loại, decompose), steering vector *ghi* (cộng vào để chỉnh). Cùng một vector $v$ có thể vừa đọc vừa ghi — đây là sự thống nhất read↔write mà [linear probing](01-linear-probing.md) đã hé lộ.

---

## **2. Cơ chế / Công thức**

### 2.1 Trích steering vector — difference of means (CAA)

Cho tập cặp tương phản: prompt dương $P$ (thể hiện hành vi) và âm $N$ (không). Chạy model, lấy activation tại layer $\ell$ và một vị trí token (thường là token cuối). Steering vector là hiệu hai trung bình:

$$ v_\ell = \frac{1}{|P|}\sum_{x \in P} a_\ell(x) \;-\; \frac{1}{|N|}\sum_{x \in N} a_\ell(x) $$

trong đó $a_\ell(x) \in \mathbb{R}^{d}$ là activation của input $x$ tại layer $\ell$, và $v_\ell$ là hiệu của hai mean. $v_\ell$ trỏ từ tâm cụm âm sang tâm cụm dương — ước lượng hướng mà khái niệm biến thiên. Đây là phương pháp **Contrastive Activation Addition (CAA)** của Rimsky et al. (2024); nó là một ước lượng *Fisher direction không whitening* (LDA bỏ qua covariance).

### 2.2 Áp dụng — activation addition

Tại inference, can thiệp vào forward pass bằng cách cộng vector đã chuẩn hóa:

$$ a_\ell \leftarrow a_\ell + \alpha\, \hat{v}_\ell, \qquad \hat{v}_\ell = \frac{v_\ell}{\lVert v_\ell\rVert} $$

trong đó $\hat{v}_\ell$ là hướng đơn vị, $\alpha$ là **hệ số steering** (độ mạnh, dấu quyết định chiều). Cộng ở mọi vị trí token sau prompt, tại một (hoặc vài) layer. $\alpha$ lớn → đẩy mạnh nhưng rủi ro sụp đổ; $\alpha$ nhỏ → an toàn nhưng yếu. Đây là can thiệp $do(\cdot)$ theo hướng đúng như [note nhân quả](04-causal-intervention-vs-observational.md) mô tả: $z' = z + \alpha\hat{v}$.

### 2.3 Quan hệ với latent arithmetic

Steering chính là [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) với một concept direction: $a + \alpha(\bar{a}_\text{pos} - \bar{a}_\text{neg})$ là phép "cộng khái niệm" $z_\text{king} - z_\text{man} + z_\text{woman}$ tổng quát hóa. Difference-of-means chính là cách *ước lượng* vector hiệu từ nhiều mẫu thay vì một cặp đơn lẻ — robust hơn nhờ trung bình hóa nhiễu.

---

## **3. Biến thể — cách trích hướng**

| Phương pháp | Cách trích hướng | Đặc điểm |
|---|---|---|
| **CAA** (Rimsky 2024) | Mean diff trên nhiều cặp tương phản | Robust nhờ trung bình; chuẩn hiện hành |
| **ActAdd** (Turner 2023) | Hiệu một cặp prompt đơn (vd "Love"−"Hate") | Cực rẻ; nhiễu hơn, phụ thuộc cặp prompt |
| **RepE reading vector** (Zou 2023) | PCA trên tập hiệu activation tương phản | Bắt hướng phương sai chính; khung tổng quát |
| **Probe / CAV weight** | Pháp tuyến linear classifier ([probe](01-linear-probing.md), [TCAV](03-tcav.md)) | Hướng *phân biệt* tối ưu thay vì hiệu mean |
| **SAE feature** | Cột $W_\text{dec}$ của một feature ([SAE](07-sparse-autoencoder.md)) | Mono-semantic, ít leak hơn |

Mọi phương pháp đều cho ra *một hướng* để cộng vào latent — khác nhau ở cách ước lượng. Mean-difference rẻ và mạnh đáng ngạc nhiên; probe/CAV cho hướng phân biệt tốt hơn khi cụm chồng lấn; SAE feature sạch nhất nhưng cần train SAE trước.

So với [activation patching](05-activation-patching.md): patching *tráo* activation thật từ một run khác (interchange, giá trị từ phân phối thật); steering *cộng* một hướng nhân tạo cố định. Patching localize "ở đâu"; steering chỉnh "theo hướng nào" — bổ sung nhau.

---

## **4. Giới hạn / Khi nào thất bại**

**Không ổn định giữa các hành vi.** Tan et al. (2024) cho thấy hiệu ứng steering *biến thiên mạnh* giữa các hành vi và thường *thiếu tin cậy* hoặc *phản tác dụng* — một vector hoạt động tốt cho hành vi này có thể vô dụng cho hành vi khác. Steering không phải công cụ đáng tin phổ quát; phải đánh giá từng trường hợp.

**Hệ số $\alpha$ cố định.** Rimsky et al. chỉ ra dùng một $\alpha$ cố định cho mọi prompt và vị trí token gây **under-steering** (yếu) hoặc **over-steering** (quá đà) tùy input. Nó cũng *bỏ qua* việc model đã thể hiện sẵn bao nhiêu hành vi đó trong ngữ cảnh — kết quả sau steering không hoàn toàn xác định bởi $\alpha$.

**Sụp đổ chất lượng sinh.** Tăng $\alpha$ để đẩy mạnh hơn làm **giảm chất lượng output**: lặp từ, văn bản vô nghĩa, collapse. Có một cửa sổ $\alpha$ hẹp giữa "không đủ tác dụng" và "hỏng output".

**Off-manifold ở $\alpha$ lớn.** Cộng $\alpha\hat{v}$ lớn đẩy activation ra ngoài manifold model từng thấy — cùng vấn đề off-manifold của [can thiệp](04-causal-intervention-vs-observational.md). Output khi đó là artifact của vùng latent vô nghĩa, không phải biểu hiện sạch của khái niệm.

**Giả định tuyến tính + leak do superposition.** Steering giả định khái niệm là *một hướng*. Nếu khái niệm mã hóa phi tuyến, hoặc nằm trong [superposition](06-superposition-hypothesis.md) (gần trực giao với feature khác), cộng theo $\hat{v}$ sẽ *rò* sang feature lân cận (interference) — đẩy khái niệm mục tiêu kèm tác dụng phụ. Đây là lý do SAE feature (mono-semantic) steer sạch hơn mean-diff thô.

**Phụ thuộc tập calibration và layer.** Hướng $v$ đổi theo cặp prompt tương phản, layer chọn, và vị trí token. Thiết kế tập calibration kém cho hướng lệch; chọn sai layer cho hiệu ứng yếu.

---

## **5. Liên hệ với Latent-Anything**

Steering vector là **primitive manipulation cốt lõi của Layer B**, và là điểm hội tụ read→write của toàn tầng 5:

- **`LatentSpace.steer(z, direction, alpha, layer=ℓ)`** → thực thi $a_\ell \leftarrow a_\ell + \alpha\hat{v}$, primitive ghi chung. `direction` *nhận từ bất kỳ method Layer A nào*: mean-diff, [probe](01-linear-probing.md) weight, [CAV](03-tcav.md), hay [SAE feature](07-sparse-autoencoder.md) — thống nhất "đọc" và "ghi" quanh cùng một đối tượng hướng.
- **`LatentSpace.contrastive_direction(pos, neg, layer=ℓ)`** → tính difference-of-means (CAA), trả về `direction` dùng được ngay cho `steer`. Đây là cách rẻ nhất sinh một concept direction, không cần train.
- **Gác cổng an toàn**: vì steering dễ off-manifold và sụp đổ, `steer` phải trả kèm cảnh báo khi $\alpha$ đẩy $z'$ quá xa manifold ([Mahalanobis](../../04-latent-computation/research/05-mahalanobis-distance.md)), và khuyến nghị $\alpha$ thích ứng theo ngữ cảnh thay vì cố định — đúng bài học Rimsky/Tan.
- **Ưu tiên SAE feature khi cần sạch**: API nên cho phép chọn nguồn hướng, và cảnh báo rằng mean-diff thô có thể leak do [superposition](06-superposition-hypothesis.md); hướng từ SAE feature mono-semantic ít tác dụng phụ hơn.
- **Ràng buộc `ModelAdapter`**: steering cần hook *ghi* activation tại layer trung gian lúc forward — cùng yêu cầu như [activation patching](05-activation-patching.md), nhẹ hơn (chỉ cộng một hướng, không cần cache interchange).

Quan trọng nhất: steering vector hợp nhất toàn bộ tầng 5 thành một câu chuyện — mọi method introspection (probe, TCAV, SAE, dictionary learning) đều sinh ra *một hướng*, và steering là hành động *dùng* hướng đó để chỉnh latent. Nó là cây cầu cuối từ "hiểu latent" sang "điều khiển latent", đặt nền cho mọi pipeline manipulation của Layer B.

---

## Liên quan

- [Linear probing (mục 01 — tầng này)](01-linear-probing.md) — probe weight là một nguồn steering direction; read↔write quanh cùng một hướng.
- [TCAV (mục 03 — tầng này)](03-tcav.md) — CAV là concept direction dùng được để steer.
- [Causal intervention vs observational (mục 04 — tầng này)](04-causal-intervention-vs-observational.md) — steering là can thiệp $do(z+\alpha\hat{v})$; off-manifold là giới hạn chung.
- [Sparse autoencoder (mục 07 — tầng này)](07-sparse-autoencoder.md) — SAE feature là steering direction mono-semantic, ít leak hơn mean-diff.
- [Latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) — steering là phép số học vector với concept direction.

## Tham khảo

- A. Zou, L. Phan, S. Chen, et al., *Representation Engineering: A Top-Down Approach to AI Transparency* (2023, arXiv:2310.01405). — Khung RepE: reading vector + control vector; steering ở mức biểu diễn population.
- A. M. Turner, L. Thiergart, G. Leech, et al., *Activation Addition: Steering Language Models Without Optimization* (2023, arXiv:2308.10248). — ActAdd: hiệu một cặp prompt, cộng vào forward pass, không tối ưu.
- N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, A. M. Turner, *Steering Llama 2 via Contrastive Activation Addition* (ACL 2024, arXiv:2312.06681). — CAA: difference-of-means trên nhiều cặp; phân tích giới hạn hệ số cố định.
- D. Tan, D. Chanin, et al., *Analyzing the Generalization and Reliability of Steering Vectors* (2024, arXiv:2407.12404). — Hiệu ứng steering không ổn định giữa các hành vi, đôi khi phản tác dụng.
