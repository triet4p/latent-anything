# Autoencoder

Mô hình Tự mã hóa (Autoencoder - AE) nguyên thủy nhất là một mạng nơ-ron nhân tạo được thiết kế cho các tác vụ học không giám sát (unsupervised learning) và giảm chiều dữ liệu (dimensionality reduction). Thay vì dự đoán một nhãn $Y$ từ đầu vào $X$, mục tiêu tối thượng của AE là học một hàm đồng nhất (identity function) sao cho đầu ra được tái tạo gần giống với đầu vào nhất có thể, tức là $f(x) \approx x$. 

## **1. Kiến trúc thắt nút cổ chai (Bottleneck Architecture)**
Để hiểu tại sao Autoencoder có thể học được các biểu diễn hữu ích, chúng ta phải nhìn vào kiến trúc của nó. Một Autoencoder bao gồm hai thành phần mạng nơ-ron chính được nối với nhau thông qua một lớp thắt nút cổ chai:

*   **Bộ mã hóa (Encoder - $g_\phi$):** Có nhiệm vụ tiếp nhận dữ liệu đầu vào $x$ (thường có số chiều rất cao) và biến đổi nó thành một biểu diễn ẩn (latent representation) $z$. Các lớp nơ-ron trong bộ mã hóa sẽ thu hẹp dần dần số chiều của dữ liệu đầu vào. Về mặt toán học, biểu diễn này được viết là $z = g_\phi(x)$, với $\phi$ là các trọng số của bộ mã hóa.
*   **Lớp thắt cổ chai (Bottleneck / Latent Space):** Đây là cốt lõi của cấu trúc Autoencoder. Thay vì sử dụng một hàm mục tiêu toán học để phạt việc giữ lại thông tin như hàm Lagrangian trong nguyên lý Nghẽn thông tin (Information Bottleneck) mà chúng ta đã thảo luận trước đó, AE nguyên thủy đạt được sự nén này một cách vật lý: số lượng nơ-ron ở lớp ẩn trung gian được thiết kế ít hơn rất nhiều so với lớp đầu vào. Nút thắt này ép dữ liệu phải được chiếu (project) xuống một không gian đặc trưng (feature space) nhỏ hơn.
*   **Bộ giải mã (Decoder - $f_\theta$):** Nhận vector nén $z$ từ lớp cổ chai và cố gắng khôi phục lại dữ liệu gốc ban đầu $x'$, thường thông qua các lớp có số chiều tăng dần để khớp với kích thước của đầu vào. Về mặt toán học, $x' = f_\theta(z)$, với $\theta$ là trọng số của bộ giải mã.

## **2. Mục tiêu tái cấu trúc (Reconstruction Objective)**
Để mô hình học được các trọng số $\phi$ và $\theta$, AE sử dụng chính dữ liệu đầu vào làm "nhãn" (labels) để huấn luyện, khiến nó trở thành một phương pháp học không cần nhãn thủ công (unsupervised). 

Quá trình tối ưu hóa xoay quanh việc giảm thiểu **sai số tái cấu trúc (reconstruction loss)**, đại lượng đo lường độ chênh lệch giữa dữ liệu đầu vào gốc $x$ và bản phục hồi $x'$ ở đầu ra. Hàm mất mát phổ biến nhất được sử dụng cho AE nguyên thủy là Trung bình bình phương sai số (Mean Squared Error - MSE):
$\mathcal{L}_{\text{AE}}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (x^{(i)} - f_\theta(g_\phi(x^{(i)})))^2$
Hoặc viết dưới dạng kỳ vọng toán học: $\mathcal{L}(\theta, \phi) = \mathbb{E}[||x - f_\theta(g_\phi(x))||^2]$.

## **Tại sao sự kết hợp giữa Reconstruction Objective và Bottleneck lại quan trọng?**
Nếu không có nút thắt cổ chai (ví dụ: lớp ẩn có số chiều bằng hoặc lớn hơn đầu vào), mạng nơ-ron có thể dễ dàng học cách "ghi nhớ" hoặc sao chép nguyên xi dữ liệu từ đầu vào ra đầu ra mà không cần trích xuất bất kỳ đặc trưng hữu ích nào. Trạng thái này không mang lại lợi ích gì cho việc học biểu diễn.

Tuy nhiên, bằng cách áp đặt cấu trúc bottleneck kết hợp với áp lực phải tái cấu trúc lại ảnh gốc thông qua hàm loss, mô hình bị ép buộc (forced) phải khám phá ra cấu trúc ẩn bên trong dữ liệu. Khả năng truyền dẫn thông tin bị bóp nghẹt khiến mạng không thể truyền toàn bộ điểm ảnh (pixels). Thay vào đó, nó buộc phải vứt bỏ các nhiễu dư thừa và chỉ nén lại những đặc trưng, quy luật cốt lõi nhất (generative factors) – chẳng hạn như các cạnh, đường nét – đủ để bộ giải mã có thể suy luận và khôi phục lại hình ảnh ban đầu. Điều này có sự tương đồng triết lý rất lớn với *Information Bottleneck principle*, nơi mô hình phải lọc bỏ thông tin ngoại cảnh để giữ lại một biểu diễn tối giản, nhưng ở đây mục tiêu của nó là tái tạo lại chính $X$ chứ không phải dự đoán một nhãn $Y$ bên ngoài.

## **Hạn chế của Autoencoder nguyên thủy (Mở đường cho VAE)**
Mặc dù Autoencoder nguyên thủy thực hiện rất tốt việc nén (compression) và giảm chiều dữ liệu, bản chất của nó là một mô hình **tất định (deterministic)**. Khi học, bộ mã hóa ánh xạ một đầu vào $X$ thành một tọa độ điểm cố định duy nhất $Z$ trong không gian ẩn, hoàn toàn thiếu vắng cấu trúc xác suất (probabilistic structure). 

Hệ quả là các điểm biểu diễn của dữ liệu huấn luyện nằm rải rác, tạo ra các "khoảng trống" (latent gaps) vô nghĩa trong không gian ẩn. Vì mạng chưa bao giờ được huấn luyện để giải mã các khoảng trống này, nếu bạn lấy mẫu ngẫu nhiên một điểm bất kỳ trong không gian ẩn và đưa cho bộ giải mã, nó sẽ sinh ra những hình ảnh vô nghĩa hoặc dị dạng. Đó là lý do tại sao Autoencoder nguyên thủy không thể được sử dụng làm mô hình sinh (Generative Model) để tạo ra dữ liệu mới, mà phải cần đến những kiến trúc áp đặt phân phối xác suất lên không gian ẩn như Variational Autoencoder (VAE) sau này.

---

## Liên quan

- [Information Bottleneck](01-information-bottleneck.md) — nền tảng lý thuyết của việc nén qua bottleneck.
- [VAE](03-vae.md) — bổ sung cấu trúc xác suất để biến AE thành mô hình sinh.
- [Giả thuyết Đa tạp](../../01-space-representation/research/03-manifold-hypothesis.md) — AE học chiếu dữ liệu xuống đa tạp chiều thấp.

## Tham khảo

- Hinton, Salakhutdinov, *Reducing the Dimensionality of Data with Neural Networks* (Science, 2006).
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press, 2016), Chương 14 — Autoencoders.
- Vincent et al., *Extracting and Composing Robust Features with Denoising Autoencoders* (ICML, 2008).