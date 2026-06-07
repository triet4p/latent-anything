# FlatVI

> **TL;DR.** FlatVI thêm một **hàm phạt phẳng hóa** ép [pullback metric](06-pullback-metric.md) ≈ ma trận đơn vị, "san phẳng" latent để [geodesic](05-geodesic.md) ≈ đường thẳng Euclidean. Nhờ đó lerp, k-NN và Optimal Transport chạy đúng mà không cần solver geodesic đắt đỏ.

## Flat Variational Inference (FlatVI)

FlatVI là một khung huấn luyện (training framework) dành cho các mô hình VAE để thiết kế để xử lý dữ liệu thưa và rời rạc như scRNA-seq trong sinh học

Để hiểu rõ cơ chế của FlatVI một cách trực quan, hãy hình dung lại ví dụ về "tấm bản đồ 2D" (không gian ẩn) và "địa hình đồi núi thực tế 3D" (không gian dữ liệu) mà chúng ta vừa bàn ở phần Mêtric kéo lùi (Pullback Metric).

Ở các mô hình VAE thông thường, nếu bạn kẻ một đường thẳng bằng thước kẻ nối điểm A và điểm B trên bản đồ, khi chiếu ra thực địa, đường thẳng đó có thể đi đâm xuyên qua một ngọn núi hoặc rơi xuống vực sâu (không tuân theo quy luật tự nhiên của dữ liệu). 

Thay vì cố gắng tìm các đường vòng vèo trắc địa đắt đỏ để né ngọn núi, **cơ chế cốt lõi của FlatVI là dùng một hình phạt toán học ép bộ giải mã (decoder) phải "san phẳng" mọi đồi núi trên thực địa sao cho phù hợp tuyệt đối với tấm bản đồ phẳng.**

Cụ thể, cơ chế này diễn ra qua 3 bước căn bản sau:

**1. Tính toán Mêtric kéo lùi (Pullback Metric) tại mỗi điểm**
FlatVI (được thiết kế đặc biệt cho dữ liệu giải trình tự tế bào scRNA-seq) sử dụng một bộ giải mã dựa trên phân phối Nhị thức âm (Negative Binomial) để phản ánh đúng bản chất rời rạc của dữ liệu tế bào. Tại bất kỳ điểm $z$ nào trong không gian ẩn, FlatVI sẽ tính toán mêtric kéo lùi (cụ thể ở đây là mêtric thông tin Fisher - Fisher Information Metric). Mêtric này, gọi là $M(z)$, chứa thông tin về độ cong và tỷ lệ biến dạng của dữ liệu thực tế tại điểm đó.

**2. Ép mêtric trở thành "Ma trận đơn vị" (Khâu quan trọng nhất)**
Trong toán học, một không gian được coi là phẳng hoàn hảo (Euclidean) khi mêtric đo lường của nó là một ma trận đơn vị ($I_d$) tại mọi nơi. Ma trận đơn vị đồng nghĩa với việc không có hướng nào bị co giãn hay bóp méo, mọi hướng đều bình đẳng. 
FlatVI đưa vào một hàm mất mát gọi là **Hàm phạt phẳng hóa (Flattening Loss - $\mathcal{L}_{flat}$)**. Hàm này sử dụng chuẩn Frobenius để trừng phạt sự khác biệt giữa mêtric kéo lùi $M(z)$ và một ma trận đơn vị được nhân với hệ số tỷ lệ $\alpha$ ($\alpha I_d$). 
Nói cách khác, nó ép $M(z) \approx \alpha I_d$. Nếu điều kiện này xảy ra, khoảng cách trắc địa (geodesic) tự động biến thành đường thẳng Euclidean (linear). Tham số $\alpha$ được cho phép học tự động để mô hình có không gian co giãn linh hoạt, miễn là sự co giãn đó đồng đều ở mọi hướng.

**3. Tối ưu hóa song song (Joint Training)**
Hàm phạt phẳng hóa này không đứng một mình mà được cộng trực tiếp vào hàm mất mát tiêu chuẩn của VAE (hàm ELBO). 
Quá trình huấn luyện sẽ là một cuộc giằng co:

*   ELBO ép mô hình phải tái tạo lại dữ liệu tế bào cho đúng.
*   Hàm Flattening ép không gian ẩn phải phẳng.
Kết quả là mô hình phải tìm ra cách sắp xếp các điểm dữ liệu trong không gian ẩn sao cho vừa giữ được đặc tính sinh học, vừa tạo ra một bề mặt hình học phẳng hoàn hảo.

**Tại sao cơ chế này lại vượt trội?**

*   **Chấm dứt sự phức tạp:** Các phương pháp khác (như GAGA hay NeuralFIM) chấp nhận không gian bị cong và phải dùng một mạng nơ-ron chuyên biệt (Neural ODE) cực kỳ nặng nề để dò dẫm giải phương trình tìm đường cong trắc địa. Quá trình này rất chậm, không ổn định và hay tạo ra các biểu hiện gene sai lệch (dao động bất thường).
*   **Nội suy bằng đường thẳng (LERP):** Vì FlatVI đã dọn dẹp sẵn và san phẳng không gian, người dùng giờ đây chỉ cần dùng phép nối đường thẳng (linear interpolation) cực kỳ rẻ và cơ bản. Các đường thẳng này, khi được giải mã, sẽ sinh ra quỹ đạo biến đổi tế bào hợp lý, tự nhiên, và bám sát đa tạp dữ liệu thực tế.
*   **Phù hợp với các công cụ có sẵn:** Rất nhiều công cụ phân tích hiện tại (như Vận chuyển Tối ưu Optimal Transport hay k-NN) mặc định giả định dữ liệu tính bằng khoảng cách thẳng Euclidean. Bằng cách cung cấp một không gian ẩn thực sự "phẳng", FlatVI giúp các công cụ hạ nguồn này hoạt động chính xác hơn rất nhiều.

---

## Liên quan

- [Pullback Metric](06-pullback-metric.md) — đại lượng mà FlatVI ép về ma trận đơn vị.
- [Geodesic](05-geodesic.md) — khi không gian phẳng, geodesic trở thành đường thẳng.
- [VAE](../../02-representation-learning/research/03-vae.md) — FlatVI cộng phạt phẳng hóa vào hàm ELBO.
- [Normalizing Flows](../../03-geometry-structure/research/06-normalizing-flows.md) — một hướng khác để "làm thẳng/re-coordinatize" không gian.
- [Slerp](../../03-geometry-structure/research/05-slerp.md) — giải pháp nội suy thay thế khi không làm phẳng được.

## Tham khảo

- *FlatVI: Enforcing Latent Euclidean Geometry in Single-Cell VAEs for Manifold Interpolation* (2025, arXiv:2507.11789).
- Arvanitidis, Hansen, Hauberg, *Latent Space Oddity* (ICLR 2018, arXiv:1710.11379) — nền tảng pullback metric.
- Lopez et al., *Deep generative modeling for single-cell transcriptomics* (scVI, Nature Methods, 2018) — decoder Negative Binomial cho scRNA-seq.