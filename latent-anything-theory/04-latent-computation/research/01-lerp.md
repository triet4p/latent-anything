# Lerp — Linear Interpolation trong Latent Space

> **TL;DR.** Lerp là phép nội suy tuyến tính $z(t) = (1-t)\cdot z_1 + t\cdot z_2$ — đơn giản, nhanh, và là baseline cho mọi thao tác nối hai điểm trong latent space. Nó tối ưu khi không gian phẳng (Euclidean), nhưng thất bại khi latent vector chuẩn hóa (nằm trên hypersphere): điểm giữa bị kéo vào bên trong quả cầu, rơi khỏi vùng có mật độ cao của prior, khiến decode cho ra kết quả nhòa hoặc không coherent. Khi nào latent không chuẩn hóa và nằm trong cùng cluster thì lerp thường đủ tốt; còn lại, cần xem xét slerp hoặc geodesic-aware alternative.

Phép nội suy tuyến tính (*linear interpolation*, hay lerp) là phép toán nền tảng nhất trong latent space: cho hai điểm $z_1, z_2$, tạo ra chuỗi điểm trung gian nằm trên đoạn thẳng nối hai điểm đó. Mọi hệ thống thao tác latent đều bắt đầu từ đây — lerp là geodesic (đường ngắn nhất) trong không gian Euclidean phẳng, và là điểm khởi đầu tự nhiên trước khi bàn đến các alternative phức tạp hơn. Vấn đề là latent space của các model thực tế thường không phẳng, và khi đó lerp bắt đầu vi phạm các giả định hình học cơ bản.

---

## **1. Trực giác / Định nghĩa**

Hình dung hai ảnh trong latent space: ảnh chó $z_1$ và ảnh mèo $z_2$. Lerp tạo ra chuỗi điểm đi thẳng từ $z_1$ đến $z_2$. Nếu latent space "lành mạnh", decoder sẽ render chuỗi đó thành hình ảnh chuyển dần từ chó sang mèo.

Trực giác này *đúng* khi không gian phẳng và các điểm nằm trong vùng có mật độ cao. Trực giác *vỡ* khi:

- Đường thẳng đi qua vùng nằm ngoài manifold dữ liệu (decoder chưa thấy vùng đó lúc train).
- Các vector bị chuẩn hóa về độ dài đơn vị — đường thẳng đi *bên trong* hình cầu, không phải *trên* bề mặt.
- Hai điểm thuộc hai cluster ngữ nghĩa khác nhau, khiến đường thẳng đi qua "vùng trống" giữa chúng.

---

## **2. Cơ chế / Công thức**

Cho hai điểm $z_1, z_2 \in \mathbb{R}^d$ và tham số $t \in [0, 1]$:

$$z(t) = (1 - t)\cdot z_1 + t\cdot z_2$$

trong đó $t = 0$ cho $z_1$, $t = 1$ cho $z_2$, và $t \in (0, 1)$ cho các điểm trung gian. Công thức tuyến tính trong $t$ này là nghiệm của bài toán "di chuyển với vận tốc không đổi trên đường thẳng" trong Euclidean space — tức là geodesic Euclidean.

**Norm của điểm giữa khi $z_1, z_2$ là unit vector.** Nếu $\|z_1\| = \|z_2\| = 1$ và góc giữa chúng là $\theta$:

$$\|z(0.5)\| = \left\|\frac{z_1 + z_2}{2}\right\| = \sqrt{\frac{1 + \cos\theta}{2}} = \cos\!\left(\frac{\theta}{2}\right)$$

kết quả nhỏ hơn 1 khi $\theta > 0$. Điểm giữa lerp *luôn* nằm sâu hơn bên trong quả cầu so với hai đầu mút. Ví dụ: $\theta = 60°$ cho $\|z(0.5)\| \approx 0.866$; $\theta = 90°$ cho $\|z(0.5)\| \approx 0.707$.

**Bất biến quan trọng:** lerp bảo toàn tính lồi — nếu $z_1, z_2$ đều hợp lệ trong convex hull của dữ liệu, $z(t)$ cũng hợp lệ. Nhưng latent manifold thực tế thường không lồi, nên bất biến này yếu hơn nhiều so với vẻ ngoài.

---

## **3. Khi nào lerp đủ tốt, khi nào fail**

| Điều kiện | Lerp ổn? | Lý do |
|---|---|---|
| Latent không chuẩn hóa, phân phối Gaussian isotropic | ✓ | Đường thẳng đi qua vùng có mật độ cao, prior phẳng |
| Nội suy trong cùng một cluster ngữ nghĩa | ✓ | Manifold cục bộ gần phẳng, decoder đã thấy vùng đó |
| Latent học bởi normalizing flow | ✓ | Flow đảm bảo base space là Gaussian phẳng |
| Latent không chuẩn hóa, phân phối anisotropic | ⚠️ | Điểm giữa vẫn trong convex hull nhưng có thể xa mode |
| Latent chuẩn hóa (unit-norm, VAE với spherical prior) | ✗ | Điểm giữa bị kéo vào trong quả cầu, ra ngoài prior |
| Nội suy qua nhiều cluster khác nhau | ✗ | Đường thẳng qua "vùng trống", decoder không coherent |
| Latent có chiều cao, norm chuẩn hóa (diffusion model) | ✗ | Concentration of measure: mass tập trung ở bề mặt cầu |

---

## **4. Giới hạn / Khi nào thất bại**

**Vấn đề hình học: điểm giữa rơi khỏi prior.** Trong VAE chuẩn với prior $\mathcal{N}(0, I)$, ở chiều cao hầu hết probability mass tập trung gần bề mặt một hypersphere bán kính $\approx \sqrt{d}$ (concentration of measure). Lerp ở $t = 0.5$ cho $\|z(0.5)\| = \cos(\theta/2) < \|z_1\|$, tức rơi vào vùng bên trong có mật độ thấp — decoder khi đó decode ra vùng nó chưa được train, thường cho kết quả nhòa hoặc không nhất quán. White (2016) là người đầu tiên phân tích rõ hiện tượng này với GAN latent space và chỉ ra rằng slerp tạo ra kết quả sắc nét hơn đáng kể.

**Vấn đề ngữ nghĩa: đường thẳng không phải geodesic trên manifold.** Ngay cả khi $\|z(0.5)\|$ hợp lệ về mặt norm, điểm giữa có thể decode ra ảnh "hỗn hợp" không có nghĩa. Ví dụ: lerp giữa ảnh mặt nhìn trái và nhìn phải thường cho ra ảnh nhòa, thay vì một góc nhìn trung gian hợp lệ, vì geodesic trên face manifold không phải đường thẳng trong latent Euclidean.

**Tốc độ di chuyển không đều trong semantic space.** Lerp di chuyển với vận tốc không đổi trong Euclidean, nhưng tốc độ thay đổi ngữ nghĩa không đều: một số đoạn của đường thẳng tương ứng với thay đổi lớn về nội dung, một số đoạn khác gần như không đổi. Điều này gây ra hiệu ứng "nhảy cóc" khi dùng lerp để tạo animation.

**Nội suy qua cluster boundary.** Nếu $z_1$ và $z_2$ thuộc hai mode khác biệt (ví dụ: chó và máy bay), đường lerp đi qua "vùng trống" mà model không thấy lúc train. Các điểm ở giữa đường không có điểm training data tương ứng gần nhất, khiến decoder extrapolate theo cách không có ý nghĩa.

**Lerp không phải metric operation.** Lerp không tính đến hình dạng phân phối của latent: nếu phân phối anisotropic (co dài theo một hướng), điểm $z(0.5)$ theo Euclidean có thể xa hơn nhiều so với khi dùng Mahalanobis-weighted interpolation. Điều này đặc biệt quan trọng khi dùng lerp để so sánh khoảng cách ngữ nghĩa.

---

## **5. Liên hệ với Latent-Anything**

Lerp là operation cơ bản nhất của **Layer B (manipulation)** trong Latent-Anything. Nó là baseline cho:

- **`Trajectory` primitive**: nội suy giữa hai latent state để tạo keyframe hoặc dense sequence.
- **Latent blending**: trộn hai concept trong latent, dùng trong image editing và style mixing.
- **Benchmark health của latent space**: nếu lerp decode ra kết quả coherent, không gian đó đủ "lành mạnh" để dùng cho thao tác đơn giản.

Trong framework, lerp sẽ là implementation mặc định của `LatentSpace.interpolate(method="lerp")`, đi kèm với kiểm tra tự động: nếu norm của latent vector gần 1.0 (unit-norm), framework sẽ cảnh báo và đề xuất chuyển sang slerp. Phần tiếp theo — **Slerp** (mục 02 của tầng này) — trình bày implementation và lý thuyết chi tiết hơn phần giới thiệu ở tầng 3, đặc biệt là cách xử lý edge case khi $\theta \to 0$ và $\theta \to \pi$.

---

## Liên quan

- **Slerp (mục 02 — tầng này)** — thay thế lerp trên hypersphere bằng cách đi theo geodesic của hình cầu; trực tiếp khắc phục vấn đề norm được mô tả ở mục 4.
- [Slerp (tầng 3 — giới thiệu)](../../03-geometry-structure/research/05-slerp.md) — giới thiệu ban đầu về slerp trong ngữ cảnh hình học Riemannian; mục 02 tầng này sẽ đi sâu hơn về implementation.
- [Geodesic](../../01-space-representation/research/05-geodesic.md) — lerp là geodesic trong Euclidean; khi manifold cong, geodesic thực sự là đường khác.
- [Riemannian geometry cơ bản](../../03-geometry-structure/research/04-riemannian-geometry.md) — cơ sở lý thuyết giải thích tại sao điểm giữa lerp rơi khỏi manifold cong.
- [Manifold hypothesis](../../01-space-representation/research/03-manifold-hypothesis.md) — lý do dữ liệu thực nằm trên submanifold có chiều thấp, khiến lerp dễ lạc ra ngoài.
- [Isotropy & anisotropy](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — phân phối anisotropic làm lerp Euclidean thiếu chính xác về mặt ngữ nghĩa.

## Tham khảo

- T. White, *Sampling Generative Networks* (arXiv 2016, arXiv:1609.04468). — Phân tích đầu tiên so sánh lerp và slerp trong latent space GAN; chỉ ra hiện tượng "blurry midpoint" do norm sụt giảm.
- D. P. Kingma, M. Welling, *Auto-Encoding Variational Bayes* (ICLR 2014, arXiv:1312.6114). — VAE gốc; prior $\mathcal{N}(0, I)$ là nguồn gốc của vấn đề norm trong mục 4.
- T. Davidson et al., *Hyperspherical Variational Auto-Encoders* (UAI 2018, arXiv:1804.00891). — VAE với von Mises–Fisher prior trên hypersphere; trường hợp lerp hoàn toàn không phù hợp vì latent nằm trực tiếp trên $S^{d-1}$.
