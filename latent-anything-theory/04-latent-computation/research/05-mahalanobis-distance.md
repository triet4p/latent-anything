# Mahalanobis Distance trong Latent Computation

> **TL;DR.** Khoảng cách Mahalanobis $D_M(z) = \sqrt{(z - \mu)^T \Sigma^{-1} (z - \mu)}$ là Euclidean distance đo sau khi "nắn thẳng" không gian theo covariance — nó scale-invariant và tính đến tương quan giữa các chiều. Trong latent computation, cần dùng thay Euclidean khi latent space anisotropic: retrieval, OOD detection, arithmetic, và projection đều cho kết quả sai lệch nghiêm trọng nếu một số chiều có variance lớn hơn hàng nghìn lần chiều khác. Giới hạn chính là chi phí $O(d^3)$ để tính $\Sigma^{-1}$ và instability khi số mẫu ít hơn chiều không gian.

> **Nền tảng toán học:** Bài này tập trung vào *ứng dụng* trong latent computation. Công thức, chứng minh, và covariance matrix được cover trong [Mahalanobis distance — Tầng 1](../../01-space-representation/research/02-mahalanobis-distance.md).

Latent space của hầu hết model học được là *anisotropic*: một số chiều có variance cao (chứa thông tin quan trọng), số khác gần như phẳng. Khi đó, Euclidean distance bị dominated bởi các chiều high-variance — hai điểm có thể "gần nhau" theo Euclidean nhưng rất khác nhau về ngữ nghĩa, chỉ vì chúng tình cờ có cùng giá trị theo chiều có variance lớn. Mahalanobis distance giải quyết điều này bằng cách normalize theo covariance trước khi đo khoảng cách, đưa latent space về dạng isotropic ("trắng hóa" — whitening) rồi đo Euclidean.

---

## **1. Trực giác / Định nghĩa**

Hình dung latent space 2D với phân phối hình ellipse: trục X có variance = 100, trục Y có variance = 1. Hai điểm $z_1 = (5, 0)$ và $z_2 = (0, 1)$ có Euclidean distance $\approx 5.1$ và $\approx 1.4$ so với gốc tọa độ, khiến $z_2$ có vẻ "gần" hơn nhiều. Nhưng $z_1$ thực ra cách gốc chỉ $5/\sqrt{100} = 0.5$ sigma, còn $z_2$ cách $1/\sqrt{1} = 1$ sigma — về mặt thống kê $z_1$ thuộc về phân phối tốt hơn. Mahalanobis distance trả về đúng quan hệ này.

**Quan hệ với whitening:** Mahalanobis distance tương đương với Euclidean distance trong không gian đã whitened. Nếu định nghĩa:

$$z' = \Sigma^{-1/2}(z - \mu)$$

trong đó $\mu$ là trung bình và $\Sigma^{-1/2}$ là nghịch căn bậc hai của covariance matrix (có thể tính qua eigendecomposition: $\Sigma^{-1/2} = V\Lambda^{-1/2}V^T$), thì:

$$D_M(z_1, z_2) = \|z_1' - z_2'\|_2$$

Whitened vectors $z'$ có covariance là identity — tức không gian trở nên isotropic, và mọi thao tác Euclidean trong $z'$-space đều tương đương với thao tác Mahalanobis trong $z$-space.

**Annotate:** $\Sigma \in \mathbb{R}^{d \times d}$ là covariance matrix của latent distribution; $\mu \in \mathbb{R}^d$ là mean; $\Sigma^{-1/2}$ là inverse square root (whitening matrix). Kết quả $z'$ là tọa độ trong không gian đã chuẩn hóa.

---

## **2. Cơ chế / Công thức**

### 2.1 Ba cách whitening

Có ba cách phổ biến để tính whitening matrix, khác nhau ở chỗ có rotate hay không:

| Phương pháp | Công thức $W$ sao cho $z' = W(z - \mu)$ | Đặc điểm |
|---|---|---|
| **PCA whitening** | $W = \Lambda^{-1/2} V^T$ | Rotate về PCA axes + scale. Kết quả là disentangled (không tương quan). |
| **ZCA whitening** | $W = V\Lambda^{-1/2} V^T$ | Không rotate, chỉ scale. Kết quả gần với $z$ gốc nhất theo Frobenius norm. |
| **Cholesky whitening** | $W = L^{-1}$ ($\Sigma = LL^T$) | Triangular. Efficient cho implementation. |

trong đó $\Sigma = V\Lambda V^T$ là eigendecomposition của covariance matrix ($V$ — eigenvectors, $\Lambda$ — diagonal eigenvalues).

**PCA whitening** phù hợp khi muốn disentangle + compress (loại bỏ chiều gần-zero variance). **ZCA whitening** phù hợp khi muốn normalize mà vẫn giữ cấu trúc của $z$ gần như nguyên vẹn — lý tưởng cho latent arithmetic và subspace projection.

### 2.2 Khi nào latent computation cần Mahalanobis

| Operation | Euclidean đủ? | Khi nào cần Mahalanobis? |
|---|---|---|
| **k-NN retrieval** | Chỉ khi isotropic | Latent space anisotropic → high-variance dims dominate; kết quả retrieval lệch |
| **Latent arithmetic** ($z_a - z_b + z_c$) | Chỉ khi $\|\delta\|$ đồng đều các concept | Concept directions có magnitude khác nhau theo variance → cần normalize $\delta$ bằng $\Sigma^{-1/2}\delta$ |
| **Subspace projection** ($UU^Tz$) | Chỉ khi isotropic | Inner product $u^Tz$ không phản ánh ngữ nghĩa khi $\Sigma \neq I$; cần project trong whitened space |
| **OOD detection** | Không — fail với anisotropic | Cần $D_M(z, \mu)$ để đo xem $z$ thuộc phân phối training đến mức nào |
| **Interpolation** (lerp/slerp) | Gần đúng | Lerp với Mahalanobis metric cho path tự nhiên hơn trong Gaussian latent space |

### 2.3 OOD detection với Mahalanobis score

Lee et al. (2018) đề xuất dùng Mahalanobis distance trong feature space làm confidence score để phát hiện out-of-distribution (OOD) samples. Với $K$ class, ước lượng $\{\mu_k, \Sigma\}$ (tied covariance) từ training data, sau đó tính:

$$M(z) = \max_k \left[-(z - \mu_k)^T \hat\Sigma^{-1} (z - \mu_k)\right]$$

trong đó $\hat\Sigma$ là pooled covariance matrix (trung bình có trọng số của covariance mỗi class). Điểm $z$ có $M(z)$ thấp (khoảng cách Mahalanobis lớn với mọi class) được coi là OOD. Phương pháp này vượt trội rõ rệt so với softmax confidence và Euclidean distance trong latent space.

---

## **3. Biến thể**

### 3.1 Diagonal approximation

Khi $d$ lớn (ví dụ: latent dimension 512–4096), tính và lưu $\Sigma^{-1}$ có chi phí $O(d^2)$ bộ nhớ và $O(d^3)$ tính toán. Xấp xỉ phổ biến là dùng *diagonal covariance*:

$$\Sigma \approx \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$$

Khi đó whitening đơn giản là chia từng chiều cho std: $z'_i = (z_i - \mu_i) / \sigma_i$. Mất thông tin về tương quan giữa các chiều nhưng loại bỏ được scale bias với chi phí $O(d)$.

### 3.2 Class-conditional vs global Mahalanobis

- **Global**: một $\Sigma$ cho toàn bộ latent distribution. Phù hợp khi latent space có phân phối đồng nhất (ví dụ: VAE với Gaussian prior).
- **Class-conditional**: một $\Sigma_k$ riêng cho từng class/cluster. Phù hợp khi latent space có nhiều mode với hình dạng khác nhau. Chi phí: $O(K \cdot d^2)$ bộ nhớ.

### 3.3 Online covariance estimation

Khi cần update $\Sigma$ theo batch mới mà không retrain toàn bộ (streaming setting), có thể dùng online covariance update:

$$\Sigma_{n+1} = \frac{n}{n+1}\Sigma_n + \frac{n}{(n+1)^2}(z_{n+1} - \mu_n)(z_{n+1} - \mu_n)^T$$

Với latent space thay đổi theo thời gian (continual learning), đây là cách cập nhật metric mà không cần full recompute.

---

## **4. Giới hạn / Khi nào thất bại**

**Chi phí tính toán.** Tính $\Sigma^{-1}$ cần $O(d^3)$ với eigendecomposition, hoặc $O(d^3)$ với Cholesky — quá đắt khi $d = 4096$ (ví dụ: latent của diffusion model). Trong thực tế thường phải dùng diagonal approximation hoặc giảm chiều trước (PCA xuống $k \ll d$) rồi mới tính Mahalanobis.

**Ill-conditioning khi $n < d$.** Nếu số mẫu để ước lượng $\Sigma$ ít hơn chiều không gian ($n < d$), sample covariance matrix là singular — không invertible. Cần regularization: $\hat\Sigma_\lambda = \hat\Sigma + \lambda I$ (Ledoit-Wolf shrinkage hoặc đơn giản là thêm $\epsilon I$).

**Gaussian assumption.** Mahalanobis distance là optimal khi latent distribution là Gaussian. Nếu latent space có multiple modes, heavy tails, hoặc cấu trúc phi tuyến, Mahalanobis với một $\Sigma$ global sẽ bóp méo metric. Với multi-modal distribution, cần class-conditional Mahalanobis.

**Không bất biến với nonlinear transformation.** Whitening trong latent space của layer $l$ không có nghĩa trong latent space của layer $l+1$ nếu giữa hai layer là nonlinear activation. Metric học được ở một layer không tự động đúng với layer khác.

**Covariance drift.** Sau khi model được fine-tune hoặc khi phân phối input thay đổi, $\Sigma$ cũ không còn phản ánh đúng cấu trúc latent mới. Cần recompute định kỳ — đây là chi phí bổ sung mà Euclidean distance không có.

---

## **5. Liên hệ với Latent-Anything**

Mahalanobis distance là metric chuẩn cho **Layer B** khi làm việc với anisotropic latent space:

- **Whitening pipeline**: `LatentSpace.fit_whitening(samples)` → tính $\hat\mu$ và $\hat\Sigma^{-1/2}$ từ tập mẫu; `LatentSpace.whiten(z)` → trả về $z'$. Sau khi whiten, mọi operation Euclidean (arithmetic, projection, lerp) đều tự động Mahalanobis-correct.
- **OOD guard**: `LatentSpace.mahalanobis_score(z)` → dùng làm bộ lọc trước khi decode. Nếu score > threshold, flag là OOD và từ chối decode hoặc cảnh báo.
- **Metric-aware arithmetic**: khi `LatentSpace.arithmetic(z_a, z_b, z_c)` phát hiện latent anisotropic (đo bằng condition number của $\hat\Sigma$), tự động whiten trước arithmetic và unwhiten sau.
- **Retrieval**: `LatentSpace.nearest_neighbors(z, k)` dùng Mahalanobis metric thay Euclidean khi `whitening_fitted=True`.

Kết hợp với **Subspace projection** (mục 04): nên project trong whitened space để inner product $u^T z'$ phản ánh đúng ngữ nghĩa. Whitening trước → projection sau là pipeline chuẩn cho latent space anisotropic.

---

## Liên quan

- [Mahalanobis distance — Tầng 1 (nền tảng toán)](../../01-space-representation/research/02-mahalanobis-distance.md) — công thức, covariance matrix, và bằng chứng tại sao Euclidean fail với dữ liệu tương quan.
- [Subspace projection (mục 04 — tầng này)](04-subspace-projection.md) — projection trong whitened space cho kết quả metric-correct; mục 4 của note đó đã chỉ ra vấn đề metric mismatch.
- [Latent arithmetic (mục 03 — tầng này)](03-latent-arithmetic.md) — arithmetic trong anisotropic space cần normalize $\delta$ theo $\Sigma^{-1/2}$ để concept directions có magnitude so sánh được.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — lý do latent space hầu hết anisotropic và hệ quả cho metric.

## Tham khảo

- K. Lee, K. Lee, H. Lee, J. Shin, *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks* (NeurIPS 2018, arXiv:1807.03888). — Dùng Mahalanobis distance trong feature space của deep network làm OOD confidence score; chứng minh vượt trội so với softmax baseline.
- P. C. Mahalanobis, *On the Generalised Distance in Statistics* (Proceedings of the National Institute of Sciences of India, 1936). — Paper gốc giới thiệu metric.
- O. Ledoit, M. Wolf, *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices* (Journal of Multivariate Analysis, 2004). — Ledoit-Wolf shrinkage estimator cho covariance trong điều kiện $n < d$; cần thiết khi latent dimension lớn.
