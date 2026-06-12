# Density Estimation trong Latent Space

> **TL;DR.** Density estimation trong latent space ước lượng phân phối $p(z)$ của tập training để phân biệt điểm $z$ *thuộc về* phân phối (in-distribution) và điểm *bất thường* (out-of-distribution — OOD). Phương pháp phổ biến gồm Gaussian Mixture Model ($p(z) = \sum_k \pi_k \mathcal{N}(z \mid \mu_k, \Sigma_k)$) và normalizing flow (biến đổi khả nghịch cho phép tính $\log p(z)$ chính xác); caveat quan trọng: deep generative model đôi khi gán likelihood *cao* cho dữ liệu OOD — cần dùng likelihood ratio hoặc complexity correction thay vì raw likelihood để phát hiện OOD đáng tin cậy.

Latent space của một model học được không phải là một không gian rỗng ngẫu nhiên — nó có cấu trúc: các điểm training tập trung quanh một số "cluster" hoặc "mode" nhất định, và những vùng khác gần như trống. Biết được $p(z)$ — mật độ xác suất tại mỗi điểm — cho phép trả lời các câu hỏi thực tiễn: Điểm $z_{\text{new}}$ này có xuất phát từ cùng phân phối với tập training không? Latent nào là "điển hình" và latent nào là "bất thường"? Nếu decode một $z$ có $p(z)$ thấp, kết quả có đáng tin không?

Density estimation là bước tự nhiên tiếp theo sau [Mahalanobis distance](05-mahalanobis-distance.md): Mahalanobis giả định phân phối Gaussian đơn (single Gaussian), còn density estimation cho phép mô hình hóa các phân phối phức tạp hơn với nhiều mode.

---

## **1. Trực giác / Định nghĩa**

Hình dung latent space 2D của một model ảnh khuôn mặt: các điểm training tập trung thành nhiều cụm — nam/nữ, trẻ/già, ảnh sáng/tối. Giữa các cụm là vùng thưa thớt hoặc trống. Density estimation vẽ bản đồ này: vùng đỏ (mật độ cao, in-distribution), vùng xanh (mật độ thấp, OOD hoặc bất thường).

Khi nhận một $z_{\text{new}}$, ta có thể:

1. Tính $\log p(z_{\text{new}})$ — nếu thấp hơn ngưỡng $\tau$, flag là OOD.
2. Tính softmax assignment $p(k \mid z) = \pi_k \mathcal{N}(z \mid \mu_k, \Sigma_k) / p(z)$ — biết $z$ thuộc cluster nào với xác suất bao nhiêu.
3. Dùng gradient của log-density $\nabla_z \log p(z)$ — hướng về phía "trung tâm" của phân phối, dùng để "kéo" $z$ vào vùng in-distribution nếu nó bị trôi ra ngoài.

**Phân biệt với Mahalanobis:** Mahalanobis distance tính khoảng cách từ $z$ đến *một* Gaussian duy nhất — nhanh nhưng không capture được cấu trúc multi-modal. Density estimation tổng quát hơn: fit nhiều Gaussian (GMM), hoặc một phân phối phi Gaussian hoàn toàn (normalizing flow).

---

## **2. Cơ chế / Công thức**

### 2.1 Gaussian Mixture Model (GMM)

GMM là phương pháp đơn giản nhất và phổ biến nhất cho latent space có cấu trúc cluster rõ ràng:

$$p(z) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(z \mid \mu_k, \Sigma_k)$$

trong đó $K$ là số component; $\pi_k \geq 0$, $\sum_k \pi_k = 1$ là mixing weight; $\mu_k \in \mathbb{R}^d$ và $\Sigma_k \in \mathbb{R}^{d \times d}$ là mean và covariance của component $k$. Kết quả $p(z)$ là mật độ xác suất tại điểm $z$ — tổng có trọng số của $K$ Gaussian.

**Fitting bằng EM algorithm:**

- **E-step:** Tính posterior $r_{nk} = p(k \mid z_n) = \pi_k \mathcal{N}(z_n \mid \mu_k, \Sigma_k) / p(z_n)$ — xác suất mỗi điểm $z_n$ thuộc component $k$.
- **M-step:** Cập nhật parameters: $\pi_k = \frac{1}{N}\sum_n r_{nk}$, $\mu_k = \frac{\sum_n r_{nk} z_n}{\sum_n r_{nk}}$, $\Sigma_k = \frac{\sum_n r_{nk}(z_n - \mu_k)(z_n - \mu_k)^T}{\sum_n r_{nk}}$.

EM hội tụ về local maximum của log-likelihood $\sum_n \log p(z_n)$. Để chọn $K$, dùng BIC hoặc cross-validate trên held-out set. Với latent space cao chiều ($d > 100$), cần giảm chiều trước (PCA) hoặc dùng diagonal covariance để tránh overfitting.

**Log-likelihood cho OOD detection:**

$$\text{score}(z) = \log p(z) = \log \sum_{k=1}^{K} \pi_k \mathcal{N}(z \mid \mu_k, \Sigma_k)$$

Điểm có $\text{score}(z) < \tau$ được flag là OOD. Ngưỡng $\tau$ chọn từ percentile thấp (ví dụ: 5th percentile) của score trên tập training.

### 2.2 Normalizing Flow trên latent space

Normalizing flow (NF) học một ánh xạ khả nghịch $f_\theta: \mathbb{R}^d \to \mathbb{R}^d$ từ một phân phối đơn giản $p_0$ (thường là $\mathcal{N}(0, I)$) sang latent distribution $p(z)$. Công thức change-of-variables:

$$\log p(z) = \log p_0(f_\theta^{-1}(z)) + \log \left|\det J_{f_\theta^{-1}}(z)\right|$$

trong đó $f_\theta^{-1}$ là inverse của flow; $J_{f_\theta^{-1}}(z) = \frac{\partial f_\theta^{-1}}{\partial z}$ là Jacobian của inverse flow tại $z$; $\log |\det J|$ là log-determinant Jacobian — đo "how much volume the inverse mapping compresses or expands at $z$". Kết quả: $\log p(z)$ là log-density *chính xác*, không phải xấp xỉ.

Để tính $\log |\det J|$ hiệu quả, flow phải có cấu trúc đặc biệt:

- **Coupling layers** (Real NVP, Glow): Chia $z$ thành hai phần $[z_1, z_2]$; transform $z_2$ bằng hàm phụ thuộc $z_1$. Jacobian là triangular → $\det J = $ tích các phần tử đường chéo → $O(d)$.
- **Continuous flow** (FFJORD): Dùng Neural ODE. $\log |\det J|$ tính bằng Hutchinson trace estimator: $\log |\det J| = -\int_{t_0}^{t_1} \text{tr}(J_f) \, dt \approx -\int_{t_0}^{t_1} \epsilon^T J_f \epsilon \, dt$ với $\epsilon \sim \mathcal{N}(0, I)$. Đắt hơn coupling layers nhưng không cần cấu trúc đặc biệt.

Training NF bằng maximum likelihood: $\mathcal{L}(\theta) = \frac{1}{N} \sum_n \log p_\theta(z_n)$.

### 2.3 OOD detection với Likelihood Ratio

Nalisnick et al. (2019) phát hiện rằng deep generative model (flow, VAE) đôi khi gán likelihood **cao hơn** cho dữ liệu OOD so với in-distribution data. Lý do: model học được low-level statistics (background complexity, pixel entropy) thay vì semantic structure. Một tập OOD đơn giản hơn về mặt thống kê có thể có likelihood cao hơn.

Ren et al. (2019) đề xuất **likelihood ratio**:

$$\text{LLR}(z) = \log p_{\theta}(z) - \log p_{\text{bg}}(z)$$

trong đó $p_{\text{bg}}$ là một background model (ví dụ: GMM được fit trên toàn bộ không gian, không chỉ training data). LLR loại bỏ phần "background complexity" khỏi likelihood, để lại chỉ phần "semantic signal" — tốt hơn raw likelihood cho OOD detection.

Heuristic đơn giản hơn: thay vì raw likelihood, dùng **typicality test** — điểm $z$ bất thường khi:

$$\left| \log p(z) - H[p] \right| > \epsilon$$

trong đó $H[p] = -\mathbb{E}_{z \sim p}[\log p(z)]$ là entropy của phân phối training. Điểm "điển hình" có $\log p(z) \approx -H[p]$ — không quá cao cũng không quá thấp.

---

## **3. Biến thể**

| Phương pháp | Mật độ | Scalability | Đặc điểm |
|---|---|---|---|
| **Single Gaussian** | Chính xác (Gaussian) | $O(d^2)$ / $O(d^3)$ | Nhanh, tương đương Mahalanobis; fail với multi-modal distribution |
| **GMM** | Xấp xỉ (mixture) | $O(K d^2)$ | Capture multi-modal; cần chọn $K$; EM có thể bị local optima |
| **KDE** | Non-parametric | $O(N d)$ mỗi query | Không cần chọn structure; không scale được với $N$ lớn hoặc $d$ cao |
| **Normalizing Flow** | Chính xác | Phụ thuộc kiến trúc | Expressive nhất; nhưng fail với OOD (Nalisnick) nếu dùng raw likelihood |
| **VampPrior** | Xấp xỉ qua pseudo-inputs | $O(M d)$ | Học $p(z)$ cùng lúc với VAE; phù hợp khi model là VAE |
| **Score model** | Qua $\nabla_z \log p(z)$ | $O(d)$ mỗi query | Không cần tính $p(z)$ trực tiếp; dùng Langevin để sample; hiện đại hơn |

**VampPrior** (Tomczak & Welling, 2018): Thay vì prior $\mathcal{N}(0, I)$ đơn giản, dùng $p(z) = \frac{1}{M}\sum_m q_\phi(z \mid u_m)$ trong đó $\{u_m\}$ là tập "pseudo-inputs" được học cùng với encoder. Kết quả: prior fit sát hơn với aggregate posterior $q_\phi(z) = \frac{1}{N}\sum_n q_\phi(z \mid x_n)$, giảm posterior collapse và cải thiện ước lượng $p(z)$ trong latent space VAE.

**Score-based / Energy-based:** Thay vì ước lượng $p(z)$ trực tiếp, ước lượng score $s_\theta(z) = \nabla_z \log p(z)$ bằng score matching. Không cần normalizing constant. Dùng Langevin dynamics để sample: $z_{t+1} = z_t + \frac{\eta}{2} s_\theta(z_t) + \sqrt{\eta} \epsilon_t$ với $\epsilon_t \sim \mathcal{N}(0, I)$.

---

## **4. Giới hạn / Khi nào thất bại**

**Nalisnick phenomenon (deep NF fail for OOD).**  Flow model fit trên CIFAR-10 gán log-likelihood cao hơn cho ảnh SVHN (dataset rất khác) so với CIFAR-10 test set. Nguyên nhân: flow học được entropy thống kê của pixel — SVHN background đơn giản hơn, dễ nén hơn, nên flow "thích" hơn dù ngữ nghĩa hoàn toàn ngoài phân phối. Giải pháp: dùng likelihood ratio (Ren et al.), typicality test, hoặc kết hợp với Mahalanobis score ở feature space thay vì pixel space.

**Curse of dimensionality cho GMM.** Với $d \geq 100$, mỗi Gaussian component cần ít nhất $O(d^2)$ parameters cho full covariance. Với $K$ component và $d = 512$, đây là $K \times 262144$ parameters chỉ cho covariance — không thực tế. Phải dùng diagonal covariance (mất thông tin correlation) hoặc giảm chiều trước (PCA xuống $\tilde d \ll d$).

**KDE không scale.** Kernel density estimation có query cost $O(N)$ trong đó $N$ là số điểm training — không dùng được khi $N$ lớn (hàng triệu latent vectors từ dataset lớn). Approximations như KD-tree hay random sampling làm giảm độ chính xác.

**Local optima của EM.** EM cho GMM hội tụ về local maximum, không guaranteed global. Kết quả phụ thuộc vào khởi tạo. Nên chạy nhiều lần với khởi tạo khác nhau (ví dụ: k-means++), sau đó chọn kết quả có log-likelihood cao nhất.

**Drift sau fine-tune.** Sau khi model gốc bị fine-tune hoặc domain shift xảy ra, $p(z)$ đã học không còn đúng với latent distribution mới. Phải re-fit — đây là chi phí bảo trì không nhỏ cho hệ thống production.

**Không detect distribution shift theo hướng ngữ nghĩa.** GMM và NF fit theo geometric structure của $z$. Nếu model encoder encode hai concept khác nhau về ngữ nghĩa vào cùng vùng không gian (concept entanglement), density estimation sẽ không phân biệt được. Cần kết hợp với supervised probing (Tầng 5) để phát hiện semantic shift.

---

## **5. Liên hệ với Latent-Anything**

Density estimation là building block của **Layer B** (manipulation guard) và **Layer A** (introspection):

- **OOD guard trước decode:** `LatentSpace.log_prob(z)` → tính $\log p(z)$ theo model đã fit (GMM hoặc NF). Nếu thấp hơn ngưỡng, `LatentSpace.is_ood(z, tau=True)` trả về `True` và pipeline từ chối decode (hoặc kèm warning).
- **Density-guided sampling:** Kết hợp với Langevin dynamics, pipeline có thể "kéo" latent OOD về phía vùng in-distribution: `LatentSpace.project_to_distribution(z, n_steps=50)` — lặp gradient ascent trên $\log p(z)$.
- **Cluster assignment cho conditional generation:** `LatentSpace.assign_cluster(z)` → GMM posterior $p(k \mid z)$ — biết điểm $z$ thuộc cluster nào (ví dụ: style cluster) để chọn đúng decoder head hoặc conditioning vector.
- **Monitoring và alerting:** Chạy `LatentSpace.log_prob()` trên batch latent vectors theo thời gian để phát hiện distribution drift sớm — log-likelihood trung bình giảm đáng kể là dấu hiệu domain shift.

Pipeline tích hợp đề xuất cho latent space anisotropic: (1) **Whiten** bằng ZCA whitening ([Mahalanobis distance](05-mahalanobis-distance.md)); (2) **Giảm chiều** bằng PCA xuống $\tilde d$ chiều chứa 95% variance; (3) **Fit GMM** với $K = 5{-}20$ component trên $\tilde d$-D whitened space; (4) Dùng GMM log-prob làm **OOD score**.

---

## Liên quan

- [Mahalanobis distance (mục 05 — tầng này)](05-mahalanobis-distance.md) — single Gaussian case của density estimation; whitening trước khi fit GMM là pipeline chuẩn.
- [Subspace projection (mục 04 — tầng này)](04-subspace-projection.md) — giảm chiều bằng PCA trước density estimation để tránh curse of dimensionality.
- [Normalizing Flows](../../03-geometry-structure/research/06-normalizing-flows.md) — nền tảng toán học của flow-based density estimation: change-of-variables, coupling layers.
- [VAE](../../02-representation-learning/research/03-vae.md) — VampPrior là extension trực tiếp của VAE prior; aggregate posterior $q_\phi(z)$ là density estimation ẩn trong VAE.
- [Giả thuyết Đa tạp](../../01-space-representation/research/03-manifold-hypothesis.md) — lý do density estimation trong latent space có ý nghĩa: tập training chiếm một manifold thấp chiều, không phải toàn bộ $\mathbb{R}^d$.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — latent space anisotropic cần whitening trước khi fit GMM để các chiều được đối xử công bằng.

## Tham khảo

- C. M. Bishop, *Pattern Recognition and Machine Learning*, Chapter 9 — Mixture Models and EM (Springer, 2006). — Sách giáo khoa chuẩn cho GMM và EM algorithm; chứng minh E-step / M-step và điều kiện hội tụ.
- L. Dinh, J. Sohl-Dickstein, S. Bengio, *Density Estimation Using Real NVP* (ICLR 2017, arXiv:1605.08803). — Coupling-layer flow cho phép tính exact log-likelihood trong $O(d)$ thời gian.
- D. P. Kingma, P. Dhariwal, *Glow: Generative Flow with Invertible 1×1 Convolutions* (NeurIPS 2018, arXiv:1807.03039). — Flow-based model với invertible 1×1 convolution; tractable density estimation cho ảnh.
- W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever, D. Duvenaud, *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models* (ICLR 2019, arXiv:1810.01367). — Continuous normalizing flow dùng Neural ODE; không cần coupling structure; Hutchinson trace estimator cho log-det.
- E. Nalisnick, A. Matsukawa, Y. W. Teh, D. Görür, B. Lakshminarayanan, *Do Deep Generative Models Know What They Don't Know?* (ICLR 2019, arXiv:1810.09136). — Phát hiện flow model gán likelihood cao cho OOD data; typicality test là giải pháp được đề xuất.
- J. Ren, P. J. Liu, E. Fertig, J. Snoek, R. Poplin, M. A. DePristo, J. V. Dillon, B. Lakshminarayanan, *Likelihood Ratios for Out-of-Distribution Detection* (NeurIPS 2019, arXiv:1906.02845). — Likelihood ratio loại bỏ background complexity; cải thiện OOD detection đáng kể so với raw log-likelihood.
- J. M. Tomczak, M. Welling, *VAE with a VampPrior* (AISTATS 2018, arXiv:1705.07120). — VampPrior: prior học được từ pseudo-inputs; fit sát aggregate posterior; giảm posterior collapse.
