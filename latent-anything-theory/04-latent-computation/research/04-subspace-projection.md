# Subspace Projection — Tách và phân tích thành phần trong Latent Space

> **TL;DR.** Subspace projection tách $z \in \mathbb{R}^d$ thành hai phần vuông góc: $z = z_{\text{concept}} + z_{\text{residual}}$, trong đó $z_{\text{concept}} = UU^T z$ là hình chiếu lên subspace $k$-chiều được span bởi các cột của $U$. Phần khái niệm giữ đúng $k$ hướng ngữ nghĩa quan tâm; phần dư $z_{\text{residual}} = (I - UU^T)z$ giữ tất cả còn lại. Phương pháp chỉ có nghĩa khi các khái niệm được encode tuyến tính và không dùng chung hướng với các khái niệm không muốn giữ.

Latent space thường có hàng trăm hoặc hàng nghìn chiều, nhưng hầu hết ngữ nghĩa quan trọng nằm trong một subspace có chiều thấp hơn rất nhiều. Subspace projection khai thác điều này: thay vì làm việc với $z$ đầy đủ, phân tách $z$ thành "phần ứng với khái niệm X" và "phần không liên quan đến X", sau đó thao tác trực tiếp từng phần một. Đây là bước tự nhiên tiếp theo sau latent arithmetic — arithmetic dùng một *hướng* duy nhất $\delta$, còn projection dùng một *subspace* $k$-chiều, cho phép capture các khái niệm phức tạp hơn đòi hỏi nhiều hơn một chiều để biểu diễn.

---

## **1. Trực giác / Định nghĩa**

Hình dung latent space của một model ảnh mặt người. Toàn bộ không gian 512-chiều chứa thông tin về danh tính, cảm xúc, góc nhìn, ánh sáng, tuổi tác, … Nếu muốn *chỉnh* cảm xúc mà không làm thay đổi danh tính, cần tìm một subspace mà cảm xúc nằm trong đó nhưng danh tính thì không.

**Subspace projection** cung cấp công cụ cho điều đó:

1. Tìm một tập basis $\{u_1, \ldots, u_k\}$ (orthonormal) span subspace cần thiết.
2. Project $z$ lên subspace để lấy phần khái niệm: $z_{\text{concept}} = \sum_{i=1}^k (z \cdot u_i)\, u_i$.
3. Phần bổ sung $z_{\text{residual}} = z - z_{\text{concept}}$ không chứa thông tin nằm trong subspace đó.

Khi $k = 1$ (subspace một chiều), projection trùng với steering vector — $z_{\text{concept}} = (z \cdot \hat\delta)\,\hat\delta$. Khi $k > 1$, projection capture được khái niệm có cấu trúc đa chiều như "phong cách họa sĩ" hay "điều kiện thời tiết".

---

## **2. Cơ chế / Công thức**

### 2.1 Phép chiếu lên subspace

Cho basis orthonormal $U = [u_1 \mid \ldots \mid u_k] \in \mathbb{R}^{d \times k}$ (các cột orthonormal: $U^TU = I_k$). Projection matrix lên subspace này là:

$$P = UU^T \in \mathbb{R}^{d \times d}$$

trong đó $P$ là ma trận đối xứng, idempotent ($P^2 = P$), và rank $k$. Mỗi $z$ được tách thành:

$$z = \underbrace{Pz}_{z_{\text{concept}}} + \underbrace{(I - P)z}_{z_{\text{residual}}}$$

trong đó $z_{\text{concept}} \perp z_{\text{residual}}$ (vuông góc), $\|z\|^2 = \|z_{\text{concept}}\|^2 + \|z_{\text{residual}}\|^2$ (Pythagorean identity). Phần trăm năng lượng của $z$ nằm trong subspace là $\|Pz\|^2 / \|z\|^2 \in [0, 1]$.

**Annotate:** $u_i \in \mathbb{R}^d$ là basis vector thứ $i$, $z \cdot u_i$ là tọa độ của $z$ theo hướng $u_i$. Kết quả $z_{\text{concept}}$ là điểm trong subspace gần $z$ nhất theo Euclidean distance — đây là đặc tính tối ưu của orthogonal projection.

### 2.2 Concept negation (xóa concept)

Để loại bỏ hoàn toàn một concept khỏi $z$, giữ lại phần residual:

$$z_{\text{no-concept}} = (I - UU^T)\,z$$

Đây là phép chiếu lên *không gian bổ sung* (null space của $U^T$). Kết quả $z_{\text{no-concept}}$ giữ nguyên tất cả thông tin trong $z$ ngoại trừ thành phần dọc theo các hướng concept.

### 2.3 Tương tự hoá concept (concept transfer)

Thay thế phần concept của $z_{\text{target}}$ bằng phần concept của $z_{\text{source}}$:

$$z_{\text{transfer}} = (I - UU^T)\,z_{\text{target}} + UU^T\,z_{\text{source}}$$

trong đó $(I - UU^T)z_{\text{target}}$ giữ nội dung của target, còn $UU^Tz_{\text{source}}$ mang phong cách/concept của source. Đây là generalization của analogy arithmetic khi concept cần nhiều hơn một chiều.

---

## **3. Cách tìm basis U**

Ba họ phương pháp, tùy vào có label hay không và loại model:

| Phương pháp | Cần label? | Ưu điểm | Giới hạn |
|---|---|---|---|
| PCA (unsupervised) | Không | Tìm hướng high-variance tự động | Variance ≠ ngữ nghĩa; có thể bỏ sót concept nhỏ |
| Mean-difference / SVM (supervised) | Có | Trực tiếp target concept cụ thể | Cần annotated examples; chỉ 1 chiều (mean-diff) |
| SeFa / weight-space factorization | Không | Không cần data — chỉ cần model weights | Chỉ áp dụng cho GAN; concept không selectable |

**PCA (GANSpace style):** Thu thập $N$ latent vector $\{z_i\}$, stack thành ma trận $Z \in \mathbb{R}^{N \times d}$. Tính SVD: $Z - \bar Z = V\Sigma W^T$ (trong đó các cột của $W$ là right singular vectors). Top-$k$ columns của $W$ là basis cho subspace có variance cao nhất:

$$U = W_{:,1:k}$$

Härkönen et al. (2020) — GANSpace — áp dụng PCA trên feature activations thay vì trực tiếp trên $z$, cho kết quả interpretable hơn vì feature space của StyleGAN gần với "nhân tố độc lập" hơn latent space gốc.

**Mean-difference (supervised, 1-D):** Lấy $\delta = \bar z_{\text{with}} - \bar z_{\text{without}}$, normalize: $u_1 = \delta / \|\delta\|$. Đây là $k=1$ special case. Để có $k > 1$, tìm thêm direction trong null space của $u_1$ bằng PCA trên residuals.

**CAV (Concept Activation Vector — TCAV):** Train linear SVM trên tập $\{z_i\}$ với label có/không có concept. Normal vector của hyperplane (nghiệm $w$ của SVM) là CAV, normalized thành $u_1 = w/\|w\|$. Kim et al. (2018) dùng cách này để đo ảnh hưởng của concept đến prediction thông qua đạo hàm có hướng (*directional derivative*).

**SeFa (Closed-form Factorization):** Thay vì phân tích data, phân tích trực tiếp weight matrix $A$ của layer đầu tiên của generator (thường là linear map từ $z$ sang feature). Top-$k$ right singular vectors của $A$ là các hướng mà generator nhạy cảm nhất:

$$z_{\text{edit}} = z + \alpha \, v_k, \quad \text{với } v_k \text{ là right singular vector thứ } k \text{ của } A$$

Shen & Zhou (2021) cho thấy cách này recover được các attribute như tuổi, ánh sáng, góc nhìn mà không cần bất kỳ nhãn nào.

---

## **4. Giới hạn / Khi nào thất bại**

**Linearity assumption.** Subspace projection giả định concept có thể capture bởi một subspace tuyến tính của $\mathbb{R}^d$. Nếu "phong cách Van Gogh" nằm trên một manifold cong trong latent space (không phải flat subspace), projection tuyến tính sẽ chỉ capture một phần — và phần bị bỏ sót sẽ còn lại trong $z_{\text{residual}}$ theo cách không kiểm soát được.

**Subspace overlap và concept leakage.** Nếu "cảm xúc" và "danh tính" chia sẻ một số hướng chung, xóa emotion subspace cũng làm biến dạng danh tính một phần. Đây là phiên bản đa chiều của vấn đề concept leakage đã thấy ở latent arithmetic. Kiểm tra bằng cách đo $\|U_{\text{emotion}}^T U_{\text{identity}}\|_F$ — nếu gần 0, hai subspace gần như vuông góc và an toàn để dùng cùng lúc.

**PCA không bằng semantics.** PCA tìm hướng có variance cao nhất, không phải hướng ngữ nghĩa quan trọng nhất. Nếu một concept quan trọng (ví dụ: "nguy hiểm") nhưng hiếm trong dữ liệu, nó sẽ nằm ở singular value thấp và bị bỏ qua hoàn toàn bởi PCA. Supervised method (CAV, mean-difference) giải quyết vấn đề này nhưng đòi hỏi label.

**Instability khi $k$ nhỏ.** Với $k = 1$, basis rất nhạy với noise trong tập training. Nếu chỉ có 10 mẫu "with concept" và 10 mẫu "without concept", mean-difference sẽ bị dominated bởi noise trong các chiều không liên quan đến concept. Dùng ít nhất 50–100 mẫu mỗi class để ổn định.

**Metric mismatch.** Phép chiếu $z_{\text{concept}} = UU^T z$ tối ưu theo Euclidean distance. Nếu latent space anisotropic (có covariance $\Sigma \neq I$), Euclidean distance không phản ánh đúng "khoảng cách ngữ nghĩa". Trong trường hợp này, cần whitening trước: $z' = \Sigma^{-1/2} z$, thực hiện projection trong $z'$-space, rồi unwhiten kết quả. Đây là mối liên hệ trực tiếp với **Mahalanobis distance** (mục tiếp theo).

---

## **5. Liên hệ với Latent-Anything**

Subspace projection là thao tác cốt lõi của **Layer B** và là nền tảng của **Layer A (introspection)**:

- **Concept injection / erasure**: `LatentSpace.project(z, U)` → $UU^T z$ và `LatentSpace.erase(z, U)` → $(I - UU^T) z$.
- **Concept transfer**: `LatentSpace.transfer(z_source, z_target, U)` → hoán đổi phần concept giữa hai latent.
- **Diagnostic**: `LatentSpace.concept_coverage(z, U)` → $\|UU^T z\|^2 / \|z\|^2$ — đo bao nhiêu % latent nằm trong concept subspace.
- **Interpretability pipeline**: `Layer A` sẽ dùng PCA hoặc CAV để tìm $U$ tự động từ tập labelled hoặc unlabelled examples, rồi expose `ConceptSubspace` object.

Subspace projection cũng là building block của các Layer A method phức tạp hơn ở tầng 5 (probe, intervention): linear probe chính là tìm $u_1$ bằng logistic regression; TCAV đo $\nabla_z \hat{y} \cdot u_1$ để đánh giá ảnh hưởng của concept lên output; activation patching hoán đổi $z_{\text{concept}}$ giữa hai forward pass.

---

## Liên quan

- [Latent arithmetic (mục 03 — tầng này)](03-latent-arithmetic.md) — trường hợp $k = 1$: subspace một chiều. Steering vector = concept direction duy nhất.
- [Cấu trúc tuyến tính](../../03-geometry-structure/research/01-linear-structure.md) — lý do subspace tuyến tính tồn tại trong latent space của model học tốt.
- [Tính tách biệt biểu diễn (disentanglement)](../../03-geometry-structure/research/02-disentanglement.md) — latent disentangled = các concept subspace vuông góc với nhau → projection sạch không leakage.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — khi không gian anisotropic, cần whitening trước khi projection để Euclidean metric có nghĩa.

## Tham khảo

- B. Kim, M. Wattenberg, J. Gilmer, C. Cai, J. Wexler, F. Viégas, R. Sayres, *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)* (ICML 2018, arXiv:1711.11279). — Giới thiệu CAV: normal vector của SVM hyperplane trong activation space; dùng directional derivative để đo ảnh hưởng concept.
- E. Härkönen, A. Hertzmann, J. Lehtinen, S. Paris, *GANSpace: Discovering Interpretable GAN Controls* (NeurIPS 2020, arXiv:2004.02546). — PCA trên latent/feature space của GAN để tìm interpretable edit directions mà không cần label.
- Y. Shen, B. Zhou, *Closed-Form Factorization of Latent Semantics in GANs* (CVPR 2021, arXiv:2007.06600). — Phân tích weight matrix của generator để tìm semantic directions mà không cần data; closed-form, dưới 1 giây để chạy trên bất kỳ GAN nào.
