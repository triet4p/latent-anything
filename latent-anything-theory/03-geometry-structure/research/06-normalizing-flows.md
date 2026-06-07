# Normalizing Flows

**Normalizing Flow (Luồng chuẩn hóa)** là một họ mô hình sinh học một **ánh xạ khả nghịch (bijective)** giữa một phân phối đơn giản đã biết (base distribution — thường là Gaussian $\mathcal{N}(0, I)$) và một phân phối phức tạp của dữ liệu (hoặc của latent). Nhờ tính khả nghịch, nó cho phép **ước lượng mật độ chính xác (exact density estimation)** — điều mà VAE (chỉ tối ưu cận dưới ELBO) và GAN (không có likelihood) không làm được.

Tên gọi đến từ chính cơ chế: dữ liệu được "chảy" (flow) qua một chuỗi phép biến đổi, mỗi bước được "chuẩn hóa" (normalize) lại thể tích bằng định thức Jacobian, cho tới khi biến thành phân phối base chuẩn tắc.

---

## **1. Công thức đổi biến (Change of Variables)**

Đây là nền tảng toán học của toàn bộ phương pháp. Cho $z \sim p_Z$ (base, ví dụ $\mathcal{N}(0,I)$) và một hàm khả nghịch, khả vi $f$ với $x = f(z)$. Khi đó mật độ của $x$ là:

$$
p_X(x) = p_Z\big(f^{-1}(x)\big)\,\left|\det J_{f^{-1}}(x)\right|
= p_Z(z)\,\left|\det J_{f}(z)\right|^{-1}
$$

trong đó $J_f = \partial f / \partial z$ là **ma trận Jacobian** của phép biến đổi.

* **Vai trò của định thức Jacobian:** $|\det J|$ đo **mức thay đổi thể tích** cục bộ do phép biến đổi gây ra. Khi $f$ kéo giãn một vùng không gian, mật độ ở đó phải giảm tương ứng để tổng xác suất vẫn bằng 1 — định thức Jacobian chính là hệ số điều chỉnh đó. Đây là lý do có chữ "normalizing".

Lấy log để huấn luyện ổn định:

$$
\log p_X(x) = \log p_Z\big(f^{-1}(x)\big) + \log\left|\det J_{f^{-1}}(x)\right|
$$

* **Huấn luyện = maximum likelihood trực tiếp.** Vì có likelihood chính xác, ta tối ưu trực tiếp $\log p_X(x)$ trên dữ liệu — **không cần cận dưới** như ELBO của [VAE](../../02-representation-learning/research/03-vae.md). Đây là khác biệt cốt lõi.

---

## **2. Hai chiều của luồng**

Một flow đã train dùng được theo cả hai chiều:

* **Chiều chuẩn hóa (normalizing): $x \to z$.** Đưa dữ liệu phức tạp về base đơn giản. Dùng để **đánh giá mật độ** $p_X(x)$ và để huấn luyện.
* **Chiều sinh (generative): $z \to x$.** Lấy mẫu $z \sim \mathcal{N}(0,I)$ rồi đẩy qua $f$ để **sinh dữ liệu mới**.

Cùng một bộ tham số phục vụ cả hai — đây là tính đối xứng đẹp mà VAE/GAN không có.

---

## **3. Ghép nhiều phép biến đổi (Composition)**

Một phép biến đổi đơn giản không đủ biểu diễn phân phối phức tạp. Giải pháp: **ghép một chuỗi** các phép khả nghịch $f = f_M \circ \dots \circ f_2 \circ f_1$. Hợp của các bijection vẫn là bijection, và log-định thức cộng dồn:

$$
\log\left|\det J_f\right| = \sum_{i=1}^{M} \log\left|\det J_{f_i}\right|
$$

Mỗi bước "uốn" phân phối thêm một chút; xếp chồng đủ nhiều bước cho phép biểu diễn phân phối tùy ý phức tạp.

### Hai ràng buộc thiết kế bắt buộc
Mỗi $f_i$ phải đồng thời:
1. **Khả nghịch** — để đi được cả hai chiều.
2. **Có định thức Jacobian tính được rẻ** — định thức tổng quát tốn $O(D^3)$, không khả thi ở chiều cao.

Toàn bộ "nghệ thuật" của normalizing flows nằm ở việc thiết kế các lớp thỏa mãn cả hai mà vẫn đủ biểu cảm.

---

## **4. Các kiến trúc tiêu biểu**

### Coupling layers (lớp ghép cặp)
Mẹo cốt lõi: chia vector đầu vào làm hai phần, giữ nguyên một nửa và dùng nửa đó để biến đổi nửa còn lại.

* **NICE (Dinh et al., 2014):** additive coupling — chỉ dịch chuyển (shift), định thức Jacobian = 1.
* **RealNVP (Dinh et al., 2017):** **affine coupling** — thêm cả scale. Đây là kiến trúc kinh điển nhất:

$$
y_{1:d} = x_{1:d}, \qquad y_{d+1:D} = x_{d+1:D} \odot \exp\big(s(x_{1:d})\big) + t(x_{1:d})
$$

  Vì nửa sau chỉ phụ thuộc nửa đầu (đã giữ nguyên), **Jacobian là ma trận tam giác dưới**, nên định thức = tích đường chéo:

$$
\log\left|\det J\right| = \sum_j s(x_{1:d})_j
$$

  Phép nghịch đảo cũng dạng đóng:

$$
x_{1:d} = y_{1:d}, \qquad x_{d+1:D} = \big(y_{d+1:D} - t(y_{1:d})\big) \odot \exp\big(-s(y_{1:d})\big)
$$

  **Điểm thông minh:** nghịch đảo *không* cần đảo $s$ hay $t$, nên $s, t$ có thể là mạng neural tùy ý phức tạp mà vẫn khả nghịch. Cần xen kẽ **permutation** giữa các lớp để mọi chiều đều được biến đổi.

* **Glow (Kingma & Dhariwal, 2018):** RealNVP + **invertible 1×1 convolution** (thay permutation cố định bằng phép trộn kênh học được) + actnorm. Đạt kết quả sinh ảnh chất lượng cao.

### Autoregressive flows
Biến đổi từng chiều, mỗi chiều phụ thuộc các chiều trước → Jacobian tam giác tự nhiên.
* **MAF (Masked Autoregressive Flow):** đánh giá mật độ nhanh (một lượt), nhưng lấy mẫu chậm (tuần tự $D$ bước).
* **IAF (Inverse Autoregressive Flow):** ngược lại — lấy mẫu nhanh, đánh giá mật độ chậm. IAF hay dùng để **cải thiện posterior của VAE**.

### Các hướng khác
* **Neural Spline Flows (Durkan et al., 2019):** thay biến đổi affine bằng spline hữu tỉ bậc hai (rational-quadratic), tăng mạnh biểu cảm cho mỗi lớp.
* **Continuous Normalizing Flows / FFJORD:** định nghĩa flow bằng một ODE (Neural ODE) thay vì chuỗi lớp rời rạc; định thức được thay bằng tích phân vết (trace) ước lượng bằng Hutchinson estimator.

---

## **5. Giới hạn và cạm bẫy**

### Bảo toàn số chiều
Vì cần khả nghịch, base $z$ và dữ liệu $x$ **phải cùng số chiều**. Khác với VAE/autoencoder, flow **không nén được chiều** — nó không học một bottleneck. Điều này tốn kém ở dữ liệu chiều cao và mâu thuẫn với [giả thuyết đa tạp](../../01-space-representation/research/03-manifold-hypothesis.md) (dữ liệu thật nằm trên submanifold chiều thấp).

### Ràng buộc topology
Một phép đồng phôi (homeomorphism — bijective liên tục, nghịch đảo liên tục) **bảo toàn topology**. Vì base thường là $\mathbb{R}^D$ liên thông, flow **không thể** biểu diễn chính xác phân phối có topology khác — ví dụ phân phối đa thành phần rời nhau (multi-modal tách biệt) hay có "lỗ". Mô hình buộc phải kéo những vùng đáng lẽ rời nhau dính vào nhau bằng các "cầu" mật độ thấp giả tạo.

### Thất bại trong OOD detection
Trực giác "likelihood cao = in-distribution" **không đáng tin** với flows. Nghiên cứu kinh điển *"Why Normalizing Flows Fail to Detect Out-of-Distribution Data"* (Nalisnick et al. / Kirichenko et al., NeurIPS 2020) cho thấy flow train trên CIFAR-10 lại gán **likelihood cao hơn** cho ảnh SVHN (out-of-distribution). Nguyên nhân: flow học các đặc trưng cục bộ chung chung (như độ mượt của pixel) thay vì ngữ nghĩa. → Dùng flow cho OOD cần cẩn trọng, thường phải estimate mật độ **trên latent của một encoder** chứ không trên pixel thô.

---

## **6. So sánh nhanh với VAE và GAN**

| Tiêu chí | Normalizing Flow | VAE | GAN |
|---|---|---|---|
| Likelihood | **Chính xác** | Cận dưới (ELBO) | Không có |
| Encoder/decoder | Một bijection (chung) | Stochastic, tách rời | Chỉ generator |
| Nén chiều | Không (giữ nguyên $D$) | Có (bottleneck) | Có |
| Latent → data | Khả nghịch, xác định | Decode (stochastic) | Generate |
| Điểm yếu | Ràng buộc kiến trúc, topology | Output mờ | Không có likelihood, mode collapse |

Flow và VAE còn **lai ghép** được: dùng một flow (như IAF) làm posterior linh hoạt $q(z|x)$ cho VAE, nới lỏng giả định Gaussian.

---

## **7. Liên hệ với Latent-Anything**

* **Ước lượng mật độ trong latent (Tầng 4):** flow là công cụ chính để học $p(z)$ chính xác trên latent của một model khác — phục vụ **in/out-of-distribution detection** và phát hiện điểm bất thường trong Layer A (introspection). (Lưu ý cạm bẫy OOD ở mục 5.)
* **Công cụ "làm thẳng" không gian:** flow ánh xạ latent cong về một base Gaussian đẳng hướng, nơi [lerp](04-riemannian-geometry.md) và khoảng cách Euclidean lại có nghĩa — bổ trợ cho [slerp](05-slerp.md) và pullback-geodesic. Cùng tinh thần với [FlatVI](../../01-space-representation/research/07-flatvi.md) ở Tầng 1.
* **Manipulation khả nghịch:** vì bijective, mọi thao tác trong không gian base đều ánh xạ ngược lại được về latent gốc một cách xác định — thuận lợi cho các phép can thiệp có kiểm soát ở Layer B.

---

## Tham khảo

* Dinh, Sohl-Dickstein, Bengio, *Density Estimation using Real NVP*, ICLR 2017 — [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
* Kingma, Dhariwal, *Glow: Generative Flow with Invertible 1×1 Convolutions*, NeurIPS 2018.
* Papamakarios et al., *Normalizing Flows for Probabilistic Modeling and Inference*, JMLR 2021 — [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)
* Kobyzev et al., *Normalizing Flows: An Introduction and Review of Current Methods* — [arXiv:1908.09257](https://arxiv.org/abs/1908.09257)
* Kirichenko, Izmailov, Wilson, *Why Normalizing Flows Fail to Detect Out-of-Distribution Data*, NeurIPS 2020 — [paper](https://proceedings.neurips.cc/paper/2020/file/ecb9fe2fbb99c31f567e9823e884dbec-Paper.pdf)
