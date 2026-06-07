# Slerp — Spherical Linear Interpolation

> **TL;DR.** Slerp nội suy theo **cung tròn lớn** trên mặt siêu cầu thay vì dây cung thẳng — bảo toàn norm và đi với vận tốc góc đều. Dùng thay lerp khi latent sống trên vỏ cầu (prior Gaussian chiều cao); suy biến về lerp khi góc giữa hai điểm nhỏ.

**Slerp (Spherical Linear Interpolation — nội suy tuyến tính cầu)** là phép nội suy giữa hai vector đi men theo **cung tròn lớn (great circle)** trên mặt siêu cầu đơn vị, thay vì đi theo dây cung thẳng băng qua lòng cầu như Lerp. Đây chính là hiện thực hóa cụ thể của đường trắc địa ([geodesic](../../01-space-representation/research/05-geodesic.md)) trên trường hợp đặc biệt — và quan trọng nhất với deep learning — là mặt cầu.

Thuật ngữ "Slerp" được Ken Shoemake đặt ra trong bài *"Animating Rotation with Quaternion Curves"* (SIGGRAPH 1985) để nội suy mượt các phép quay 3D biểu diễn bằng quaternion. Tuy ra đời trong computer graphics, công thức của nó áp dụng được cho **bất kỳ vector đơn vị nào trong một không gian tích trong (inner product space)** — đó là lý do nó trở thành công cụ tiêu chuẩn cho nội suy trong latent space.

---

## **1. Trực giác hình học**

Cho hai vector đơn vị $p_0$ và $p_1$ trên mặt siêu cầu, hợp với nhau một góc $\Omega$ tại tâm cầu.

* **Lerp** vẽ một đường thẳng (dây cung) nối hai điểm: $\text{lerp}(p_0, p_1; t) = (1-t)\,p_0 + t\,p_1$. Đường này **chui vào trong lòng cầu** — các điểm giữa có chuẩn (norm) nhỏ hơn, đặc biệt tại $t=0.5$ chuẩn bị suy giảm mạnh nhất.
* **Slerp** đi **men theo bề mặt cầu**, dọc theo cung tròn lớn — chính là đường ngắn nhất (geodesic) nối hai điểm trên cầu. Mọi điểm trung gian đều nằm đúng trên mặt cầu, **bảo toàn chuẩn**.

Có thể hiểu Slerp như quay vector $p_0$ dần dần về phía $p_1$ với một **vận tốc góc không đổi**: tại $t$, vector đã quét được đúng góc $t\Omega$ trong tổng góc $\Omega$.

---

## **2. Công thức**

Công thức Slerp (dạng của Glenn Davis, được Shoemake trích dẫn):

$$
\text{slerp}(p_0, p_1; t) = \frac{\sin\big((1-t)\,\Omega\big)}{\sin \Omega}\, p_0 \;+\; \frac{\sin\big(t\,\Omega\big)}{\sin \Omega}\, p_1
$$

trong đó:

* $t \in [0, 1]$ là tham số nội suy ($t=0$ cho $p_0$, $t=1$ cho $p_1$).
* $\Omega$ là góc giữa hai vector, tính qua tích vô hướng của hai vector **đã chuẩn hóa**: $\cos\Omega = p_0 \cdot p_1$.
* Hệ số $1/\sin\Omega$ là chuẩn hóa để đảm bảo kết quả luôn nằm trên cung và đi với vận tốc góc đều.

Một dạng tương đương cho quaternion (và mọi vector đơn vị):

$$
\text{slerp}(q_0, q_1; t) = q_0\,(q_0^{-1} q_1)^{t}
$$

dạng này thể hiện rõ bản chất "quay dần" — nâng phép quay từ $q_0$ sang $q_1$ lên lũy thừa $t$.

### Liên hệ exp/log map

Slerp chính là công thức geodesic $\gamma(t) = \text{Exp}_{p_0}\!\big(t \cdot \text{Log}_{p_0}(p_1)\big)$ viết riêng cho mặt cầu, nơi exponential map và logarithmic map có dạng đóng (closed form). So với pullback geodesic phải giải bài toán tối ưu năng lượng, Slerp **rẻ và tức thời** vì hình học của cầu đã biết trước.

---

## **3. Các tính chất quan trọng**

### Bảo toàn chuẩn (norm preservation)
Vì mọi điểm nội suy luôn nằm trên mặt cầu đơn vị, $\|\text{slerp}(p_0,p_1;t)\| = 1$ với mọi $t$. Đây là tính chất quyết định khiến Slerp vượt trội Lerp trong latent space (xem phần 4).

### Vận tốc góc không đổi (constant angular velocity)
Góc quét được là hàm tuyến tính của $t$. Điều này tạo ra chuyển động/biến đổi **đều và mượt** — không bị tăng tốc rồi giảm tốc như khi chuẩn hóa Lerp (nlerp).

### Suy biến về Lerp khi góc nhỏ
Khi $\Omega \to 0$, công thức Slerp tiến đúng về Lerp đối xứng $(1-t)p_0 + t\,p_1$. Tức là khi hai điểm rất gần nhau, hai phép nội suy gần như trùng nhau — đường thẳng và cung tròn lớn gần như không phân biệt.

### Đối xứng
$\text{slerp}(p_0, p_1; t) = \text{slerp}(p_1, p_0; 1-t)$.

### Đường ngắn nhất / cung lớn
Slerp đi theo geodesic của mặt cầu (cung tròn lớn) — tương đương đoạn thẳng của hình học phẳng nhưng trên không gian cong.

---

## **4. Khi nào dùng Slerp thay Lerp?**

Slerp phù hợp khi **dữ liệu sống trên (hoặc gần) một mặt siêu cầu** — đây là tình huống cực kỳ phổ biến trong các mô hình sinh:

* Prior Gaussian đẳng hướng $\mathcal{N}(0, I)$ trong VAE, GAN, diffusion. Do hiện tượng **tập trung độ đo (concentration of measure)** ở số chiều cao, gần như toàn bộ khối lượng xác suất tụ trên một lớp **vỏ siêu cầu mỏng** có bán kính $\approx \sqrt{d}$, chứ không ở gốc tọa độ. (Xem thêm [curse of dimensionality](../../01-space-representation/research/04-curse-of-dimensionality.md).)

Trong tình huống đó, **Lerp thất bại** vì:

1. **Suy giảm chuẩn (norm degradation):** dây cung Lerp chui vào lõi cầu, điểm giữa có chuẩn nhỏ hơn $\sqrt{d}$ — rơi vào vùng **xác suất cực thấp** mà decoder chưa từng thấy. Hệ quả: ảnh trung gian mờ nhòe, mất chi tiết (hiện tượng "tent-pole": rõ ở hai đầu, mờ ở giữa).
2. **Off-manifold:** vì rời khỏi vỏ cầu mật độ cao, điểm nội suy nằm ngoài đa tạp dữ liệu.

**Slerp khắc phục cả hai** bằng cách giữ điểm nội suy luôn trên vỏ cầu mật độ cao → biến đổi sắc nét, mượt, ngữ nghĩa hợp lý. Đây là lý do Slerp là lựa chọn mặc định khi nội suy **noise vector của diffusion** hay **latent của GAN**.

**Khi nào Lerp là đủ:** khi latent **không** có cấu trúc cầu (ví dụ không gian gần Euclidean, các điểm rất gần nhau, hoặc latent đã được "làm phẳng"), Lerp vừa đủ tốt vừa rẻ hơn. Khi $\Omega$ nhỏ, hai phép gần như đồng nhất.

---

## **5. Lưu ý khi cài đặt (edge cases)**

* **Góc gần 0 ($\sin\Omega \approx 0$):** mẫu số $\sin\Omega$ tiến về 0 gây mất ổn định số học. Giải pháp: khi $\Omega$ dưới một ngưỡng nhỏ, **rơi về Lerp** (rồi chuẩn hóa nếu cần) — vì lúc này hai kết quả gần như trùng nhau.
* **Điểm đối cực / đường dài (antipodal):** nếu $\cos\Omega < 0$ (góc tù), Slerp có thể đi vòng đường dài. Với quaternion (do tính phủ kép — $q$ và $-q$ cùng một phép quay) người ta **đảo dấu một đầu** khi tích vô hướng âm để ép $-90° \le \Omega \le 90°$, đảm bảo đi đường ngắn. Khi $\Omega \to 180°$ (đối cực thực sự) đường geodesic không xác định duy nhất.
* **Chuẩn hóa đầu vào:** công thức giả định $p_0, p_1$ là vector đơn vị. Nếu latent chưa chuẩn hóa, cần chuẩn hóa trước (và lưu lại norm nếu muốn khôi phục độ lớn) — vì Slerp thuần túy thao tác trên *hướng*.
* **nlerp (normalized lerp)** là phương án xấp xỉ rẻ hơn: làm Lerp rồi chuẩn hóa về cầu. Nó bảo toàn chuẩn nhưng **không** giữ vận tốc góc đều — đường đi vẫn là cung lớn nhưng tốc độ không đều.

---

## **6. Giới hạn của Slerp và hướng mở rộng**

Slerp giải quyết tốt bài toán **hai điểm, prior zero-mean unit-covariance** $\mathcal{N}(0, I)$. Nhưng nó không tổng quát cho:

* **Mean khác 0 ($\mu \neq 0$):** Slerp giả định tâm cầu ở gốc tọa độ; với phân phối lệch tâm, việc giữ chuẩn quanh gốc không còn khớp với vùng mật độ cao.
* **Hiệp phương sai khác $I$:** với $\Sigma \neq I$ đa tạp mật độ cao là một **ellipsoid** chứ không phải mặt cầu, Slerp không còn bám đúng.
* **Nhiều hơn 2 vector:** Slerp khó tổng quát sang centroid, trung bình có trọng số, hay dựng subspace từ nhiều latent.

**Hướng tổng quát hơn:** thay vì chỉ bảo toàn chuẩn, các phương pháp gần đây chuyển sang **khớp phân phối (distribution matching)** — đảm bảo mọi tổ hợp tuyến tính của các latent vẫn tuân theo đúng phân phối gốc $\mathcal{N}(\mu, \Sigma)$. Ví dụ phương pháp **LOL (Latent Optimal Linear combinations)** dùng biến đổi dạng đóng để hiệu chỉnh cả mean và covariance, mở rộng được cho centroid và subspace, không chỉ nội suy 2 điểm. Với đa tạp cong phức tạp không phải cầu, quay về [pullback geodesic / FlatVI](../../01-space-representation/research/06-pullback-metric.md).

---

## **7. Liên hệ với Latent-Anything**

* Slerp là một **operation cốt lõi của Layer B (manipulation)** — sẽ được implement song song với Lerp như hai phương án nội suy của `Trajectory`/`LatentSpace`.
* Nó cũng là cầu nối thực tiễn giữa lý thuyết geodesic (tầng 1) và code: là trường hợp geodesic có dạng đóng, không cần solver.
* Quyết định **chọn Lerp hay Slerp** nên dựa trên hình học của latent space cụ thể (có phải cầu/anisotropic không — xem [isotropy vs anisotropy](03-isotropy-anisotropy.md)), nên framework cần phát hiện hoặc cho người dùng khai báo cấu trúc này.

---

## Tham khảo

* Ken Shoemake, *Animating Rotation with Quaternion Curves*, SIGGRAPH 1985.
* [Spherical linear interpolation — Wikipedia](https://en.wikipedia.org/wiki/Slerp)
* [Game Math: Deriving the Slerp Formula — Allen Chou](https://allenchou.net/2018/05/game-math-deriving-the-slerp-formula/)
* [Linear combinations of latents in generative models: subspaces and beyond (LOL), arXiv:2408.08558](https://arxiv.org/html/2408.08558)
* [Addressing degeneracies in latent interpolation for diffusion models, arXiv:2505.07481](https://arxiv.org/html/2505.07481v1)
