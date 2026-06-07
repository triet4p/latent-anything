# Slerp — Spherical Linear Interpolation: Implementation & Depth

> **TL;DR.** Slerp là geodesic trên hypersphere: $\text{slerp}(p_0, p_1; t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,p_0 + \frac{\sin(t\Omega)}{\sin\Omega}\,p_1$, với $\Omega = \arccos(p_0 \cdot p_1)$ là góc giữa hai vector đơn vị. Khác lerp, slerp bảo toàn norm và di chuyển với vận tốc góc đều, khiến mọi điểm trung gian nằm đúng trên vỏ cầu mật độ cao của prior. Hai edge case cần xử lý kỹ trong code: $\Omega \to 0$ (suy biến số học) và $\Omega \to \pi$ (geodesic không duy nhất).

Note này là phần thực hành của [Slerp (tầng 3 — giới thiệu)](../../03-geometry-structure/research/05-slerp.md), tập trung vào ba nội dung mà note tầng 3 chỉ phác thảo: chứng minh công thức từ hình học, cài đặt đầy đủ với xử lý edge case, và so sánh định lượng với lerp và nlerp.

---

## **1. Trực giác / Định nghĩa**

Vấn đề: cho hai vector đơn vị $p_0, p_1 \in S^{d-1}$ (mặt cầu đơn vị $d-1$ chiều) và tham số $t \in [0,1]$, tìm điểm $z(t)$ nằm trên **cung tròn lớn (great-circle arc)** nối $p_0$ và $p_1$ sao cho:

1. $z(0) = p_0$, $z(1) = p_1$
2. $\|z(t)\| = 1$ với mọi $t$ (bảo toàn norm)
3. Vận tốc góc không đổi: góc đã quét tại $t$ là đúng $t\Omega$ trong tổng $\Omega$

Lerp vi phạm điều kiện 2 và 3. Slerp thoả mãn cả ba.

---

## **2. Chứng minh công thức từ hình học**

**Bước 1 — Xây trục tọa độ trong mặt phẳng 2D chứa arc.**

Đặt $e_1 = p_0$. Tạo vector trực giao $e_2$ trong mặt phẳng chứa $p_0, p_1$:

$$e_2 = \frac{p_1 - (p_1 \cdot p_0)\,p_0}{\|p_1 - (p_1 \cdot p_0)\,p_0\|} = \frac{p_1 - \cos\Omega\cdot p_0}{\sin\Omega}$$

trong đó $\cos\Omega = p_0 \cdot p_1$. Khi đó $\{e_1, e_2\}$ là cơ sở trực chuẩn của mặt phẳng đó.

**Bước 2 — Viết điểm trên arc.**

Điểm trên arc đơn vị ở góc $t\Omega$ so với $p_0$ là:

$$z(t) = \cos(t\Omega)\cdot e_1 + \sin(t\Omega)\cdot e_2$$

Thay $e_2$ vào, thu được:

$$z(t) = \cos(t\Omega)\cdot p_0 + \frac{\sin(t\Omega)}{\sin\Omega}\cdot p_1 - \frac{\sin(t\Omega)\cos\Omega}{\sin\Omega}\cdot p_0$$

$$= \frac{\cos(t\Omega)\sin\Omega - \sin(t\Omega)\cos\Omega}{\sin\Omega}\cdot p_0 + \frac{\sin(t\Omega)}{\sin\Omega}\cdot p_1$$

Nhận ra tử số là công thức trừ góc $\sin(\Omega - t\Omega) = \sin((1-t)\Omega)$:

$$\boxed{\text{slerp}(p_0, p_1; t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,p_0 + \frac{\sin(t\Omega)}{\sin\Omega}\,p_1}$$

trong đó $\Omega = \arccos(\text{clip}(p_0 \cdot p_1, -1, 1))$ là góc giữa hai vector, và $\sin\Omega$ ở mẫu đóng vai trò chuẩn hóa để kết quả có norm 1. **Công thức này đúng với vector đơn vị bất kỳ trong không gian tích trong bất kỳ chiều nào** — không chỉ quaternion.

---

## **3. Cài đặt đầy đủ — xử lý edge case**

### 3.1. Thuật toán

```python
def slerp(p0: np.ndarray, p1: np.ndarray, t: float,
          eps: float = 1e-7) -> np.ndarray:
    """
    Slerp giữa hai vector (có thể không chuẩn hóa sẵn).
    Args:
        p0, p1: vector đầu vào, shape (..., d)
        t:      tham số nội suy ∈ [0, 1]
        eps:    ngưỡng cho edge case gần-0 và gần-pi
    Returns:
        vector nội suy, cùng shape và norm như p0/p1
    """
    # 1. Chuẩn hóa và lưu norm
    n0 = np.linalg.norm(p0, axis=-1, keepdims=True)
    n1 = np.linalg.norm(p1, axis=-1, keepdims=True)
    u0 = p0 / np.clip(n0, eps, None)   # unit vector
    u1 = p1 / np.clip(n1, eps, None)

    # 2. Tính góc Ω
    dot = np.clip(np.sum(u0 * u1, axis=-1), -1.0, 1.0)
    omega = np.arccos(dot)              # ∈ [0, π]

    # 3. Edge case: Ω gần 0 → suy biến về lerp (rồi chuẩn hóa)
    # (khi sin(Ω) ≈ 0, mẫu số không ổn định; lerp cho kết quả gần đúng)
    near_zero = np.abs(omega) < eps
    sin_omega = np.sin(omega)

    # 4. Tính hệ số
    coeff0 = np.where(near_zero, 1.0 - t,
                      np.sin((1.0 - t) * omega) / sin_omega)
    coeff1 = np.where(near_zero, t,
                      np.sin(t * omega) / sin_omega)

    # 5. Nội suy trên unit sphere
    result_unit = coeff0[..., None] * u0 + coeff1[..., None] * u1

    # 6. Khôi phục norm (nội suy tuyến tính trong scale)
    norm_out = (1 - t) * n0 + t * n1
    return result_unit * norm_out
```

### 3.2. Ba edge case quan trọng

**Edge case 1: $\Omega \to 0$ (hai vector gần như trùng nhau).**

Khi $\Omega < \varepsilon$ (ví dụ $\varepsilon = 10^{-7}$), $\sin\Omega \approx \Omega \to 0$ — mẫu số gần bằng 0 và phép chia mất ổn định số học. Giải pháp: khi $\Omega < \varepsilon$, thay thế bằng lerp (rồi chuẩn hóa nếu cần). Lúc này sai số giữa slerp và lerp là $O(\Omega^2)$ — không đáng kể.

**Edge case 2: $\Omega \to \pi$ (gần đối cực).**

Khi hai vector gần đối cực ($\cos\Omega \approx -1$), có vô số great-circle arc nối chúng (tất cả đều có cùng độ dài). Slerp không xác định duy nhất — kết quả phụ thuộc vào hướng vector $e_2$, dễ không ổn định. Trong thực tế, khi $\Omega > \pi - \varepsilon$: hoặc báo lỗi, hoặc chọn một mặt phẳng cố định (ví dụ thêm nhiễu nhỏ vào một chiều).

**Edge case 3: Đảo dấu (quaternion, góc tù).**

Với quaternion biểu diễn phép quay 3D, $q$ và $-q$ đại diện cho cùng một phép quay (double cover). Khi $\cos\Omega < 0$ (góc tù), slerp đi đường dài (> $\pi$). Fix: nếu $p_0 \cdot p_1 < 0$, đảo dấu $p_1 \leftarrow -p_1$ để ép $\Omega \in [0, \pi/2]$, đảm bảo đi đường ngắn. **Với latent vector thông thường** (không phải quaternion), đảo dấu không có ý nghĩa — chỉ áp dụng nếu latent space có tính đối xứng rõ ràng.

---

## **4. Biến thể — Nlerp (Normalized Lerp)**

Nlerp là xấp xỉ rẻ của slerp: làm lerp rồi chuẩn hóa kết quả về cầu:

$$\text{nlerp}(p_0, p_1; t) = \frac{(1-t)\,p_0 + t\,p_1}{\|(1-t)\,p_0 + t\,p_1\|}$$

| Thuộc tính | Lerp | Nlerp | Slerp |
|---|---|---|---|
| Norm output = 1 | ✗ | ✓ | ✓ |
| Vận tốc góc đều | ✓ (Euclidean) | ✗ | ✓ |
| Bám đúng arc lớn | ✗ | ✓ | ✓ |
| Chi phí tính toán | O(d) | O(d) | O(d) |
| Ổn định khi $\Omega \to 0$ | ✓ | ✓ | cần xử lý |
| Ổn định khi $\Omega \to \pi$ | ✓ | ✗ (mẫu → 0) | cần xử lý |

**Khi nlerp đủ tốt:** nội suy để render hình ảnh, làm animation — mắt người không nhận ra tốc độ không đều nếu bước đủ nhỏ. Với $\Omega \le 60°$, sai lệch góc tại $t=0.5$ giữa nlerp và slerp nhỏ hơn 1°.

**Khi cần slerp chính xác:** tính đạo hàm theo $t$ (gradient phải trơn đều), optimization trên không gian tham số, hoặc khi bước lớn ($\Omega > 90°$).

---

## **5. So sánh định lượng: lerp — nlerp — slerp**

Để làm rõ mức độ khác biệt, xét hai vector đơn vị trong $\mathbb{R}^{512}$ với góc $\Omega = 90°$ (điển hình trong high-dimensional latent space):

| Phương pháp | Norm tại $t=0.5$ | Sai lệch góc so với slerp tại $t=0.5$ |
|---|---|---|
| Lerp | $\cos(45°) \approx 0.707$ | $0°$ (cùng hướng, sai norm) |
| Nlerp | $1.0$ (chuẩn hóa) | $< 0.01°$ (hướng gần đúng) |
| Slerp | $1.0$ (chính xác) | $0°$ (chuẩn) |

Với $\Omega = 90°$, lerp mất 29% norm — khoảng cách từ bề mặt cầu bán kính $\sqrt{512} \approx 22.6$ xuống $\approx 16.0$: rơi sâu vào vùng zero-probability của prior. Nlerp và slerp giống nhau về hướng (sai lệch góc < 0.01°), nhưng nlerp rẻ hơn vì tránh được lượng giác.

---

## **6. Giới hạn / Khi nào thất bại**

**Giả định: prior là spherical.** Slerp chỉ đúng khi density cao nằm trên mặt cầu (prior $\mathcal{N}(0,I)$, VAE chuẩn, GAN noise). Nếu prior là Gaussian có $\Sigma \neq I$, mặt đẳng mật độ là ellipsoid, không phải cầu — slerp bảo toàn sai bề mặt.

**Chỉ hai điểm.** Slerp không tổng quát tự nhiên sang hơn 2 vector. Mở rộng (Fréchet mean trên $S^{d-1}$, Squad cho quaternion) tốn kém hơn nhiều.

**Không giải quyết vấn đề cross-cluster.** Nếu $p_0$ và $p_1$ thuộc hai mode ngữ nghĩa khác nhau, slerp vẫn đi qua "vùng trống" ngữ nghĩa — chỉ theo đường trên cầu thay vì qua lõi cầu. Hình học của prior được bảo toàn, nhưng hình học của posterior (tức phân phối thực của latent có điều kiện trên dữ liệu) không được xét đến.

**Slerp với latent không chuẩn hóa.** Cần chuẩn hóa trước, slerp, rồi khôi phục norm. Bước khôi phục norm thường dùng lerp tuyến tính trên norm (như trong pseudocode ở mục 3), nhưng đây là lựa chọn tùy ý — không có cơ sở lý thuyết mạnh.

---

## **7. Liên hệ với Latent-Anything**

Slerp là **implementation chuẩn mực của `LatentSpace.interpolate(method="slerp")`** trong Layer B, bên cạnh lerp:

- **Quy trình lựa chọn tự động:** framework kiểm tra mean norm của latent sample trong `LatentSpace`. Nếu $\bar{\|z\|} \approx \sqrt{d}$ (với $d$ = dim latent), đây là dấu hiệu prior spherical → slerp là mặc định. Nếu norm biến thiên rộng → lerp.
- **Trajectory primitive:** slerp giữa hai keyframe state cho ra path trơn trên vỏ cầu, tránh "blurry middle" khi render.
- **Latent arithmetic an toàn:** thay vì $z_a - z_b + z_c$ (lerp hai lần), có thể dùng slerp để giữ kết quả trong vùng valid của prior.

Phần tiếp theo — **Latent arithmetic (mục 03)** — bàn đến điều kiện để phép cộng-trừ vector trong latent space có nghĩa ngữ nghĩa: không phải lúc nào cũng thay bằng slerp là đủ.

---

## Liên quan

- [Lerp (mục 01 — tầng này)](01-lerp.md) — phép nội suy baseline mà slerp thay thế khi latent unit-norm; đọc trước để hiểu vấn đề norm dip.
- [Slerp (tầng 3 — giới thiệu khái niệm)](../../03-geometry-structure/research/05-slerp.md) — trình bày trực giác và tính chất; note này là phần implementation đi sâu hơn.
- [Geodesic](../../01-space-representation/research/05-geodesic.md) — slerp là trường hợp geodesic dạng đóng trên $S^{d-1}$; geodesic tổng quát hơn trên manifold cong phức tạp cần solver.
- [Pullback metric & FlatVI](../../01-space-representation/research/06-pullback-metric.md) — khi latent manifold không phải cầu, cần geodesic dưới metric Riemannian do decoder cảm sinh — phức tạp hơn nhưng chính xác hơn slerp.
- [Isotropy & anisotropy](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — nếu phân phối latent anisotropic ($\Sigma \neq I$), slerp bảo toàn sai ellipsoid; cần biết phân phối có spherical không trước khi chọn slerp.
- [Curse of dimensionality](../../01-space-representation/research/04-curse-of-dimensionality.md) — giải thích tại sao ở chiều cao prior $\mathcal{N}(0,I)$ tập trung trên vỏ cầu, làm slerp trở thành lựa chọn đúng đắn.

## Tham khảo

- K. Shoemake, *Animating Rotation with Quaternion Curves* (SIGGRAPH 1985). — Bài gốc đặt tên và công thức slerp cho quaternion; derivation của Glenn Davis (trích dẫn trong paper) là dạng được dùng cho vector đơn vị tổng quát.
- J. Blow, *Understanding Slerp, Then Not Using It* (The Inner Product, 2004). — Phân tích kỹ nlerp vs slerp, lập luận nlerp đủ tốt cho graphics; hữu ích để biết khi nào slerp không cần thiết.
- T. White, *Sampling Generative Networks* (arXiv 2016, arXiv:1609.04468). — Áp dụng slerp cho GAN latent space; bằng chứng thực nghiệm đầu tiên slerp cho ảnh sắc nét hơn lerp.
- Wikipedia, *Spherical linear interpolation*. — Tổng hợp các dạng công thức và edge case.
