# Lie groups và Lie algebra

> **TL;DR.** Lie group là một nhóm mà đồng thời cũng là một đa tạp trơn, nên có thể vừa **compose** như một phép biến đổi hình học, vừa **vi phân** như một đối tượng giải tích. Ý tưởng cốt lõi là thay vì tối ưu trực tiếp trên ma trận quay hay rigid transform như một vector Euclidean, ta làm việc cục bộ trong **Lie algebra** — không gian tiếp tuyến tại identity — rồi đi ngược về group bằng exponential map, ví dụ $\exp(\hat\omega) \in SO(3)$ hoặc $\exp(\hat\xi) \in SE(3)$. Caveat là đây không phải vector space toàn cục: composition không giao hoán, log map có nhánh, và các xấp xỉ tuyến tính chỉ tốt khi sai lệch đủ nhỏ.

Trong roadmap của Latent-Anything, đây là phần toán nền cho hai hướng rất cụ thể: (1) biểu diễn rotation và rigid motion trong robotics, camera pose, trajectory; (2) tham số hóa quay/pose của Gaussian và object trong 3D representation. Nếu bỏ qua Lie group structure và coi rotation/pose như vector thường, rất dễ tối ưu ra ma trận không hợp lệ, gặp singularity, hoặc cộng-trừ sai hình học.

---

## 1. Trực giác / Định nghĩa

Một rotation 3D không phải là một vector 3 chiều "bình thường". Nếu lấy hai rotation hợp lệ rồi cộng từng phần tử, kết quả thường **không còn là rotation hợp lệ**. Vấn đề gốc rễ là không gian của rotation không phẳng như $\mathbb{R}^n$, mà có cấu trúc cong và có phép nhân nhóm riêng.

Đó là lý do Lie group xuất hiện:

- nó là **group**: có phép composition, identity, inverse;
- nó là **manifold trơn**: có thể nói tới tiếp tuyến, đạo hàm, exponential map.

Ví dụ quan trọng nhất:

$$
SO(3) = \{R \in \mathbb{R}^{3 \times 3} \mid R^\top R = I,\ \det R = 1\},
$$

trong đó $R$ là ma trận quay 3D, điều kiện $R^\top R = I$ nói rằng các cột trực chuẩn, còn $\det R = 1$ loại bỏ phản xạ. Đây là không gian của mọi orientation hợp lệ của một rigid body trong 3D.

Nhóm rigid motion đầy đủ là

$$
SE(3) =
\left\{
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
\;\middle|\;
R \in SO(3),\ t \in \mathbb{R}^3
\right\},
$$

trong đó $R$ là rotation và $t$ là translation. Đây là đối tượng chuẩn trong robotics, camera pose estimation, và mọi bài toán "vừa quay vừa tịnh tiến".

---

## 2. Cơ chế / Công thức

### 2.1. Lie algebra là tiếp tuyến tại identity

Ý tưởng lớn của Lie theory là: thay vì làm việc trực tiếp trên manifold cong, ta linearize quanh identity và làm việc trong tiếp tuyến.

Với $SO(3)$, Lie algebra là

$$
\mathfrak{so}(3) =
\{\Omega \in \mathbb{R}^{3 \times 3} \mid \Omega^\top = -\Omega\},
$$

nghĩa là tập các ma trận skew-symmetric. Mọi vector $\omega = (\omega_1,\omega_2,\omega_3)^\top \in \mathbb{R}^3$ có thể được đổi sang ma trận qua toán tử "hat":

$$
\hat\omega =
\begin{bmatrix}
0 & -\omega_3 & \omega_2 \\
\omega_3 & 0 & -\omega_1 \\
-\omega_2 & \omega_1 & 0
\end{bmatrix}.
$$

trong đó $\omega$ là vector vận tốc góc / axis-angle infinitesimal, còn $\hat\omega$ là dạng ma trận dùng được trong exponential map và commutator. Ý nghĩa thực tế là: Lie algebra biến rotation nhỏ thành một đối tượng tuyến tính dễ cộng, dễ tối ưu, dễ vi phân.

Với rigid motion, Lie algebra của $SE(3)$ là

$$
\mathfrak{se}(3) =
\left\{
\begin{bmatrix}
\hat\omega & v \\
0 & 0
\end{bmatrix}
\middle|\;
\omega, v \in \mathbb{R}^3
\right\},
$$

trong đó $\omega$ là thành phần quay, $v$ là thành phần tịnh tiến, và cặp $(\omega, v)$ thường được gọi là một **twist**.

### 2.2. Exponential map trên SO(3)

Exponential map đưa ta từ algebra quay trở lại rotation thật:

$$
R = \exp(\hat\omega).
$$

Nếu đặt $\theta = \|\omega\|$ và $K = \hat\omega / \theta$ khi $\theta \neq 0$, ta có công thức Rodrigues:

$$
\exp(\hat\omega)
=
I + \sin\theta\, K + (1-\cos\theta)\,K^2.
$$

trong đó $\theta$ là góc quay, $K$ mã hóa trục quay, và ba hạng lần lượt là identity, thành phần bậc một, và correction bậc hai để bảo đảm kết quả vẫn nằm đúng trên $SO(3)$. Công thức này là lý do axis-angle và rotation vector đặc biệt tiện: chỉ cần 3 số trong algebra, rồi `exp` sẽ trả về rotation hợp lệ.

Ngược lại, logarithmic map lấy rotation thật về lại tiếp tuyến cục bộ:

$$
\hat\omega = \log(R).
$$

Điều này cực quan trọng cho optimization: loss thường được đo trong tangent space, vì ở đó mới có phép cộng-trừ tuyến tính "đúng kiểu".

### 2.3. Exponential map trên SE(3)

Với một twist

$$
\hat\xi =
\begin{bmatrix}
\hat\omega & v \\
0 & 0
\end{bmatrix},
$$

ta có rigid motion

$$
\exp(\hat\xi)
=
\begin{bmatrix}
\exp(\hat\omega) & J(\omega)v \\
0 & 1
\end{bmatrix},
$$

trong đó $J(\omega)$ là ma trận phụ thuộc vào phần quay:

$$
J(\omega)
=
I
+ \frac{1-\cos\theta}{\theta^2}\hat\omega
+ \frac{\theta-\sin\theta}{\theta^3}\hat\omega^2.
$$

Ở đây $\theta = \|\omega\|$ như trước; $J(\omega)v$ là translation đã được hiệu chỉnh theo việc rotation và translation xảy ra đồng thời chứ không phải độc lập. Đây là điểm mà trực giác Euclidean rất dễ sai: trong $SE(3)$, quay và tịnh tiến không chỉ là "nối hai vector vào nhau", mà phải được compose theo hình học nhóm.

### 2.4. Vì sao composition không giao hoán

Một tính chất quan trọng của cả $SO(3)$ lẫn $SE(3)$ là:

$$
AB \neq BA
$$

trong nhiều trường hợp. Hai rotation đổi thứ tự sẽ cho orientation khác nhau; quay rồi tịnh tiến cũng khác tịnh tiến rồi quay. Điều này giải thích vì sao:

- cộng trực tiếp Euler angles thường gây hiểu lầm;
- cập nhật pose theo thứ tự sai sẽ tạo trajectory sai;
- linearization phải giữ đúng frame convention (left/right update, body/spatial twist).

Đó cũng là vai trò của Lie bracket và BCH formula ở phía sau: chúng mô tả "sai số do không giao hoán" khi ghép các chuyển động nhỏ.

---

## 3. Biến thể / Trường hợp

| Đối tượng | Nó biểu diễn gì? | Lie algebra tương ứng | Khi nào dùng |
|---|---|---|---|
| **SO(3)** | rotation thuần túy | $\mathfrak{so}(3)$ | orientation, camera attitude, quay ellipsoid |
| **SE(3)** | rigid motion: quay + tịnh tiến | $\mathfrak{se}(3)$ | robot pose, object pose, camera extrinsics |
| **Quaternion** | representation của rotation | không tự nó thay thế Lie algebra local | tốt cho lưu trữ/quay 3D, nhưng vẫn cần log/exp để tối ưu hình học sạch |

Quaternion rất hữu ích để tránh singularity kiểu Euler angles, nhưng không xóa đi nhu cầu Lie group thinking. Ngay cả khi lưu rotation bằng quaternion, bài toán local update, interpolation ngắn nhất, hay pose error trong control vẫn thường được diễn đạt trong tangent space.

---

## 4. Giới hạn / Khi nào thất bại

**Không phải vector space toàn cục.** Có thể cộng hai twist nhỏ rồi `exp`, nhưng không thể coi toàn bộ manifold như $\mathbb{R}^6$ và nội suy tùy tiện mà vẫn đúng hình học.

**Log map có nhánh và không ổn định gần góc $\pi$.** Với rotation gần 180 độ, việc chọn trục quay không còn duy nhất; số học dễ nhạy hơn hẳn.

**Linearization chỉ đúng cục bộ.** Xấp xỉ $\exp(\hat\omega) \approx I + \hat\omega$ chỉ tốt khi góc rất nhỏ. Dùng nó cho chuyển động lớn sẽ phá tính trực chuẩn hoặc tích lũy drift.

**Quaternion vẫn có double cover.** $q$ và $-q$ là cùng một rotation, nên nếu không xử lý dấu nhất quán, interpolation và loss có thể nhảy nhánh.

**Sai convention là sai hoàn toàn.** Body frame hay spatial frame, left-multiply hay right-multiply, active hay passive transform: đổi convention mà không đổi công thức kéo theo bug rất khó truy ra.

---

## 5. Liên hệ với Latent-Anything

Lie groups và Lie algebra chạm trực tiếp vào các phần sau của framework:

- **3D representation / 3DGS**: orientation của Gaussian ellipsoid, camera pose, object pose đều tự nhiên sống trên $SO(3)$ hoặc $SE(3)$.
- **Robotics / world models**: end-effector pose, base motion, action space 6-DoF đều là rigid motions chứ không phải vector Euclidean thường.
- **Trajectory primitive**: khi trajectory chứa pose, operation như interpolation, averaging, smoothing, hoặc error accumulation nên diễn ra trên manifold, không phải chỉ trên tọa độ thô.
- **Optimization interfaces**: parameter update nên đi theo mô hình "giữ state trên group, tính gradient/update trong algebra", thay vì tối ưu trực tiếp trên ma trận rồi chiếu sửa hậu kỳ.

Đặc biệt, mục này nối về [Slerp](../../03-geometry-structure/research/05-slerp.md): slerp thực chất là geodesic trên mặt cầu/unit quaternion, còn Lie group formulation là phiên bản tổng quát hơn cho rotation và rigid motion. Nó cũng nối tới [Covariance Matrix trong 3DGS](../../03b-3d-representation/research/07-covariance-matrix-3dgs.md), nơi phần quay của ellipsoid không nên bị đối xử như ba số Euler vô tội vạ.

---

## Liên quan

- [Slerp](../../03-geometry-structure/research/05-slerp.md) — ví dụ cụ thể nhất của interpolation hình học trên một manifold quay.
- [Covariance Matrix trong 3DGS](../../03b-3d-representation/research/07-covariance-matrix-3dgs.md) — rotation quyết định ellipsoid Gaussian được định hướng thế nào.
- [Stochastic Transition](../../06-latent-temporal/research/03-stochastic-transition.md) — cảnh báo về việc dùng distribution Euclidean cho biến periodic, quaternion, hay manifold-valued state.
- [Kalman Filter variants](../../06-latent-temporal/research/05-kalman-filter-variants.md) — manifold-aware filtering trở nên cần thiết khi state chứa orientation/pose.

## Tham khảo

- R. M. Murray, Z. Li, S. S. Sastry, *A Mathematical Introduction to Robotic Manipulation* (CRC Press 1994).
- A. Müller, *Review of the Exponential and Cayley Map on SE(3) as relevant for Lie Group Integration of the Generalized Poisson Equation and Flexible Multibody Systems* (2023, arXiv:2303.07928).
