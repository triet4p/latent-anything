# Spherical Harmonics — basis hài cầu cho màu phụ thuộc hướng nhìn

> **TL;DR.** Spherical harmonics (SH) là một họ basis trực chuẩn trên mặt cầu, đóng vai trò giống Fourier basis nhưng cho **hàm theo hướng** thay vì hàm theo thời gian hay theo tọa độ 1D. Trong 3DGS, màu phụ thuộc hướng nhìn của mỗi Gaussian được khai triển thành $c(\mathbf{d})=\sum_{l=0}^{L}\sum_{m=-l}^{l} c_{lm}Y_l^m(\mathbf{d})$; với `sh_degree = 3`, ta có $(3+1)^2=16$ basis mỗi kênh màu, suy ra 48 hệ số RGB cho một Gaussian. Đổi lại, SH rất gọn và mượt cho tín hiệu góc nhìn tần số thấp, nhưng bị rung (ringing), tốn hệ số bậc hai theo degree, và không phải lựa chọn lý tưởng cho hiệu ứng phản xạ quá sắc.

Trong [3D Gaussian Splatting](06-3d-gaussian-splatting.md), covariance xử lý phần **geometry**: Gaussian nằm đâu, to cỡ nào, quay theo hướng nào. Nhưng scene còn thiếu một nửa quan trọng: khi camera nhìn Gaussian từ các hướng khác nhau, màu của nó có thể thay đổi do specular, Fresnel, hay phản xạ phụ thuộc góc nhìn. Đó là chỗ SH xuất hiện: nó là cách để gắn cho mỗi Gaussian một **hàm màu theo hướng** vừa đủ giàu để biểu diễn view-dependent appearance, nhưng vẫn đủ gọn để rasterize thời gian thực.

---

## **1. SH là gì: Fourier basis trên mặt cầu**

Nếu Fourier series biểu diễn một hàm 1D theo các sóng sin/cos với tần số khác nhau, thì spherical harmonics làm điều tương tự cho các hàm sống trên mặt cầu đơn vị $S^2$. Một hướng nhìn hay hướng phát xạ có thể được xem như một điểm trên sphere, nên mọi đại lượng "phụ thuộc hướng" đều có thể khai triển trên basis này.

Ký hiệu chuẩn là:

$$
Y_l^m(\theta,\phi), \qquad l \ge 0,\; -l \le m \le l
$$

trong đó $l$ là **degree/band** và $m$ là **order** trong band đó. Biến $(\theta,\phi)$ là hai góc cầu mô tả hướng trên sphere. Ý nghĩa của chỉ số:

- $l$ càng lớn thì basis càng dao động nhanh theo góc nhìn, tức là biểu diễn được chi tiết góc nhìn tần số cao hơn.
- Với một degree cố định $l$, có đúng $2l+1$ basis khác nhau.
- Nếu giữ tất cả basis từ $l=0$ đến $l=L$, tổng số basis là:

$$
\sum_{l=0}^{L}(2l+1) = (L+1)^2
$$

trong đó $L$ là degree cao nhất được giữ lại. Công thức này có nghĩa là số hệ số tăng **bậc hai** theo degree, chứ không tăng tuyến tính.

Theo Sloan, real spherical harmonics là "spherical analog to the Fourier basis on the unit circle", tức là phiên bản Fourier dành cho các hàm trên sphere chứ không phải trên đường tròn hay trục 1D.

---

## **2. Từ basis đến khai triển hàm theo hướng**

Một hàm vô hướng theo hướng $f(\mathbf{d})$, với $\mathbf{d}\in S^2$ là vector đơn vị, có thể được xấp xỉ bằng tổng có trọng số của các basis SH:

$$
f(\mathbf{d}) \approx \sum_{l=0}^{L}\sum_{m=-l}^{l} a_{lm}Y_l^m(\mathbf{d})
$$

trong đó $a_{lm}$ là hệ số của basis $Y_l^m$, còn $L$ là degree cắt ngắn tối đa. Công thức này nói rằng thay vì lưu nguyên một hàm liên tục trên mọi hướng, ta chỉ lưu một vector hệ số hữu hạn.

Nếu $f$ là **màu**, ta làm điều đó riêng cho từng kênh:

$$
\mathbf{c}(\mathbf{d}) =
\begin{bmatrix}
c_r(\mathbf{d})\\
c_g(\mathbf{d})\\
c_b(\mathbf{d})
\end{bmatrix}
\approx
\sum_{l=0}^{L}\sum_{m=-l}^{l}
\mathbf{c}_{lm}\,Y_l^m(\mathbf{d})
$$

trong đó $\mathbf{c}_{lm}\in\mathbb{R}^3$ là bộ hệ số RGB của basis $(l,m)$. Kết quả là một Gaussian không còn chỉ có "một màu cố định", mà có cả một **hàm màu phụ thuộc hướng nhìn** được mã hóa bằng các hệ số SH.

Điểm quan trọng: vì basis là trực chuẩn, các hệ số $a_{lm}$ có diễn giải khá sạch về mặt tần số góc nhìn. Band thấp nắm thành phần mượt, band cao nắm chi tiết thay đổi nhanh theo hướng.

---

## **3. SH xuất hiện trong 3DGS như thế nào**

Trong 3DGS, mỗi Gaussian mang position, opacity, covariance, và thêm một bộ hệ số SH để biểu diễn view-dependent color. Implementation chính thức cho phép chọn:

- `--sh_degree`: degree tối đa của spherical harmonics, mặc định `3`;
- `feature_lr`: learning rate cho spherical harmonics features.

Từ công thức đếm basis ở trên, nếu `sh_degree = 3` thì số basis mỗi kênh màu là:

$$
(3+1)^2 = 16
$$

trong đó 16 là tổng số basis từ band 0 tới band 3. Vì màu có 3 kênh RGB, suy ra mỗi Gaussian cần:

$$
3 \times 16 = 48
$$

hệ số màu. Đây là **suy ra toán học** từ số basis SH và cũng khớp với cách implementation chính thức cấp phát tensor `features` có kích thước `(N, 3, (sh_degree + 1)^2)`.

Một chi tiết implementation rất đáng chú ý:

- lúc khởi tạo từ point cloud, code chuyển màu RGB ban đầu sang hệ số SH bậc 0 bằng `RGB2SH(...)`;
- chỉ số `0` trong tensor features được gán màu DC ban đầu;
- các hệ số còn lại khởi tạo bằng `0`;
- `active_sh_degree` bắt đầu từ `0`, rồi mới tăng dần lên tới `max_sh_degree`.

Trực giác của thiết kế này rất hay:

- band 0 (DC term) tương ứng với **màu trung bình không phụ thuộc hướng nhìn**;
- các band cao hơn dần thêm vào phần phụ thuộc góc nhìn;
- huấn luyện có thể đi từ biểu diễn đơn giản, ổn định hơn, rồi mới mở thêm năng lực biểu diễn.

---

## **4. Degree thấp nghĩa là gì về mặt trực giác**

Mỗi band thêm vào cho phép hàm màu thay đổi theo hướng nhìn tinh vi hơn.

| Degree cao nhất $L$ | Số basis | Trực giác |
|---|---|---|
| `0` | `1` | chỉ có màu hằng, hoàn toàn view-independent |
| `1` | `4` | thêm biến thiên tuyến tính thô theo hướng |
| `2` | `9` | đã mô tả được cấu trúc góc nhìn mượt khá tốt |
| `3` | `16` | đủ giàu cho nhiều hiệu ứng view-dependent mượt trong 3DGS gốc |

Ramamoorthi và Hanrahan chỉ ra rằng với diffuse irradiance dưới ánh sáng xa, chỉ 9 hệ số đầu tiên (tức tới `l <= 2`) đã đủ cho sai số trung bình rất thấp, vì irradiance là tín hiệu góc nhìn tần số thấp. Điều này rất quan trọng về mặt trực giác: SH đặc biệt hợp với những gì **thay đổi mượt theo hướng**, chứ không phải mọi hiện tượng phản xạ.

Liên hệ lại với 3DGS:

- nhiều phần appearance phụ thuộc hướng trong scene thật là tương đối mượt ở mức local Gaussian;
- vì vậy degree thấp như `3` thường là điểm cân bằng tốt giữa chất lượng và tốc độ;
- không cần MLP riêng cho màu của từng Gaussian, chỉ cần đánh giá vài basis SH.

---

## **5. Vì sao SH hợp với 3DGS**

So với việc gắn cho mỗi Gaussian một MLP nhỏ để nhận hướng nhìn rồi trả màu, SH có một số ưu điểm rất hợp với rasterization thời gian thực:

1. **Đánh giá rẻ.**  
   Với degree thấp, ta chỉ cần tính một số ít basis tại hướng nhìn hiện tại rồi nhân với hệ số đã học.

2. **Gradient sạch.**  
   Màu là hàm tuyến tính theo hệ số SH, nên backprop qua các hệ số appearance khá gọn.

3. **Tách geometry khỏi appearance.**  
   [Covariance matrix trong 3DGS](07-covariance-matrix-3dgs.md) xử lý hình học cục bộ; SH xử lý biến thiên theo hướng. Hai phần này gần như độc lập khái niệm.

4. **Phù hợp với primitive explicit.**  
   Mỗi Gaussian đã là một latent token hình học; gắn thêm một vector SH vào nó vẫn giữ được tính explicit và dễ inspect.

Có thể xem SH trong 3DGS là tương tự [Positional Encoding](04-positional-encoding.md) ở NeRF theo một nghĩa hạn chế: cả hai đều là basis expansion. Nhưng khác biệt lớn là:

- positional encoding mở rộng **tọa độ không gian**;
- SH mở rộng **hướng nhìn trên sphere**.

---

## **6. Giới hạn: khi nào SH không còn là lựa chọn tốt**

SH rất mạnh cho tín hiệu mượt, nhưng không phải vũ khí vạn năng.

### **6.1. Tín hiệu góc nhìn quá sắc**

Specular highlight rất hẹp, reflection rất sắc, hoặc các hiện tượng góc nhìn tần số cao cần band cao hơn để khớp tốt. Nếu degree bị cắt thấp, SH chỉ tạo được một bản làm mượt của tín hiệu đó.

### **6.2. Ringing khi cắt ngắn chuỗi**

Vì SH là basis toàn cục trên sphere, khi cố biểu diễn tín hiệu có biên sắc bằng một số band hữu hạn, dễ xuất hiện hiện tượng **ringing**: dao động giả quanh vùng thay đổi mạnh. Đây là họ hàng trực tiếp của Gibbs-like behavior trong Fourier truncation.

### **6.3. Số hệ số tăng bậc hai**

Muốn tăng degree để bắt chi tiết góc nhìn sắc hơn thì số basis tăng theo $(L+1)^2$. Điều này làm:

- bộ nhớ tăng nhanh;
- chi phí evaluate cũng tăng;
- mỗi Gaussian trở nên nặng hơn.

Điểm này đủ quan trọng để các paper sau 3DGS tìm cách thay SH bằng biểu diễn khác. Chẳng hạn SG-Splatting mở đầu bằng nhận xét rằng việc dựa vào third-degree SH cho màu làm tăng đáng kể storage và computational overhead.

### **6.4. Không tự nhiên cho mọi loại material**

SH phù hợp nhất khi view-dependent color khá mượt. Với material quá gương, quá lấp lánh, hoặc có directional spikes hẹp, các basis toàn cục bậc thấp sẽ khá gượng.

---

## **7. Liên hệ với Latent-Anything**

Nếu Gaussian set là latent space của một world model kiểu LeWM, thì SH cho thấy một ý rất quan trọng: latent không chỉ là vị trí và shape, mà còn chứa **appearance function**.

- Mean + covariance trả lời: Gaussian nằm đâu và có hình học gì.
- Opacity trả lời: Gaussian đóng góp mạnh tới đâu.
- SH coefficients trả lời: Gaussian trông như thế nào khi nhìn từ các hướng khác nhau.

Như vậy, một Gaussian primitive trong latent space có thể được xem như:

$$
z_i = (\mu_i, \Sigma_i, o_i, \{ \mathbf{c}_{lm} \})
$$

trong đó $\{\mathbf{c}_{lm}\}$ là toàn bộ hệ số SH của Gaussian thứ $i$. Biểu thức này có nghĩa là appearance cũng là một phần của latent state, không phải thứ chỉ sinh ra ở decoder cuối cùng.

Điều này mở ra các hướng thao tác rất tự nhiên cho Latent-Anything:

- probe band thấp vs band cao để xem Gaussian encode màu trung bình hay hiệu ứng view-dependent mạnh;
- regularize degree hay sparsity của SH coefficients;
- thay SH bằng basis khác khi cần hiệu ứng góc nhìn sắc hơn;
- tách riêng geometry latent và appearance latent trong adapter.

Note này cũng đặt nền trực tiếp cho **Gaussian rasterization** ở mục tiếp theo: khi đã có ellipse 2D từ covariance và có màu theo hướng từ SH, rasterizer chỉ còn việc evaluate màu tại hướng nhìn và alpha-composite theo thứ tự độ sâu.

---

## Liên quan

- [3D Gaussian Splatting](06-3d-gaussian-splatting.md) — SH là nửa appearance của mỗi Gaussian trong 3DGS.
- [Covariance Matrix trong 3DGS](07-covariance-matrix-3dgs.md) — covariance xử lý geometry cục bộ, SH xử lý view-dependent color.
- [NeRF](02-nerf.md) — NeRF cũng tách nhánh theo viewing direction để mô hình hóa màu phụ thuộc hướng nhìn.
- [Positional Encoding](04-positional-encoding.md) — đều là basis expansion, nhưng PE áp lên tọa độ không gian còn SH áp lên hướng nhìn trên sphere.
- [Instant-NGP](05-instant-ngp.md) — Instant-NGP tăng tốc encoding tọa độ; SH giải bài toán encoding theo hướng nhìn trong 3DGS.

## Tham khảo

- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (ACM Transactions on Graphics 2023, arXiv:2308.04079).
- Graphdeco-Inria, *gaussian-splatting* official implementation (GitHub repository, `--sh_degree`, `active_sh_degree`, `features_dc`, `features_rest`).
- Ramamoorthi, Hanrahan, *An Efficient Representation for Irradiance Environment Maps* (SIGGRAPH 2001).
- Sloan, *Efficient Spherical Harmonic Evaluation* (JCGT 2013).
- Wang, Chen, Yi, *SG-Splatting: Accelerating 3D Gaussian Splatting with Spherical Gaussians* (arXiv 2025, arXiv:2501.00342).
