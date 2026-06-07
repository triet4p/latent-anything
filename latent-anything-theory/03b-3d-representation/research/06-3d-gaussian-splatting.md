# 3D Gaussian Splatting — Biểu diễn scene bằng tập Gaussian tường minh

> **TL;DR.** 3D Gaussian Splatting (3DGS) biểu diễn một scene như một tập Gaussian 3D dị hướng $G_i=(\mu_i,\Sigma_i,o_i,\mathbf{a}_i)$ thay vì một MLP toàn cục, rồi render bằng cách chiếu từng Gaussian xuống ảnh 2D và alpha-composite theo thứ tự độ sâu. Công thức cốt lõi là $\Sigma_i = R_i S_i S_i^\top R_i^\top$ để giữ covariance hợp lệ, và sau phép chiếu cục bộ ta thu được $\Sigma_i' = J_i W \Sigma_i W^\top J_i^\top$ trên image plane nên toàn bộ pipeline vẫn khả vi. Đổi lại, 3DGS rất nhanh và chỉnh sửa cục bộ tốt hơn [NeRF](02-nerf.md), nhưng tốn bộ nhớ, cần sắp xếp theo độ sâu, và bản gốc chỉ nhắm tới scene tĩnh với camera pose tốt.

3DGS (Kerbl et al., 2023) là bước ngoặt của họ biểu diễn 3D hậu-NeRF: thay vì hỏi một mạng "điểm này có màu và mật độ gì?" hàng triệu lần dọc theo mỗi tia, nó lưu luôn scene thành một **tập phần tử tường minh**. Mỗi phần tử là một ellipsoid mềm trong không gian, có vị trí, hình dạng, độ đục và appearance riêng. Kết quả là scene vừa giữ được tinh thần **radiance field** của [NeRF](02-nerf.md), vừa ăn khớp với pipeline rasterization vốn rất hợp GPU.

---

## **1. Trực giác: thay một hàm toàn cục bằng nhiều primitive cục bộ**

Nếu [Neural Implicit Representation](01-neural-implicit-representation.md) và [NeRF](02-nerf.md) coi scene là **một hàm liên tục toàn cục**, thì 3DGS coi scene là **một tập primitive địa phương**:

$$
\mathcal{S} = \{G_i\}_{i=1}^{N}, \qquad G_i = (\mu_i, \Sigma_i, o_i, \mathbf{a}_i)
$$

trong đó $\mu_i \in \mathbb{R}^3$ là tâm của Gaussian thứ $i$, $\Sigma_i \in \mathbb{R}^{3\times 3}$ là covariance mô tả kích thước và hướng của ellipsoid, $o_i \in [0,1]$ là opacity nền, và $\mathbf{a}_i$ là tham số appearance của Gaussian đó. Phương trình này nói rằng scene không còn nằm trong một trọng số mạng duy nhất, mà được tách thành $N$ "hạt" mềm có ý nghĩa hình học rõ ràng.

Với một điểm không gian $\mathbf{x}$, ảnh hưởng hình học của Gaussian $i$ được đo bằng hàm:

$$
g_i(\mathbf{x}) = \exp\!\left(-\frac{1}{2}(\mathbf{x}-\mu_i)^\top \Sigma_i^{-1}(\mathbf{x}-\mu_i)\right)
$$

trong đó $(\mathbf{x}-\mu_i)^\top \Sigma_i^{-1}(\mathbf{x}-\mu_i)$ là bình phương khoảng cách Mahalanobis từ $\mathbf{x}$ tới tâm $\mu_i$. Công thức này có nghĩa là Gaussian đóng góp mạnh gần tâm và mờ dần theo đúng hình ellipsoid mà covariance quy định.

Trực giác quan trọng nhất của 3DGS là:

- NeRF phân bố thông tin trong trọng số một MLP toàn cục, nên muốn sửa một vùng nhỏ thường vẫn phải đi qua cả hàm.
- 3DGS lưu scene dưới dạng các phần tử rời rạc, nên thêm, bớt, di chuyển, clone hay split một vùng là thao tác tự nhiên hơn nhiều.
- Không gian rỗng gần như không tốn compute, vì chỉ render nơi có Gaussian hiện diện thay vì march qua mọi mẫu dọc tia như [Volume Rendering & Ray Marching](03-volume-rendering-ray-marching.md).

---

## **2. Hình học của một Gaussian: ellipsoid mềm thay cho voxel hay MLP**

Một Gaussian 3D được hiểu như một ellipsoid xác suất trong không gian. Covariance của nó không được tối ưu trực tiếp theo từng phần tử ma trận, mà thường được parameterize thành:

$$
\Sigma_i = R_i S_i S_i^\top R_i^\top
$$

trong đó $R_i \in \mathbb{R}^{3\times 3}$ là ma trận quay, còn $S_i=\operatorname{diag}(s_{i,1}, s_{i,2}, s_{i,3})$ là ma trận scale chéo. Công thức này đảm bảo $\Sigma_i$ luôn đối xứng và bán xác định dương, nghĩa là ellipsoid luôn hợp lệ về mặt hình học; đồng thời nó tách rõ "quay đi đâu" khỏi "phình theo trục nào".

Đây là một quyết định thiết kế rất quan trọng:

- nếu tối ưu trực tiếp 6 hay 9 phần tử của covariance, gradient rất dễ tạo ra ma trận không hợp lệ hoặc khó ổn định;
- còn khi tách thành rotation + scale, mỗi Gaussian trở thành một primitive dễ diễn giải: một ellipsoid có tâm, có hướng, có bán trục.

Trong biểu diễn gốc của Kerbl et al., appearance thường được mã hóa bằng **spherical harmonics (mục tiếp theo)** để cho phép màu phụ thuộc hướng nhìn mà không cần một MLP riêng cho mỗi Gaussian.

---

## **3. Render: chiếu Gaussian 3D thành ellipse 2D rồi alpha-composite**

Khác với [NeRF](02-nerf.md), 3DGS không tích phân hàng trăm mẫu dọc mỗi tia. Thay vào đó, mỗi Gaussian được chiếu trực tiếp xuống ảnh thành một Gaussian 2D.

Mean được chiếu bởi camera transform và phép phối cảnh:

$$
\mu_i' = \pi(W\mu_i)
$$

trong đó $W$ là phép biến đổi từ world space sang camera space, còn $\pi(\cdot)$ là phép chiếu phối cảnh. Kết quả $\mu_i'$ là tâm của ellipse trên image plane.

Covariance 3D được đẩy qua phép chiếu cục bộ bằng xấp xỉ tuyến tính bậc nhất:

$$
\Sigma_i' = J_i W \Sigma_i W^\top J_i^\top
$$

trong đó $J_i$ là Jacobian của phép chiếu phối cảnh tại vị trí của Gaussian thứ $i$. Công thức này nói rằng một ellipsoid 3D, khi nhìn từ camera, trở thành một ellipse 2D với hình dạng phụ thuộc cả geometry của chính Gaussian lẫn góc nhìn hiện tại.

Tại một pixel $p$, độ đục hiệu dụng của Gaussian $i$ là:

$$
\alpha_i(p) = o_i \exp\!\left(-\frac{1}{2}(p-\mu_i')^\top {\Sigma_i'}^{-1}(p-\mu_i')\right)
$$

trong đó $o_i$ là opacity học được còn phần mũ Gaussian mô tả footprint 2D của ellipse trên màn hình. Công thức này có nghĩa là Gaussian chỉ ảnh hưởng mạnh tới các pixel nằm gần tâm ảnh chiếu của nó.

Màu cuối cùng của pixel được tính bằng alpha compositing theo thứ tự gần camera:

$$
C(p) = \sum_{i} T_i(p)\,\alpha_i(p)\,c_i(p),
\qquad
T_i(p) = \prod_{j<i}\big(1-\alpha_j(p)\big)
$$

trong đó $c_i(p)$ là màu mà Gaussian $i$ đóng góp tại pixel $p$, và $T_i(p)$ là transmittance còn lại sau các Gaussian phía trước. Kết quả này rất giống tinh thần của [Volume Rendering & Ray Marching](03-volume-rendering-ray-marching.md), nhưng thay vì lấy mẫu dọc tia rồi cộng tích phân, 3DGS alpha-blend trực tiếp các ellipse 2D đã được sắp xếp theo độ sâu.

Hệ quả thực dụng là:

- render trở thành bài toán rasterization, hợp với phần cứng GPU hơn volume integration;
- backward pass đi qua phép chiếu, footprint 2D và alpha blend, nên vẫn tối ưu end-to-end từ photometric loss;
- chi phí compute tỉ lệ với số Gaussian thực sự chạm vào tile/pixel, không tỉ lệ với số mẫu ray trong không gian rỗng.

---

## **4. Huấn luyện: tối ưu primitive và densify khi scene còn thiếu chi tiết**

3DGS thường khởi tạo từ sparse point cloud do camera calibration / Structure-from-Motion sinh ra. Mỗi điểm ban đầu trở thành một Gaussian thô, rồi hệ thống đồng thời học:

- vị trí $\mu_i$;
- rotation $R_i$ và scale $S_i$;
- opacity $o_i$;
- appearance $\mathbf{a}_i$.

Loss chính vẫn là photometric reconstruction giữa ảnh render và ảnh thật từ nhiều góc nhìn, giống tinh thần huấn luyện [NeRF](02-nerf.md). Điểm mới lớn nằm ở **interleaved optimization + density control**:

- Gaussian nào có gradient lớn nhưng còn quá nhỏ có thể được clone để tăng mật độ cục bộ.
- Gaussian nào đang che phủ quá rộng một vùng chi tiết có thể bị split thành nhiều primitive nhỏ hơn.
- Gaussian nào ít đóng góp, opacity thấp, hoặc dư thừa có thể bị prune.

Nhờ đó, scene không bị khóa cứng bởi point cloud khởi tạo. Một point cloud thưa ban đầu có thể dần mọc thành một biểu diễn dày hơn, đúng chỗ hơn, và vẫn giữ tính explicit.

---

## **5. Vị trí của 3DGS trong họ radiance field**

| | NeRF | Instant-NGP | 3DGS |
|---|---|---|---|
| Biểu diễn chính | MLP toàn cục | Hash grid + MLP nhỏ | Tập Gaussian tường minh |
| Cách render | Volume rendering dọc tia | Volume rendering dọc tia | Rasterize ellipse 2D + alpha blend |
| Chi phí ở không gian rỗng | Cao | Giảm bớt nhờ occupancy grid | Thấp, vì không cần march vùng rỗng |
| Chỉnh sửa cục bộ | Khó | Khó | Tự nhiên hơn |
| Bộ nhớ | Nghiêng về trọng số mạng | Thêm bộ nhớ cho grid | Tăng theo số Gaussian |
| Tốc độ novel-view | Chậm | Nhanh hơn NeRF nhiều | Real-time là mục tiêu gốc |

NeRF và [Instant-NGP](05-instant-ngp.md) vẫn nằm trong họ "query một trường liên tục rồi render". 3DGS dịch bài toán sang "lưu thẳng các primitive có thể vẽ được". Đây là khác biệt nền tảng khiến 3DGS đặc biệt hấp dẫn cho các pipeline cần thao tác trực tiếp trên scene representation.

---

## **6. Giới hạn / Khi nào thất bại**

- **Tốn bộ nhớ theo số primitive.** Chất lượng cao thường đòi hỏi hàng trăm nghìn tới hàng triệu Gaussian, nên representation có thể khá nặng.
- **Vẫn là per-scene optimization.** 3DGS gốc không phải một mô hình tổng quát hóa scene mới; mỗi scene vẫn cần tối ưu riêng.
- **Nhạy với pose và khởi tạo SfM.** Nếu point cloud ban đầu hoặc camera pose sai, Gaussian sẽ mọc sai hình học ngay từ đầu.
- **Xử lý transparency, mỏng, và occlusion phức tạp chưa hoàn hảo.** Alpha blending theo thứ tự độ sâu là một xấp xỉ khả dụng, nhưng không phải lời giải vật lý hoàn chỉnh cho mọi hiện tượng quang học.
- **Bản gốc giả định scene tĩnh.** Motion, deformation và temporal consistency cần các biến thể **Dynamic 3DGS (mục sau)**.
- **Không tự sinh topology rõ ràng như mesh.** 3DGS rất tốt cho render và quan sát, nhưng kém trực tiếp hơn nếu mục tiêu là CAD, collision, hay chỉnh sửa topology chính xác.

---

## **7. Liên hệ với Latent-Anything**

3DGS rất gần với cách Latent-Anything muốn đối xử với latent space: không chỉ là một vector, mà là một **đối tượng có cấu trúc, có thể inspect và manipulate**.

- Mỗi Gaussian là một latent primitive cục bộ với state riêng: vị trí, kích thước, hướng, opacity, appearance.
- Toàn bộ scene là một **set latent variable** tường minh, phù hợp với các operation kiểu add/remove/merge/split ở Layer B.
- Rasterizer của 3DGS là một decoder gần-deterministic: từ Gaussian set sang ảnh 2D mà không cần một MLP nặng trong vòng lặp.
- Đây là nền trực tiếp cho các mục tiếp theo của roadmap: **Covariance matrix trong 3DGS**, **Spherical harmonics**, **Gaussian rasterization**, và đặc biệt là ý tưởng **Gaussian parameters là latent variable** cho LeWM-style world model.

Với định hướng đó, 3DGS không chỉ là một kỹ thuật render nhanh hơn NeRF. Nó là ví dụ rất mạnh cho thấy một latent space có thể được tổ chức thành **tập phần tử có ý nghĩa hình học**, từ đó mở ra các thao tác world-model tự nhiên hơn hẳn latent vector thuần túy.

---

## Liên quan

- [Neural Implicit Representation](01-neural-implicit-representation.md) — cùng bài toán biểu diễn scene 3D, nhưng dùng hàm toàn cục thay vì primitive tường minh.
- [NeRF](02-nerf.md) — đối chiếu trực tiếp nhất: radiance field implicit so với Gaussian set explicit.
- [Volume Rendering & Ray Marching](03-volume-rendering-ray-marching.md) — 3DGS giữ tinh thần transmittance/alpha compositing nhưng bỏ ray marching dày đặc.
- [Instant-NGP](05-instant-ngp.md) — cùng mục tiêu tăng tốc radiance field, nhưng theo nhánh hash-grid thay vì rasterization explicit.
- [Khoảng cách Mahalanobis](../../01-space-representation/research/02-mahalanobis-distance.md) — hàm Gaussian dùng đúng khoảng cách Mahalanobis do covariance quy định.

## Tham khảo

- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (ACM Transactions on Graphics 2023, arXiv:2308.04079).
- Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng, *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis* (ECCV 2020, arXiv:2003.08934).
- Zwicker, Pfister, van Baar, Gross, *EWA Splatting* (IEEE TVCG 2002) — nền point-based rendering cho footprint Gaussian trên image plane.
