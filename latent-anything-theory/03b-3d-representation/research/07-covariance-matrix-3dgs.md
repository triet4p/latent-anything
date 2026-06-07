# Covariance Matrix trong 3D Gaussian Splatting

> **TL;DR.** Trong 3DGS, covariance matrix $\Sigma$ quyết định mỗi Gaussian là một ellipsoid 3D "to cỡ nào" và "quay theo hướng nào", nên nó chính là phần mang hình học cục bộ của scene. Thay vì tối ưu trực tiếp $\Sigma$, phương pháp gốc parameterize nó thành $\Sigma = R S S^\top R^\top$ với $R$ là rotation và $S$ là scale, để giữ $\Sigma$ luôn hợp lệ và dễ tối ưu hơn; sau đó chiếu xuống ảnh qua $\Sigma' = J W \Sigma W^\top J^\top$. Đổi lại, covariance học sai sẽ dẫn tới splat quá to, quá mỏng, hoặc định hướng sai, gây blur, floater và occlusion artifact.

Trong [3D Gaussian Splatting](06-3d-gaussian-splatting.md), mean $\mu$ cho biết Gaussian nằm ở đâu, opacity $o$ cho biết nó "đục" tới mức nào, còn covariance $\Sigma$ cho biết Gaussian **chiếm không gian theo hình dạng nào**. Nếu mean là vị trí của một hạt, thì covariance là phần quyết định hạt đó là chấm tròn nhỏ, ellipsoid kéo dài, hay một vệt mỏng nghiêng theo bề mặt. Vì vậy, để hiểu vì sao 3DGS render nhanh mà vẫn giữ geometry tốt, phải hiểu covariance trước.

---

## **1. Covariance nói gì về một Gaussian 3D**

Với một Gaussian 3D tâm $\mu \in \mathbb{R}^3$, mật độ của nó có dạng:

$$
g(\mathbf{x}) = \exp\!\left(-\frac{1}{2}(\mathbf{x}-\mu)^\top \Sigma^{-1}(\mathbf{x}-\mu)\right)
$$

trong đó $\mathbf{x}$ là một điểm bất kỳ trong không gian và $\Sigma \in \mathbb{R}^{3\times 3}$ là covariance matrix. Biểu thức bậc hai ở giữa chính là bình phương khoảng cách Mahalanobis, nên các tập điểm có cùng giá trị của $g(\mathbf{x})$ tạo thành các mặt ellipsoid đồng mức.

Hệ quả hình học:

- trị riêng lớn của $\Sigma$ nghĩa là Gaussian trải rộng mạnh theo một hướng;
- trị riêng nhỏ nghĩa là Gaussian bị nén theo hướng đó;
- vector riêng của $\Sigma$ cho biết các trục chính của ellipsoid quay về đâu.

Nói ngắn gọn, $\Sigma$ mang đúng hai thứ 3DGS cần cho geometry cục bộ:

- **anisotropy**: surface không phải lúc nào cũng đẳng hướng như quả cầu;
- **orientation**: cùng một độ lớn nhưng quay khác đi thì footprint trên ảnh khác hẳn.

Đây là lý do paper gốc nhấn mạnh việc tối ưu **anisotropic covariance**, không chỉ isotropic radius.

---

## **2. Vì sao không tối ưu trực tiếp ma trận $\Sigma$**

Về mặt hình thức, một ma trận đối xứng $3\times 3$ có 6 bậc tự do, nên có vẻ hoàn toàn có thể tối ưu thẳng 6 số đó. Vấn đề là không phải ma trận đối xứng nào cũng là covariance hợp lệ. Để định nghĩa một Gaussian thật sự, $\Sigma$ phải ít nhất là **đối xứng bán xác định dương**; nếu không, ellipsoid có thể méo theo kiểu không vật lý, và $\Sigma^{-1}$ trở nên bất ổn hoặc không tồn tại.

Điều này dẫn tới ràng buộc:

$$
\mathbf{v}^\top \Sigma \mathbf{v} \ge 0 \qquad \forall \mathbf{v}\in\mathbb{R}^3
$$

trong đó $\mathbf{v}$ là một vector bất kỳ. Bất đẳng thức này có nghĩa là covariance không được tạo ra "phương sai âm" theo bất kỳ hướng nào.

Suy ra thực dụng là:

- nếu cập nhật trực tiếp các phần tử của $\Sigma$ bằng gradient descent, rất dễ đi ra ngoài miền hợp lệ;
- ngay cả khi còn khả nghịch, conditioning của ma trận cũng có thể cực xấu, làm projection và rasterization mất ổn định;
- việc giải thích parameter học được cũng mơ hồ hơn hẳn so với một cặp "quay + scale".

Ý "không tối ưu trực tiếp" ở đây là suy luận hình học từ ràng buộc covariance, đồng thời nhất quán với cách parameterize trong phương pháp gốc và implementation tham chiếu của tác giả.

---

## **3. Parameterization chuẩn: $\Sigma = R S S^\top R^\top$**

3DGS dùng decomposition:

$$
\Sigma = R S S^\top R^\top
$$

trong đó $R \in \mathbb{R}^{3\times 3}$ là ma trận quay và $S=\operatorname{diag}(s_1,s_2,s_3)$ là ma trận scale chéo. Công thức này nói rằng: trước hết tạo một ellipsoid trục chuẩn với ba bán trục do $s_1,s_2,s_3$ quyết định, rồi quay nó vào world space bằng $R$.

Vì $S S^\top$ là ma trận chéo có phần tử không âm trên đường chéo, và $R$ là trực giao, nên $\Sigma$ sinh ra theo cách này luôn đối xứng và bán xác định dương. Đây là lợi ích lớn nhất của parameterization.

Có thể nhìn decomposition này dưới hai lớp ý nghĩa:

- **scale** giữ phần "kích thước theo từng trục";
- **rotation** giữ phần "định hướng của các trục chính".

Trong implementation thực tế, rotation thường được lưu bằng quaternion rồi chuyển thành ma trận quay $R$. Cách này tránh singularity kiểu Euler angle và gọn hơn khi tối ưu. Điều này không đổi bản chất của decomposition: Gaussian vẫn được xác định bởi một ellipsoid chuẩn bị kéo dãn rồi đem quay.

---

## **4. Từ covariance 3D sang ellipse 2D trên image plane**

Một Gaussian chỉ hữu ích cho render nếu geometry 3D của nó được chuyển thành footprint 2D trên màn hình. 3DGS làm điều này bằng một xấp xỉ affine cục bộ của phép chiếu phối cảnh:

$$
\Sigma' = J W \Sigma W^\top J^\top
$$

trong đó $W$ là phép biến đổi từ world space sang camera space, còn $J$ là Jacobian của phép chiếu phối cảnh tại tâm Gaussian. Kết quả $\Sigma'$ là covariance $2\times 2$ của ellipse trên image plane.

Ý nghĩa của từng thành phần:

- $W\Sigma W^\top$ đưa ellipsoid từ hệ world sang hệ camera;
- $J$ tuyến tính hóa phép chiếu quanh tâm hiện tại của Gaussian;
- $\Sigma'$ cho biết ellipse trên ảnh sẽ rộng hẹp và nghiêng thế nào.

Đây là bước then chốt biến "geometry 3D mềm" thành "primitive rasterizable 2D". Nếu không có covariance hợp lệ và ổn định, bước chiếu này sẽ tạo footprint bất thường, kéo theo lỗi blending và gradient.

Liên hệ với [3D Gaussian Splatting](06-3d-gaussian-splatting.md): mean $\mu$ quyết định tâm ảnh chiếu, còn covariance $\Sigma$ quyết định hình dạng của splat sau khi lên màn hình.

---

## **5. Trực giác eigenvalue: surface patch cần Gaussian kiểu nào**

Một cách rất hữu ích để đọc covariance là nhìn nó như một "surface patch mềm".

- Nếu ba trị riêng gần bằng nhau, Gaussian gần như isotropic, giống một đốm cầu mềm.
- Nếu hai trị riêng lớn và một trị riêng rất nhỏ, Gaussian giống một mảnh lá mỏng, phù hợp với local tangent patch của bề mặt.
- Nếu một trị riêng quá lớn so với hai trị riêng còn lại, Gaussian giống một cây kim kéo dài, dễ gây artifact khi nhìn từ góc xiên.

Với scene thật, nhiều bề mặt cục bộ gần giống một miếng phẳng nhỏ. Vì vậy Gaussian dị hướng là hợp lý hơn Gaussian cầu: nó có thể trải rộng theo hai phương của mặt phẳng và mỏng theo pháp tuyến, nên cần ít primitive hơn để phủ đúng geometry.

Đây cũng là chỗ 3DGS thắng các point representation đẳng hướng cũ: thay vì tăng số điểm để bù cho việc mỗi điểm quá "ngu", nó làm từng điểm mạnh hơn về hình học.

---

## **6. So sánh các cách parameterize hình học Gaussian**

| | Tối ưu trực tiếp $\Sigma$ | Isotropic radius | $R S S^\top R^\top$ |
|---|---|---|---|
| Bậc tự do hình học | Cao | Thấp | Vừa đủ |
| Đảm bảo covariance hợp lệ | Khó | Dễ | Dễ |
| Mô tả anisotropy | Có | Không | Có |
| Dễ diễn giải | Kém | Rất dễ | Dễ |
| Phù hợp surface patch mỏng | Có, nhưng khó ổn định | Kém | Tốt |

Vì thế, decomposition của 3DGS là một điểm cân bằng khá đẹp:

- mạnh hơn sphere-based splat vì có anisotropy;
- ổn định hơn optimize ma trận thẳng vì có cấu trúc hình học rõ ràng.

---

## **7. Giới hạn / Khi nào covariance học sai**

- **Scale phình quá lớn**: một Gaussian che phủ quá nhiều vùng ảnh, làm scene mờ hoặc dính các mặt không nên dính.
- **Scale quá nhỏ**: cần rất nhiều Gaussian để lấp đầy surface, làm chi phí bộ nhớ và tối ưu tăng mạnh.
- **Rotation sai**: ellipsoid nghiêng lệch so với local surface, dẫn tới footprint trên ảnh sai và blending sai.
- **Conditioning xấu**: một trục quá lớn, một trục quá bé làm phép chiếu và gradient nhạy cảm với nhiễu số.
- **Occlusion artifact**: covariance lớn không đúng chỗ khiến một splat chen vào vùng lẽ ra bị Gaussian khác che khuất.
- **Floaters**: primitive lơ lửng trong không gian rỗng nhưng vẫn có opacity đủ để hiện lên ở một số góc nhìn.

Phần lớn các cơ chế clone/split/prune trong 3DGS thực ra là để sửa các failure mode hình học này chứ không chỉ để tăng số primitive.

---

## **8. Liên hệ với Latent-Anything**

Nếu LeWM-style adapter dùng Gaussian set làm latent, thì covariance không phải chi tiết phụ. Nó là một phần của **latent state**.

- Mean $\mu$ nói latent object nằm ở đâu.
- Covariance $\Sigma$ nói latent object chiếm không gian và định hướng ra sao.
- Opacity và appearance nói latent object hiện ra như thế nào khi decode.

Từ góc nhìn đó, một Gaussian không phải chỉ là "điểm có màu", mà là một latent token hình học đầy đủ. Điều này rất hợp với triết lý của Latent-Anything:

- Layer A có thể probe geometry bằng phân phối trị riêng, anisotropy, orientation statistics.
- Layer B có thể manipulate bằng cách co giãn, quay, merge, hoặc regularize covariance.
- Layer C có thể dùng rasterizer như decoder gần-deterministic từ latent set sang observation.

Note này cũng đặt nền trực tiếp cho hai mục tiếp theo của roadmap:

- **Spherical harmonics** sẽ xử lý phần appearance phụ thuộc hướng nhìn;
- **Gaussian rasterization** sẽ đi sâu vào cách các covariance đã chiếu được sắp xếp theo tile và alpha-blend trên GPU.

---

## Liên quan

- [3D Gaussian Splatting](06-3d-gaussian-splatting.md) — note mẹ của toàn bộ biểu diễn 3DGS.
- [Khoảng cách Mahalanobis](../../01-space-representation/research/02-mahalanobis-distance.md) — covariance đi vào hàm Gaussian đúng qua quadratic form Mahalanobis.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — 3DGS thành công một phần vì dùng primitive dị hướng thay vì isotropic sphere.
- [Hình học Riemannian](../../03-geometry-structure/research/04-riemannian-geometry.md) — covariance có thể được đọc như local metric xấp xỉ cho hình học cục bộ, dù 3DGS không dựng đầy đủ manifold metric.

## Tham khảo

- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (ACM Transactions on Graphics 2023, arXiv:2308.04079).
- Zwicker, Pfister, van Baar, Gross, *EWA Splatting* (IEEE TVCG 2002).
