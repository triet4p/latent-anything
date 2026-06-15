# COLMAP & Các phương pháp SfM/MVS cổ điển

> **TL;DR.** Structure-from-Motion (SfM) tái dựng camera pose và point cloud thưa từ một tập ảnh bằng cách ghép nối các ràng buộc hình học qua nhiều view; Multi-View Stereo (MVS) dày hoá point cloud từ camera pose đã biết. COLMAP (Schönberger & Frahm, 2016) là hiện thực tiêu chuẩn của pipeline SfM+MVS incremental, dùng bundle adjustment phi tuyến để tinh chỉnh đồng thời camera và điểm 3D. Pipeline này chính xác về mặt hình học nhưng cần overlap view lớn, texture tốt, không có vật trong suốt/phản chiếu, và chạy incremental nên chậm — các hạn chế này trực tiếp dẫn đến sự ra đời của phương pháp feed-forward learning-based như Dust3R, VGGT.

COLMAP là hiện thân tiêu biểu nhất của trường phái **geometry-first** trong tái dựng 3D: mọi quyết định đều dựa trên ràng buộc hình học thuần tuý (epipolar geometry, photometric consistency), không cần bất kỳ prior học máy nào. Hiểu COLMAP là bắt buộc để đánh giá được *tại sao* các phương pháp học sâu — từ NeRF đến 3DGS đến VGGT — tồn tại và giải quyết vấn đề gì.

---

## **1. Trực giác: ghép ảnh như ghép puzzle hình học**

Khi chụp một toà nhà từ nhiều góc, mỗi ảnh là một hình chiếu 2D của cùng một thế giới 3D. Nếu tìm được các điểm tương ứng giữa hai ảnh (vd: góc cửa sổ này trong ảnh A khớp với góc cửa sổ đó trong ảnh B), hình học epipolar cho phép tính ra **vị trí tương đối** giữa hai camera và **toạ độ 3D** của điểm đó.

Pipeline SfM cổ điển làm đúng việc này, nhưng với *hàng trăm đến hàng nghìn ảnh*:

1. **Feature extraction & matching** — tìm keypoint (SIFT) và match chúng giữa các cặp ảnh.
2. **Geometric verification** — kiểm tra match có nhất quán với một epipolar geometry (fundamental/essential matrix) không.
3. **Incremental reconstruction** — khởi tạo từ một cặp ảnh tốt, rồi *đăng ký* từng ảnh mới vào mô hình 3D hiện có.
4. **Bundle adjustment (BA)** — sau mỗi lần thêm ảnh, tối ưu lại toàn bộ camera pose và điểm 3D để giảm sai số tái chiếu.

Kết quả SfM là một **point cloud thưa** + camera pose cho mọi ảnh. Sau đó MVS dày hoá point cloud thành **dense reconstruction**.

---

## **2. Cơ chế: từ ảnh đến mô hình 3D**

### 2.1 Structure-from-Motion (SfM)

SfM giải bài toán: cho $N$ ảnh $I_1,\dots,I_N$, tìm camera pose $\{R_i, \mathbf{t}_i\}$ và toạ độ 3D $\{\mathbf{X}_j\}$ của các điểm khớp.

Với một cặp ảnh $(I_a, I_b)$ và một điểm 3D $\mathbf{X}_j$ quan sát được trong cả hai, **sai số tái chiếu** (reprojection error) là:

$$ e_{ij} = \|\mathbf{x}_{ij} - \pi(R_i \mathbf{X}_j + \mathbf{t}_i, K_i)\|_2 $$

trong đó $\mathbf{x}_{ij}$ là vị trí 2D quan sát được của điểm $j$ trong ảnh $i$, $\pi(\cdot, K_i)$ là phép chiếu phối cảnh với nội tham $K_i$, còn $R_i\mathbf{X}_j + \mathbf{t}_i$ biến đổi điểm 3D về hệ toạ độ camera. Tổng bình phương sai số này trên *tất cả* ảnh và điểm chính là **Bundle Adjustment objective**:

$$ \min_{\{R_i,\mathbf{t}_i\},\{\mathbf{X}_j\}} \sum_i \sum_j \rho\left(e_{ij}^2\right) $$

với $\rho$ là một robust loss function (Huber) để giảm ảnh hưởng của outlier.

COLMAP tối ưu bài toán này bằng Levenberg-Marquardt, khai thác cấu trúc thưa của Jacobian (mỗi điểm chỉ quan sát được trong vài ảnh) để giải hiệu quả.

### 2.2 Multi-View Stereo (MVS)

Sau khi có camera pose từ SfM, MVS trả lời: với mỗi pixel trong ảnh tham chiếu, điểm 3D tương ứng nằm ở đâu?

Nguyên lý **photometric consistency**: nếu một điểm 3D là bề mặt thật, thì khi chiếu nó sang các ảnh lân cận, màu sắc quan sát được phải *giống nhau* (giả định bề mặt Lambertian). COLMAP dùng **PatchMatch stereo** — thuật toán tìm depth + normal cho mỗi pixel bằng cách lan truyền thông tin trong không gian ảnh, kết hợp photometric consistency cost.

MVS đầu ra một **dense depth map** cho mỗi ảnh; fusion các depth map này cho **dense point cloud** hoặc mesh (vd qua Poisson surface reconstruction).

### 2.3 Geometric verification & pose graph

Trước khi chạy SfM incremental, cần biết cặp ảnh nào "liên quan" với nhau. COLMAP:

1. Dùng **SIFT** trích keypoint và descriptor.
2. Match descriptor giữa mọi cặp ảnh (có thể dùng vocabulary tree để giảm $O(N^2)$).
3. **Geometric verification**: với mỗi cặp, ước lượng fundamental matrix $F$ (uncalibrated) hoặc essential matrix $E$ (calibrated) từ các match. Tỉ lệ inlier quyết định cặp đó có geometric overlap không.
4. Xây dựng **pose graph**: node = ảnh, edge = cặp đã verify. Chọn cặp khởi tạo tốt nhất (nhiều inlier, baseline đủ lớn).

---

## **3. COLMAP: hiện thực incremental SfM tiêu chuẩn**

COLMAP chọn chiến lược **incremental** (thêm từng ảnh một) thay vì **global** (tối ưu tất cả một lần):

| | Incremental (COLMAP) | Global SfM |
|---|---|---|
| Khởi tạo | Cặp ảnh tốt nhất | Tất cả relative pose cùng lúc |
| Robustness | Cao — từng bước lọc outlier | Thấp — một cặp sai làm hỏng toàn bộ |
| Tốc độ | Chậm — BA lại sau mỗi ảnh | Nhanh hơn — BA một lần cuối |
| Độ chính xác | Cao nhất | Phụ thuộc chất lượng khởi tạo |

Quy trình incremental của COLMAP:

1. **Khởi tạo**: chọn cặp ảnh có nhiều inlier nhất, baseline đủ lớn. Ước lượng relative pose, triangulate điểm 3D ban đầu.
2. **Đăng ký ảnh mới**: chọn ảnh kế tiếp thấy nhiều điểm 3D đã biết nhất. Dùng PnP solver (Perspective-n-Point) để ước lượng pose từ 2D–3D correspondence.
3. **Triangulate điểm mới**: mọi điểm 2D chưa có 3D mà xuất hiện trong ≥2 ảnh đã đăng ký → triangulate.
4. **Bundle Adjustment**: tối ưu lại toàn bộ camera + điểm. COLMAP dùng local BA (chỉ ảnh mới + lân cận) rồi global BA định kỳ.
5. **Lọc outlier**: xoá điểm có reprojection error cao hoặc góc triangulation quá hẹp.
6. Lặp 2–5 đến khi không còn ảnh để thêm.

---

## **4. Giới hạn — nơi COLMAP thất bại**

Đây là phần quan trọng nhất để hiểu *tại sao* cần Dust3R, NeRF, 3DGS, VGGT:

- **Phụ thuộc texture**: SIFT cần góc, cạnh, pattern rõ ràng. Tường trắng, mặt nước, bầu trời → không có keypoint → không reconstruct được. Hậu quả trực tiếp: các bề mặt không texture trở thành "lỗ" trong point cloud.
- **Vật liệu không Lambertian**: photometric consistency giả định màu không đổi khi thay đổi góc nhìn. Vật trong suốt, gương, kim loại bóng → giả định vỡ → MVS thất bại. CVPR 2026 có hẳn bài **3DReflecNet** (INDEX #1) benchmark riêng cho vấn đề này.
- **Occlusion và thin structure**: lưới, cành cây, tóc — quá mảnh để match SIFT ổn định.
- **Cảnh động**: COLMAP giả định scene tĩnh. Người đi bộ, xe chạy, lá cây đung đưa → outlier, gây nhiễu BA.
- **Tốc độ**: incremental SfM + MVS mất hàng giờ cho vài trăm ảnh. Bundle adjustment là nút thắt: $O((M+N)^3)$ trên lý thuyết dù sparse solver giảm đáng kể.
- **Scale ambiguity**: từ ảnh đơn thuần, SfM chỉ khôi phục được hình dạng *tương đối*, không có scale tuyệt đối (trừ khi có thông tin bổ sung như GPS, IMU).
- **Không có prior ngữ nghĩa**: COLMAP không "biết" cái ghế trông thế nào. Nếu thiếu dữ liệu, nó để trống — không "điền" bằng prior như deep learning.

---

## **5. Liên hệ với Latent-Anything**

COLMAP nằm ở vị trí nền móng cho toàn bộ tầng 3D:

- **Đối trọng với feed-forward 3D**: COLMAP là **optimization per-scene**, trong khi Dust3R/Must3R/VGGT là **feed-forward một lần truyền**. Hiểu COLMAP giúp đánh giá *cái gì* được học (và cái gì bị mất) khi chuyển từ optimization sang learning.
- **Camera pose là precondition cho 3DGS**: [3D Gaussian Splatting](../../03b-3d-representation/research/06-3d-gaussian-splatting.md) cần camera pose chính xác — thường lấy từ COLMAP. Sai số camera pose từ SfM lan thẳng vào chất lượng Gaussian.
- **Data engine cho world model**: SfM+MVS tạo ra 3D ground-truth cho việc huấn luyện world model từ video thực tế.
- **Latent của geometry**: COLMAP lưu geometry dưới dạng point cloud + camera — một dạng latent *explicit*, khác với latent *implicit* của NeRF hay Gaussian parameters. Đây là một lát cắt quan trọng trong câu hỏi "biểu diễn 3D nào là latent tốt nhất cho world model?"

---

## Liên quan

- [Dust3R, Must3R & VGGT-Ω — Feed-forward 3D Reconstruction](02-feed-forward-3d-recon.md) — thế hệ kế tiếp, thay optimization bằng one-pass inference.
- [3D Gaussian Splatting](../../03b-3d-representation/research/06-3d-gaussian-splatting.md) — hưởng lợi trực tiếp từ camera pose COLMAP.
- [NeRF](../../03b-3d-representation/research/02-nerf.md) — giải quyết vấn đề texture-less của COLMAP bằng prior học máy.
- [CVPR 2026: Từ "nhìn thấy gì" đến "làm gì"](04-vision-for-action.md) — COLMAP là ví dụ hoàn hảo của paradigm "nhìn để dựng bản đồ", paradigm đang bị thách thức bởi feed-forward direct action.

## Tham khảo

- Schönberger & Frahm, *Structure-from-Motion Revisited* (CVPR 2016).
- Schönberger et al., *Pixelwise View Selection for Unstructured Multi-View Stereo* (ECCV 2016).
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (Cambridge University Press, 2004).
- Furukawa & Hernández, *Multi-View Stereo: A Tutorial* (Foundations and Trends in CG & Vision, 2015).
