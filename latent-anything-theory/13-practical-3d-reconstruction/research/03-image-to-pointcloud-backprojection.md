# Image-to-Point Cloud Feature Back-Projection — Cầu nối 2D–3D cho Multimodal Learning

> **TL;DR.** Phương pháp back-projection chiếu đặc trưng 2D từ ảnh lên point cloud 3D bằng cách dùng depth geometry để ánh xạ mỗi pixel vào toạ độ 3D, sau đó gán feature 2D cho điểm 3D tương ứng. Khác với late fusion (gộp ở cuối pipeline), back-projection cho phép 2D feature đi thẳng vào 3D backbone ở tầng đầu — giữ được spatial alignment chi tiết và cho phép end-to-end training. Hạn chế chính: phụ thuộc chất lượng depth estimation; sai số depth tạo ra sai lệch feature assignment; occlusion và vùng không thấy trong ảnh không có feature 2D tương ứng.

Bài toán cốt lõi của multimodal 3D learning là: **làm sao để kết hợp thông tin từ ảnh 2D (texture, semantic chi tiết) với point cloud 3D (geometry chính xác, bất biến viewpoint)?** Feature back-projection là một trong những cách trực tiếp nhất — và được Han et al. (CVPR 2026) phát triển thành một cơ chế huấn luyện multimodal cho 3D semantic understanding.

---

## **1. Trực giác: feature 2D không phải là "rác" khi có 3D**

Trong pipeline multimodal truyền thống, ảnh 2D thường được xử lý qua một 2D backbone (CNN/ViT) để sinh feature map, trong khi point cloud qua 3D backbone (PointNet++, sparse conv). Kết quả của hai nhánh được **fuse** ở cuối pipeline — thường là concatenate feature vector của object proposal hoặc voxel. Cách này có hai vấn đề:

1. **Mất spatial alignment**: feature 2D đến từ một vùng ảnh cụ thể, nhưng khi fuse ở cuối, mối liên kết không gian giữa pixel ảnh và điểm 3D đã bị mờ.
2. **Hai backbone hoạt động độc lập**: không có tín hiệu gradient nào từ 3D backbone ảnh hưởng đến cách 2D backbone trích xuất feature (và ngược lại) — trừ khi fuse ở đầu.

Back-projection giải quyết cả hai vấn đề này bằng cách **đưa feature 2D vào không gian 3D ngay từ đầu**.

---

## **2. Cơ chế: chiếu ngược từ ảnh lên point cloud**

### 2.1 Phép chiếu

Cho một ảnh $I \in \mathbb{R}^{H \times W \times 3}$ và một point cloud $\mathcal{P} = \{\mathbf{p}_i \in \mathbb{R}^3\}$, giả sử đã biết camera pose $(R, \mathbf{t})$ và nội tham $K$:

1. **Trích xuất feature 2D**: dùng một 2D backbone $\Phi_{2D}$ để thu được feature map $F_{2D} \in \mathbb{R}^{H' \times W' \times C}$.
2. **Chiếu điểm 3D lên ảnh**: với mỗi điểm $\mathbf{p}_i$, tính toạ độ ảnh tương ứng:

   $$ \mathbf{u}_i = \pi(K(R\mathbf{p}_i + \mathbf{t})) $$

   trong đó $\pi$ là phép chiếu phối cảnh (divide by z). $\mathbf{u}_i$ là toạ độ pixel (có thể là số thực, nội suy bilinear).

3. **Back-project feature**: lấy feature 2D tại $\mathbf{u}_i$ qua bilinear sampling và gán cho điểm $\mathbf{p}_i$:

   $$ F_{3D}(\mathbf{p}_i) = \text{sample}(F_{2D}, \mathbf{u}_i) $$

4. **Tổng hợp**: $F_{3D}$ giờ đã có feature từ ảnh, có thể được đưa vào 3D backbone $\Phi_{3D}$ cùng với toạ độ hình học $\mathbf{p}_i$.

Toàn bộ quá trình này khả vi (bilinear sampling differentiable), cho phép gradient từ $\Phi_{3D}$ chảy ngược về $\Phi_{2D}$.

### 2.2 Multimodal training objective

Han et al. training pipeline tối ưu đồng thời 2D backbone và 3D backbone với một shared objective (vd: 3D semantic segmentation loss):

$$ \mathcal{L} = \mathcal{L}_{\text{3D}}(\Phi_{3D}([\mathbf{p}_i; F_{3D}(\mathbf{p}_i)])) + \lambda \mathcal{L}_{\text{align}} $$

trong đó $\mathcal{L}_{\text{align}}$ là một loss phụ để đảm bảo feature 2D và 3D nằm trong cùng một không gian biểu diễn (vd: contrastive loss giữa feature 2D và 3D của cùng một điểm).

---

## **3. Tại sao back-projection hiệu quả hơn late fusion**

| | Late Fusion | Back-Projection |
|---|---|---|
| Feature alignment | Cuối pipeline | Đầu pipeline |
| Spatial granularity | Thô (object/voxel level) | Mịn (per-point) |
| Gradient flow | 2D → classifier, 3D → classifier (tách biệt) | 3D → 2D qua bilinear sampling |
| Missing points | Không ảnh hưởng (feature vector đã gộp) | Không có feature cho điểm không thấy trong ảnh |
| Depth quality dependency | Thấp (chỉ cần biết object nào) | Cao (sai depth → sai feature assignment) |
| End-to-end | Không thực sự | Có |

Mấu chốt là **gradient flow**: khi feature 2D được gán trực tiếp vào từng điểm 3D, tín hiệu từ 3D loss chảy ngược qua bilinear sampling về 2D backbone, dạy nó trích xuất feature *có ích cho 3D task*, không chỉ "feature tốt cho ảnh".

---

## **4. Giới hạn và câu hỏi mở**

- **Phụ thuộc depth/camera pose**: mọi pixel cần được map chính xác vào 3D. Sai số từ depth estimation (hoặc từ [COLMAP](01-colmap-sfm-mvs.md)) gây misalignment feature-point. Đây là lý do back-projection hưởng lợi trực tiếp từ chất lượng của [feed-forward 3D reconstruction](02-feed-forward-3d-recon.md).
- **Occlusion**: điểm 3D bị che khuất trong ảnh không nhận được feature 2D → cần fallback (zero vector, learnable token, hoặc feature từ ảnh khác).
- **Nhiều ảnh → một điểm**: một điểm 3D có thể thấy trong nhiều ảnh. Làm sao aggregate feature từ nhiều view? Mean, max, attention-weighted, hay dùng confidence?
- **Domain gap 2D–3D**: feature 2D từ ImageNet-pretrained backbone có thể không hợp với 3D task → cần joint training hoặc adapter.

---

## **5. Liên hệ với Latent-Anything**

Back-projection là một **adapter cầu nối modality** — chính xác là loại cơ chế mà ModelAdapter trong Latent-Anything cần:

- **2D → 3D là một phép biến đổi latent**: feature 2D được "nâng" lên không gian 3D qua phép chiếu hình học. Đây là một instance của **cross-modal latent alignment** — cùng một điểm trong thế giới thực có hai biểu diễn latent (pixel feature và point feature), và back-projection là một cách đơn giản nhưng khả vi để map giữa chúng.
- **Liên kết với 3DGS**: [3D Gaussian Splatting](../../03b-3d-representation/research/06-3d-gaussian-splatting.md) làm ngược lại — chiếu 3D xuống 2D. Back-projection chiếu 2D lên 3D. Cả hai tạo thành một cặp **project-unproject** hoàn chỉnh, có thể dùng để xây dựng [Gaussian parameters là latent variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md).
- **Từ "nhìn" đến "làm"**: Trong [triết lý CVPR 2026](04-vision-for-action.md), back-projection là một cách để feature 2D *phục vụ trực tiếp* cho 3D task — không chỉ "nhìn để dựng bản đồ", mà nhìn để *gán ý nghĩa* cho geometry.

---

## Liên quan

- [COLMAP & SfM/MVS cổ điển](01-colmap-sfm-mvs.md) — nơi cung cấp camera pose cho back-projection.
- [Dust3R, Must3R & VGGT-Ω](02-feed-forward-3d-recon.md) — cung cấp depth/camera pose dày đặc, giảm sai số back-projection.
- [3D Gaussian Splatting](../../03b-3d-representation/research/06-3d-gaussian-splatting.md) — project 3D→2D, cặp đối xứng của back-projection.
- [Gaussian Parameters là Latent Variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) — từ ảnh đến Gaussian set, một dạng "nâng" 2D feature lên 3D.
- [CVPR 2026: Từ "nhìn thấy gì" đến "làm gì"](04-vision-for-action.md) — triết lý nền cho việc dùng feature để phục vụ action.

## Tham khảo

- Han et al., *Image-to-Point Cloud Feature Back-Projection for Multimodal Training of 3D Semantic Understanding* (CVPR 2026).
- Qi et al., *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space* (NeurIPS 2017).
- Choy et al., *4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks* (CVPR 2019).
