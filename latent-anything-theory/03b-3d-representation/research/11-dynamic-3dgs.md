# Dynamic 3DGS — mở rộng 3D Gaussian Splatting cho scene có chuyển động

> **TL;DR.** Dynamic 3DGS mở rộng [3D Gaussian Splatting](06-3d-gaussian-splatting.md) từ scene tĩnh sang scene biến thiên theo thời gian bằng cách làm cho Gaussian có **state phụ thuộc thời gian** hoặc sinh ra từ một **canonical Gaussian set + deformation field**. Ý tưởng cốt lõi có thể viết gọn là $z_i(t)=D_\theta(\tilde{z}_i,t)$, trong đó $\tilde{z}_i$ là Gaussian ở canonical space còn $D_\theta$ dự đoán trạng thái của nó tại thời điểm $t$; hoặc theo nhánh persistent tracking, mỗi Gaussian tự mang quỹ đạo và quay riêng qua thời gian. Đổi lại, mô hình động phải giải quyết thêm temporal consistency, correspondence drift, occlusion, và các chuyển động/phân rã topology mà 3DGS tĩnh không gặp.

3DGS gốc cực mạnh cho novel-view synthesis của **một scene tĩnh**: mỗi Gaussian có vị trí, shape, opacity và appearance cố định, rồi được rasterize ra ảnh từ góc nhìn mới. Nhưng video động không chỉ thay camera; chính scene cũng đổi theo thời gian. Nếu giữ nguyên toàn bộ Gaussian set qua mọi frame, mô hình sẽ buộc một representation tĩnh phải giải thích chuyển động thật, và lỗi xuất hiện gần như ngay lập tức dưới dạng blur, ghosting hoặc geometry không nhất quán giữa các thời điểm.

Vì vậy, Dynamic 3DGS đặt thêm một chiều thời gian lên representation: không chỉ hỏi "Gaussian này ở đâu?", mà còn hỏi "Gaussian này **biến thành gì** khi thời gian trôi đi?". Đây là bước biến 3DGS từ một radiance-field explicit cho scene tĩnh thành một representation 4D có khả năng mô tả motion, deformation và temporal coherence.

---

## **1. Trực giác: từ scene state cố định sang scene state phụ thuộc thời gian**

Với 3DGS tĩnh, scene được lưu như:

$$
\mathcal{S} = \{z_i\}_{i=1}^{N},
\qquad
z_i = (\mu_i,\Sigma_i,o_i,\mathbf{a}_i)
$$

trong đó mỗi Gaussian $z_i$ có một mean $\mu_i$, covariance $\Sigma_i$, opacity $o_i$ và appearance $\mathbf{a}_i$ cố định cho toàn bộ scene. Biểu thức này phù hợp khi chỉ camera thay đổi, còn vật thể trong scene không tự chuyển động.

Với scene động, representation tự nhiên hơn là:

$$
\mathcal{S}(t)=\{z_i(t)\}_{i=1}^{N_t}
$$

trong đó $z_i(t)$ là trạng thái của Gaussian thứ $i$ tại thời điểm $t$. Công thức này có nghĩa là scene không còn là một tập primitive bất biến, mà là một **quỹ đạo của Gaussian set trong thời gian**.

Ngay ở mức trực giác đã có hai chiến lược lớn:

- **persistent Gaussians**: cùng một Gaussian được theo dõi qua thời gian, tự mang motion và rotation của nó;
- **canonical + deformation**: học một Gaussian set chuẩn ở canonical space, rồi dùng deformation field để sinh ra trạng thái tại từng thời điểm.

Hai hướng này khác nhau về cách tổ chức state, nhưng cùng giải quyết một vấn đề chung: làm sao để representation vẫn explicit và render nhanh, trong khi scene không còn đứng yên.

---

## **2. Persistent Dynamic Gaussians: cùng một Gaussian bám cùng một vùng vật lý**

Một nhánh rất trực tiếp, được đại diện rõ bởi **Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis** (Luiten et al., 2023), là cho mỗi Gaussian tự chuyển động và quay theo thời gian, đồng thời giữ một số thuộc tính bền vững như màu, opacity và kích thước.

Khái niệm này có thể viết abstract như:

$$
z_i(t) = (\mu_i(t), R_i(t), S_i, o_i, \mathbf{a}_i)
$$

trong đó $\mu_i(t)$ là quỹ đạo vị trí của Gaussian theo thời gian, $R_i(t)$ là quay theo thời gian, còn scale $S_i$, opacity $o_i$ và appearance $\mathbf{a}_i$ được giữ persistent hoặc chỉ thay đổi ít hơn. Biểu thức này có nghĩa là mục tiêu không chỉ là render đúng từng frame, mà là giữ cho Gaussian thứ $i$ thực sự tiếp tục đại diện **cùng một mảnh vật lý** của scene qua nhiều thời điểm.

Điểm quan trọng của Dynamic 3D Gaussians là:

- motion và rotation được regularize bằng **local rigidity constraints**;
- dense 6-DOF tracking xuất hiện như hệ quả của representation persistent;
- không cần correspondence hay flow supervision đầu vào để từng Gaussian trở thành một "tracker" cục bộ.

Hướng này đặc biệt hấp dẫn khi muốn theo dõi đối tượng hay muốn world model giữ được notion về identity của primitive qua thời gian.

---

## **3. Canonical Gaussian set + deformation field**

Nhánh thứ hai, rất phổ biến trong các biến thể 4DGS và deformable GS, là không cho Gaussian mang quỹ đạo tự do trực tiếp ở mọi frame, mà học một **canonical state** rồi deform nó theo thời gian.

Một cách viết ngắn gọn là:

$$
\tilde{\mathcal{S}}=\{\tilde{z}_i\}_{i=1}^{N},
\qquad
z_i(t)=D_\theta(\tilde{z}_i,t)
$$

trong đó $\tilde{z}_i$ là Gaussian thứ $i$ trong canonical space, còn $D_\theta$ là deformation field hay deformation network phụ thuộc thời gian. Công thức này có nghĩa là geometry động không cần được lưu riêng cho từng frame; thay vào đó, một Gaussian chuẩn được "uốn" thành trạng thái tại từng mốc thời gian.

Đây là đúng tinh thần của:

- **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction** (Yang et al., 2023), nơi scene được học trong canonical space rồi deform để mô hình hóa dynamic monocular scenes;
- **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** (Wu et al., 2023/2024), nơi 3D Gaussians kết hợp với 4D neural voxels và một lightweight MLP để dự đoán deformation tại novel timestamps.

Lợi ích chính của canonical + deformation:

- tiết kiệm hơn so với lưu một scene 3DGS độc lập cho mỗi frame;
- dễ nội suy theo thời gian hơn, vì motion được parameterize thành một hàm trơn;
- cho phép tách "cấu trúc gốc của object/scene" khỏi "biến thiên theo thời gian".

Nhược điểm là chất lượng phụ thuộc mạnh vào deformation model: nếu deformation field không đủ mạnh hoặc không đủ ổn định, dynamic artifacts sẽ tích lũy rất nhanh.

---

## **4. So sánh hai chiến lược chính**

| | Persistent dynamic Gaussians | Canonical + deformation field |
|---|---|---|
| Ý tưởng lõi | cùng một Gaussian được track qua thời gian | Gaussian canonical được deform để sinh frame hiện tại |
| Điểm mạnh | identity và tracking rõ ràng | gọn hơn, nội suy theo thời gian tự nhiên hơn |
| Điểm khó | correspondence drift, rigidity regularization | deformation field khó học, dễ under/overfit motion |
| Hợp với bài toán | tracking, compositional editing, motion-aware state | dynamic novel-view synthesis, time interpolation |
| Temporal consistency | đến từ persistence của primitive | đến từ smoothness của deformation function |

Thực tế, nhiều paper nằm đâu đó giữa hai đầu này:

- vẫn có canonical state,
- nhưng cũng cố giữ một dạng persistence hay regularization để Gaussian không "nhảy identity" giữa các frame.

Vì vậy `Dynamic 3DGS` nên được hiểu như một họ representation hơn là một công thức duy nhất.

---

## **5. Render vẫn là Gaussian rasterization, nhưng state đã là 4D**

Một điểm rất đẹp của Dynamic 3DGS là decoder không cần thay đổi bản chất. Sau khi lấy được trạng thái tại thời gian $t$:

$$
\mathcal{S}(t)=\{z_i(t)\}
$$

ta vẫn render bằng [Gaussian Rasterization](09-gaussian-rasterization.md):

1. project Gaussian 3D tại thời điểm $t$ xuống image plane;
2. lấy footprint ellipse 2D;
3. sort theo depth trong tile;
4. front-to-back alpha composite.

Điều thay đổi nằm ở **state generator**, không nằm ở render core. Vì vậy Dynamic 3DGS thường được nhìn như:

$$
\text{dynamic representation} = \text{static Gaussian decoder} + \text{temporal state model}
$$

trong đó "temporal state model" có thể là trajectory parameters, deformation field, 4D voxel features, hay một motion MLP nhẹ. Biểu thức này có nghĩa là thành phần thời gian có thể được thêm vào mà không phải vứt bỏ toàn bộ ưu thế render nhanh của 3DGS.

Đó cũng là lý do nhiều biến thể dynamic vẫn giữ được real-time hoặc near-real-time rendering ở giai đoạn inference.

---

## **6. Temporal consistency thực ra là bài toán trung tâm**

Trong scene động, render đẹp ở từng frame riêng lẻ là chưa đủ. Cần thêm một điều kiện mạnh hơn: representation phải nhất quán theo thời gian.

Ở mức khái niệm, temporal consistency đòi hỏi:

$$
z_i(t+\Delta t) \approx \text{một biến thiên hợp lý của } z_i(t)
$$

trong đó "hợp lý" có thể nghĩa là:

- chuyển động mượt thay vì giật frame-to-frame;
- rotation nhất quán với local motion;
- opacity và appearance không nhảy lung tung nếu vùng vật lý đó vẫn cùng một bề mặt;
- neighborhood của các Gaussian không vỡ cấu trúc quá mức giữa hai thời điểm gần nhau.

Dynamic 3D Gaussians xử lý việc này bằng persistence + local rigidity. Deformable 3DGS và 4DGS xử lý bằng canonical representation, deformation regularization, và smoothness qua thời gian. Dù cơ chế khác nhau, vấn đề cốt lõi là giống nhau:

- nếu temporal consistency yếu, sẽ thấy jitter,
- nếu correspondence yếu, Gaussian sẽ drift,
- nếu deformation quá tự do, representation sẽ "giải thích lại" scene mỗi frame thay vì thực sự theo dõi cùng scene đó.

Nói ngắn gọn, scene động không chỉ thêm chiều thời gian; nó thêm yêu cầu về **identity qua thời gian**.

---

## **7. Giới hạn / Khi nào Dynamic 3DGS thất bại**

- **Correspondence drift.** Gaussian hôm nay không còn đại diện đúng cùng vùng vật lý ngày mai, nhất là khi motion phức tạp hoặc occlusion mạnh.
- **Topology change khó.** Tóc, nước, khói, vải mỏng, hay vật thể xuất hiện/biến mất nhanh làm canonical assumptions yếu đi.
- **Occlusion và disocclusion.** Phần mới lộ ra theo thời gian rất khó gán ổn định vào Gaussian persistent sẵn có.
- **Monocular ambiguity.** Với một camera, motion và depth dễ lẫn nhau; deformation field có thể học lời giải sai nhưng vẫn render tạm ổn.
- **Storage vs flexibility trade-off.** Lưu nhiều state/time-specific parameters thì nặng; dùng deformation quá nén thì motion phức tạp khó biểu diễn.
- **Training khó hơn 3DGS tĩnh.** Ngoài photometric loss còn phải cân đối smoothness, rigidity, temporal regularization, và đôi khi cả scene-flow-like priors.

Đó là lý do Dynamic 3DGS thường được xem là một bước khó hơn đáng kể so với 3DGS tĩnh: representation explicit giúp render nhanh, nhưng motion làm bài toán state trở nên khó hơn nhiều.

---

## **8. Liên hệ với Latent-Anything**

Dynamic 3DGS là chỗ 3B bắt đầu chạm trực tiếp sang tầng 6 của roadmap: latent giờ không chỉ represent một state, mà còn phải **tiến hóa theo thời gian**.

Nếu Gaussian parameters đã là latent variable như ở [Gaussian parameters là latent variable](10-gaussian-parameters-latent-variable.md), thì Dynamic 3DGS thêm đúng một lớp còn thiếu:

$$
\mathcal{Z}_{t+1} = F(\mathcal{Z}_t, a_t)
$$

trong đó $\mathcal{Z}_t$ là Gaussian set tại thời điểm $t$, còn $F$ là temporal update rule hay transition model. Công thức này có nghĩa là world model kiểu Gaussian-centric không cần rời khỏi representation explicit khi đi từ state hiện tại sang state tương lai.

Điều này đặc biệt hợp với Latent-Anything:

- Layer A có thể probe motion statistics, rigidity, deformation modes, temporal drift của Gaussian set.
- Layer B có thể can thiệp lên motion field, freeze một nhóm Gaussian, hay chỉnh deformation để test phản ứng của model.
- Layer C có thể decode từng rollout step bằng rasterizer gần-deterministic, nên quan hệ giữa latent dynamics và observation rõ ràng hơn nhiều.

Dynamic 3DGS cũng đặt nền trực tiếp cho mục kế tiếp:

- **Gaussian set operations** — vì scene động gần như chắc chắn đòi hỏi add/remove/split/merge Gaussian theo thời gian nếu muốn latent state thật sự linh hoạt.

---

## Liên quan

- [3D Gaussian Splatting](06-3d-gaussian-splatting.md) — nền representation tĩnh mà Dynamic 3DGS mở rộng sang chiều thời gian.
- [Covariance Matrix trong 3DGS](07-covariance-matrix-3dgs.md) — dynamic motion không chỉ dịch chuyển mean mà còn có thể làm covariance quay và deform theo thời gian.
- [Gaussian Rasterization](09-gaussian-rasterization.md) — decoder render vẫn là rasterizer Gaussian; phần động chủ yếu nằm ở state generator theo thời gian.
- [Gaussian parameters là latent variable](10-gaussian-parameters-latent-variable.md) — Dynamic 3DGS là bước biến structured latent đó thành temporal state.
- **Gaussian set operations** ở mục tiếp theo — scene động thường đòi hỏi primitive-level operations để xử lý birth/death/split/merge của Gaussian.

## Tham khảo

- Luiten, Kopanas, Leibe, Ramanan, *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis* (arXiv:2308.09713).
- Yang, Gao, Zhou, Jiao, Zhang, Jin, *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction* (CVPR 2024, arXiv:2309.13101).
- Wu, Yi, Fang, Xie, Zhang, Wei, Liu, Tian, Wang, *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering* (CVPR 2024, arXiv:2310.08528).
- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (ACM Transactions on Graphics 2023, arXiv:2308.04079).
