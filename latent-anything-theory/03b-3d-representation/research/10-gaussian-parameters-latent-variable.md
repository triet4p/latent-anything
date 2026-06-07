# Gaussian Parameters là Latent Variable — từ primitive đồ họa sang state của world model

> **TL;DR.** Khi coi mỗi Gaussian không chỉ là một primitive để render mà là một **biến trạng thái có cấu trúc** $z_i=(\mu_i,\Sigma_i,o_i,\mathbf{a}_i)$, latent space của world model trở thành một **set state** tường minh thay vì một vector đặc trưng khó diễn giải. Ý tưởng cốt lõi là encoder dự đoán một tập tham số Gaussian, transition model cập nhật tập đó theo thời gian hay theo action, và decoder gần-deterministic là [Gaussian Rasterization](09-gaussian-rasterization.md). Đổi lại, latent kiểu này dễ inspect và thao tác hơn, nhưng kéo theo các bài toán khó như cardinality thay đổi, permutation symmetry, split/merge, và correspondence qua thời gian.

Mục này không phải tên chuẩn của một paper duy nhất. Nó là một **design principle** ngày càng rõ trong các hướng Gaussian-centric gần đây: scene không còn được nén vào một latent vector duy nhất, mà được biểu diễn bằng **nhiều phần tử có nghĩa hình học**. Trong [3D Gaussian Splatting](06-3d-gaussian-splatting.md), các Gaussian đã là primitive explicit cho render. Trong các world model mới hơn như DSG-World, GaussianWorld, GaussianDream, hay DLWM, chính không gian Gaussian đó bắt đầu được dùng như nơi mang state, dự đoán evolution, hoặc làm latent prefix cho downstream policy.

---

## **1. Trực giác: latent không còn là một vector, mà là một tập object có tham số**

Trong world model cổ điển, một quan sát $x_t$ thường được encoder nén thành một latent vector:

$$
z_t \in \mathbb{R}^d
$$

trong đó $z_t$ là trạng thái ẩn gộp của toàn bộ scene tại thời điểm $t$. Biểu thức này gọn và tiện cho học, nhưng đổi lại mọi yếu tố hình học, appearance, và object interaction đều bị trộn vào cùng một không gian tọa độ khó diễn giải.

Với Gaussian-centric representation, một state có thể được viết như:

$$
\mathcal{Z}_t = \{z_{t,i}\}_{i=1}^{N_t},
\qquad
z_{t,i} = (\mu_{t,i}, \Sigma_{t,i}, o_{t,i}, \mathbf{a}_{t,i})
$$

trong đó $\mu_{t,i}$ là vị trí, $\Sigma_{t,i}$ là shape/orientation, $o_{t,i}$ là opacity, còn $\mathbf{a}_{t,i}$ là appearance parameters của Gaussian thứ $i$ tại thời điểm $t$. Công thức này có nghĩa là latent state không còn là "một điểm trong không gian ẩn", mà là một **tập các primitive có ngữ nghĩa hình học cục bộ**.

Trực giác của bước chuyển này là:

- một object hay surface patch vốn đã có cấu trúc không gian;
- Gaussian cho phép gói cấu trúc đó thành một token có vị trí, kích thước, hướng và ảnh hưởng nhìn thấy được;
- vì vậy prediction trong latent space có thể gần hơn với prediction trên "thế giới 3D đang diễn ra", thay vì chỉ prediction trên một embedding tổng hợp.

Đây là điểm nối trực tiếp giữa đồ họa explicit và world modeling: render chỉ còn là hậu quả của state, không còn là nơi phải học lại geometry từ đầu.

---

## **2. Mỗi Gaussian mang state gì, và vì sao các tham số đó đủ giàu**

Trong nhánh 3DGS gốc, một Gaussian đã có đủ bốn thành phần cơ bản:

$$
z_i = (\mu_i,\Sigma_i,o_i,\mathbf{a}_i)
$$

trong đó $\mu_i\in\mathbb{R}^3$ là mean 3D, $\Sigma_i\in\mathbb{R}^{3\times 3}$ là covariance hình học, $o_i\in[0,1]$ là opacity, và $\mathbf{a}_i$ là appearance parameters, thường là [Spherical Harmonics](08-spherical-harmonics.md) coefficients. Biểu thức này nói rằng một Gaussian không phải chỉ là một điểm màu, mà là một latent token chứa cả geometry lẫn appearance.

Nếu unpack theo vai trò:

- **mean $\mu$** trả lời Gaussian nằm ở đâu;
- **covariance $\Sigma$** trả lời Gaussian chiếm không gian theo hình nào và quay theo hướng nào;
- **opacity $o$** trả lời Gaussian đóng góp mạnh yếu ra sao;
- **appearance $\mathbf{a}$** trả lời Gaussian trông như thế nào khi render.

Điểm mạnh của thiết kế này là mỗi phần tử latent có **semantic locality** khá rõ:

- đổi $\mu$ chủ yếu làm primitive di chuyển;
- đổi $\Sigma$ chủ yếu làm primitive co giãn hoặc quay;
- đổi $o$ chủ yếu làm primitive mờ/đậm hơn;
- đổi $\mathbf{a}$ chủ yếu làm primitive đổi vẻ ngoài.

So với latent vector dày đặc, structured latent như vậy dễ debug hơn nhiều vì tác động của từng thành phần lên observation gần như có thể dự đoán trước.

---

## **3. Từ scene encoder sang Gaussian set**

Khi coi Gaussian parameters là latent variable, encoder không còn map quan sát tới một vector duy nhất, mà tới một set structured state:

$$
E:\; x_t \mapsto \mathcal{Z}_t
$$

trong đó $x_t$ là quan sát tại thời điểm $t$, còn $\mathcal{Z}_t$ là tập Gaussian state tương ứng. Công thức này có nghĩa là quá trình encoding bây giờ bao gồm cả việc suy ra **số lượng primitive**, **vị trí của chúng**, và **tham số của từng primitive**.

Tuỳ bài toán, encoder có thể làm điều này theo vài kiểu khác nhau:

- khởi tạo từ point cloud hoặc multi-view reconstruction rồi tối ưu Gaussian set;
- dự đoán Gaussian queries trực tiếp từ quan sát 2D như một latent prefix;
- tách state theo static / dynamic components rồi encode từng nhánh riêng.

Các paper gần đây minh họa các biến thể của ý này:

- **DSG-World** xây một 3D Gaussian world model explicit từ dual-state observations và thao tác trực tiếp trong Gaussian space.
- **GaussianWorld** suy luận evolution của scene trong 3D Gaussian space cho bài toán streaming occupancy prediction.
- **GaussianDream** đưa các learnable GaussianDream Queries vào encoder để nắm current 3D structure và short-horizon future evolution.
- **DLWM** dùng Gaussian-centric scene representation làm nền cho dual latent world models trong autonomous driving.

Những paper này không hoàn toàn giống nhau về bài toán, nhưng cùng chia sẻ một ý cốt lõi: **trạng thái bên trong mô hình không còn là latent vector thuần túy mà đã gắn với các thành phần Gaussian có nghĩa hình học**.

---

## **4. Transition model: dự đoán evolution trong Gaussian space**

Một khi state đã là Gaussian set, dynamics có thể được viết ở mức abstract như:

$$
\mathcal{Z}_{t+1} = F(\mathcal{Z}_t, a_t)
$$

trong đó $\mathcal{Z}_t$ là Gaussian state hiện tại, $a_t$ là action hay control input, còn $F$ là transition model. Biểu thức này có nghĩa là thay vì dự đoán frame tương lai trực tiếp trong pixel space, mô hình dự đoán cách **các primitive hình học** thay đổi theo thời gian.

Ở mức primitive, transition có thể được nghĩ như:

$$
z_{t+1,i} = f(z_{t,i}, a_t, \mathcal{N}_{t,i})
$$

trong đó $\mathcal{N}_{t,i}$ biểu diễn ngữ cảnh lân cận hoặc interaction với các Gaussian khác. Công thức này nói rằng evolution của một Gaussian hiếm khi hoàn toàn độc lập: object motion, occlusion, contact, hay scene flow đều có thể phụ thuộc vào lân cận.

Ưu điểm của transition trong Gaussian space:

- predict motion như dịch chuyển mean thường tự nhiên hơn predict lại pixel;
- predict deformation như đổi covariance gắn trực tiếp với geometry;
- scene evolution có thể tách thành motion của static background, dynamic object, và newly visible regions như trong GaussianWorld.

Nói cách khác, latent dynamics không còn là “embedding drift”, mà trở thành **state transition trên các phần tử hình học**.

---

## **5. Decoder: Gaussian rasterizer gần-deterministic thay cho decoder neural nặng**

Khi state đã là Gaussian set, decoder không cần học lại toàn bộ hình học. Nó chỉ cần render state đó ra observation:

$$
\hat{x}_t = R(\mathcal{Z}_t, v_t)
$$

trong đó $R$ là rasterizer và $v_t$ là camera/viewpoint tại thời điểm $t$. Công thức này có nghĩa là phần “giải mã” chủ yếu là hình học + compositing, chứ không phải một mạng sâu phải tái tạo ảnh từ một embedding mờ nghĩa.

Trong nhánh 3DGS, decoder gần-deterministic chính là [Gaussian Rasterization](09-gaussian-rasterization.md):

- project từng Gaussian xuống image plane;
- evaluate footprint và màu;
- sort theo depth trong tile;
- front-to-back alpha composite.

Lợi ích rất lớn của cách thiết kế này là:

- latent state và observation model được tách bạch rõ;
- decoder có tính giải thích cao;
- lỗi observation có thể trace ngược về primitive nào gây ra sai khác.

Đây là một điểm khác biệt nền tảng với nhiều latent world model vector-based, nơi decoder thường là một neural renderer dày đặc và khó audit hơn nhiều.

---

## **6. Vì sao latent kiểu Gaussian hấp dẫn hơn vector latent trong world modeling**

| | Vector latent cổ điển | Gaussian-parameter latent |
|---|---|---|
| Cấu trúc state | một vector dày đặc | một tập token có nghĩa hình học |
| Tính giải thích | thấp | cao hơn rõ rệt |
| Chỉnh sửa cục bộ | khó | tự nhiên hơn |
| Decoder | thường là neural decoder | có thể là rasterizer gần-deterministic |
| Theo dõi dynamics | entangled | có thể gắn với motion/deformation cục bộ |
| Cardinality | cố định | có thể thay đổi |

Sức hút của Gaussian latent đến từ ba điểm:

1. **Explicit geometry.**  
   State đã mang 3D structure thay vì chỉ ngầm chứa nó.

2. **Locality của thao tác.**  
   Split, merge, move, prune, add Gaussian là những operation tự nhiên.

3. **Observation model rẻ hơn.**  
   Rendering không cần query MLP toàn cục nhiều lần như [NeRF](02-nerf.md).

Đó là lý do các hướng mới trong driving và robotics ngày càng nghiêng về Gaussian-centric representation khi cần vừa structure, vừa renderability, vừa khả năng dự đoán evolution.

---

## **7. Giới hạn / Khi nào ý tưởng này trở nên khó**

- **Cardinality thay đổi.** Số Gaussian $N_t$ không cố định, nên state space không còn là tensor kích thước cố định đơn giản.
- **Permutation symmetry.** Thứ tự Gaussian trong set không mang nghĩa, nên transition model phải bất biến hoặc gần bất biến theo permutation.
- **Data association qua thời gian khó.** Gaussian nào ở $t$ tương ứng với Gaussian nào ở $t+1$ không phải lúc nào cũng rõ, nhất là khi có split/merge.
- **Dynamics phức tạp hơn rigid motion.** Mean dịch chuyển thì dễ hiểu, nhưng covariance, opacity, và appearance cùng đổi theo occlusion/deformation sẽ khó ổn định hơn.
- **State quá chi tiết có thể nặng.** Dùng hàng chục nghìn hay hàng triệu Gaussian làm latent state khiến planning hoặc long-horizon rollout trở nên đắt.
- **Không phải decoder nào cũng hoàn toàn deterministic.** Nếu upstream cần hallucinate unseen regions, completion module hay prediction head vẫn có thể cần học thêm.

Vì vậy, “Gaussian parameters là latent variable” rất mạnh về mặt cấu trúc, nhưng không miễn phí. Nó đổi một latent vector dễ đóng gói lấy một state space giàu nghĩa nhưng khó quản hơn.

---

## **8. Liên hệ với Latent-Anything**

Mục này gần như là chìa khóa để 3B chạm trực tiếp vào kiến trúc của Latent-Anything.

- Layer A có thể probe distribution của mean, covariance, opacity, hoặc appearance như những biến state thực thụ.
- Layer B có thể can thiệp bằng các operation có nghĩa: move, scale, split, merge, clone, prune.
- Layer C có thể rollout state qua thời gian trong Gaussian space rồi decode bằng rasterizer.

Nếu viết theo primitive của framework, một `LatentSpace` kiểu Gaussian-centric có thể không còn là `Tensor[d]`, mà là một object:

$$
\texttt{LatentSpace} \equiv \{\texttt{GaussianPrimitive}_i\}_{i=1}^{N}
$$

trong đó mỗi `GaussianPrimitive` mang một bộ field có thể inspect, manipulate, và render. Biểu thức này có nghĩa là latent space được nâng từ “mảng số” lên thành “tập phần tử có schema”.

Đây cũng là bước mở đường trực tiếp cho các mục sau:

- **Dynamic 3DGS** để hiểu Gaussian state biến thiên theo thời gian;
- **Gaussian set operations** để formalize add/remove/merge/split như primitive-level latent algebra;
- các tầng 6–7 của roadmap, nơi latent không chỉ represent mà còn phải **transition** và **plan**.

---

## Liên quan

- [3D Gaussian Splatting](06-3d-gaussian-splatting.md) — cung cấp primitive cơ sở $(\mu,\Sigma,o,\mathbf{a})$ để biến thành structured latent state.
- [Covariance Matrix trong 3DGS](07-covariance-matrix-3dgs.md) — covariance là phần geometry của latent token Gaussian.
- [Spherical Harmonics](08-spherical-harmonics.md) — appearance phụ thuộc hướng nhìn là phần latent appearance của mỗi Gaussian.
- [Gaussian Rasterization](09-gaussian-rasterization.md) — decoder gần-deterministic từ Gaussian set sang ảnh 2D.
- **Latent transition model** ở tầng 6 — sẽ là bản world-model tổng quát hơn của ý transition trong Gaussian space khi note đó được viết.

## Tham khảo

- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (ACM Transactions on Graphics 2023, arXiv:2308.04079).
- Hu, Wen, Li, Wang, *DSG-World: Learning a 3D Gaussian World Model from Dual State Videos* (arXiv:2506.05217).
- Zuo, Zheng, Huang, Zhou, Lu, *GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction* (CVPR 2025, arXiv:2412.10373).
- Zhang, Jiang, Cheng, Li, Liu, Zhao, Luo, Zhou, Yu, *GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation* (arXiv:2605.20752).
- Zhu, Xue, Zhang, Jiang, Zhou, Yan, Gao, Cai, Liu, Li, Shen, *DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving* (CVPR 2026, arXiv:2604.00969).
