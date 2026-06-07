# Neural Implicit Representation

**Neural Implicit Representation (NIR)** — hay **Implicit Neural Representation (INR)** — là cách biểu diễn một tín hiệu (hình dạng 3D, ảnh, video, trường vật lý) bằng **một hàm liên tục được tham số hóa bởi mạng neural**, thay vì bằng một cấu trúc rời rạc tường minh (explicit) như lưới voxel, mesh, hay point cloud.

Ý tưởng cốt lõi: một mạng MLP nhận **tọa độ** làm đầu vào và trả về **giá trị của tín hiệu tại tọa độ đó**:

$$
f_\theta : \mathbb{R}^n \to \mathbb{R}^m, \qquad f_\theta(\mathbf{x}) = \text{giá trị tại } \mathbf{x}
$$

Với hình học 3D, $f_\theta(x, y, z)$ trả về thông tin về hình dạng tại điểm $(x,y,z)$. Hình dạng không được lưu dưới dạng một danh sách đỉnh/voxel, mà **chính là bộ trọng số $\theta$ của mạng** — bản thân mạng *là* hình dạng. Đây gọi là mạng **coordinate-based** (dựa trên tọa độ).

> Đây là prerequisite cho **NeRF** (mục tiếp theo của tầng 3B) và 3D Gaussian Splatting: hiểu được "biểu diễn 3D = một hàm" là nền tảng để hiểu vì sao một scene có thể được nhồi vào latent và render khả vi.

---

## **1. Explicit vs Implicit**

| | Explicit (tường minh) | Implicit (ẩn) |
|---|---|---|
| Lưu trữ | Voxel grid, mesh, point cloud | Trọng số $\theta$ của MLP |
| Độ phân giải | Cố định, gắn với lưới | **Liên tục, vô hạn** (query tọa độ bất kỳ) |
| Bộ nhớ | Tăng theo $O(N^3)$ với voxel | Tách rời khỏi độ phân giải, rất nhỏ gọn |
| Topology | Khó thay đổi với voxel/mesh | Linh hoạt, biểu diễn topology tùy ý |
| Khả vi | Thường không trơn | **Khả vi hoàn toàn** theo tọa độ và $\theta$ |

Điểm mấu chốt: với explicit, muốn tăng độ chi tiết phải tăng số phần tử (bùng nổ bộ nhớ — liên hệ [lời nguyền chiều](../../01-space-representation/research/04-curse-of-dimensionality.md)). Với implicit, độ chi tiết do **dung lượng mạng** quyết định, không do độ phân giải lưu trữ — nên có thể query ở *bất kỳ* độ phân giải nào sau khi train.

---

## **2. Mạng biểu diễn cái gì? (các loại trường)**

Tùy bài toán, $f_\theta$ học một trong các **trường (field)** sau:

* **Signed Distance Function (SDF):** $f_\theta(\mathbf{x}) \to d \in \mathbb{R}$, trả về khoảng cách *có dấu* tới bề mặt gần nhất (âm = bên trong, dương = bên ngoài). **Bề mặt là tập mức không (zero-level set)** $\{\mathbf{x} : f_\theta(\mathbf{x}) = 0\}$. Đại diện: **DeepSDF** (Park et al., 2019).
* **Occupancy:** $f_\theta(\mathbf{x}) \to [0,1]$, xác suất điểm nằm *bên trong* vật thể. Bề mặt là mặt mức $0.5$. Đại diện: **Occupancy Networks** (Mescheder et al., 2019), **IM-Net** (Chen & Zhang, 2019).
* **Density + Color (radiance):** $f_\theta(\mathbf{x}, \mathbf{d}) \to (\sigma, \mathbf{c})$, trả về mật độ và màu phụ thuộc hướng nhìn $\mathbf{d}$. Đây chính là **NeRF** — chủ đề mục tiếp theo.

Để trích ra hình học tường minh (mesh) từ SDF/occupancy, dùng thuật toán **Marching Cubes** trên tập mức.

---

## **3. Bài toán spectral bias và cách khắc phục**

Một MLP ReLU thuần "ngây thơ" học INR rất **mờ**: nó mắc **spectral bias (thiên kiến phổ)** — ưu tiên khớp các thành phần **tần số thấp** (hình dạng tổng thể, chuyển màu mượt) và **rất khó học chi tiết tần số cao** (cạnh sắc, kết cấu). Đây là rào cản lớn nhất của INR.

Hai giải pháp chuẩn:

* **Positional Encoding / Fourier Features:** ánh xạ tọa độ đầu vào lên không gian tần số cao bằng các hàm sin/cos nhiều tần số trước khi đưa vào MLP:
$$
\gamma(\mathbf{x}) = \big(\sin(2^0\pi \mathbf{x}), \cos(2^0\pi \mathbf{x}), \dots, \sin(2^{L-1}\pi \mathbf{x}), \cos(2^{L-1}\pi \mathbf{x})\big)
$$
  Việc "trải" tọa độ ra nhiều tần số giúp MLP học được chi tiết sắc nét (Tancik et al., 2020 — *Fourier Features*). NeRF dùng đúng kỹ thuật này.
* **Periodic activation (SIREN):** thay ReLU bằng hàm kích hoạt **sin** ($\sin(\omega_0 \cdot)$). Mạng SIREN (Sitzmann et al., 2020) biểu diễn trực tiếp được tín hiệu tần số cao và cả đạo hàm của chúng, rất hợp để học SDF (cần gradient trơn).

---

## **4. Điều kiện hóa bằng latent code — cầu nối tới latent space**

Một MLP đơn lẻ chỉ biểu diễn **một** vật thể. Để biểu diễn **cả một họ hình dạng** (và sinh ra hình mới), ta **điều kiện hóa hàm bằng một latent code** $\mathbf{z}$:

$$
f_\theta(\mathbf{x}, \mathbf{z}) \to \text{giá trị tại } \mathbf{x} \text{ cho hình dạng } \mathbf{z}
$$

* Mỗi hình dạng tương ứng một latent vector $\mathbf{z}$; **di chuyển trong không gian $\mathbf{z}$ = nội suy/biến đổi hình dạng**.
* DeepSDF dùng kiến trúc **auto-decoder**: không có encoder, latent code của mỗi hình được **tối ưu trực tiếp** cùng với $\theta$ (giống tinh thần [VAE](../../02-representation-learning/research/03-vae.md)/[autoencoder](../../02-representation-learning/research/02-autoencoder.md) nhưng bỏ encoder).
* Đây chính là lý do NIR quan trọng với Latent-Anything: nó biến **hình học 3D thành một điểm trong latent space**, nơi ta có thể introspect và manipulate — và decoder (ở đây là MLP + bộ render) là **deterministic**.

---

## **5. Ưu điểm và hạn chế**

**Ưu điểm:**
- Liên tục, **không phụ thuộc độ phân giải**; nhỏ gọn (memory tách khỏi resolution).
- Khả vi → nhúng được vào pipeline tối ưu gradient (đặc biệt khi kết hợp render khả vi như NeRF).
- Biểu diễn topology tùy ý, không bị ràng buộc lưới.

**Hạn chế:**
- **Tối ưu theo từng scene** (per-scene/per-shape optimization) thường chậm — đây là nhược điểm mà Instant-NGP và 3DGS sau này khắc phục.
- Spectral bias cố hữu (cần positional encoding/SIREN).
- Là hàm **toàn cục**: thay đổi một vùng nhỏ có thể ảnh hưởng toàn bộ; khó chỉnh sửa cục bộ và khó cho các phép toán tập hợp (add/remove) — ngược với biểu diễn tường minh như Gaussian Splatting.

---

## **6. Liên hệ với Latent-Anything**

* NIR là dạng **decoder hình học**: latent $\mathbf{z}$ → trường liên tục → (qua render/marching cubes) → hình 3D. Đây là khuôn mẫu để thiết kế `ModelAdapter` cho các model 3D.
* Vì hình dạng = một điểm latent, mọi công cụ ở Tầng 3–4 (nội suy, [slerp](../../03-geometry-structure/research/05-slerp.md), số học latent) áp dụng được cho biến đổi hình học.
* Đặt nền cho việc hiểu vì sao **NeRF** (density+color field) và sau đó 3D Gaussian Splatting (biểu diễn tường minh, amenable với world model) là các lựa chọn thiết kế latent khác nhau.

---

## Liên quan

- [Giả thuyết Đa tạp](../../01-space-representation/research/03-manifold-hypothesis.md) — một hình dạng cũng là một đa tạp được hàm $f_\theta$ mã hóa.
- [Autoencoder](../../02-representation-learning/research/02-autoencoder.md) — auto-decoder của DeepSDF là AE bỏ encoder.
- [VAE](../../02-representation-learning/research/03-vae.md) — điều kiện hóa hàm bằng latent code để sinh hình mới.
- [Lời nguyền chiều](../../01-space-representation/research/04-curse-of-dimensionality.md) — vì sao voxel grid bùng nổ bộ nhớ còn INR thì không.

## Tham khảo

- Park, Florence, Straub, Newcombe, Lovegrove, *DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation* (CVPR 2019, arXiv:1901.05103).
- Mescheder, Oechsle, Niemeyer, Nowozin, Geiger, *Occupancy Networks: Learning 3D Reconstruction in Function Space* (CVPR 2019, arXiv:1812.03828).
- Chen, Zhang, *Learning Implicit Fields for Generative Shape Modeling* (IM-Net) (CVPR 2019).
- Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions* (SIREN) (NeurIPS 2020, arXiv:2006.09661).
- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains* (NeurIPS 2020, arXiv:2006.10739).
- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis* (ECCV 2020, arXiv:2003.08934).
- Xie et al., *Neural Fields in Visual Computing and Beyond* (Eurographics STAR survey, 2022).
