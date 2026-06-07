# NeRF — Neural Radiance Fields

> **TL;DR.** NeRF biểu diễn cả một *scene* 3D bằng một MLP ánh xạ tọa độ 5D **(vị trí $x,y,z$ + hướng nhìn $\theta,\phi$) → (màu $c$, mật độ $\sigma$)**, rồi tổng hợp ảnh từ góc nhìn mới bằng **volume rendering khả vi**. Vì render khả vi nên chỉ cần một tập ảnh có pose để tối ưu bằng photometric loss. Caveat: train/render **chậm**, **per-scene**, scene **tĩnh**.

NeRF (Mildenhall et al., 2020) là bước kế thừa trực tiếp của [Neural Implicit Representation](01-neural-implicit-representation.md): thay vì hàm SDF/occupancy mô tả *hình học*, NeRF học một **trường bức xạ (radiance field)** mô tả cả hình học lẫn diện mạo phụ thuộc hướng nhìn, và "đọc" nó ra ảnh bằng phép tích phân quang học. Đây là lời giải cho bài toán **novel view synthesis**: cho vài chục ảnh chụp một vật/cảnh ở các góc đã biết, sinh ra ảnh ở góc nhìn hoàn toàn mới.

---

## **1. Đầu vào/đầu ra: hàm 5D**

NeRF là một MLP $F_\Theta$:

$$
F_\Theta : (\underbrace{x,y,z}_{\text{vị trí } \mathbf{x}},\ \underbrace{\theta,\phi}_{\text{hướng } \mathbf{d}}) \;\longmapsto\; (\mathbf{c}, \sigma)
$$

trong đó $\mathbf{c}=(r,g,b)$ là màu phát ra và $\sigma$ là **mật độ thể tích (volume density)** — hiểu như xác suất vi phân tia sáng bị "dừng" tại điểm đó.

**Một chi tiết thiết kế then chốt:** mật độ $\sigma$ chỉ phụ thuộc **vị trí** $\mathbf{x}$ (hình học không đổi theo góc nhìn), còn màu $\mathbf{c}$ phụ thuộc **cả vị trí lẫn hướng nhìn** $\mathbf{d}$ (để tái hiện phản xạ bóng, specular). Kiến trúc vì thế tách hai nhánh: MLP xử lý $\mathbf{x}$ cho ra $\sigma$ + một vector đặc trưng, rồi vector này nối với $\mathbf{d}$ để cho ra $\mathbf{c}$.

---

## **2. Volume rendering — biến trường thành pixel**

Để tô màu một pixel, NeRF bắn một **tia** $\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ từ tâm camera qua pixel đó, rồi tích phân màu·mật độ dọc tia (mô hình hấp thụ-phát xạ cổ điển):

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt,
\qquad
T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)
$$

trong đó $[t_n, t_f]$ là khoảng gần–xa của tia, và $T(t)$ là **độ truyền qua tích lũy (accumulated transmittance)** — xác suất tia đi được tới $t$ mà *chưa* va vào vật gì. Trực giác: điểm vừa có mật độ cao ($\sigma$ lớn) vừa chưa bị che ($T$ lớn) đóng góp nhiều nhất vào màu pixel.

Tích phân được xấp xỉ rời rạc bằng cách lấy $N$ mẫu dọc tia và dùng **alpha compositing**:

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i\,\big(1 - e^{-\sigma_i \delta_i}\big)\,\mathbf{c}_i,
\qquad
T_i = \exp\!\left(-\sum_{j=1}^{i-1}\sigma_j \delta_j\right)
$$

với $\delta_i = t_{i+1}-t_i$ là khoảng cách giữa hai mẫu liên tiếp. Đại lượng $\alpha_i = 1 - e^{-\sigma_i \delta_i}$ chính là **độ đục (opacity)** của đoạn $i$; $T_i$ là phần ánh sáng còn sót lại khi tới mẫu $i$. Công thức này **khả vi theo mọi mẫu**, nên gradient của photometric loss chảy ngược được về trọng số $\Theta$.

---

## **3. Positional encoding — chống spectral bias**

Nếu đưa thẳng tọa độ vào MLP, NeRF sẽ bị **spectral bias** (xem [Neural Implicit Representation](01-neural-implicit-representation.md)) và cho ảnh mờ. NeRF ánh xạ mỗi tọa độ qua **positional encoding** tần số cao trước khi vào mạng:

$$
\gamma(p) = \big(\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\big)
$$

trong đó $p$ là một thành phần tọa độ và $L$ là số dải tần. Việc "trải" tọa độ ra nhiều tần số cho MLP truy cập trực tiếp các hàm cơ sở tần số cao → tái hiện được cạnh sắc, kết cấu chi tiết.

---

## **4. Hierarchical sampling — lấy mẫu thông minh**

Lấy mẫu đều dọc tia rất phí: phần lớn không gian là rỗng. NeRF train **hai mạng**:

* **Coarse:** lấy mẫu thưa, đều dọc tia để ước lượng xem mật độ tập trung ở đâu.
* **Fine:** dùng phân bố mật độ từ mạng coarse làm hàm trọng số để **importance sampling** thêm mẫu vào đúng vùng bề mặt, rồi render lại.

Loss tối ưu đồng thời tổng sai số tái tạo của cả hai mạng.

---

## **5. Huấn luyện**

Đầu vào chỉ là **một tập ảnh có pose camera đã biết** (thường ước lượng bằng COLMAP). Hàm mất mát là **photometric loss** — bình phương sai số giữa màu pixel render ra và màu pixel thật:

$$
\mathcal{L} = \sum_{\mathbf{r}} \big\| \hat{C}_{\text{coarse}}(\mathbf{r}) - C_{\text{gt}}(\mathbf{r}) \big\|^2 + \big\| \hat{C}_{\text{fine}}(\mathbf{r}) - C_{\text{gt}}(\mathbf{r}) \big\|^2
$$

Không cần nhãn 3D, mesh, hay depth — chỉ ảnh 2D + pose. Toàn bộ hình học 3D **tự nổi lên** từ ràng buộc đa góc nhìn (multi-view consistency).

---

## **6. Giới hạn (where it breaks)**

* **Chậm:** train một scene mất hàng giờ–vài ngày; render một ảnh cần hàng trăm nghìn lần query MLP → không real-time. Đây là nhược điểm bị tấn công mạnh nhất.
* **Per-scene:** mỗi scene là một mạng riêng, không tổng quát hóa sang scene mới (không như một decoder điều kiện hóa latent).
* **Tĩnh:** NeRF gốc giả định scene bất động và ánh sáng cố định ("baked"); cảnh động/đổi sáng cần biến thể riêng.
* **Cần pose chính xác** và đủ nhiều góc nhìn; pose sai làm hỏng tái tạo.
* **Aliasing** khi ảnh train/test ở độ phân giải/khoảng cách khác nhau (Mip-NeRF khắc phục).

## **7. Các biến thể đáng chú ý**

| Biến thể | Giải quyết |
|---|---|
| **Instant-NGP** (Müller et al., 2022) | Thay positional encoding bằng **multiresolution hash grid** → train từ vài ngày xuống **vài giây/phút** |
| **Mip-NeRF** (Barron et al., 2021) | Render theo hình nón thay vì tia đơn → **chống aliasing** đa tỉ lệ |
| **Mip-NeRF 360** | Scene **không giới hạn** (unbounded, 360°) |

So sánh ngắn ba cách biểu diễn 3D trong tầng này:

| | NIR (SDF/occupancy) | NeRF (radiance field) | 3D Gaussian Splatting |
|---|---|---|---|
| Mạng học | trường hình học | trường màu + mật độ | **không có MLP** — tập Gaussian tường minh |
| Render | marching cubes | volume rendering (tích phân) | rasterize (splatting) |
| Tốc độ render | trung bình | chậm | **real-time** |
| Chỉnh sửa cục bộ | khó | khó (hàm toàn cục) | dễ (phần tử rời rạc) |

(3D Gaussian Splatting là mục tiếp theo của tầng 3B.)

---

## **8. Liên hệ với Latent-Anything**

* NeRF là một **decoder hình học khả vi**: trọng số $\Theta$ (hoặc một latent điều kiện hóa NeRF) *là* biểu diễn của scene. Đây là khuôn mẫu cho `ModelAdapter` của các model 3D dạng radiance field.
* Vì render khả vi, có thể đưa NeRF vào pipeline tối ưu/can thiệp ở Layer B–C.
* Đặt cùng với NIR và (sắp tới) 3D Gaussian Splatting, NeRF cho thấy **các lựa chọn thiết kế latent khác nhau** cho cùng mục tiêu: nhồi một scene 3D render được vào latent space — đánh đổi giữa tính liên tục/nhỏ gọn (NeRF) và tốc độ/khả năng chỉnh sửa (3DGS).

---

## Liên quan

- [Neural Implicit Representation](01-neural-implicit-representation.md) — prerequisite: ý tưởng "scene = một hàm theo tọa độ" và spectral bias.
- [Giả thuyết Đa tạp](../../01-space-representation/research/03-manifold-hypothesis.md) — một scene cũng là một đa tạp được hàm mã hóa.
- [Slerp](../../03-geometry-structure/research/05-slerp.md) — nội suy giữa các góc nhìn/latent của scene.

## Tham khảo

- Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng, *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis* (ECCV 2020, arXiv:2003.08934).
- Müller, Evans, Schied, Keller, *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding* (Instant-NGP) (ACM TOG / SIGGRAPH 2022, arXiv:2201.05989).
- Barron et al., *Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields* (ICCV 2021, arXiv:2103.13415).
- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains* (NeurIPS 2020, arXiv:2006.10739) — nền của positional encoding.
- Kajiya, Von Herzen, *Ray Tracing Volume Densities* (SIGGRAPH 1984) — mô hình volume rendering cổ điển NeRF kế thừa.
