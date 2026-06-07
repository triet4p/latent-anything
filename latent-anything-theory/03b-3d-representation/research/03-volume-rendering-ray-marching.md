# Volume Rendering & Ray Marching

> **TL;DR.** Volume rendering biến một trường $(\sigma, c)$ thành màu pixel bằng cách **tích phân màu·mật độ dọc một tia**, trọng số theo độ truyền qua $T(t)$. Tích phân được xấp xỉ bằng **ray marching** (lấy $N$ mẫu dọc tia) + **alpha compositing** $\alpha_i = 1-e^{-\sigma_i\delta_i}$. Vì mọi bước là hàm trơn của $\sigma, c$ nên toàn bộ **khả vi** — đó là điều cho phép [NeRF](02-nerf.md) học trường chỉ từ ảnh 2D.

Đây là "cơ chế đọc" của các trường thể tích: [Neural Implicit Representation](01-neural-implicit-representation.md) và NeRF lưu scene dưới dạng hàm; volume rendering là cách chiếu hàm đó ra ảnh. Note này đi sâu vào *phép tích phân, cách lấy mẫu, và vì sao nó khả vi* — phần mà note NeRF chỉ tóm tắt.

---

## **1. Mô hình phát xạ–hấp thụ (emission–absorption)**

Coi không gian là một môi trường tham gia (participating medium): mỗi điểm vừa **phát ra** màu $c$ vừa **hấp thụ** ánh sáng với mật độ $\sigma$. Màu của một tia $\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ là:

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,c(\mathbf{r}(t),\mathbf{d})\,dt,
\qquad
T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)
$$

trong đó $T(t)$ là **độ truyền qua tích lũy** — xác suất ánh sáng đi tới $t$ mà chưa bị hấp thụ. Trực giác: điểm đóng góp vào pixel khi vừa **đặc** ($\sigma$ lớn) vừa **chưa bị che** ($T$ lớn). Đây là mô hình quang học cổ điển (Kajiya & Von Herzen 1984; Max 1995).

---

## **2. Rời rạc hóa: ray marching + alpha compositing**

Không tính được tích phân giải tích, ta **march** dọc tia: lấy $N$ mẫu tại $t_1<\dots<t_N$, đặt $\delta_i = t_{i+1}-t_i$. Quy tắc cầu phương (quadrature) cho:

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i\,\alpha_i\,c_i,
\qquad
\alpha_i = 1 - e^{-\sigma_i \delta_i},
\qquad
T_i = \prod_{j=1}^{i-1}(1-\alpha_j)
$$

trong đó $\alpha_i$ là **độ đục (opacity)** của đoạn $i$ (xác suất tia bị dừng trong đoạn đó), $T_i$ là phần ánh sáng còn sót khi tới mẫu $i$, và $w_i = T_i\alpha_i$ là **trọng số đóng góp** của mẫu $i$. Công thức này *chính xác* là alpha compositing "over" trong đồ họa. Tổng $\sum_i w_i \le 1$; phần thiếu là nền (background).

* **Bản đồ độ sâu (depth)** rơi ra miễn phí: $\hat{D}(\mathbf{r}) = \sum_i w_i\,t_i$ — kỳ vọng vị trí dừng của tia.

---

## **3. Vì sao khả vi — và vì sao điều đó quan trọng**

Mỗi $\alpha_i$ là hàm mũ trơn của $\sigma_i$; $T_i$ là tích các $(1-\alpha_j)$; $\hat{C}$ là tổng có trọng số của $c_i$. Tất cả đều khả vi theo $\sigma_i, c_i$, nên gradient của photometric loss $\|\hat{C}-C_{\text{gt}}\|^2$ chảy ngược được về trường $(\sigma,c)$ và về trọng số mạng. **Không có khâu khả vi này thì không thể học hình học chỉ từ ảnh 2D** — đó là khác biệt cốt lõi so với render rasterize truyền thống.

---

## **4. Lấy mẫu dọc tia**

Lấy mẫu quyết định cả chất lượng lẫn chi phí.

* **Stratified sampling (lấy mẫu phân tầng):** chia $[t_n, t_f]$ thành $N$ ô đều, rồi lấy *ngẫu nhiên đều một điểm trong mỗi ô*. Nhờ vậy, qua các bước huấn luyện, mạng bị query ở *vô số* vị trí liên tục thay vì một lưới cố định → tránh học "vá" theo các điểm rời rạc, cho trường liên tục.
* **Hierarchical sampling (lấy mẫu phân cấp):** lấy mẫu đều rất phí vì phần lớn tia là không gian rỗng. NeRF train **hai mạng**:
  1. *Coarse:* march thưa, tính các trọng số $w_i$ → chuẩn hóa thành một **phân phối xác suất** dọc tia (nơi nào nhiều khối lượng = nơi có bề mặt).
  2. *Fine:* dùng **inverse transform sampling** trên phân phối đó để lấy thêm mẫu *tập trung quanh bề mặt*, rồi render lại.

  Loss tối ưu đồng thời cả coarse và fine. Đây là **importance sampling**: dồn ngân sách mẫu vào vùng quan trọng.

---

## **5. Giới hạn (where it breaks)**

* **Đắt:** mỗi pixel cần hàng chục–trăm lần query MLP; một ảnh = hàng trăm nghìn tia → render chậm. Đây là nút thắt mà [Instant-NGP](05-instant-ngp.md) (bỏ phần lớn mẫu rỗng bằng occupancy grid + lookup nhanh) và 3D Gaussian Splatting (bỏ hẳn ray marching, chuyển sang splatting) tấn công.
* **Không gian giới hạn:** $[t_n,t_f]$ giả định scene nằm trong một hộp; scene vô hạn (bầu trời, nền xa) cần reparameterize (Mip-NeRF 360).
* **Aliasing:** lấy mẫu điểm dọc tia không "biết" độ rộng pixel → răng cưa khi đổi tỉ lệ; khắc phục bằng [integrated positional encoding](04-positional-encoding.md) của Mip-NeRF.
* **Sai số cầu phương:** quá ít mẫu → tích phân thô, bề mặt mỏng bị bỏ sót.

---

## **6. Liên hệ với Latent-Anything**

* Volume rendering là **decoder khả vi** chuẩn cho mọi biểu diễn trường (NeRF, neural fields): latent → trường $(\sigma,c)$ → (qua render) → ảnh. Đây là khuôn mẫu để gắn một "renderer" deterministic vào Layer C.
* Vì trọng số $w_i$ là một phân phối dọc tia, nó còn cho **độ sâu** và **độ bất định** — tín hiệu hữu ích cho introspection (Layer A) khi phân tích scene latent.

---

## Liên quan

- [NeRF](02-nerf.md) — dùng trực tiếp volume rendering để học radiance field.
- [Neural Implicit Representation](01-neural-implicit-representation.md) — trường được render; SDF/occupancy là biến thể không cần tích phân thể tích.
- [Positional encoding](04-positional-encoding.md) — quyết định trường có sắc nét để render hay không.
- [Instant-NGP](05-instant-ngp.md) — tăng tốc bằng cách bỏ qua không gian rỗng khi march.

## Tham khảo

- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis* (ECCV 2020, arXiv:2003.08934) — stratified + hierarchical sampling.
- Max, N., *Optical Models for Direct Volume Rendering* (IEEE TVCG 1995) — mô hình phát xạ–hấp thụ.
- Kajiya, Von Herzen, *Ray Tracing Volume Densities* (SIGGRAPH 1984).
