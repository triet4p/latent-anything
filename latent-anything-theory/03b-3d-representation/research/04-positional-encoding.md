# Positional Encoding (Fourier Features)

> **TL;DR.** MLP theo tọa độ có **spectral bias** — học tần số thấp dễ, tần số cao gần như không. Positional encoding ánh xạ tọa độ qua một dải sin/cos nhiều tần số $\gamma(p)=(\sin 2^l\pi p, \cos 2^l\pi p)_l$ trước khi vào mạng, "trao" cho MLP các hàm cơ sở tần số cao. Đây là điều biến [NeRF](02-nerf.md) từ mờ thành sắc; "scale" của encoding là **núm điều khiển băng thông** — quá thấp thì mờ, quá cao thì nhiễu/aliasing.

Note này đi sâu vào *vì sao* cần encoding (góc nhìn NTK), *cách điều khiển* băng thông, và các biến thể (random Fourier, integrated PE của Mip-NeRF). Phần "spectral bias là gì" đã thấy thực nghiệm trong [Neural Implicit Representation](01-neural-implicit-representation.md).

---

## **1. Vì sao MLP thuần bị spectral bias (góc nhìn NTK)**

Một MLP đủ rộng, huấn luyện bằng gradient descent, hành xử như **kernel regression** với **Neural Tangent Kernel (NTK)**. NTK của MLP ReLU trên tọa độ thô là một kernel có **phổ suy giảm rất nhanh theo tần số**: các thành phần (eigenmode) tần số cao có trị riêng cực nhỏ → học cực chậm. Hệ quả thực tế: mạng khớp hình dạng tổng thể trước, chi tiết sắc nét gần như không bao giờ tới trong ngân sách huấn luyện. Đây là **spectral bias** (Rahaman et al. 2019; Tancik et al. 2020).

---

## **2. Positional encoding = Fourier features**

Giải pháp: nâng tọa độ lên không gian tần số trước khi vào MLP. Dạng NeRF (axis-aligned, theo lũy thừa 2):

$$
\gamma(p) = \big(\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\big)
$$

với $p$ là một thành phần tọa độ và $L$ là số dải tần. Tancik et al. (2020) chứng minh đây là **trường hợp riêng của Fourier features** và giải thích tác dụng qua NTK: ánh xạ Fourier biến NTK thành một **kernel dừng (stationary) có băng thông điều chỉnh được** — không còn thiên vị mạnh về tần số thấp, nên MLP học được chi tiết tần số cao.

Dạng tổng quát (random Fourier features):

$$
\gamma(\mathbf{v}) = \big(\cos(2\pi \mathbf{B}\mathbf{v}),\ \sin(2\pi \mathbf{B}\mathbf{v})\big),
\qquad \mathbf{B}_{ij}\sim \mathcal{N}(0,\,s^2)
$$

trong đó mỗi hàng của $\mathbf{B}$ là một tần số ngẫu nhiên, và **$s$ (scale/độ lệch chuẩn) là núm băng thông**: $s$ đặt "tần số trung bình" mà mạng nhìn thấy.

---

## **3. Núm băng thông và đánh đổi**

| Băng thông ($L$ lớn hoặc $s$ lớn) | Hệ quả |
|---|---|
| Quá thấp | Kết quả **mờ** — vẫn còn spectral bias, mất chi tiết |
| Vừa | Sắc nét, khớp đúng dải tần của tín hiệu |
| Quá cao | **Nhiễu hạt / aliasing / overfit** — mạng dựng được tần số cao hơn cả tín hiệu, nội suy xấu giữa các điểm |

Vì thế $L$/$s$ là siêu tham số phải chỉnh theo độ chi tiết của dữ liệu — một điểm yếu (chỉnh tay) mà các phương pháp học đặc trưng như hash grid sau này tránh được.

**SIREN** là hướng thay thế: thay vì encoding đầu vào, dùng *hàm kích hoạt $\sin$* trong mạng (xem [Neural Implicit Representation](01-neural-implicit-representation.md)) — đạt cùng mục tiêu biểu diễn tần số cao theo cách khác.

---

## **4. Integrated Positional Encoding (Mip-NeRF) — encoding biết tỉ lệ**

PE chuẩn lấy encoding tại **một điểm vô cùng nhỏ**, nên không "biết" pixel đang phủ vùng to hay nhỏ → **aliasing** khi đổi độ phân giải/khoảng cách. **Mip-NeRF** (Barron et al. 2021) thay tia bằng **hình nón**, chia thành các *conical frustum*, xấp xỉ mỗi frustum bằng một Gaussian, rồi tính **kỳ vọng của PE trên Gaussian đó** (dạng đóng) — gọi là **Integrated Positional Encoding (IPE)**:

$$
\gamma_{\text{IPE}} = \mathbb{E}_{\mathbf{x}\sim\mathcal{N}(\mu,\Sigma)}[\gamma(\mathbf{x})]
$$

Điểm hay: khi vùng tích phân **rộng** so với chu kỳ của một tần số, kỳ vọng của sin/cos tần số đó **co về 0** một cách tự động. Tức IPE **tự tắt các tần số cao** ở vùng nhìn thô → chống aliasing, và encoding *mã hóa luôn kích thước vùng* (scale-aware). Đây là cách đưa khái niệm mipmap vào neural field.

---

## **5. Giới hạn (where it breaks)**

* **Băng thông cố định, chỉnh tay:** $L$/$s$ phải đoán trước; sai là mờ hoặc nhiễu.
* **Cùng băng thông cho mọi nơi:** không thích nghi theo vùng (vùng phẳng vs vùng nhiều chi tiết) — IPE và nhất là **hash grid học được** của [Instant-NGP](05-instant-ngp.md) khắc phục bằng cách *học* đặc trưng thay vì cố định sin/cos.
* **Tốn chiều đầu vào:** encoding làm phình input của MLP.

---

## **6. Liên hệ với Latent-Anything**

* Positional encoding là một **lớp tiền xử lý chuẩn** cho mọi coordinate-network adapter: muốn decoder hình học (NIR/NeRF) sắc nét thì gần như bắt buộc.
* Núm băng thông là một tham số introspectable (Layer A): nó đặt trần tần số mà latent→trường có thể biểu diễn — hữu ích khi chẩn đoán vì sao một scene latent ra mờ.

---

## Liên quan

- [Neural Implicit Representation](01-neural-implicit-representation.md) — nơi spectral bias được minh họa thực nghiệm; SIREN là hướng thay thế.
- [NeRF](02-nerf.md) — positional encoding là thành phần làm NeRF sắc nét.
- [Volume rendering & ray marching](03-volume-rendering-ray-marching.md) — encoding quyết định trường có sắc để render hay không.
- [Instant-NGP](05-instant-ngp.md) — thay encoding cố định bằng đặc trưng học được trên hash grid.

## Tham khảo

- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains* (NeurIPS 2020, arXiv:2006.10739).
- Mildenhall et al., *NeRF* (ECCV 2020, arXiv:2003.08934) — positional encoding nguyên bản.
- Rahaman et al., *On the Spectral Bias of Neural Networks* (ICML 2019).
- Barron et al., *Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields* (ICCV 2021, arXiv:2103.13415) — integrated positional encoding.
