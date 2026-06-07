# Instant-NGP — Multiresolution Hash Encoding

> **TL;DR.** Instant-NGP tăng tốc NeRF từ *vài ngày* xuống *vài giây* bằng cách **chuyển dung lượng từ MLP lớn sang một bảng đặc trưng học được**: một **multiresolution hash encoding** tra cứu $O(1)$ vào $L$ lưới đặc trưng đa độ phân giải (mỗi lưới là một hash table $T$ phần tử, vector $F$ chiều), nội suy trilinear, nối lại rồi đưa qua một **MLP tí hon**. Đánh đổi: va chạm hash + tốn bộ nhớ bảng.

Đây là câu trả lời cho nút thắt của [NeRF](02-nerf.md): MLP lớn + [positional encoding](04-positional-encoding.md) cố định, bị query hàng triệu lần ([volume rendering](03-volume-rendering-ray-marching.md)) → cực chậm. Instant-NGP giữ nguyên pipeline trường + render, chỉ thay **cách mã hóa tọa độ**.

---

## **1. Ý tưởng cốt lõi: feature grid học được thay cho sin/cos cố định**

Positional encoding cấp cho MLP các hàm cơ sở *cố định* (sin/cos), rồi MLP phải gánh toàn bộ việc dựng scene → MLP phải lớn và chậm. Instant-NGP lật ngược: **lưu các vector đặc trưng *học được* tại các đỉnh lưới không gian**, để bản thân encoding mang phần lớn dung lượng. MLP chỉ còn việc giải mã đặc trưng cục bộ → có thể *tí hon* (vài lớp, vài chục neuron) và rất nhanh.

---

## **2. Multiresolution hash encoding**

Ba siêu tham số: **$L$** (số mức độ phân giải), **$T$** (kích thước mỗi hash table), **$F$** (số chiều vector đặc trưng).

1. **Đa độ phân giải:** dựng $L$ lưới (mặc định $L=16$) với độ phân giải tăng theo **cấp số nhân** từ $N_{\min}$ tới $N_{\max}$. Mức thô bắt cấu trúc lớn, mức mịn bắt chi tiết.
2. **Hash table mỗi mức:** mỗi mức có một bảng tối đa $T$ vector đặc trưng $F$ chiều (thường $F=2$, $T\in[2^{14}, 2^{24}]$), toàn bộ **trainable**.
3. **Tra cứu (lookup):** với điểm $\mathbf{x}$, ở mỗi mức tìm ô lưới chứa nó; mỗi đỉnh trong $2^d$ đỉnh được ánh xạ vào bảng qua **hàm hash không gian**:

$$
h(\mathbf{x}) = \Big(\bigoplus_{i=1}^{d} x_i\,\pi_i\Big) \bmod T
$$

   trong đó $\oplus$ là XOR theo bit, và $\pi_i$ là các số nguyên tố lớn (ví dụ $\pi=[1,\,2654435761,\,805459861]$). Lấy vector đặc trưng tại các đỉnh rồi **nội suy trilinear** theo vị trí của $\mathbf{x}$ trong ô.
4. **Ghép:** nối đặc trưng từ cả $L$ mức → vector $L\times F$ chiều, đưa vào MLP tí hon → $(\sigma, c)$.

Ở mức thô, số đỉnh < $T$ nên **không va chạm** (ánh xạ 1–1, bỏ qua hash). Ở mức mịn, số đỉnh > $T$ nên *phải* hash và xảy ra va chạm.

---

## **3. Va chạm hash được xử lý thế nào?**

Instant-NGP **không giải va chạm tường minh**. Nhiều ô không gian (ở mức mịn) trỏ vào cùng một phần tử bảng; gradient từ tất cả các ô đó **cộng dồn** lên cùng vector đặc trưng. Mấu chốt: các điểm bề mặt thật (mật độ cao) đóng góp gradient mạnh và nhất quán, lấn át các ô rỗng va chạm cùng ô nhớ; đồng thời **các mức thô (không va chạm)** cấp ngữ cảnh để MLP **phân biệt (disambiguate)** các va chạm còn lại. Kết quả: artifact nhỏ, đổi lại tốc độ cực lớn.

---

## **4. Vì sao nhanh đến vậy**

* **Lookup $O(1)$** thay cho một MLP sâu mã hóa toàn bộ không gian.
* **MLP tí hon** (fully-fused, chạy gọn trong cache GPU) → mỗi query rẻ hơn nhiều bậc.
* **Occupancy grid:** một lưới nhị phân thô đánh dấu ô rỗng, cho phép **bỏ qua phần lớn mẫu rỗng** khi ray marching (liên hệ [volume rendering](03-volume-rendering-ray-marching.md)).
* Tất cả song song hóa tốt → train một scene trong **giây–phút** thay vì giờ–ngày.

---

## **5. Giới hạn (where it breaks)**

* **Tốn bộ nhớ:** các hash table chiếm RAM/VRAM đáng kể (đánh đổi memory lấy tốc độ).
* **Va chạm gây artifact:** ở scene rất chi tiết, va chạm hash để lại nhiễu nhỏ; không phải biểu diễn "sạch" như một hàm trơn.
* **Không còn là hàm giải tích gọn:** encoding giờ là một bảng tra cứu lớn — kém "đẹp" về lý thuyết, khó phân tích phổ như positional encoding.
* Vẫn là **per-scene** như NeRF (không tự tổng quát hóa sang scene mới).

---

## **6. Liên hệ với Latent-Anything**

* Instant-NGP cho thấy một **trục thiết kế latent quan trọng**: đặt dung lượng ở đâu — trong trọng số MLP (NeRF) hay trong một **bảng đặc trưng không gian học được** (NGP). Bảng đặc trưng này *chính là* một dạng latent có cấu trúc không gian, rất hợp để introspect/manipulate cục bộ theo vùng.
* Tốc độ của nó là điều kiện thực tế để dùng neural field như decoder trong vòng lặp huấn luyện/can thiệp của Layer C.

---

## Liên quan

- [NeRF](02-nerf.md) — Instant-NGP tăng tốc, giữ nguyên pipeline trường + render.
- [Positional encoding](04-positional-encoding.md) — hash grid học được thay cho encoding sin/cos cố định.
- [Volume rendering & ray marching](03-volume-rendering-ray-marching.md) — occupancy grid bỏ qua mẫu rỗng khi march.
- [Neural Implicit Representation](01-neural-implicit-representation.md) — cùng tinh thần "mạng/đặc trưng là tín hiệu".

## Tham khảo

- Müller, Evans, Schied, Keller, *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding* (ACM TOG / SIGGRAPH 2022, arXiv:2201.05989).
- Mildenhall et al., *NeRF* (ECCV 2020, arXiv:2003.08934) — baseline mà Instant-NGP tăng tốc.
