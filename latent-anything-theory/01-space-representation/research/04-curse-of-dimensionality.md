# Curse of Dimensionality

Lời nguyền đa chiều là thuật ngữ chỉ hiện tượng xảy ra khi phân tích và mô hình hóa dữ liệu trong không gian có số chiều lớn.

## Hiện tượng thể tích tập trung ở rìa (Volume Concentration)

Khi số chiều của không gian tăng lên, thể tích của nó tăng theo cấp số nhân, khiến dữ liệu bên trong trở nên cực kỳ thưa thớt. **Gần như toàn bộ thể thích của không gian chiều cao lại bị đẩy ra sát rìa và góc**.

## Sự tập trung khoảng cách (Distance Concentration).

- **Khoảng cách đồng đều**: Trong không gian chiều cao, khoảng cách giữa các cặp điểm ngẫu nhiên có xu hướng hội tụ về cùng một giá trị, dẫn tới **mất khả năng phân biệt**.

- **Chi phí duyệt bùng nổ**: Để tìm được một lượng láng giềng nhỏ (ví dụ $k=10$) hộp tìm kiểm trong không gian 1000 chiều sẽ phải mở rộng tới mức bao trùm 99,54% chiều dài của mỗi chiều, làm cho thuật toán thoái hóa thành duyệt toàn bộ.

- **Hiện tượng Hubness**: Sự phân bố khoảng cách bị méo mó sinh ra các điểm "Hub" ngẫu nhiên, chúng xuất hiện trong danh sách láng giềng của hầu hết các điểm khác, trong khi đại đa số các điểm còn lại bị cô lập, làm sai lệch kết quả phân cụm và phân loại.

## Nén dữ liệu (Compression/Dimensionality Reduction)

Để hệ thống học máy có thể hoạt động hiệu quả ta bắt buộc phải nén dữ liệu (AE, PCA, UMAP, t-SNE,...) để:

1. Loại bỏ chiều nhiễu để khôi phục khoảng cách
2. Chống overfitting do dữ liệu thưa
3. Khai thác cấu trúc đa tạp (intrinsic dimension), tôi ưu chi phí tính toán, không gian lưu trữ.

---

## Liên quan

- [Metric Space & Vector Space](01-metric-space-vector-space.md) — định nghĩa khoảng cách bị ảnh hưởng ở chiều cao.
- [Giả thuyết Đa tạp](03-manifold-hypothesis.md) — lối thoát: dữ liệu thật sống ở chiều nội tại thấp.
- [Khoảng cách Mahalanobis](02-mahalanobis-distance.md) — chuẩn hóa khoảng cách theo phân phối.
- [Slerp](../../03-geometry-structure/research/05-slerp.md) — hiện tượng vỏ siêu cầu (concentration of measure) ở chiều cao.

## Tham khảo

- Bellman, R., *Adaptive Control Processes: A Guided Tour* (Princeton, 1961) — nguồn gốc thuật ngữ.
- Beyer et al., *When Is "Nearest Neighbor" Meaningful?* (ICDT, 1999).
- Aggarwal, Hinneburg, Keim, *On the Surprising Behavior of Distance Metrics in High Dimensional Space* (ICDT, 2001).
- Radovanović, Nanopoulos, Ivanović, *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data* (JMLR, 2010) — hiện tượng hubness.