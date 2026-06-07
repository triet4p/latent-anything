# Manifold Hypothesis & Tangent Space

## **Giả thuyết Đa tạp (Manifold Hypothesis)** 

Giả thuyết đa tạp phát biểu rằng mặc dù dữ liệu thực tế được biểu diễn trong một không gian có số chiều cực kỳ lớn (không gian ngoại tại - ambient space), chúng thực chất lại phân bố tập trung dày đặc trên hoặc xung quanh các đa tạp con (submanifolds) có số chiều nội tại (intrinsic dimension) thấp hơn rất nhiều.

**Tại sao dữ liệu thực lại nằm trên submanifold có chiều thấp?**
Nguyên nhân cốt lõi đến từ các quy luật tự nhiên, vật lý hoặc ngữ nghĩa chi phối quá trình tạo ra dữ liệu. Dữ liệu thực tế không bao giờ là tập hợp của những giá trị hoàn toàn ngẫu nhiên và độc lập.

*   Hãy lấy ví dụ về hình ảnh một khuôn mặt người: Dù bức ảnh có độ phân giải hàng triệu pixel (hàng triệu chiều), cấu trúc của khuôn mặt lại bị ràng buộc khắt khe bởi các yếu tố biến thiên liên tục (degrees of freedom) như góc chiếu sáng, tư thế quay đầu (pose), hoặc các biểu cảm. 
*   Trong không gian vật lý, quỹ đạo chuyển động của một cánh tay robot tuy được ghi nhận ở không gian nhiều chiều nhưng thực chất bị giới hạn bởi số lượng khớp nối cơ học. 

Chính những ràng buộc vật lý, cơ học hay quy luật quang học này đã "khóa" dữ liệu vào các mức độ tự do thấp, cản trở chúng phân bố đều khắp không gian ngoại tại khổng lồ, và ép chúng nằm trên một bề mặt hình học phi tuyến tính (manifold). 

Việc giả định dữ liệu có cấu trúc đa tạp chiều thấp cho phép các mô hình máy học hiện đại (như Autoencoder, GAN, hay t-SNE) dễ dàng trích xuất đặc trưng, tối ưu hóa tính toán và tổng quát hóa tốt hơn thay vì bị ảnh hưởng bởi "lời nguyền đa chiều".

## **Cấu trúc Cục bộ (Local structure) và Không gian Tiếp tuyến (Tangent space)**

Theo định nghĩa hình học vi phân, một không gian chỉ được coi là đa tạp (manifold) nếu **ở phạm vi cục bộ (local)**, nó trông giống hệt như một không gian phẳng Euclidean. 
*   **Không gian tiếp tuyến (Tangent space - $T_pM$):** Để hiểu và phân tích cấu trúc cục bộ này, người ta dùng không gian tiếp tuyến. Tại bất kỳ điểm $p$ nào trên đa tạp, không gian tiếp tuyến là một không gian vectơ tuyến tính đóng vai trò xấp xỉ tốt nhất phần đa tạp ngay tại điểm đó. Định nghĩa một cách chặt chẽ, nó là tập hợp tất cả các vận tốc (đạo hàm) của mọi đường cong trơn nằm hoàn toàn trên đa tạp và đi ngang qua điểm $p$. 
*   Ý nghĩa của cấu trúc cục bộ là cho phép chúng ta thực hiện các phép đo hình học (như khoảng cách ngắn, góc lệch) thông qua một mêtric Riemannian đặt trực tiếp trên không gian tiếp tuyến phẳng này, bất chấp việc toàn bộ đa tạp đang uốn cong trong không gian lớn.

**Cấu trúc Toàn cục (Global structure)**
Trong khi cấu trúc cục bộ giúp ta xấp xỉ tuyến tính tại một điểm, **Cấu trúc toàn cục (Global structure)** mô tả hình dáng và sự liên kết tổng thể của toàn bộ dữ liệu.

*   Cấu trúc này được hình thành bằng cách mô tả cách mà các vùng không gian tiếp tuyến cục bộ (được gọi là các hiến chương - charts) liên kết, gộp nối (stitch) lại với nhau thông qua một "tập bản đồ" (atlas) có các miền chồng lấn để tạo nên hình dạng phi tuyến hoàn chỉnh của đa tạp. 
*   Ở góc nhìn toàn cục, đa tạp sẽ thể hiện các thuộc tính topo đặc trưng mà phân tích cục bộ không thể thấy được, chẳng hạn như tính đóng, sự tồn tại của ranh giới (boundaries), hay tính định hướng (orientability - xác định xem đa tạp có bị xoắn như dải Möbius hay chai Klein hay không). 

Tóm lại, thông qua **Không gian tiếp tuyến** để xấp xỉ **cấu trúc cục bộ** và liên kết chúng lại để hiểu **cấu trúc toàn cục**, các mô hình máy học có khả năng khai phá và mô hình hóa thành công giả thuyết đa tạp chiều thấp tiềm ẩn bên trong dữ liệu phức tạp.

---

## Liên quan

- [Lời nguyền chiều](04-curse-of-dimensionality.md) — lý do dữ liệu buộc phải nằm trên đa tạp chiều thấp.
- [Geodesic](05-geodesic.md) — đường ngắn nhất *trên* đa tạp khác đường thẳng ambient.
- [Pullback Metric](06-pullback-metric.md) — đo hình học đa tạp ẩn qua Jacobian của decoder.
- [Hình học Riemannian](../../03-geometry-structure/research/04-riemannian-geometry.md) — độ cong và dịch chuyển song song trên đa tạp.
- [Autoencoder](../../02-representation-learning/research/02-autoencoder.md) — model khai thác trực tiếp giả thuyết đa tạp.

## Tham khảo

- Fefferman, Mitter, Narayanan, *Testing the Manifold Hypothesis* (Journal of the AMS, 2016).
- Bengio, Courville, Vincent, *Representation Learning: A Review and New Perspectives* (IEEE TPAMI, 2013).
- do Carmo, M., *Riemannian Geometry* (Birkhäuser, 1992) — tangent space, manifold, atlas/charts.