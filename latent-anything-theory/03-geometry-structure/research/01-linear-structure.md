# Linear Structure in Latent Space

## Giả thuyết hướng ẩn (Latent Direction Hypothesis)

Giả thuyết này khẳng định các mô hình học sâu tham số hóa cao (overparameterized) không mã hóa các hành vi hoặc khái niệm trừu tượng một cách phân tán phức tạp mà có xu hướng mã hóa các ***nhân tố biến thiên cốt lõi (factors of variation)*** dưới dạng các **vector định hướng có thể truy cập tuyến tính** trong không gian ẩn.

Ví dụ:

Khi bạn bắt AI học hàng triệu bức ảnh khuôn mặt, nếu nó chỉ "học vẹt" (nhớ từng điểm ảnh), nó sẽ bị quá tải. Để hoàn thành nhiệm vụ nén dữ liệu, AI buộc phải tìm ra "công thức" hoặc "quy luật" chung nhất tạo nên các khuôn mặt.

Nó nhận ra rằng: "À, hàng vạn bức ảnh này thực ra giống hệt nhau, chỉ khác mỗi cái là người trong ảnh đang cười hay không". Thế là AI gom tất cả sự thay đổi về nụ cười đó lại, tạo thành một "thanh trượt" (một hướng vector thẳng). Khi bạn kéo thanh trượt này (di chuyển dọc theo hướng vector), khuôn mặt trong ảnh sẽ từ mếu chuyển dần sang cười mà màu tóc hay độ tuổi không bị ảnh hưởng. Việc biến thế giới phức tạp thành các "thanh trượt" thẳng thớm này gọi là quá trình "trải phẳng đa tạp dữ liệu".

## Hệ quả từ cách mô hình học

Cả tích cực và tiêu cực

### **1. Khả năng thực hiện số học vector (Latent Arithmetic) và tính nhân quả:**
Bởi vì mô hình đã chuyển hóa các quan hệ logic-ngữ nghĩa phức tạp thành các dịch chuyển hình học song song, chúng ta có thể thực hiện các phép toán số học vector vô cùng hiệu quả (như phép toán nổi tiếng $z_{\text{king}} - z_{\text{man}} + z_{\text{woman}} \approx z_{\text{queen}}$). Khi có một sự can thiệp (intervention) vào dữ liệu, sự thay đổi đó có thể được cô lập và biểu diễn chính xác dưới dạng một vector hiệu số (Causal Delta Embedding) mà không bị ảnh hưởng bởi các thuộc tính ngữ nghĩa không liên quan khác.

### **2. Tránh hiện tượng học hàm đồng nhất nhờ cơ chế điều hòa (Regularization):**
Nếu mô hình quá lớn mà không có cơ chế điều hòa, nó sẽ có nguy cơ học một hàm đồng nhất (identity function), tức là chỉ sao chép đầu vào mà không trích xuất được đặc trưng hữu ích. Việc thêm các hình phạt như suy giảm trọng số (weight decay), ràng buộc thưa thớt (sparsity) hoặc nhiễu Gaussian là yếu tố then chốt ép không gian ẩn trở nên chặt chẽ và hình thành các hướng tuyến tính bền vững trước biến động. 

### **3. Sự sụp đổ hình học và Hiện tượng Bất đẳng hướng (Anisotropy):**
Một hệ quả tiêu cực từ cách học của các mô hình ngôn ngữ (như Transformer) là **hiện tượng thoái hóa biểu diễn** (representation degeneration). Do tác động của ma trận tự chú ý (self-attention) hoạt động như một toán tử co rút, cộng với việc liên kết trọng số (weight tying) đẩy các từ xuất hiện thường xuyên ra xa, các biểu diễn tuyến tính không phân bổ đều (đẳng hướng) mà **bị dồn ép, co cụm lại thành một hình nón cực kỳ hẹp**. Sự phân bố bất đẳng hướng này làm giảm số chiều hiệu dụng, tạo ra độ tương đồng cosine cực cao giữa các vector không liên quan và bóp méo hình học ngữ nghĩa của không gian ẩn.

### **4. Sự thất bại của phép nội suy tuyến tính (Lerp) khi đa tạp bị cong:**
Mặc dù mô hình cố gắng "trải phẳng" không gian, cấu trúc thực tế do bộ giải mã (decoder) định hình lại thường là một **đa tạp khả vi cong Riemannian**. Do đó, việc di chuyển theo một đường thẳng tuyến tính Euclidean (lerp) giữa hai điểm ở xa nhau sẽ vạch ra một quỹ đạo cắt ngang qua lòng đa tạp cong, đi xuyên qua các vùng "ngoại đa tạp" (off-manifold). 
*   Đây là những vùng lõi có mật độ xác suất dữ liệu thực tế cực thấp. 
*   **Hệ quả:** Khi bộ giải mã tiếp nhận các vector nằm trên đoạn thẳng tuyến tính này (nơi có chuẩn vector bị suy giảm nghiêm trọng), nó sẽ sinh ra các mẫu hình ảnh trung gian bị mờ, mất chi tiết, méo mó hoặc không thực tế. Điều này đòi hỏi phải sử dụng các phép nội suy trên hình cầu (Slerp) hoặc tính toán đường trắc địa (Geodesic) uốn cong theo mật độ dữ liệu để khắc phục.

---

## Liên quan

- [Tính tách biệt biểu diễn](02-disentanglement.md) — lý tưởng mỗi hướng tuyến tính = một factor độc lập.
- [Đẳng hướng & Bất đẳng hướng](03-isotropy-anisotropy.md) — mặt tiêu cực: hướng co cụm thành hình nón.
- [Hình học Riemannian](04-riemannian-geometry.md) — khi nào đa tạp cong đủ để lerp fail.
- [Slerp](05-slerp.md) — nội suy trên mặt cầu khắc phục norm degradation.
- [Geodesic](../../01-space-representation/research/05-geodesic.md) — nội suy theo đường trắc địa.

## Tham khảo

- Mikolov, Yih, Zweig, *Linguistic Regularities in Continuous Space Word Representations* (NAACL, 2013) — latent arithmetic.
- Radford, Metz, Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs* (ICLR, 2016) — vector arithmetic trong latent GAN.
- Bengio, Courville, Vincent, *Representation Learning: A Review and New Perspectives* (IEEE TPAMI, 2013).
- Gao et al., *Representation Degeneration Problem in Training Natural Language Generation Models* (ICLR, 2019).