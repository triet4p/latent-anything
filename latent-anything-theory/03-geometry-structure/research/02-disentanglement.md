# Disentanglement (Tính tác biệt biểu diễn)

Trong học máy, **Disentanglement (tính tách biệt biểu diễn)** là mục tiêu tìm ra một không gian ẩn (latent space) mà ở đó, các nguồn biến thiên cốt lõi (factors of variation) của thế giới thực được tách rời hoàn toàn và biểu diễn bằng các tọa độ độc lập. 

Theo định nghĩa chuẩn mực, một biểu diễn được coi là "disentangled" (đã gỡ rối) khi **một sự thay đổi ở một biến ẩn chỉ tương ứng với sự thay đổi của duy nhất một nhân tố sinh dữ liệu**, đồng thời hoàn toàn bất biến trước sự thay đổi của các nhân tố khác. Dưới góc độ toán học của lý thuyết nhóm (Group Theory), điều này đạt được khi không gian ẩn phân rã thành các không gian con độc lập, và một phép biến đổi (ví dụ: đổi màu) chỉ tác động lên đúng không gian con chịu trách nhiệm cho màu sắc mà không ảnh hưởng đến các không gian con khác (như hình dáng hay vị trí).

Lý tưởng là vậy, nhưng để đo lường xem một AI thực sự đạt được mức độ lý tưởng "một chiều kiểm soát một factor" đến đâu, các nhà nghiên cứu đã phát triển ba họ thước đo (metrics) định lượng chính:

### 1. Thước đo dựa trên Thông tin (Mutual Information)

Hệ thống thước đo này định lượng trực tiếp **lượng thông tin tương hỗ (Mutual Information - MI)** chia sẻ giữa các biến ẩn và các nhân tố sinh dữ liệu thực tế.

*   **Mutual Information Gap (MIG):** Đây là thước đo nổi bật nhất trong nhóm này. Đối với một nhân tố dữ liệu cụ thể, MIG tính toán sự chênh lệch (gap) giữa lượng thông tin tương hỗ của biến ẩn chứa nhiều thông tin nhất (top 1) và biến ẩn chứa nhiều thông tin thứ hai (top 2), sau đó chuẩn hóa bằng entropy của chính nhân tố đó. 

*   **Ý nghĩa:** Một giá trị MIG cao (gần 1) chứng tỏ **thông tin về một nhân tố được tập trung cao độ và độc quyền vào một chiều duy nhất**, thay vì bị rò rỉ (spillover) và phân tán ra nhiều chiều khác nhau.

*   **Điểm mù và cách khắc phục:** MIG mặc định giả định rằng các nhân tố trong thế giới thực là hoàn toàn độc lập với nhau. Tuy nhiên, nếu dữ liệu có các nhân tố tương quan (ví dụ: màu sắc và hình dạng thường đi kèm nhau), MIG sẽ đánh giá sai và cho điểm thấp. Để giải quyết, biến thể **DMIG (Dependency-aware MIG)** được sử dụng bằng cách thay thế mẫu số bằng entropy có điều kiện, giúp đo lường chính xác ngay cả khi các thuộc tính bị phụ thuộc lẫn nhau.

### 2. Thước đo dựa trên Bộ dự đoán (DCI Score)

Thay vì dùng lý thuyết thông tin, DCI sử dụng các mô hình học máy phụ trợ (như Lasso hoặc Random Forest) để dự đoán các nhân tố thực tế từ các biến ẩn, qua đó xây dựng một **ma trận tầm quan trọng (importance matrix)**. DCI chia nhỏ việc đánh giá thành 3 tiêu chí riêng biệt:

*   **Disentanglement (Tính tách biệt - D):** Đo lường mức độ một biến ẩn chỉ mang tính quyết định đối với *một* nhân tố duy nhất. Điểm D cao nghĩa là một chiều không bị "gánh" nhiều thuộc tính cùng lúc (ngăn chặn tình trạng một thanh trượt vừa làm đổi màu vừa làm xoay hình). Nó được tính toán dựa trên entropy của các hàng trong ma trận.

*   **Completeness (Tính trọn vẹn/Nhỏ gọn - C):** Đo lường mức độ một nhân tố sinh dữ liệu chỉ được dự đoán bởi *một* biến ẩn duy nhất. Điểm C cao đảm bảo một thuộc tính không bị phân mảnh rải rác ra nhiều chiều khác nhau (tính toán qua entropy của các cột).

*   **Informativeness (Tính thông tin - I):** Đánh giá độ chính xác tổng thể của mô hình khi dự đoán các nhân tố từ không gian ẩn.
*   **Hệ quả & Hạn chế:** Điểm DCI rất trực quan nhưng **bị phụ thuộc mạnh mẽ vào năng lực của thuật toán dự đoán (predictor) được chọn**. Nếu dùng một bộ phân loại quá thông minh (phi tuyến tính), nó có thể dự đoán đúng nhân tố ngay cả khi không gian ẩn chưa thực sự được "gỡ rối" cấu trúc hình học, dẫn đến việc cho điểm DCI cao một cách mù quáng.

### 3. Thước đo dựa trên Can thiệp (Intervention Effect)
Nhóm này mang đậm tính nhân quả (causality), đánh giá bằng cách **can thiệp trực tiếp vào quá trình sinh dữ liệu hoặc vào chính các biến ẩn** để quan sát sự thay đổi có tính ổn định hay không.

*   **Tiếp cận từ Dữ liệu ($\beta$-VAE & FactorVAE metrics):** Kỹ thuật này hoạt động bằng cách chủ động cố định một nhân tố sinh dữ liệu (ví dụ: giữ nguyên góc quay mặt), nhưng lấy mẫu ngẫu nhiên các nhân tố còn lại (màu tóc, ánh sáng) để tạo ra các cặp dữ liệu. Khi đưa qua mô hình mã hóa, nếu một chiều ẩn có **phương sai nhỏ nhất** (hoặc độ lệch tuyệt đối nhỏ nhất), nó sẽ được xác định là chiều đang biểu diễn cho nhân tố bị cố định đó.
*   **Tiếp cận Causal Delta & Cấu trúc nhân quả:** Trong một mô hình lý tưởng, khi ta thực hiện một hành động can thiệp (như "mở cửa" hay "đổi màu"), sự biến thiên này phải được biểu diễn bằng một vector hiệu số (Causal Delta) độc lập, không bị vướng bận bởi các chi tiết nền của bối cảnh (như ánh sáng hay vị trí camera).
*   **Unconfoundedness (UC) & Counterfactual Generativeness (CG):** Đây là các thước đo can thiệp tiên tiến nhất nhằm đánh giá mức độ tách biệt nhân quả.

    *   **UC (Tính không nhiễu):** Kiểm tra xem các nhân tố khác nhau có bị gán chồng chéo lên cùng một tập hợp biến ẩn hay không, nhằm loại bỏ các yếu tố gây nhiễu (confounders).
    *   **CG (Khả năng sinh dữ liệu giả định):** Can thiệp trực tiếp (toán tử *do*) vào các chiều ẩn cụ thể để sinh ra hình ảnh giả định (counterfactuals). Nó đo lường **tác động nhân quả trung bình (Average Causal Effect)** của riêng chiều ẩn đó lên ảnh được tạo ra, nhằm đảm bảo rằng khi kéo một "thanh trượt" trong latent space, nó tác động đúng lên nhân tố mục tiêu mà không phá hỏng bất kỳ thuộc tính nào khác.

---

## Liên quan

- [Cấu trúc tuyến tính](01-linear-structure.md) — mỗi hướng tuyến tính lý tưởng kiểm soát một factor.
- [VAE](../../02-representation-learning/research/03-vae.md) — β-VAE và áp lực KL thúc đẩy disentanglement.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) — nén thông tin về prior độc lập.

## Tham khảo

- Bengio, Courville, Vincent, *Representation Learning: A Review and New Perspectives* (IEEE TPAMI, 2013).
- Higgins et al., *β-VAE* (ICLR, 2017).
- Chen et al., *Isolating Sources of Disentanglement in VAEs* (β-TCVAE, MIG) (NeurIPS, 2018).
- Eastwood, Williams, *A Framework for the Quantitative Evaluation of Disentangled Representations* (DCI) (ICLR, 2018).
- Kim, Mnih, *Disentangling by Factorising* (FactorVAE) (ICML, 2018).
- Locatello et al., *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations* (ICML, 2019).