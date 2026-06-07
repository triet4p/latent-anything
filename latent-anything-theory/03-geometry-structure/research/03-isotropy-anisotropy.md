# Isotropy & Anisotropy

**Isotropy (Tính đẳng hướng)** và **Anisotropy (Tính bất đẳng hướng)** là hai khái niệm hình học mô tả cách các vector biểu diễn (embeddings) phân bố trong không gian ẩn (latent space). 

Trong một không gian ẩn lý tưởng và phân phối đều (Isotropic), các vector biểu diễn sẽ dàn trải đồng đều theo mọi hướng. Cấu trúc hình cầu này giúp tối đa hóa entropy, tận dụng toàn bộ dung lượng thông tin của không gian và bảo toàn tính đồng nhất cho các phép đo khoảng cách.

Tuy nhiên, câu trả lời cho câu hỏi "latent có phân phối đều theo mọi hướng không?" là **Không**. Hầu hết các mô hình học máy hiện đại, đặc biệt là các mô hình ngôn ngữ lớn dựa trên kiến trúc Transformer (LLMs), đều mắc phải hiện tượng **Bất đẳng hướng (Anisotropy)**, hay còn gọi là vấn đề **Thoái hóa biểu diễn (Representation Degeneration Problem)**. Thay vì tỏa đều, các vector bị co cụm lại, dồn ép vào một không gian con (subspace) có số chiều thấp hơn nhiều so với thiết kế ban đầu và tạo thành một **"hình nón hẹp" (narrow cone)**.

## Tại sao hầu hết các model lại bị Anisotropic?

Hiện tượng này không phải là lỗi ngẫu nhiên mà là hệ quả tất yếu từ kiến trúc mạng và động lực học tối ưu (learning dynamics) trong quá trình huấn luyện:

### **1. Sự co rút của cơ chế Tự chú ý (Self-Attention Contraction):** 
Bản thân cơ chế cốt lõi của Transformer là nguyên nhân sinh ra bất đẳng hướng. Ma trận tự chú ý trong mỗi lớp hoạt động giống như một toán tử co rút (contraction operator) đối với phương sai góc. Khi dữ liệu đi qua hàng chục lớp xếp chồng lên nhau, quá trình này liên tục ép các vector lại gần nhau về phía một trục trung tâm, tước đi sự phân bố đồng đều của không gian.

### **2. Tác động của Tần suất từ vựng (Frequency-Induced Variance Collapse):**
Các token xuất hiện thường xuyên (ví dụ: các từ nối như "the", "a", "is") có xu hướng bị ép chặt phương sai và giảm độ lớn (norm). Chúng bị "ghim" chặt vào một tọa độ trung tâm, dao động trong một thể tích rất nhỏ của không gian. Ngược lại, những token hiếm gặp thì phân tán hơn. Sự sụp đổ phương sai do tần suất này làm không gian bị biến dạng mạnh mẽ. 

### **3. Thiên kiến Cập nhật Gradient (Gradient Bias Toward Tangent Directions):**
Trong giai đoạn đầu huấn luyện, các thuật toán như Gradient Descent ưu tiên khuếch đại các hướng tiếp tuyến (tangent) có nhiều dữ liệu và triệt tiêu các hướng pháp tuyến (normal). Trọng số của mạng sẽ tự động căn chỉnh theo một số ít hướng chính này. Điều này tạo ra một vòng lặp tự củng cố: hướng nào đã mạnh sẽ càng nhận được nhiều gradient và càng mạnh thêm, trong khi các hướng khác bị "bỏ đói" và thui chột, khiến không gian bị bẹp lại thành dạng hình nón.

### **4. Suy hao thông tin và mục tiêu Cross-Entropy:**
Các từ hiếm trong hàm mất mát Cross-Entropy rất ít khi nhận được gradient tích cực; chúng liên tục bị các từ phổ biến đẩy đi, khiến toàn bộ các vector bị lùa về chung một hướng. Hơn nữa, việc truyền tín hiệu qua các lớp mạng sâu mà không có cơ chế bảo toàn thông tin phù hợp sẽ dẫn đến sự suy hao liên tục, làm sụp đổ các chiều biểu diễn.

## Hệ quả của không gian Anisotropic

Sự sụp đổ hình học này mang lại những hệ quả sâu sắc, bao gồm cả các mặt tiêu cực lẫn một số góc nhìn tích cực.

**Hệ quả Tiêu cực:**

*   **Điểm tương đồng Cosine ảo (High Cosine Similarity):** Do tất cả các vector đều nằm chật chội trong một hình nón hẹp, góc giữa bất kỳ hai vector nào cũng rất nhỏ. Điều này khiến độ tương đồng Cosine giữa các từ hoàn toàn không liên quan cũng cao chót vót, làm hỏng khả năng phân biệt ngữ nghĩa tinh tế của mô hình.
*   **Bóp méo khoảng cách Euclidean:** Hình học bị bóp méo khiến các phép tìm kiếm lân cận gần nhất (nearest-neighbor) hoặc phân cụm bị sai lệch nghiêm trọng. 
*   **Hiện tượng "Stolen Probability":** Vì các vector hợp với nhau những góc nhọn và có norm nhỏ, mô hình gặp khó khăn trong việc gán xác suất cao tuyệt đối cho một token cụ thể trong một số ngữ cảnh nhất định, làm giảm độ tự tin và tính đa dạng khi sinh văn bản.

**Góc nhìn Tích cực (Tại sao model vẫn hoạt động tốt?):**

Mặc dù bị coi là "thoái hóa", các nghiên cứu gần đây chỉ ra rằng Anisotropy không hẳn là một căn bệnh cần loại bỏ hoàn toàn.

*   **Giảm chiều ẩn (Implicit Dimension Reduction):** Không gian bất đẳng hướng có thể là cách mô hình tự động giảm số chiều hiệu dụng (intrinsic dimensionality) để lọc bỏ nhiễu, giúp các mạng nơ-ron khổng lồ có khả năng khái quát hóa (generalize) tốt hơn trên lượng dữ liệu có hạn. 
*   **Mã hóa Cú pháp:** Cấu trúc hình nón này thường là nơi chứa các quy luật cú pháp ngôn ngữ (syntax). Các mô hình học cách nhóm các đặc trưng cấu trúc chung vào một số ít trục tọa độ lớn, giúp việc phân loại tuyến tính dễ dàng hơn. Do đó, việc cố gắng ép mô hình trở nên hoàn toàn Isotropic đôi khi lại làm hỏng hiệu suất ở các tác vụ cần phân tách cụm từ vựng.

**Tóm lại:** Không gian ẩn không hề đẳng hướng (Isotropic) mà bị kéo giãn và dồn ép khốc liệt thành dạng bất đẳng hướng (Anisotropic). Đây là kết quả tất yếu của cơ chế Attention, sự chênh lệch tần suất dữ liệu và thiên kiến của Gradient. Mặc dù nó gây khó khăn cho việc đo lường bằng hình học phẳng (Euclidean), nhưng chính sự "co cụm" này lại là cách mô hình nén các quy luật ngôn ngữ để thực hiện suy luận. Các kỹ thuật khắc phục (như phạt tương phản hay kéo giãn bằng hình học Simplicial) hiện đang được nghiên cứu để mở rộng không gian này nhằm tối ưu hóa sự đa dạng mà không làm mất đi các quy luật cấu trúc đã học.

---

## Liên quan

- [Cấu trúc tuyến tính](01-linear-structure.md) — anisotropy là hệ quả tiêu cực của cách model học hướng tuyến tính.
- [Metric Space & Vector Space](../../01-space-representation/research/01-metric-space-vector-space.md) — cosine similarity bị thổi phồng khi anisotropic.
- [Slerp](05-slerp.md) — thao tác trên mặt siêu cầu khi latent có cấu trúc cầu.
- [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md) — chuẩn hóa L2 ép code về đẳng hướng trên mặt cầu.

## Tham khảo

- Gao et al., *Representation Degeneration Problem in Training Natural Language Generation Models* (ICLR, 2019).
- Mu, Viswanath, *All-but-the-Top: Simple and Effective Postprocessing for Word Representations* (ICLR, 2018).
- Ethayarajh, *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings* (EMNLP, 2019).