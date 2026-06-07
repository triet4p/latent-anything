# Vector-Quantized VAE (VQ-VAE)

> **TL;DR.** VQ-VAE thay latent liên tục bằng một **codebook rời rạc**: encoder xuất vector, mỗi vector bị "ép" về vector gần nhất trong bảng mã (nearest-neighbor + straight-through estimator để gradient chảy qua được). Lợi: hết blur, tránh posterior collapse, biến ảnh thành chuỗi token cho Transformer. Khó: **codebook collapse** (nhiều mã chết) — khắc phục bằng EMA, reset, hoặc chuẩn hóa L2.

VQ-VAE được giới thiệu bởi van den Oord và các cộng sự vào năm 2017, đại diện cho một bước chuyển mình mang tính bước ngoặt trong lĩnh vực học máy: dịch chuyển từ các không gian ẩn liên tục (như trong VAE truyền thống) sang **một không gian ẩn rời rạc (discrete latent space)**

## Tại sao cần VQ-VAE?

Nó sinh ra để giải quyết các hạn chế của VAE:

- **Khắc phục hiện tượng bị mờ (Blurry Reconstructions)**: Trong VAE, phân phối hậu nghiệm được giả định là Gaussian và để tính Recon Loss, nó phải tính trung bình tích phân của các cấu hình khả dĩ -> triệt tiêu hình học tần số cao.

- **Sự tương đồng với bản chất dữ liệu thực tế**: Trong tự nhiên, nhiều loại dữ liệu mang tính rời rạc cốt lõi thay vì liên tục. Ví dụ, ngôn ngữ được cấu thành từ các từ vựng/kí tự rời rạc, âm thanh cấu thành từ các âm vị, và suy luận logic cũng mang tính biểu tượng rời rạc

- **Ngăn chặn sụp đổ hậu nghiệm (Posterior Collapse)**: Trong VAE truyền thống, mô hình đôi khi học cách phớt lờ hoàn toàn không gian ẩn $Z$ khiến biến ẩn trở nên vô dụng (đặc biệt khi dùng chung với các bộ giải mã tự hồi quy mạnh).

- **Mở đường cho các Mô hình ngôn ngữ / Tự hồi quy trên ảnh**: Bằng cách biến một bức ảnh thành một chuỗi các chỉ số rời rạc (các "tô-ken" hình ảnh), VQ-VAE cho phép ta coi hình ảnh giống hệt như một chuỗi văn bản. Điều này cho phép các kiến trúc mạnh mẽ như Transformer hay PixelCNN học cách sinh ảnh một cách dễ dàng và mở ra kỷ nguyên của các mô hình sinh đa phương tiện quy mô lớn (như VQGAN, DALL-E)

## Các thành phần cốt lõi của VQ-VAE

### A. Codebook (Bảng mã) và Không gian ẩn rời rạc
Thay vì trích xuất ra một vector liên tục tùy ý, VQ-VAE duy trì một **Codebook (bảng mã)** ký hiệu là $C = \{c_1, c_2, \dots, c_K\}$. Codebook này chứa $K$ vector (hay "từ vựng" đại diện cho các đặc trưng hình ảnh), mỗi vector có số chiều là $D$. 
Khi bộ mã hóa xử lý đầu vào, thay vì đẩy đầu ra trực tiếp cho bộ giải mã, nó buộc phải chọn ra các vector có sẵn trong Codebook này để làm đại diện. 

### B. Lượng tử hóa Vector (Vector Quantization)
Quá trình chuyển đổi từ biểu diễn liên tục sang rời rạc được gọi là Vector Quantization. Cụ thể:

1. Bộ mã hóa nhận ảnh $X$ và xuất ra một mạng lưới các vector biểu diễn liên tục $z_e$.
2. Tại mỗi vị trí không gian, mô hình tính toán khoảng cách (thường là khoảng cách Euclidean) giữa $z_e$ và toàn bộ $K$ vector trong Codebook.
3. Vector lượng tử hóa $z_q$ được xác định bằng cách lấy vector $c_k$ có khoảng cách gần nhất với $z_e$ (thông qua phép toán `argmin`). 

### C. Bộ Ước lượng Straight-Through Estimator (STE)

Một trở ngại toán học chí mạng xuất hiện tại bước Lượng tử hóa: hàm `argmin` là một hàm bậc thang không liên tục và không thể lấy đạo hàm. Điều này làm đứt gãy chuỗi lan truyền ngược (backpropagation) của mạng nơ-ron, khiến gradient không thể chảy từ bộ giải mã về bộ mã hóa. 

Để giải quyết, các tác giả sử dụng một thủ thuật gọi là **Straight-Through Estimator (STE)**. Trong pha lan truyền ngược (backward pass), STE xem quá trình lượng tử hóa như một hàm đồng nhất (identity function) và **sao chép trực tiếp gradient từ đầu vào của bộ giải mã ($z_q$) truyền thẳng về đầu ra của bộ mã hóa ($z_e$)**. 
Về mặt lập trình, thủ thuật này được triển khai khéo léo bằng toán tử `stop_gradient` (ký hiệu là `sg`): 
$z_q = z_e + \text{sg}[z_q - z_e]$. 
Nhờ đó, giá trị trong pha truyền tiến (forward pass) vẫn là $z_q$, nhưng đạo hàm trong pha truyền ngược lại đi xuyên qua $z_e$.

### D. Thiết lập Hàm mất mát (Loss Function)
Vì việc sử dụng STE khiến việc huấn luyện Codebook bằng gradient tiêu chuẩn gặp khó khăn, VQ-VAE phải kết hợp 3 thành phần trong hàm mất mát để mọi thứ có thể hội tụ:

1. **Reconstruction Loss (Sai số tái cấu trúc):** Đo lường sự sai lệch giữa ảnh gốc và ảnh được khôi phục bởi bộ giải mã từ các vector rời rạc $z_q$. Nó tối ưu hóa trọng số của cả bộ mã hóa và giải mã.
2. **VQ Loss / Codebook Loss:** Vì STE gradient đi qua lớp lượng tử hóa mà không cập nhật các vector trong Codebook, chúng ta cần một số hạng để kéo các vector Codebook ($c_i$) xích lại gần đầu ra của bộ mã hóa ($z_e$). Số hạng này được định nghĩa là $\|\text{sg}[z_e] - z_q\|_2^2$. *(Lưu ý: Trong thực tế, việc cập nhật codebook hiện nay thường dùng thuật toán Trung bình trượt lũy thừa - EMA thay cho VQ Loss vì tính ổn định cao hơn).*
3. **Commitment Loss (Sai số cam kết):** Do không gian Codebook là hữu hạn, nếu đầu ra của bộ mã hóa dao động quá mạnh, nó có thể nhảy liên tục giữa các vector lượng tử khác nhau gây mất ổn định. Commitment loss phạt bộ mã hóa nếu nó đẩy biểu diễn $z_e$ đi quá xa so với vector Codebook $z_q$ đã được chọn, công thức là $\beta \|z_e - \text{sg}[z_q]\|_2^2$ với $\beta$ là hệ số kiểm soát (thường từ 0.25 đến 2).

## EMA - Sự thay thế cho STE

Mặc dù VQ Loss ở trên có thể dùng để cập nhật Codebook thông qua gradient descent (như Adam), các nhà nghiên cứu phát hiện ra rằng việc sử dụng **cơ chế EMA** lại mang lại hiệu quả vượt trội và ổn định hơn rất nhiều trong thực tế. Khi dùng EMA, người ta thường loại bỏ hoàn toàn VQ Loss ra khỏi hàm mục tiêu.

*   **Bản chất của EMA:** EMA trong VQ-VAE thực chất là một phiên bản trực tuyến (online) và liên tục của thuật toán phân cụm K-means.
*   **Cách hoạt động:** Thay vì cập nhật Codebook bằng gradient, EMA duy trì hai đại lượng thống kê trượt cho mỗi vector mã $c_j$:

    1.  **Số lượng gán ($N_j$):** Đếm xem có bao nhiêu đầu ra của bộ mã hóa được gán vào vector mã $c_j$ trong batch hiện tại.
    2.  **Tổng vector ($m_j$):** Tính tổng tất cả các vector đầu ra của bộ mã hóa được gán cho $c_j$.
    Hai đại lượng này được cập nhật liên tục qua từng batch bằng phép trung bình trượt với hệ số suy giảm $\gamma$ (thường bằng 0.99). Giá trị mới của vector Codebook đơn giản sẽ bằng phép chia: $c_j = m_j / N_j$.

**Tại sao EMA lại tốt hơn so với dùng VQ Loss + Optimizer (Adam)?**

*   **Hội tụ nhanh hơn:** Thay vì từ từ kéo vector Codebook về phía dữ liệu thông qua gradient (vốn phụ thuộc vào learning rate), EMA tính trung bình và "nhảy" trực tiếp vector Codebook về tâm của cụm dữ liệu phân bổ.
*   **Ổn định cao hơn:** EMA không bị ảnh hưởng bởi tốc độ học (learning rate) của mạng chính hay các cú sốc gradient đột ngột, do đó nó hoạt động cực kỳ mượt mà.
*   **Tiết kiệm bộ nhớ:** Việc dùng EMA không yêu cầu lưu trữ các trạng thái tối ưu hóa phức tạp (như moment bậc 1 và bậc 2 của Adam) cho riêng bảng mã, giúp giảm độ phức tạp tính toán. *(Lưu ý thêm: Về mặt toán học, sử dụng SGD trên VQ Loss với learning rate bằng 1 chính là tương đương với việc cập nhật bằng EMA).*

## Codebook Collapse và giải pháp

**Sụp đổ bảng mã (Codebook Collapse)** hay "Index Collapse" là một hiện tượng cực kỳ phổ biến và là một trong những trở ngại lớn nhất khi huấn luyện các mạng thần kinh lượng tử hóa như VQ-VAE. 

Hiện tượng này xảy ra khi chỉ có một phần rất nhỏ các vector trong Codebook thực sự được sử dụng để lượng tử hóa đầu ra của bộ mã hóa (encoder), trong khi phần lớn các vector còn lại không bao giờ được chọn làm láng giềng gần nhất. Vì không được chọn, các "vector chết" này sẽ không nhận được bất kỳ gradient nào trong quá trình lan truyền ngược (do thuật toán Straight-Through Estimator chỉ cập nhật vector được gán), khiến chúng vĩnh viễn bị loại khỏi quá trình huấn luyện và làm giảm nghiêm trọng dung lượng biểu diễn của mô hình.

Nguyên nhân gốc rễ và các kỹ thuật khắc phục thực tiễn:

### Nguyên nhân cốt lõi gây sụp đổ Codebook

**1. Sự trượt phân phối (Non-stationarity / Internal Covariate Shift)**
Trong quá trình huấn luyện, trọng số của bộ mã hóa (encoder) liên tục được cập nhật, khiến phân phối đặc trưng đầu ra của nó thay đổi theo thời gian (phi trạm). Tuy nhiên, các vector Codebook không được chọn sẽ giữ nguyên vị trí cũ. Khi đầu ra của encoder dịch chuyển sang vùng không gian khác, các vector bị bỏ lại phía sau sẽ ngày càng xa dữ liệu thực tế và trở nên hoàn toàn lỗi thời.

**2. Tính bất đối xứng của hàm mất mát (Asymmetric Commitment Loss)**
Hàm mất mát cam kết (Commitment loss) thực chất mang tính "tìm kiếm chế độ" (mode-seeking). Nó chỉ kéo các vector Codebook *đã được chọn* về phía đầu ra của encoder, mà bỏ mặc hoàn toàn nhóm vector không được chọn. Những vector không được chọn không nhận được gradient, không được cập nhật, dẫn đến hiện tượng phân đôi (bifurcation) trong Codebook và gây sụp đổ.

**3. Số chiều không gian lượng tử hóa quá lớn**
Sử dụng các vector mã có số chiều (dimensions) quá cao thường làm trầm trọng thêm tình trạng codebook không được tận dụng hết, gây cản trở quá trình gán cụm ổn định. 


### Các kỹ thuật khắc phục khi triển khai thực tế

Để giải quyết vấn đề này, cộng đồng nghiên cứu đã phát triển nhiều phương pháp từ tinh chỉnh siêu tham số đến thiết kế lại đồ thị tính toán:

#### **1. Tái khởi động Codebook (Codebook Reset / Replacement Policy)**
Đây là phương pháp thực dụng và phổ biến nhất. Khởi tạo Codebook bằng K-means (thay vì phân phối ngẫu nhiên) giúp Codebook bám sát dữ liệu ban đầu. Trong quá trình huấn luyện, bạn có thể thiết lập chính sách "Ít được sử dụng nhất" (Least-Recently-Used - LRU): theo dõi tần suất sử dụng của các vector trong một khoảng thời gian (ví dụ: vài chục batch); nếu một vector không được kích hoạt, hãy chủ động xóa nó đi và **thay thế bằng một vector đặc trưng ngẫu nhiên được trích xuất từ batch dữ liệu hiện tại**. Cách này giúp làm "sống lại" các vector chết mà không làm hỏng các vector đang hoạt động tốt.

#### **2. Tái tham số hóa Affine (Affine Re-parameterization)**
Bằng cách chia sẻ chung một bộ tham số tỷ lệ (scale) và dịch chuyển (shift/bias) trên toàn bộ Codebook, các vector mã chưa được gán vẫn nhận được gradient gián tiếp. Nghĩa là, khi encoder cập nhật và làm thay đổi trung bình/phương sai của không gian ẩn, sự thay đổi này được cập nhật vào tham số Affine chung. Nhờ đó, cả bảng mã (bao gồm cả các vector chết) đều dịch chuyển theo phân phối của encoder, giúp chúng duy trì khoảng cách đủ gần với dữ liệu để có cơ hội được kích hoạt trở lại.

#### **3. Giảm số chiều của Codebook (Lower Codebook Dimensionality)**
Thực nghiệm cho thấy, nếu bạn cần tăng dung lượng biểu diễn của mô hình, **việc tăng số lượng vector (Codebook size) hiệu quả hơn rất nhiều so với việc tăng số chiều (Codebook dimension)**. Ví dụ, thay vì dùng vector 64 hay 128 chiều, việc thu hẹp số chiều xuống thấp (như 8 chiều) giúp các vector dễ dàng bao quát không gian hơn, thúc đẩy mô hình sử dụng bộ mã phong phú và ổn định hơn.

#### **4. Tăng kích thước Batch (Larger Batch Size)**
Trong những bước huấn luyện đầu tiên, một batch size lớn sẽ cung cấp đa dạng các mẫu dữ liệu, giúp bao phủ nhiều vùng không gian và "kích hoạt" được nhiều vector Codebook hơn. Điều này giúp giảm rủi ro sụp đổ mã sớm.

#### **5. Các phương pháp lan truyền gradient nâng cao (NS-VQ / TransVQ)**
Nếu muốn can thiệp sâu vào cấu trúc:
*   **NS-VQ (Non-Stationary Vector Quantization):** Cải tiến thuật toán lan truyền ngược để phân phối một phần gradient của đầu ra encoder cho cả những Codebook không được chọn, dựa trên khoảng cách của chúng tới dữ liệu (sử dụng hàm nhân Kernel).
*   **TransVQ:** Đặt một mạng Transformer nhỏ (1 lớp) để làm hàm ánh xạ tự động biến đổi (transform) toàn bộ các điểm trong Codebook mỗi khi encoder có sự thay đổi, giúp bảng mã luôn thích ứng linh hoạt.

#### **6. Đưa các vector mã hóa và quantization về cùng mặt cầu chuẩn hóa $L2$**

Trong giới nghiên cứu, việc ép chuẩn L2 (L2 Normalization) cho cả đầu ra của bộ mã hóa ($z_e$) và các vector trong bảng mã ($z_q$) để chúng nằm trên cùng một mặt cầu đa chiều (unit hypersphere) thường được biết đến dưới tên gọi **L2-normalized codes** hoặc **Cosine Similarity Lookup**.

Thực chất, phương pháp này chính là một trong những cải tiến cốt lõi làm nên sự thành công của kiến trúc **ViT-VQGAN** (do Yu và cộng sự đề xuất năm 2021). Nhóm nghiên cứu đã thay thế mạng CNN bằng Vision Transformer (ViT) và áp dụng chuẩn hóa L2 ngay tại lớp lượng tử hóa, qua đó cải thiện đáng kể độ ổn định huấn luyện cũng như chất lượng tái cấu trúc hình ảnh. 

Dưới góc nhìn toán học và tối ưu hóa, việc ép cả hai vector lên mặt cầu giải quyết được rất nhiều rào cản của VQ-VAE truyền thống nhờ các cơ chế sau:

**Giới hạn không gian phân phối (Bounded Measure Space)**
Như chúng ta đã thảo luận về hiện tượng *Internal Codebook Covariate Shift* (sự trượt phân phối nội bộ), nguyên nhân khiến bảng mã sụp đổ là do phân phối đặc trưng của bộ mã hóa ($P_z$) liên tục thay đổi và trôi đi rất nhanh, khiến các vector bảng mã ($C_z$) không đuổi kịp và bị "bỏ rơi".
Việc áp dụng chuẩn hóa L2 thiết lập một không gian đo lường bị giới hạn nghiêm ngặt. Bất kể trọng số của bộ mã hóa có lớn đến đâu, đầu ra $z_e$ luôn bị ép nằm trên bề mặt của mặt cầu. Điều này giới hạn đáng kể phạm vi di chuyển của phân phối biểu diễn, giúp các vector bảng mã $z_q$ dễ dàng bám sát, căn chỉnh và đồng bộ với phân phối của bộ mã hóa trong suốt quá trình huấn luyện.

**Chuyển đổi từ Khoảng cách Euclidean sang Tương đồng Góc (Cosine Similarity)**
Khi cả $z_e$ và $z_q$ đều có độ dài bằng 1 ($\|z_e\|_2 = \|z_q\|_2 = 1$), bình phương khoảng cách Euclidean (MSE) giữa chúng sẽ có công thức:
$\|z_e - z_q\|_2^2 = \|z_e\|_2^2 + \|z_q\|_2^2 - 2(z_e \cdot z_q) = 2 - 2\cos(\theta)$
Có thể thấy, việc cực tiểu hóa khoảng cách Euclidean trên mặt cầu lúc này hoàn toàn tương đương toán học với việc **tối đa hóa độ tương đồng Cosine (Cosine Similarity)** giữa hai vector. 

Bằng cách triệt tiêu hoàn toàn sự biến thiên về mặt "độ lớn" (radial variations), mô hình không thể "lười biếng" mã hóa thông tin bằng cách kéo dãn vector cho dài ra. Thay vào đó, nó bị ép phải học các đặc trưng có tính phân tách tốt trong không gian góc (angular space). Sự tương đồng về mặt hướng (direction) thường mang nhiều ý nghĩa về mặt ngữ nghĩa (semantic) hơn là độ lớn của vector.

**Thúc đẩy việc tận dụng tối đa Bảng mã (Codebook Utilization)**
Trong thực tiễn triển khai (như các thiết lập cấu hình của kiến trúc mạng), người ta thường kết hợp kỹ thuật chuẩn hóa L2 này với việc **giảm mạnh số chiều của vector mã** (ví dụ: dùng $D=16$ hoặc $D=32$ chiều thay vì 256) và tăng kích thước bảng mã (ví dụ $K=16384$). Kinh nghiệm từ cộng đồng cho thấy, sự kết hợp giữa *L2 normalized codes*, số chiều nhỏ và số lượng mã lớn giúp tỷ lệ sử dụng bảng mã đạt mức tối đa và loại bỏ hoàn toàn các vector chết.

**Một điểm đánh đổi cần lưu ý (The Trade-off)**
Mặc dù phương pháp ép L2 lên mặt cầu mang lại sự ổn định tuyệt vời cho các mô hình sinh dữ liệu (Generative Models) và cải thiện sai số tái cấu trúc, một số phân tích chỉ ra rằng nó có thể **làm giảm hiệu năng trên các tác vụ phân loại (classification)**. 
Nguyên nhân là do việc triệt tiêu độ lớn của vector (magnitude) đồng nghĩa với việc vứt bỏ đi một phần thông tin. Trong các tác vụ phân loại sử dụng hàm mất mát nhạy cảm với biên độ (như soft-max cross-entropy), độ lớn của vector biểu diễn thường tỷ lệ thuận với "độ tự tin" (confidence) của mô hình. Tuy nhiên, nếu mục tiêu là học biểu diễn nén (compression) hoặc sinh ảnh/video, thì việc chiếu lên mặt cầu là một lựa chọn rất hợp lý và bắt kịp chuẩn mực của các mô hình SOTA hiện tại.

---

## Liên quan

- [VAE](03-vae.md) — VQ-VAE thay latent liên tục bằng codebook rời rạc.
- [VQGAN](05-vqgan.md) — VQ-VAE + perceptual loss + adversarial loss.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — chuẩn hóa L2 đưa code lên mặt siêu cầu.

## Tham khảo

- van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning* (NeurIPS 2017, arXiv:1711.00937) — VQ-VAE gốc.
- Razavi, van den Oord, Vinyals, *Generating Diverse High-Fidelity Images with VQ-VAE-2* (NeurIPS, 2019).
- Yu et al., *Vector-quantized Image Modeling with Improved VQGAN* (ViT-VQGAN, 2021) — L2-normalized codes.