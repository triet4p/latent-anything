# Riemannian Geometry Fundamental

> **TL;DR.** Hai trụ cột: **parallel transport** (dịch chuyển vector mà giữ "song song" trên đa tạp cong) và **curvature** (độ cong). Holonomy — vector bị quay lệch sau khi đi một vòng kín — đúng bằng lượng độ cong bên trong vòng. Khi latent cong đủ, lerp đi ra ngoài đa tạp / sụt norm → phải dùng [slerp](05-slerp.md) hoặc pullback geodesic.

Trong hình học Riemannian, **Dịch chuyển song song (Parallel Transport)** và **Độ cong (Curvature)** là hai khái niệm nền tảng để định hình cách chúng ta đo lường khoảng cách và phương hướng trên các không gian không bằng phẳng (đa tạp cong).

## **1. Dịch chuyển song song (Parallel Transport)**
Dịch chuyển song song là phương pháp di chuyển một vector dọc theo một đường cong trên đa tạp sao cho nó luôn giữ nguyên "hướng" và "độ lớn" một cách tương đối so với không gian cục bộ đó. 

*   Trong không gian phẳng Euclidean, việc giữ một vector song song với chính nó khi di chuyển là rất hiển nhiên. Tuy nhiên, trên một bề mặt cong (như mặt cầu), hướng của vector sẽ phải liên tục thay đổi để bám sát theo bề mặt (tuân theo một quy tắc gọi là đạo hàm hiệp biến - covariant derivative bằng không).
*   Một hệ quả kinh điển của dịch chuyển song song trên không gian cong là hiện tượng **Holonomy (Tính toàn chỉnh)**. Nếu bạn cầm một vector và dịch chuyển song song nó đi theo một vòng khép kín rồi quay về điểm xuất phát (ví dụ: đi từ xích đạo lên cực Bắc, rẽ ngang sang một kinh tuyến khác, rồi đi thẳng về xích đạo), vector khi quay về sẽ bị lệch đi một góc so với ban đầu. 

## **2. Độ cong (Curvature)**
Góc lệch sinh ra từ hiện tượng Holonomy tỷ lệ thuận với lượng độ cong của không gian nằm bên trong vòng khép kín đó. Độ cong đo lường mức độ đa tạp sai lệch so với không gian phẳng Euclidean.

*   Toán học biểu diễn sự sai lệch này thông qua **Tensor độ cong Riemann (Riemann Curvature Tensor)**, một đại lượng cho biết chính xác các vector sẽ bị biến đổi như thế nào khi đi theo các hướng khác nhau.
*   Trên các không gian cong này, "đường thẳng" thực chất là các **đường trắc địa (Geodesics)**. Đường trắc địa được định nghĩa chính là quỹ đạo mà vector tiếp tuyến của nó tự dịch chuyển song song với chính nó dọc theo đường đi (không có gia tốc bẻ lái sang hai bên). 

---

## Khi nào đa tạp trong không gian ẩn (Latent Space) cong đủ để Lerp thất bại?

Lerp (Linear Interpolation) là phép nội suy tuyến tính, thực chất là việc vẽ một đường thẳng tắp theo chuẩn Euclidean giữa hai điểm $z_1$ và $z_2$ trong không gian. Phép Lerp sẽ thất bại nghiêm trọng, sinh ra những kết quả nội suy tồi tệ, khi đa tạp bị cong mạnh rơi vào hai kịch bản hình học sau:

### **1. Đường thẳng cắt ngang qua "vùng rỗng" ngoại đa tạp (Off-manifold)**
Không gian ẩn thực tế do bộ giải mã (decoder) học được không phải là một khối đặc mà là một đa tạp khả vi cong, bám theo sự phân bố của dữ liệu thực tế. Khi sự thay đổi ngữ nghĩa của dữ liệu mang tính phi tuyến (non-linear), đa tạp này uốn lượn rất phức tạp.

*   Nếu bạn dùng Lerp để nối hai điểm cách xa nhau, quỹ đạo đường thẳng này sẽ cắt ngang qua lòng đa tạp cong, đi xuyên qua các vùng "ngoại đa tạp" (off-manifold).
*   Đây là những "khoảng trống" có mật độ xác suất dữ liệu cực kỳ thấp, nơi bộ giải mã chưa từng được huấn luyện để hiểu. Hệ quả là các điểm nằm giữa đường Lerp sẽ khiến mô hình sinh ra những hình ảnh/dữ liệu trung gian mờ nhòe, mất chi tiết, méo mó hoặc không thực tế.

### **2. Hiện tượng vỏ siêu cầu (Hypersphere Shell) và Suy giảm chuẩn (Norm Degradation)**
Trong các mô hình sinh có số chiều rất lớn (như VAE, Diffusion, GAN), phân phối tiên nghiệm thường là phân phối Gaussian đẳng hướng $\mathcal{N}(0, I)$. Do hiện tượng tập trung độ đo (concentration of measure) trong không gian siêu chiều, hầu hết khối lượng dữ liệu không nằm ở gốc tọa độ mà tụ tập rất chặt chẽ trên một lớp vỏ siêu cầu mỏng.

*   Khi đa tạp tập trung trên vỏ cầu, quỹ đạo Lerp nối hai điểm trên vỏ sẽ tạo thành một dây cung đâm xuyên trực tiếp qua không gian bên trong (lõi) của mặt cầu. 
*   Tại điểm chính giữa của đoạn thẳng Lerp ($t = 0.5$), **chuẩn (norm) của vector bị suy giảm nghiêm trọng nhất**. Việc sụt giảm độ lớn vector này đẩy dữ liệu vào vùng lõi có xác suất cực thấp của phân phối. Bộ giải mã khi nhận các vector có chuẩn nhỏ bất thường này sẽ thất bại trong việc tái tạo tính sắc nét, sinh ra hiện tượng "tent-pole" (chỉ rõ hai đầu nhưng giữa thì mờ nhòe).

### Giải pháp thay thế Lerp
Khi đa tạp ẩn bị cong ở mức độ này, các phương pháp hình học phi tuyến bắt buộc phải được sử dụng:

*   **Slerp (Spherical Linear Interpolation):** Thay vì đi đường thẳng qua lõi, Slerp di chuyển men theo cung tròn lớn trên bề mặt siêu cầu. Phương pháp này bảo toàn tuyệt đối độ lớn của vector trong suốt quá trình nội suy, giúp giữ cho các điểm trung gian luôn nằm trên vỏ đa tạp có mật độ cao, tạo ra sự chuyển đổi sắc nét và mượt mà.
*   **Pullback Geodesics (Đường trắc địa đo bằng Pullback):** Đối với các đa tạp cong cực kỳ phức tạp không đồng đều, độ đo Pullback (Pullback metric) được dùng để biến không gian ẩn thành một đa tạp Riemannian thực thụ. Quá trình nội suy lúc này là việc giải bài toán tối ưu năng lượng để tìm đường trắc địa. Thuật toán sẽ tự động bẻ cong quỹ đạo nội suy, né tránh các vùng dữ liệu rỗng và chỉ trườn men theo các "dãy đồi" có mật độ dữ liệu cao, từ đó đảm bảo đầu ra luôn đạt độ chân thực tối đa.