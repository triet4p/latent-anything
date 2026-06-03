# Variational AutoEncoder (VAE)

## Probabilistic Generative Models (Mô hình sinh xác suất)

Trong học máy, các mô hình sinh xác suất là một lớp mô hình được thiết kế để học cấu trúc và phân phối tiềm ẩn của dữ liệu, từ đó có khả năng sinh (generate) những mẫu dữ liệu mới tương tự như dữ liệu gốc trong tập huấn luyện.

Khác với mô hình tất định (như Autoencoder cổ điển chỉ ánh xạ một đầu vào thành một điểm cố định duy nhất), mô hình sinh xác suất thường mã hóa không gian ẩn dưới dạng các phân phối xác suất liên tục. Quá trình sinh của chúng được định nghĩa chặt chẽ qua các bước:

1. Trước tiên, một vector ẩn $z$ được lấy mẫu ngẫu nhiên từ một phân phối tiên nghiệm (prior distribution) $p_\theta(z)$.
2. Sau đó, một điểm dữ liệu mới $x sinh ra từ phân phối có điều kiện $p_\theta(x|z)$ (đóng vai trò là probabilistic decoder).

## Marginal Likelihood (Hàm khả di biên)

Trong bài toán suy diễn xác suất, đại lượng cốt lõi để đánh giá mô hình là hàm khả dĩ biên, ký hiệu $P(X)$ hoặc $p_\theta(x)$.

$P(X)$ đại diện cho xác suất biên (xác suất tổng thể để dữ liệu quan sát được xuất hiện) cuả dữ liệu dưới một mô hình với bộ tham số $\theta$. Vì chúng ta giả định dữ liệu $x$ sinh ra từ một biến ẩn $z$ chưa biết, để tính được xác suất thực sự của $x$, ta phải tính tổng/tichs phân xác suất của $x$ trên toàn bộ các cấu hình khả dĩ của không gian ẩn $Z$. Công thức toán là:

$$ P(X) = p_\theta(x) = \int p_\theta(x, z)dz=\int p_\theta(x|z)p_\theta(z)dz$$

Trong các tài liệu, nó còn được gọi là ***evidence***, vì nếu chúng ta chọn đúng mô hình và bộ $\theta$ phù hợp với thực tế, thì xác suất biên của dữ liệu quan sát $x$ phải rất cao.

Trong tính toán, ta thường dùng $\log P(X)$ để biến tích thành tổng.

## Tại sao không thể tối ưu trực tiếp $P(X)$

Có một số lý do khiến ta không thể tối ưu trực tiếp $\log P(X)$.

- **Tích phân bất khả thi**: Theo định lý xác suất toàn phần, để tính $P(X)$ cần lấy tích phân của toàn bộ $Z$. Trong mạng neural, không gian ẩn $Z$ thường rất lớn và có mối quan hệ phi tuyến phức tạp với $X$. Do vậy việc duyệt hết là bất khả thi về mặt tính toán.

- **Curse of dimensionality**: Nếu cố gắng dùng Maximum Likelihood Estimation (MLE) trên không gian dữ liệu gốc, cần một lượng khổng lồ mẫu dữ liệu.

- **Không thể tính phân phối hậu nghiệm thực tế (True Posterior)**: Hệ quả khi $P(X)$ không tính thực tế được.

## Variational Inference (suy diễn biến phân)

Vì không thể tính tích phân trực tiếp, VI biến bài toán integration thành optimization. 

Ý tưởng cốt lõi là giới thiệu một **phân phối xấp xỉ khả thi** $q(Z|X)$ (do encoder đảm nhận) để bám đuổi và bắt chước phân phối $p(Z|X)$ nhất, nghĩa là đi tối ưu độ chệch Kullback-Leibler (KL Divergence):

$$ D_{KL}(q(Z|X) || p(Z|X)) $$

### ELBO (Evidence Lower Bound)

Do cái mục tiêu $p(Z|X)$ bản thân là không tính được, nên để tối ưu, ta buộc phải tìm cách khác chỉ dựa vào Encoder, Decoder và $Z, q$.

Ta đi từ định nghĩa

$$ D_{KL}(q(Z|X)||p(Z|X)) = \int q(Z|X)\log \frac{q(Z|X)}{p(Z|X)}dZ$$

Áp dụng định lý Bayes thì

$$ D_{KL} = \int q(Z|X)\log \frac{q(Z|X)p(X)}{p(X, Z)} dZ$$

Khai triển log với chú ý $\log P(X)$ không phụ thuộc vào $Z$, ta có

$$ \log P(X) = \mathbb{E}_{q(Z|X)}\left[\log \frac{p(X,Z)}{q(Z|X)} \right] + D_{KL}(q(Z|X)||p(Z|X))$$

Từ đó có thể thấy được thay vì tối ưu thẳng $D_{KL}$, ta hoàn toàn có thể tối ưu

$$ \text{ELBO}= \mathbb{E}_{q(Z|X)}\left[\log \frac{p(X,Z)}{q(Z|X)} \right] = \mathbb{E}_{q(Z|X)}[\log p(X|Z)] - D_{KL}(q(Z|X) || p(Z)) $$

1. Số hạng thứ nhất chính là Reconstruction Loss, ép mô hình phải giữ các thông tin cốt lõi của $X$ vào $Z$.

2. Số hạng thứ hai chính là Regularization/KL Penalty. Nó bắt $q(Z|X)$ do encoder sinh ra không được đi quá xa so với phân phối tiên nghiệm chuẩn hóa của $p(Z)$, thường là $\mathcal{N}(0, I)$.

## β-VAE: Kiểm soát Tỷ lệ Thông tin

VAE tiêu chuẩn sử dụng hệ số $\beta = 1$ trước số hạng KL trong hàm ELBO. Mô hình **β-VAE** (Higgins et al., 2017) tổng quát hóa điều này bằng cách đưa vào một siêu tham số $\beta \geq 0$:

$$\text{ELBO}_\beta = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$

### Ý nghĩa của $\beta$

Hệ số $\beta$ đóng vai trò là **Lagrange multiplier** kiểm soát tỷ lệ thông tin đi qua không gian ẩn:

- **$\beta = 0$**: Không có regularization → mô hình trở về Autoencoder cổ điển. Không gian ẩn bị phân mảnh, không thể sinh mẫu mới một cách có ý nghĩa.
- **$\beta = 1$**: VAE tiêu chuẩn. Cân bằng giữa chất lượng tái cấu trúc và cấu trúc không gian ẩn.
- **$\beta > 1$**: Tăng áp lực regularization. Không gian ẩn trở nên nhỏ gọn hơn (ít thông tin hơn), đổi lại chất lượng tái cấu trúc giảm xuống. Ở chế độ này, mô hình có xu hướng học các biểu diễn **disentangled** (tách biệt yếu tố) — mỗi chiều $z_i$ kiểm soát độc lập một yếu tố sinh dữ liệu.

Với $\beta > 1$, phương trình nghiệm tối ưu của variance encoder tại mỗi điểm dữ liệu là:

$$\sigma^2_{\text{opt}}(\beta) = \frac{\beta \tau^2}{1 + \beta \tau^2}$$

trong đó $\tau^2$ là variance của decoder. Khi $\beta$ tăng, $\sigma^2_{\text{opt}} \to \tau^2$ — encoder buộc phải trải rộng xác suất ra hơn để thỏa mãn ràng buộc KL.

### Disentanglement (Tách biệt yếu tố)

Một biểu diễn ẩn được gọi là **disentangled** khi mỗi chiều $z_i$ kiểm soát độc lập một yếu tố sinh dữ liệu. Ví dụ: với dữ liệu ảnh mặt người, $z_1$ chỉ kiểm soát hướng nhìn, $z_2$ chỉ kiểm soát độ tuổi, v.v. — thay đổi $z_1$ không ảnh hưởng đến yếu tố mà $z_2$ kiểm soát.

β-VAE khuyến khích disentanglement vì số hạng KL phạt nặng các phân phối encoder có tương quan giữa các chiều. Áp lực KL đẩy $q(z|x)$ về phía phân phối tiên nghiệm độc lập $\mathcal{N}(0, I)$, nơi mọi chiều hoàn toàn độc lập nhau. Càng lớn $\beta$, áp lực độc lập giữa các chiều $z$ càng mạnh.

### Kết nối với Information Bottleneck

Mục tiêu β-VAE chính là **IB Lagrangian** được giới thiệu ở phần đầu chương này:

$$\text{ELBO}_\beta = \underbrace{\mathbb{E}[\log p(x|z)]}_{\text{sức mạnh tái cấu trúc}} - \beta \cdot \underbrace{D_{KL}(q(z|x) \| p(z))}_{\approx I(Z;X) \text{ (chi phí thông tin)}}$$

Khi $\beta$ tăng, mô hình bị ép loại bỏ nhiều thông tin hơn từ $X$ — chỉ giữ lại những gì thực sự cần thiết để tái cấu trúc. Điều này tạo ra một **đường cong Pareto** trong không gian thông tin, tương tự như Information Plane trong nguyên lý IB: mỗi giá trị $\beta$ tương ứng với một điểm vận hành trên đường biên tối ưu giữa $I(Z;X)$ và chất lượng tái cấu trúc.


