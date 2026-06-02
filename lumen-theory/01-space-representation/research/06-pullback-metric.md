# Pullback Metric

## Pullback Metric (Metric kéo lùi)

Là một công cụ toán học trong hình học vi phân cho phép "chuyển giao" (transfer) cấu trúc đo lường khoảng cách từ một không gian đích (như không gian dữ liệu gốc) ngược trở về một không gian miền (như latent space) thông qua một hàm ánh xạ trơn.

Trong học sâu, decoder chính là hàm ánh xạ từ latent space ra thực địa. Pullback metric hoạt động như một ***tỉ lệ xích cục bộ và đa hướng tại mỗi điểm trên bản đồ***. Nó giúp ta đánh giá một bược dịch cuyển cực nhỏ (infinitesimal) trong không gian ẩn sẽ thực sự tạo ra sự biến đổi lớn tới mức nào trong không gian đầu ra.

## Cơ chế toán học
Nếu không gian dữ liệu thực được trang bị một thước đo khoảng cách thông thường (Euclidean phẳng), metric kéo lùi $g_Z(z)$ tại điểm $z$ trong latent space sẽ được tính toán thông qua **ma trận Jacobian** $J_\psi(z)$ của bộ giải mã $\psi$:
$$ g_Z(z) = J_\psi(z)^T J_\psi(z) $$

Ma trận Jacobian đo lường tốc độ thay đổi của đầu ra theo đầu vào khi áp dụng metric này, độ dài của một vector dịch chuyển vi phân $dz$ trong không gian ẩn không còn là $||dz||^2$ mà bị bóp thành:

$$||dz||_{g_Z}^2=dz^T g_Z(z) \, dz$$

Metric này sẽ kéo cấu trúc cong của dữ liệu từ không gian đầu ra về để gán cho không gian ẩn.

## Ưu điểm và hạn chế

- Tính chính xác được Geosedic
- Nhưng khó khăn về tính toán (vì phải tính Jacobian liên tục).