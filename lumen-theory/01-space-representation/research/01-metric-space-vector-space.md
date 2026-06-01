# Metric Space và Vector Space

## Metric Space

Metric Space là một cặp $(X, d_X)$ trong đó $X$ là tập hợp các đối tượng và $d_X$ là một hàm khoảng cách.

Hàm khoảng cách $d_X$ cần phải tuân thủ các tiên đề về:
- Không âm
- Đồng nhất
- Phân biệt ($d(x,y)>0$ nêu $x\ne y$)
- Đối xứng.
- Bất đẳng thức tam giác.

## Vector Space

Vector Space là một không gian nơi dữ liệu được biểu diễn dưới dạng vector nhiều chiều.

Vector Space cho phép thực hiện các phép toán đại số tuyến tính như cộng hai vector hay nhân số vô hướng với vector.

Trong học máy, các điểm dữ liệu thực tế thường được số hóa và nhúng (embed) vào không gian vector này, thường gọi là Không gian biểu diễn hoặc Không gian ẩn.

## Chuẩn (Norm)

Chuẩn là một hàm toán học để đo lường **độ dài** hoặc **độ lớn** của một vector trong vector space, được ký hiệu là $||x||$. 

Mọi chuẩn đều có thể tự sinh ra một **metric khoảng cách (distance metric)** thông qua công thức 
$$ d(x, y) = ||x - y|| $$

Phổ biến nhất có 2 chuẩn là
- L2 (Chiều dài đường Euclidean)
- L1 (Chiều dài Manhattan)

## Tích vô hướng (Inner Product / Dot Product)

Tích vô hướng của 2 vector $a, b$ được định nghĩa bởi
$$a \cdot b = \Sigma a_ib_i$$
trong đó $a_i, b_i$ là các thành phần.

Cũng có thể định nghĩa

$$a \cdot b = ||a||\cdot ||b|| \cdot \cos(\theta)$$
trong đó $\theta$ là góc định hướng giữa $a, b$.

Từ đây ta có **độ tương đồng cosin (Cosin Similarity)**:
$$\cos(\theta) = \frac{a\cdot b}{||a||_2||b||_2}$$
(dùng chuẩn L2).

Cosin simlarity là thước đo đánh giá sự tương đồng về *hướng (orientation)* và bỏ qua *độ lớn (magnitude)* của chúng.