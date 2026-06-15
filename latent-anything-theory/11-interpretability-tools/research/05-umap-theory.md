# UMAP Theory (McInnes et al., 2018)

> **TL;DR.** UMAP (*Uniform Manifold Approximation and Projection*) là một phương pháp giảm chiều phi tuyến xây đồ thị lân cận có trọng số từ dữ liệu cao chiều rồi tìm một embedding thấp chiều sao cho cấu trúc topo-mờ của hai không gian khớp nhau nhất có thể. Ý tưởng cốt lõi là ghép các lân cận địa phương thành một **fuzzy simplicial graph** và tối ưu một mất mát kiểu **cross-entropy** giữa đồ thị cao chiều và thấp chiều. Điểm mạnh là nhanh, scale tốt, và thường giữ được cấu trúc cục bộ lẫn hình dạng thô toàn cục tốt hơn t-SNE khi chọn `n_neighbors` hợp lý; caveat là kết quả vẫn nhạy với hyperparameter, metric, random seed, và không nên diễn giải khoảng cách xa như chân lý hình học tuyệt đối.

Trong tầng 11, UMAP là công cụ trực quan hóa không giám sát bổ sung cho [Probing Classifiers Survey](04-probing-classifiers-survey.md). Probe trả lời "thuộc tính nào *decodable*"; UMAP trả lời "latent *nhìn như thế nào* nếu chiếu xuống 2D/3D". Với Latent-Anything, đây là mắt thường của Layer A: tìm cluster, phát hiện outlier, và so trajectory hoặc feature neighborhood trước khi đi sâu vào phân tích nhân quả.

---

## 1. Trực giác / Định nghĩa

UMAP xuất phát từ giả thuyết rằng dữ liệu thật nằm gần một manifold chiều thấp hơn, giống trực giác ở [Manifold Hypothesis](../../01-space-representation/research/03-manifold-hypothesis.md). Nếu mỗi điểm chỉ "nhìn" một vùng lân cận nhỏ quanh nó, có thể ước lượng cấu trúc manifold đó bằng một đồ thị k-nearest-neighbor có trọng số.

Khác với PCA, UMAP không cố giữ một phép chiếu tuyến tính toàn cục. Nó làm hai bước:

1. học cấu trúc địa phương trong không gian gốc bằng một đồ thị mờ;
2. đặt các điểm vào không gian thấp chiều sao cho đồ thị thấp chiều có quan hệ lân cận giống đồ thị gốc nhất có thể.

Vì vậy UMAP thuộc họ *neighbor embedding* như t-SNE, nhưng nền lý thuyết của nó đi qua hình học Riemann và topo đại số thay vì chỉ qua phân phối xác suất cặp điểm.

---

## 2. Cơ chế / Công thức

### 2.1. Từ manifold sang fuzzy simplicial graph

UMAP giả định dữ liệu được lấy mẫu từ một manifold có metric địa phương. Thay vì dùng cùng một bán kính cho mọi điểm, nó ước lượng một scale cục bộ qua `n_neighbors`: mỗi điểm có một cảm nhận riêng về "gần". Theo tài liệu chính thức, điều này tương đương với việc dùng k-nearest-neighbor để xấp xỉ metric địa phương, trong đó `n_neighbors` quyết định ta nhìn bao nhiêu là "cục bộ".

Giữa hai điểm $i, j$, ta có hai trọng số định hướng $a$ và $b$ do hai lân cận cục bộ sinh ra. UMAP hợp nhất chúng bằng fuzzy union:

$$
w_{ij} = a + b - ab.
$$

trong đó $a, b \in [0,1]$ là mức tin cậy rằng cạnh $i \leftrightarrow j$ tồn tại theo góc nhìn của từng điểm, còn $w_{ij}$ là xác suất hợp nhất của cạnh trong đồ thị mờ toàn cục. Công thức này có nghĩa là cạnh mạnh khi **ít nhất một** phía coi hai điểm là hàng xóm đáng tin.

### 2.2. Tối ưu embedding thấp chiều

Sau khi có đồ thị mờ ở không gian cao chiều, UMAP xây một đồ thị tương tự trong embedding thấp chiều và tối ưu sao cho hai đồ thị giống nhau nhất có thể. Ở mức khái niệm, mất mát là cross-entropy giữa hai fuzzy graph:

$$
\mathcal{L}
= \sum_{i \ne j}
\Big[
w_{ij}\log \frac{w_{ij}}{\hat w_{ij}}
 + (1-w_{ij})\log \frac{1-w_{ij}}{1-\hat w_{ij}}
\Big].
$$

trong đó $w_{ij}$ là trọng số cạnh trong không gian gốc, $\hat w_{ij}$ là trọng số cạnh sau khi nhúng xuống thấp chiều, và $\mathcal{L}$ đo hai cấu trúc topo-mờ lệch nhau bao nhiêu. Khi tối thiểu hóa mất mát này, điểm gần thật sẽ bị kéo lại, điểm không phải hàng xóm sẽ bị đẩy ra xa.

### 2.3. Hai hyperparameter quyết định hành vi

| Hyperparameter | Vai trò | Hệ quả trực quan |
|---|---|---|
| `n_neighbors` | Quyết định độ lớn lân cận dùng để ước lượng manifold | nhỏ: nhấn mạnh local structure, dễ vỡ thành nhiều mảnh; lớn: giữ hình dạng thô tốt hơn nhưng mất chi tiết nhỏ |
| `min_dist` | Đặt ngưỡng điểm được phép tụ gần nhau cỡ nào trong embedding | nhỏ: cụm chặt, tốt cho cluster; lớn: cụm mềm hơn, nhấn mạnh bố cục tổng thể |

Theo tài liệu `umap-learn`, `n_neighbors` là núm chỉnh trade-off local/global trực tiếp nhất, còn `min_dist` điều khiển mức "pack" của embedding.

---

## 3. UMAP khác gì t-SNE?

| Khía cạnh | UMAP | t-SNE |
|---|---|---|
| Đơn vị tối ưu | fuzzy graph / topo-mờ | phân phối xác suất cặp điểm |
| Scale | thường nhanh hơn và dễ mở rộng hơn | chậm hơn trên tập lớn nếu không dùng xấp xỉ |
| Toàn cục vs cục bộ | thường giữ bố cục thô toàn cục tốt hơn khi tăng `n_neighbors` | rất mạnh về local cluster, nhưng khoảng cách xa giữa cụm thường khó diễn giải |
| Số chiều embedding | linh hoạt hơn 2D/3D | chủ yếu dùng cho 2D/3D visualization |

Bài gốc của UMAP nói khá cẩn thận: nó "*arguably preserves more of the global structure*" so với t-SNE, không phải luôn luôn và không phải cho mọi dataset. Ý nghĩa thực tế là:

- nếu muốn nhìn **ai gần ai ở quy mô địa phương**, cả UMAP và t-SNE đều hữu ích;
- nếu muốn giữ **quan hệ giữa các cụm / nhánh lớn** tốt hơn, UMAP thường dễ điều chỉnh hơn thông qua `n_neighbors`;
- nhưng khoảng cách lớn trong bản đồ 2D vẫn là một phép xấp xỉ, không phải geodesic thật của manifold.

Với latent space, đây là điểm quan trọng: UMAP giúp xem cấu trúc thô giữa trajectory, cluster, regime hoạt động của model; còn kết luận hình học mạnh vẫn cần đối chiếu với metric gốc hoặc các phép đo riêng.

---

## 4. Giới hạn / Khi nào thất bại

**Không phải máy sự thật cho global geometry.** Dù UMAP thường giữ bố cục thô tốt hơn t-SNE, embedding 2D vẫn bóp méo manifold cao chiều. Không nên đọc khoảng cách xa như khoảng cách thật.

**Nhạy với `metric`.** Chọn `euclidean`, `cosine`, hay `correlation` sẽ thay hàng xóm ngay từ đầu; embedding thay đổi theo. Với latent chuẩn hóa norm, cosine có thể đúng hơn Euclidean.

**Nhạy với `n_neighbors`, `min_dist`, và random seed.** Hai bản đồ UMAP có thể khác nhau đáng kể nếu đổi cấu hình. Khi phân tích nghiêm túc nên quét vài giá trị và kiểm tính ổn định.

**Dễ tạo ảo giác cluster.** Mọi thuật toán chiếu xuống 2D đều có nguy cơ làm người xem tin quá mạnh vào các đám mây trực quan. Cluster thấy bằng mắt không thay thế cho đánh giá density hay kiểm định downstream.

**Không giải thích nhân quả.** UMAP chỉ là công cụ trực quan hóa. Nó cho gợi ý về cấu trúc, không cho biết feature nào gây ra hành vi hay model dùng thông tin thế nào.

---

## 5. Liên hệ với Latent-Anything

UMAP là một `Method` nền tảng của **Layer A — Introspection**: nhận latent hoặc trajectory, trả embedding 2D/3D để người dùng nhìn được cấu trúc trước khi probe hoặc can thiệp.

```python
class UmapProjection(Protocol):
    n_neighbors: int
    min_dist: float
    metric: str
    def fit_transform(self, latent: np.ndarray) -> np.ndarray: ...
    def transform(self, latent: np.ndarray) -> np.ndarray: ...
```

- **Layer A — Introspection**: chiếu latent để thấy cluster, outlier, drift theo thời gian, và feature neighborhood. Đây là bước khám phá trước khi chạy probe có giám sát.
- **Trajectory primitive**: tô màu trajectory theo thời gian trên embedding cho phép nhìn rollout, regime switch, hay collapse trực tiếp bằng mắt.
- **Layer B — Manipulation**: sau khi edit latent hoặc steer feature, có thể so embedding trước/sau để xem can thiệp đẩy điểm sang vùng nào.
- **Layer C — Runtime**: với tập activation rất lớn, runtime phải lo precompute k-NN, batching, và cache embedding để notebook/web UI phản hồi nhanh.

Trong tầng 11, UMAP mở đầu cho nhóm công cụ "nhìn latent". Mục kế tiếp là **PaCMAP**, một biến thể mới hơn tập trung cân bằng local với global bằng cách chọn các cặp điểm gần-trung gian-xa khác cách.

---

## Liên quan

- [Manifold Hypothesis](../../01-space-representation/research/03-manifold-hypothesis.md) — nền trực giác rằng dữ liệu nằm gần manifold chiều thấp.
- [Riemannian Geometry cơ bản](../../03-geometry-structure/research/04-riemannian-geometry.md) — UMAP dùng trực giác metric địa phương để ước lượng manifold.
- [Curse of Dimensionality](../../01-space-representation/research/04-curse-of-dimensionality.md) — lý do trực quan hóa hàng xóm cao chiều khó và cần phép chiếu phi tuyến.
- [Probing Classifiers — Survey](04-probing-classifiers-survey.md) — probe trả lời "encode gì", UMAP trả lời "nhìn như thế nào".

## Tham khảo

- L. McInnes, J. Healy, J. Melville, *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction* (arXiv:1802.03426, 2018).
- L. van der Maaten, G. Hinton, *Visualizing Data using t-SNE* (JMLR 2008).
- `umap-learn` documentation, *How UMAP Works* (official docs).
- `umap-learn` documentation, *Basic UMAP Parameters* (official docs).
