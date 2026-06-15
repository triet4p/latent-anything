# PaCMAP

> **TL;DR.** PaCMAP (*Pairwise Controlled Manifold Approximation Projection*) là một phương pháp giảm chiều phi tuyến được thiết kế để giữ đồng thời local structure lẫn global structure thay vì nghiêng hẳn về một phía như nhiều công cụ visualization khác. Ý tưởng cốt lõi là tối ưu embedding bằng ba loại cặp điểm khác nhau: **neighbor pairs**, **mid-near pairs**, và **further pairs**, với trọng số thay đổi theo từng pha tối ưu để trước tiên dựng bố cục thô rồi mới siết chi tiết cục bộ. Caveat là PaCMAP vẫn chỉ tạo một bản đồ 2D/3D xấp xỉ, nên rất nhạy với metric, sampling của các cặp điểm, và không thể thay thế các phép đo hình học trong latent gốc.

Sau [UMAP Theory](05-umap-theory.md), PaCMAP là bước tự nhiên tiếp theo trong tầng 11: nếu UMAP cố cân bằng local/global bằng cách chỉnh đồ thị hàng xóm mờ, thì PaCMAP cân bằng bài toán đó trực tiếp bằng cách chọn **đúng loại cặp điểm** và **đổi trọng số của chúng theo thời gian**. Với Latent-Anything, PaCMAP hữu ích khi muốn quan sát embedding của latent hoặc trajectory mà không đánh đổi quá mạnh giữa việc "nhìn rõ cluster" và "giữ được bố cục tổng thể".

---

## 1. Trực giác / Định nghĩa

PaCMAP được giới thiệu trong bài JMLR 2021 của Yingfan Wang và cộng sự như một phản ứng trực tiếp với nhược điểm quen thuộc của các công cụ visualization: t-SNE và UMAP thường rất mạnh ở local structure nhưng dễ làm méo global structure, còn các phương pháp thiên về toàn cục như TriMap lại có thể làm local neighborhood trở nên kém sắc nét hơn.

Trực giác của PaCMAP là: nếu chỉ kéo các hàng xóm thật lại gần nhau và đẩy các điểm xa ra, embedding sẽ dễ rơi vào hai cực đoan:

1. local quá mạnh: các cụm nhỏ đẹp nhưng vị trí tương đối giữa các cụm bị méo;
2. global quá mạnh: bố cục thô đúng hơn nhưng neighborhood nhỏ bị nhòe.

PaCMAP thêm một nhóm tín hiệu trung gian gọi là **mid-near pairs**. Đây không phải hàng xóm sát nhất, cũng không phải các cặp xa ngẫu nhiên, mà là các cặp "không quá gần, không quá xa". Nhóm cặp này đóng vai trò như khung xương trung cấp giúp các cluster, nhánh, hay manifold con giữ được vị trí tương đối trước khi thuật toán tinh chỉnh local structure.

---

## 2. Cơ chế / Công thức

### 2.1. Ba loại cặp điểm

PaCMAP tối ưu embedding thấp chiều $y_i$ bằng ba nhóm cặp:

- **Neighbor pairs**: các hàng xóm gần nhau trong không gian gốc, dùng để giữ local structure.
- **Mid-near pairs**: các cặp ở khoảng trung gian, dùng để neo global layout.
- **Further pairs**: các cặp không phải hàng xóm, dùng làm lực đẩy để tránh đè cụm lên nhau.

Trong implementation chính thức, neighbor pairs được lấy từ đồ thị k-nearest-neighbor đã scale khoảng cách; further pairs được lấy ngẫu nhiên nhưng tránh trùng với neighbor pairs; còn mid-near pairs được tạo bằng cách lấy 6 ứng viên ngẫu nhiên, bỏ điểm gần nhất, rồi chọn điểm gần nhất trong 5 điểm còn lại. Cách chọn này có nghĩa là mid-near pair cố tình tránh hàng xóm sát nhất nhưng vẫn không rơi sang vùng "rất xa".

### 2.2. Hàm mất mát theo cặp

Trong implementation tham chiếu của tác giả, với

$$
d_{ij} = 1 + \lVert y_i - y_j \rVert_2^2,
$$

trong đó $y_i, y_j$ là hai điểm trong embedding thấp chiều và $d_{ij}$ là bình phương khoảng cách Euclidean thấp chiều đã cộng thêm hằng số 1 để tránh suy biến, PaCMAP cộng ba hạng mất mát:

$$
\mathcal{L}
= \sum_{(i,j)\in \mathrm{NN}} w_{\mathrm{NN}}\frac{d_{ij}}{10 + d_{ij}}
+ \sum_{(i,j)\in \mathrm{MN}} w_{\mathrm{MN}}\frac{d_{ij}}{10000 + d_{ij}}
+ \sum_{(i,j)\in \mathrm{FP}} w_{\mathrm{FP}}\frac{1}{1 + d_{ij}}.
$$

trong đó `NN`, `MN`, `FP` lần lượt là tập neighbor, mid-near, và further pairs; $w_{\mathrm{NN}}, w_{\mathrm{MN}}, w_{\mathrm{FP}}$ là các trọng số theo từng pha tối ưu. Hai hạng đầu tăng theo $d_{ij}$ nên đóng vai trò lực hút: nếu cặp được chọn mà đang xa nhau trong embedding, loss tăng và gradient kéo chúng lại. Hạng cuối giảm theo $d_{ij}$ nên đóng vai trò lực đẩy: cặp further càng gần nhau thì bị phạt càng mạnh.

Điểm hay của công thức này là PaCMAP không dùng cùng một tín hiệu cho mọi khoảng cách. Nó tách rõ:

- hàng xóm gần để giữ chi tiết cục bộ;
- cặp trung gian để giữ bố cục trung cấp và toàn cục;
- cặp xa để chặn hiện tượng collapse.

### 2.3. Tối ưu theo ba pha

Một ý tưởng quan trọng khác của PaCMAP là **không giữ trọng số cố định suốt quá trình học embedding**. Trong source chính thức:

- pha 1: $w_{\mathrm{MN}}$ giảm dần từ 1000 về 3, còn $w_{\mathrm{NN}} = 2$, $w_{\mathrm{FP}} = 1$;
- pha 2: $w_{\mathrm{MN}} = 3$, $w_{\mathrm{NN}} = 3$, $w_{\mathrm{FP}} = 1$;
- pha 3: $w_{\mathrm{MN}} = 0$, $w_{\mathrm{NN}} = 1$, $w_{\mathrm{FP}} = 1$.

Ý nghĩa trực giác:

1. **đầu tiên** cho mid-near pairs ảnh hưởng rất mạnh để dựng khung toàn cục;
2. **sau đó** cân bằng mid-near với neighbor để làm rõ cấu trúc;
3. **cuối cùng** bỏ mid-near đi, chỉ giữ local refinement và repulsion cơ bản.

Vì vậy PaCMAP thường cho cảm giác "coarse-to-fine": dựng bố cục lớn trước, rồi mới mài sắc local neighborhood.

### 2.4. Hyperparameter quan trọng

Theo README chính thức của dự án:

| Hyperparameter | Vai trò | Hệ quả trực quan |
|---|---|---|
| `n_neighbors` | số hàng xóm trong k-NN graph | nhỏ: local rõ nhưng bố cục dễ vỡ; lớn: global ổn hơn nhưng dễ làm cluster mềm |
| `MN_ratio` | số mid-near pairs theo tỉ lệ với số neighbor | tăng lên giúp neo global layout mạnh hơn, nhưng quá lớn có thể làm local sắc nét kém |
| `FP_ratio` | số further pairs theo tỉ lệ với số neighbor | tăng lực đẩy, giúp tách cụm; quá mạnh có thể làm embedding bị giãn quá mức |
| `init` | cách khởi tạo embedding (`pca` hoặc `random`) | ảnh hưởng đáng kể tới hình dạng cuối, nhất là với cấu trúc toàn cục |

README cũng ghi rõ rằng việc đổi các tham số này sẽ ảnh hưởng đáng kể đến kết quả visualization.

---

## 3. PaCMAP khác gì UMAP và t-SNE?

| Khía cạnh | PaCMAP | UMAP | t-SNE |
|---|---|---|---|
| Tín hiệu tối ưu chính | ba loại cặp điểm với trọng số động | fuzzy graph cao chiều vs thấp chiều | phân phối xác suất cặp điểm |
| Cơ chế giữ global structure | mid-near pairs + lịch trọng số nhiều pha | tăng kích thước lân cận qua `n_neighbors` | tương đối yếu, chủ yếu local |
| Điều chỉnh | `n_neighbors`, `MN_ratio`, `FP_ratio`, `init` | `n_neighbors`, `min_dist`, `metric` | `perplexity`, `learning_rate`, `init` |
| Trực giác vận hành | coarse-to-fine theo pha | dựng đồ thị topo-mờ rồi khớp embedding | kéo điểm gần, đẩy điểm xa theo xác suất |

PaCMAP không "đánh bại tuyệt đối" UMAP hay t-SNE trên mọi dataset. Điểm mạnh của nó là cách diễn đạt trade-off local/global minh bạch hơn: thay vì chỉ thay kích cỡ lân cận, nó đưa hẳn một lớp quan hệ trung gian vào objective.

Với latent space, điều này khá hợp lý: nhiều cấu trúc quan trọng không phải là hàng xóm sát nhất, mà là quan hệ giữa các vùng hoạt động, phase của trajectory, hay cluster bán tách rời. Mid-near pairs là một cách ép embedding phải quan tâm đến mức cấu trúc đó.

---

## 4. Giới hạn / Khi nào thất bại

**Vẫn là visualization low-dimensional, không phải bản sao hình học thật.** Bài JMLR mở đầu bằng cảnh báo chung rằng các công cụ giảm chiều có thể tạo ra cluster hoặc khoảng cách gây hiểu lầm; PaCMAP chỉ giảm rủi ro đó chứ không loại bỏ hoàn toàn.

**Nhạy với cách lấy cặp điểm.** Nếu k-NN graph ban đầu đã sai do metric không phù hợp hoặc dữ liệu quá nhiễu, thì neighbor, mid-near, và further pairs đều bị nhiễm lỗi ngay từ đầu.

**Initialization vẫn quan trọng.** Bài JMLR dành hẳn một phần để nhấn mạnh việc khởi tạo ảnh hưởng mạnh tới khả năng giữ global structure của nhiều thuật toán DR; PaCMAP cũng không miễn nhiễm hoàn toàn với điều đó.

**Tham số thêm đồng nghĩa thêm mặt điều chỉnh.** So với UMAP, PaCMAP có thêm `MN_ratio` và `FP_ratio`. Điều này cho nhiều quyền kiểm soát hơn, nhưng cũng khiến workflow exploratory cần quét nhiều cấu hình hơn nếu muốn kết luận nghiêm túc.

**Không giải thích nguyên nhân.** Giống UMAP, PaCMAP chỉ giúp nhìn cấu trúc. Nếu embedding cho thấy hai regime tách biệt, vẫn cần probing, causal intervention, hoặc metric analysis để hiểu model thực sự encode điều gì.

---

## 5. Liên hệ với Latent-Anything

PaCMAP là một `Method` thuộc **Layer A — Introspection**: nhận latent và tạo embedding 2D/3D để khám phá cấu trúc, đặc biệt khi cần quan sát đồng thời neighborhood cục bộ và bố cục của toàn bộ manifold mẫu.

```python
class PaCMAPProjection(Protocol):
    n_neighbors: int
    mn_ratio: float
    fp_ratio: float
    init: str
    def fit_transform(self, latent: np.ndarray) -> np.ndarray: ...
```

- **Layer A — Introspection**: so embedding của nhiều checkpoint model, nhiều dataset, hoặc nhiều class để xem global arrangement có đổi hay không.
- **Trajectory primitive**: với rollout dài, PaCMAP hữu ích khi muốn giữ vừa cụm trạng thái cục bộ vừa hình dạng đường đi tổng thể giữa các regime.
- **Layer B — Manipulation**: sau một thao tác steering hay latent edit, có thể xem điểm có nhảy sang cluster khác hay chỉ dịch chuyển trong cùng neighborhood.
- **Method registry**: PaCMAP nên sống cạnh UMAP và PCA như ba baseline visualization có thiên hướng khác nhau; giao diện thống nhất cho phép notebook hoặc UI đổi method nhanh mà không đổi pipeline phân tích.

Trong thực hành của Latent-Anything, PaCMAP đặc biệt phù hợp cho các latent cloud có nhiều cluster nhưng giữa các cluster vẫn có chuyển tiếp mềm. Khi đó UMAP đôi khi làm cụm gọn hơn mức cần thiết, còn t-SNE dễ đẩy chúng xa nhau quá mức; PaCMAP là một trung gian đáng thử trước khi rút kết luận.

---

## Liên quan

- [UMAP Theory](05-umap-theory.md) — công cụ gần nhất để đối chiếu cách giữ local/global structure.
- [Manifold Hypothesis](../../01-space-representation/research/03-manifold-hypothesis.md) — nền trực giác vì sao dữ liệu cao chiều vẫn có thể visualized qua manifold thấp chiều.
- [Curse of Dimensionality](../../01-space-representation/research/04-curse-of-dimensionality.md) — giải thích vì sao việc chọn hàng xóm đáng tin trong không gian cao chiều đã khó ngay từ đầu.
- [Probing Classifiers — Survey](04-probing-classifiers-survey.md) — PaCMAP cho tín hiệu trực quan, probing cho tín hiệu có giám sát về nội dung latent encode.

## Tham khảo

- Y. Wang, H. Huang, C. Rudin, Y. Shaposhnik, *Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization* (JMLR 2021, arXiv:2012.04456).
- Yingfan Wang et al., *PaCMAP* official repository and reference implementation (GitHub).
