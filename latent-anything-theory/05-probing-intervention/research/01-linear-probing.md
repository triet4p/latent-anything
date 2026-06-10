# Linear Probing

> **TL;DR.** Linear probing huấn luyện một classifier *tuyến tính* lên latent đã đóng băng (frozen) để trả lời câu hỏi "feature này có được mã hóa **tuyến tính** trong latent không?". Probe chỉ là một lớp affine $\hat{y} = \text{softmax}(Wz + b)$ học trên cặp $(z, y)$ trong khi model gốc không đổi — accuracy cao nghĩa là thông tin tồn tại và *truy cập được tuyến tính*. Caveat chính: accuracy có thể phản ánh khả năng *ghi nhớ* của probe hơn là thông tin trong representation, nên cần kiểm soát bằng selectivity (Hewitt & Liang).

Latent của một model đã train chứa thông tin gì, và thông tin đó nằm ở dạng dễ hay khó truy cập? Linear probing là công cụ chẩn đoán đơn giản nhất để trả lời: đóng băng model, lấy latent $z$ ở một layer, rồi train một classifier tuyến tính map $z \to y$ với $y$ là thuộc tính ta quan tâm (part-of-speech, sentiment, có-kính-hay-không, class label…). Nếu một lớp tuyến tính *đơn lẻ* đã đủ để đọc ra $y$, thì $y$ được encode tuyến tính trong latent — đúng dạng cấu trúc mà [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) và steering khai thác. Đây là viên gạch nền của Layer A (introspection) trong Latent-Anything.

---

## **1. Trực giác / Định nghĩa**

Hình dung latent space như một thư viện. Probe là một người thủ thư bị *trói tay*: chỉ được phép vẽ một **siêu phẳng thẳng** (hyperplane) để chia không gian thành hai (hoặc nhiều) ngăn. Nếu người thủ thư bị trói tay này vẫn phân loại đúng "sách khoa học vs sách văn học", thì bản thân cách sắp xếp thư viện đã *tuyến tính tách biệt* theo thể loại — model đã làm hết phần việc khó. Ngược lại, nếu thủ thư phải uốn lượn phức tạp mới phân loại được (tức cần **nonlinear probing — mục tiếp theo**), thì thông tin có tồn tại nhưng bị "gói" ở dạng phi tuyến.

Điểm cốt lõi phân biệt probing với việc train một classifier thông thường: **model gốc bị đóng băng**. Ta không học representation — ta *đọc* representation đã có. Probe không được phép thay đổi $z$; gradient của probe không chảy ngược vào model. Vì vậy bất kỳ thông tin nào probe khai thác được đều phải đã có sẵn trong $z$ trước khi probe ra đời.

Alain & Bengio (2016) — người đặt tên "probe" — quan sát rằng độ tách biệt tuyến tính (linear separability) của feature **tăng đơn điệu theo độ sâu** của mạng: probe gắn ở layer càng sâu càng dễ phân loại. Đây là cách dùng kinh điển: gắn probe ở *mọi* layer để xem thông tin lớp nào "chín" ở đâu.

---

## **2. Cơ chế / Công thức**

### 2.1 Mô hình probe

Cho tập latent đã trích xuất $\{z_i\}_{i=1}^N$ với $z_i \in \mathbb{R}^d$ (output của model đóng băng trên input $x_i$) và nhãn $\{y_i\}$. Linear probe là classifier softmax:

$$ \hat{y} = \text{softmax}(Wz + b), \qquad W \in \mathbb{R}^{C \times d},\ b \in \mathbb{R}^{C} $$

trong đó $d$ là chiều latent, $C$ là số lớp của thuộc tính cần dò, $W$ và $b$ là tham số *duy nhất* được học. Probe tối thiểu hóa cross-entropy:

$$ \mathcal{L}(W, b) = -\frac{1}{N}\sum_{i=1}^{N} \log \big[\text{softmax}(W z_i + b)\big]_{y_i} $$

Kết quả $\mathcal{L}$ là negative log-likelihood trung bình; tối thiểu hóa nó đẩy probe gán xác suất cao cho nhãn đúng. Điểm mấu chốt: $z_i$ là **hằng số** trong bài toán này — chỉ $(W, b)$ là biến. Accuracy của probe trên tập test (gọi là *probing accuracy*) là đại lượng ta báo cáo.

**Diễn giải:** probing accuracy cao ⟹ tồn tại một siêu phẳng tách $y$ trong không gian $z$ ⟹ thuộc tính $y$ được mã hóa *tuyến tính khả truy cập* (linearly decodable). Đây chính là điều kiện để [hướng tuyến tính trong latent](../../03-geometry-structure/research/01-linear-structure.md) tồn tại: vector trọng số $W_c$ của lớp $c$ chính là một "hướng khái niệm" (concept direction) — di chuyển $z$ dọc theo $W_c$ làm tăng logit của lớp $c$.

### 2.2 Quy trình chuẩn

1. **Đóng băng** model, chọn layer $\ell$ cần dò.
2. **Trích xuất** $z_i = f_\ell(x_i)$ cho toàn bộ dataset, pooling nếu cần (mean-pool token, CLS token, global average pooling…).
3. **Train** linear probe trên split train, với regularization (weight decay / $\ell_2$) để tránh overfit khi $d$ lớn.
4. **Đo** accuracy trên split test → probing accuracy của layer $\ell$.
5. (Tùy chọn) lặp cho mọi layer để vẽ đường cong "thông tin theo độ sâu".

### 2.3 Selectivity — kiểm soát việc probe tự ghi nhớ

Vấn đề: nếu probe đủ mạnh, nó có thể *tự học* ánh xạ $z \to y$ ngay cả khi $z$ chẳng mã hóa $y$ — probe chỉ đang ghi nhớ. Hewitt & Liang (2019) đề xuất **control task**: gán cho mỗi *word type* (hoặc input type) một nhãn **ngẫu nhiên cố định**, độc lập với ngữ nghĩa. Theo xây dựng, control task chỉ có thể được giải bằng *khả năng ghi nhớ của bản thân probe*. Định nghĩa:

$$ \text{Selectivity} = \text{Acc}_{\text{linguistic task}} - \text{Acc}_{\text{control task}} $$

trong đó số hạng đầu là accuracy trên task thật, số hạng sau là accuracy trên task nhãn-ngẫu-nhiên. Selectivity cao nghĩa là probe giỏi task thật nhưng *dở* task ghi nhớ — tức accuracy thật phản ánh representation, không phải sức mạnh của probe. Linear probe có selectivity cao hơn hẳn MLP probe — đây là **lý do chính** chọn probe tuyến tính: nó *trung thực* hơn về điều nó đo.

---

## **3. Biến thể / So sánh**

| | Linear probe | Nonlinear (MLP) probe |
|---|---|---|
| Kiến trúc | Một lớp affine $Wz+b$ | MLP nhiều lớp + phi tuyến |
| Trả lời câu hỏi | Thông tin có *khả truy cập tuyến tính* không? | Thông tin có *tồn tại* không (bất kể dạng)? |
| Selectivity | Cao — khó ghi nhớ control task | Thấp — dễ ghi nhớ, kết quả khó diễn giải |
| Vai trò | Test cấu trúc hình học của latent | Upper bound cho lượng thông tin |
| Rủi ro | Underestimate (bỏ sót info phi tuyến) | Overestimate (probe tự giải task) |

Hai loại probe trả lời hai câu hỏi *khác nhau* chứ không phải một tốt một xấu: linear probe hỏi về **dạng** mã hóa (geometry), nonlinear probe hỏi về **sự tồn tại** thông tin (upper bound). Một thuộc tính mà nonlinear probe đọc được nhưng linear probe thì không ⟹ thông tin có đó nhưng bị "rối" phi tuyến — tín hiệu quan trọng cho việc quyết định có cần **nonlinear probing (mục tiếp theo)** hay không.

Các biến thể khác: **bilinear probe** (dùng tích bậc hai, vẫn khá selective), **probe theo layer** (vẽ đường cong accuracy theo độ sâu — cách dùng gốc của Alain & Bengio), và **amnesic probing** (xóa thông tin tuyến tính rồi đo lại để kiểm tra tính nhân quả, liên quan tới [subspace projection](../../04-latent-computation/research/04-subspace-projection.md)).

---

## **4. Giới hạn / Khi nào thất bại**

**Probe accuracy ≠ model có *dùng* thông tin đó.** Đây là giới hạn nền tảng nhất, được nhấn mạnh trong survey của Belinkov (2022): probe chỉ chứng minh thông tin *có thể được giải mã* từ latent, không chứng minh model *thực sự sử dụng* nó cho output. Tương quan trong probe không phải nhân quả — cần **causal intervention (mục sau, cùng tầng)** như activation patching để khẳng định.

**Memorization của probe.** Nếu không kiểm soát bằng selectivity, accuracy cao có thể chỉ là probe ghi nhớ. Probe càng mạnh (nhiều tham số) càng dễ ngụy tạo "thông tin" không có thực trong representation. Đây là lý do cộng đồng chuyển sang đo selectivity thay vì accuracy trần.

**Probing classifier không đáng tin cho concept removal/detection.** Ravfogel/Elazar et al. (NeurIPS 2022) chỉ ra rằng probe có thể *phát hiện* và bị dùng để *xóa* một khái niệm một cách thiếu nhất quán: xóa hướng tuyến tính mà probe tìm được không đảm bảo khái niệm thực sự biến mất khỏi representation (thông tin có thể được mã hóa dư thừa ở hướng khác).

**Nhạy cảm với chi tiết thực nghiệm.** Kết quả phụ thuộc mạnh vào: cách pooling latent, regularization, kích thước tập train của probe, và layer được chọn. Hai paper dùng cùng model có thể ra kết luận trái ngược chỉ vì khác setup probe.

**Underestimate khi info phi tuyến.** Linear probe theo định nghĩa mù với thông tin chỉ tách được phi tuyến. Một latent "kém" theo linear probe vẫn có thể giàu thông tin — cần đối chiếu với nonlinear probe trước khi kết luận.

---

## **5. Liên hệ với Latent-Anything**

Linear probing là **method đầu tiên của Layer A (introspection)** — đúng checkpoint mà [THEORY.md](https://github.com/triet4p/latent-anything/blob/main/docs/THEORY.md) đặt ra: "sau tầng 5 → có thể implement Layer A đầu tiên, rút ra `Method` interface". Nó định hình API:

- **`LatentSpace.linear_probe(labels, layer=ℓ)`** → fit probe, trả về `ProbeResult{accuracy, selectivity, direction=W}`. Việc trả về cả `selectivity` (cần một control task) là quyết định thiết kế ép người dùng không đọc accuracy trần.
- **`ProbeResult.direction`** chính là concept direction $W_c$ — nối thẳng sang Layer B: dùng làm trục cho [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md), cho [subspace projection](../../04-latent-computation/research/04-subspace-projection.md) (decompose $z = z_{\text{concept}} + z_{\text{residual}}$), và là tiền thân của **steering vector** (**mục sau, cùng tầng**).
- **Probe theo layer** → khảo sát một adapter: thông tin ngữ nghĩa chín ở layer nào, từ đó chọn layer để introspect/manipulate. Đây là chẩn đoán chuẩn khi tích hợp một `ModelAdapter` mới.
- **`Method` interface**: linear probe — fit trên frozen latent, trả direction + metric — là instance cụ thể đầu tiên để trừu tượng hóa thành interface chung cho mọi introspection method (probe, TCAV, SAE…).

Probe tuyến tính cũng là *bài kiểm tra thực nghiệm* cho [giả thuyết hướng tuyến tính](../../03-geometry-structure/research/01-linear-structure.md): nếu probe tuyến tính đạt accuracy cao và selectivity cao, giả thuyết được xác nhận cho thuộc tính đó trong adapter cụ thể.

---

## Liên quan

- [Cấu trúc tuyến tính trong latent](../../03-geometry-structure/research/01-linear-structure.md) — linear probing là cách *kiểm chứng thực nghiệm* giả thuyết hướng tuyến tính; vector $W_c$ của probe là một concept direction.
- [Latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) — concept direction tìm bằng probe là nguyên liệu cho phép số học vector.
- [Subspace projection](../../04-latent-computation/research/04-subspace-projection.md) — dùng direction của probe để tách $z$ thành thành phần concept và residual; amnesic probing xóa subspace tuyến tính.

## Tham khảo

- G. Alain, Y. Bengio, *Understanding Intermediate Layers Using Linear Classifier Probes* (ICLR Workshop 2017, arXiv:1610.01644). — Paper đặt tên "probe"; quan sát linear separability tăng đơn điệu theo độ sâu mạng.
- J. Hewitt, P. Liang, *Designing and Interpreting Probes with Control Tasks* (EMNLP 2019, arXiv:1909.03368). — Đề xuất control task và selectivity; chứng minh linear probe selective hơn MLP probe.
- Y. Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances* (Computational Linguistics 48(1), 2022, arXiv:2102.12452). — Survey nền tảng; nhấn mạnh probe accuracy không chứng minh model *dùng* thông tin.
- A. Ravfogel, M. Elazar, et al., *Probing Classifiers are Unreliable for Concept Removal and Detection* (NeurIPS 2022). — Giới hạn của probe khi dùng để xóa/phát hiện khái niệm.
