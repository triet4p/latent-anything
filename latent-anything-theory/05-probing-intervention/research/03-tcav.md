# Concept Activation Vectors — TCAV

> **TL;DR.** TCAV (Kim et al., 2018) định lượng mức độ một **khái niệm do người định nghĩa** (sọc vằn, "có chấm bi", giới tính…) ảnh hưởng tới một class output. Trước hết học một **Concept Activation Vector (CAV)** $v_C^\ell$ — pháp tuyến của siêu phẳng linear-probe tách activation "có khái niệm" khỏi activation ngẫu nhiên ở layer $\ell$; rồi đo **đạo hàm có hướng** $S_{C,k,\ell}(x)=\nabla h_{\ell,k}\!\cdot v_C^\ell$ xem nhích activation theo $v_C^\ell$ có làm tăng logit của class $k$ không. **TCAV score** = tỉ lệ ảnh trong class $k$ có đạo hàm dương. Caveat chính: CAV bất ổn (phụ thuộc concept dataset, negative set, layer) nên *bắt buộc* kiểm định ý nghĩa thống kê qua hàng trăm CAV ngẫu nhiên.

[Linear probing](01-linear-probing.md) trả lời "khái niệm này có khả truy cập tuyến tính trong latent không". TCAV đi xa hơn một bước quan trọng: nó dùng *chính cái hướng tuyến tính đó* (CAV) để hỏi "model có **dùng** khái niệm này khi quyết định một class không, và mạnh đến đâu". Đây là cầu nối từ probing (thông tin có tồn tại?) sang một thước đo *độ nhạy của output* (model có hành động theo thông tin?), và là method introspection thứ ba của Layer A — đồng thời sinh ra một concept direction dùng được cho Layer B.

---

## **1. Trực giác / Định nghĩa**

Hình dung ta muốn biết: "khi phân loại ảnh *ngựa vằn*, model có thực sự quan tâm đến khái niệm **sọc vằn** không?". Cách làm của TCAV gồm ba nhịp:

1. **Định nghĩa khái niệm bằng ví dụ.** Gom một tập ảnh có sọc (positive $P_C$) và một tập ảnh ngẫu nhiên không liên quan (negative $N$). Khái niệm ở đây không phải một nhãn có sẵn — nó được *người dùng* định nghĩa bằng cách đưa ví dụ.
2. **Tìm hướng của khái niệm trong latent (CAV).** Lấy activation ở một layer $\ell$ của cả hai tập, train một linear classifier tách chúng. Vector pháp tuyến của siêu phẳng đó — hướng vuông góc, chỉ từ "không sọc" sang "có sọc" — chính là **CAV** $v_C^\ell$. Nó là một concept direction, y hệt direction mà [linear probe](01-linear-probing.md) trả về, chỉ khác ở chỗ negative là *ảnh ngẫu nhiên* thay vì lớp đối lập.
3. **Đo độ nhạy của output theo hướng đó.** Với từng ảnh ngựa vằn, hỏi: nếu đẩy activation một chút theo $v_C^\ell$ (làm nó "có sọc hơn"), logit của class *ngựa vằn* tăng hay giảm? Nếu tăng ở phần lớn ảnh, model **nhạy** với khái niệm sọc cho class này.

Điểm cốt lõi phân biệt TCAV với linear probe: linear probe dừng ở nhịp 2 (đo *decodability* của khái niệm). TCAV thêm nhịp 3 — dùng **gradient của model gốc** để đo khái niệm có *ảnh hưởng tới quyết định* không. Đó là khác biệt giữa "thông tin nằm ở đó" và "model lấy thông tin đó ra dùng".

---

## **2. Cơ chế / Công thức**

### 2.1 Concept Activation Vector (CAV)

Gọi $f_\ell: x \mapsto f_\ell(x) \in \mathbb{R}^m$ là ánh xạ từ input tới activation ở layer $\ell$ ($m$ chiều). Cho tập positive $P_C$ (ảnh có khái niệm) và negative $N$ (ảnh ngẫu nhiên), train một binary linear classifier trên $\{f_\ell(x)\}$ để tách hai tập. **CAV** $v_C^\ell$ là vector trọng số (pháp tuyến siêu phẳng) đã chuẩn hóa của classifier đó:

$$ v_C^\ell = \frac{w}{\|w\|}, \qquad w = \arg\min_w \sum_{x} \mathcal{L}\big(\langle w, f_\ell(x)\rangle + b,\ \mathbb{1}[x \in P_C]\big) $$

trong đó $\mathcal{L}$ là loss của linear classifier (logistic hoặc SVM hinge), $\mathbb{1}[x\in P_C]$ là nhãn 1 nếu ảnh thuộc concept set. Kết quả $v_C^\ell$ là *hướng trong không gian activation* đi từ "không có khái niệm" sang "có khái niệm" — một concept direction.

### 2.2 Conceptual sensitivity (đạo hàm có hướng)

Gọi $h_{\ell,k}: \mathbb{R}^m \to \mathbb{R}$ là phần của mạng đi từ activation layer $\ell$ tới **logit của class $k$**. Độ nhạy khái niệm của ảnh $x$ là đạo hàm có hướng của logit theo hướng CAV:

$$ S_{C,k,\ell}(x) = \lim_{\epsilon\to 0}\frac{h_{\ell,k}\big(f_\ell(x)+\epsilon\,v_C^\ell\big)-h_{\ell,k}\big(f_\ell(x)\big)}{\epsilon} = \nabla h_{\ell,k}\big(f_\ell(x)\big)\cdot v_C^\ell $$

trong đó $\nabla h_{\ell,k}$ là gradient của logit class $k$ theo activation. Giá trị $S_{C,k,\ell}(x)>0$ nghĩa là: nhích activation theo hướng "có khái niệm hơn" làm **tăng** logit của class $k$ — tức khái niệm $C$ đẩy ảnh $x$ về phía class $k$.

### 2.3 TCAV score

Gộp độ nhạy trên toàn bộ ảnh của class $k$ (tập $X_k$) thành một con số duy nhất — tỉ lệ ảnh có độ nhạy dương:

$$ \text{TCAV}_{Q_{C,k,\ell}} = \frac{\big|\{x \in X_k : S_{C,k,\ell}(x) > 0\}\big|}{|X_k|} $$

trong đó tử số đếm số ảnh class $k$ mà khái niệm làm tăng logit, mẫu số là tổng số ảnh class $k$. Kết quả nằm trong $[0,1]$: **TCAV score $= 0.9$** nghĩa là với 90% ảnh ngựa vằn, khái niệm "sọc" đẩy logit lên — model nhạy mạnh và *nhất quán* với sọc. Score $\approx 0.5$ nghĩa khái niệm không có ảnh hưởng định hướng (ngẫu nhiên).

### 2.4 Kiểm định ý nghĩa thống kê (bắt buộc)

Một negative set ngẫu nhiên *bất kỳ* vẫn cho ra một CAV và một TCAV score — kể cả khi khái niệm vô nghĩa. Để loại nhiễu, TCAV train **nhiều CAV** (bài gốc dùng ~500) từ các negative set khác nhau, rồi so phân phối TCAV score của khái niệm thật với phân phối từ các **khái niệm ngẫu nhiên** bằng **two-sided t-test**. Một khái niệm chỉ được coi là có ý nghĩa nếu score của nó *khác biệt thống kê* với score của khái niệm ngẫu nhiên. Bỏ qua bước này là dùng TCAV sai.

---

## **3. So sánh với linear probe**

CAV về bản chất *là* một linear-probe direction; điểm mới của TCAV là tầng đạo hàm phía trên.

| | Linear probe | TCAV |
|---|---|---|
| Negative set | Lớp đối lập (nhãn thật) | Ảnh **ngẫu nhiên** không có khái niệm |
| Đại lượng | Accuracy → *decodability* | Đạo hàm có hướng → *độ nhạy output* |
| Câu hỏi | Khái niệm có khả truy cập tuyến tính? | Model có **dùng** khái niệm cho class $k$? |
| Output cho Layer B | Concept direction $W$ | Concept direction $v_C^\ell$ (giống vai trò) |
| Gần với nhân quả? | Không — chỉ tương quan | Gần hơn — đo phản ứng của logit, nhưng vẫn local/linear |
| Kiểm định | Selectivity (control task) | t-test trên ~500 CAV ngẫu nhiên |

Cùng một đối tượng hình học (một hướng tuyến tính), nhưng TCAV hỏi câu hỏi *mạnh hơn về mặt giải thích*: không phải "thông tin nằm đâu" mà "thông tin nào lái output". Đây là lý do TCAV được xếp ngay sau hai note probing trong tầng này.

---

## **4. Giới hạn / Khi nào thất bại**

**Giả định tuyến tính.** CAV giả định khái niệm *tách được tuyến tính* trong không gian activation của layer đã chọn. Khái niệm mã hóa phi tuyến (xem [nonlinear probing](02-nonlinear-probing.md)) cho CAV kém ý nghĩa: hướng pháp tuyến không nắm được khái niệm, và đạo hàm có hướng theo nó gây hiểu lầm.

**Bất ổn của CAV.** Ramaswamy et al. (2023) chỉ ra explanation phụ thuộc nặng vào *probe dataset*: chọn negative set khác, chọn layer khác, hoặc đổi nguồn ảnh khái niệm có thể đảo kết quả. Họ cũng cho thấy nhiều khái niệm trong probe dataset *khó học hơn* chính class chúng định giải thích — nghi ngờ tính đúng đắn của explanation. Hệ quả thực nghiệm: TCAV score có độ lệch chuẩn lớn giữa các lần chạy (báo cáo recall dao động ~19%–78% trên một số setup).

**Indicator gián đoạn → phương sai cao.** TCAV score đếm *dấu* của đạo hàm (hàm chỉ thị $\mathbb{1}[S>0]$), một phép gián đoạn khiến score nhạy với nhiễu quanh ngưỡng 0 và có phương sai không tự triệt tiêu ở vùng tới hạn. Đây là lý do *bắt buộc* dùng nhiều CAV + kiểm định, không bao giờ tin một score đơn lẻ.

**Cần concept dataset do người tạo.** TCAV chỉ kiểm tra được khái niệm mà người dùng *nghĩ ra trước và cung cấp ví dụ*. Nó không tự *khám phá* khái niệm (việc đó cần dictionary learning / **SAE — mục sau, cùng tầng**). Chất lượng explanation bị chặn bởi chất lượng và độ thiên lệch của tập ảnh khái niệm.

**Concept entanglement.** Nếu hai khái niệm tương quan trong dữ liệu (ví dụ "sọc" và "bốn chân"), CAV của chúng chồng lấn, và TCAV score quy cho khái niệm này thực ra có thể đến từ khái niệm kia. Đạo hàm có hướng là *local linear* — không tách được nhân quả khi các hướng khái niệm không trực giao.

**Vẫn không phải nhân quả chặt.** $S_{C,k,\ell}$ là độ nhạy tuyến tính cục bộ quanh điểm $f_\ell(x)$; nhích theo $v_C^\ell$ có thể đưa activation ra ngoài manifold dữ liệu, nơi gradient không phản ánh hành vi thực. TCAV gần nhân quả hơn linear probe nhưng vẫn cần **causal intervention (mục sau, cùng tầng)** như activation patching để khẳng định.

---

## **5. Liên hệ với Latent-Anything**

TCAV là method introspection thứ ba của **Layer A**, đồng thời *sản xuất* nguyên liệu cho **Layer B**:

- **`LatentSpace.cav(concept_pos, concept_neg, layer=ℓ)`** → train linear classifier trên activation, trả về `ConceptDirection{vector=v_C, accuracy}`. Hàm này tổng quát hóa `linear_probe`: chỉ khác ở chỗ negative set là ảnh ngẫu nhiên. CAV trả về dùng được ngay làm trục cho [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) và **steering vector (mục sau, cùng tầng)**.
- **`LatentSpace.tcav_score(class_inputs, cav, target_class, layer=ℓ)`** → tính đạo hàm có hướng qua gradient của adapter, trả về `TCAVResult{score, p_value, significant}`. Việc trả `p_value` (từ t-test trên nhiều CAV ngẫu nhiên) là quyết định thiết kế ép người dùng không đọc score trần — đúng bài học bất ổn từ Ramaswamy.
- **Yêu cầu với `ModelAdapter`**: TCAV cần gradient $\nabla h_{\ell,k}$ chảy *từ activation layer $\ell$ tới logit*. Đây là ràng buộc API: adapter phải expose được forward-from-layer và backward tới một scalar output. Khác với linear probe (chỉ cần activation tĩnh), TCAV cần adapter *khả vi từng phần*.
- **Quy tắc quyết định introspection**: chạy linear/nonlinear probe trước để xác nhận khái niệm *có* và *tuyến tính*; nếu đạt, chạy TCAV để biết khái niệm đó *lái* class nào. Ba note đầu tầng 5 hợp thành một pipeline introspection hoàn chỉnh: tồn tại → dạng → ảnh hưởng.

CAV cũng là một xác nhận trực tiếp cho [giả thuyết hướng tuyến tính](../../03-geometry-structure/research/01-linear-structure.md): nếu một khái niệm người-định-nghĩa cho TCAV score cao và có ý nghĩa thống kê, thì khái niệm đó thực sự tồn tại như một *hướng tuyến tính lái được* trong latent của adapter.

---

## Liên quan

- [Linear probing (mục 01 — tầng này)](01-linear-probing.md) — CAV chính là một linear-probe direction; TCAV thêm tầng đạo hàm có hướng lên trên.
- [Nonlinear probing (mục 02 — tầng này)](02-nonlinear-probing.md) — khi khái niệm mã hóa phi tuyến, giả định tuyến tính của CAV sụp đổ; cần kiểm tra trước.
- [Latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) — CAV là concept direction feed thẳng vào phép số học/steering ở Layer B.
- [Cấu trúc tuyến tính trong latent](../../03-geometry-structure/research/01-linear-structure.md) — TCAV score cao + có ý nghĩa là bằng chứng khái niệm tồn tại như hướng tuyến tính lái được.

## Tham khảo

- B. Kim, M. Wattenberg, J. Gilmer, C. Cai, J. Wexler, F. Viégas, R. Sayres, *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)* (ICML 2018, arXiv:1711.11279). — Paper gốc: định nghĩa CAV, đạo hàm có hướng, TCAV score, và kiểm định t-test trên CAV ngẫu nhiên.
- V. V. Ramaswamy, S. S. Y. Kim, R. Fong, O. Russakovsky, *Overlooked Factors in Concept-Based Explanations: Dataset Choice, Concept Learnability, and Human Capability* (CVPR 2023, arXiv:2207.09615). — Chỉ ra explanation phụ thuộc mạnh vào probe dataset, nhiều khái niệm khó học hơn class chúng giải thích, và giới hạn về số khái niệm con người đọc được.
