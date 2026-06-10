# Nonlinear Probing

> **TL;DR.** Nonlinear probing thay classifier tuyến tính bằng một **MLP** $\hat{y}=g_\theta(z)$ huấn luyện trên latent đóng băng — trả lời câu hỏi "thông tin này có *tồn tại* trong latent không?" bất kể nó được mã hóa tuyến tính hay phi tuyến, do đó là **upper bound** cho lượng thông tin khả giải mã. Cái giá: một probe đủ mạnh có thể tự giải task từ đầu thay vì đọc representation — selectivity sụt (Hewitt & Liang), khiến accuracy khó diễn giải. Hai trường phái đối nghịch: dùng probe đơn giản nhất (selectivity) vs dùng probe mạnh nhất rồi đo bằng MDL / mutual information (Pimentel; Voita & Titov).

[Linear probing](01-linear-probing.md) chỉ phát hiện được thông tin nằm ở dạng *tuyến tính khả truy cập*. Nhưng một latent có thể chứa thông tin được "gói" phi tuyến: linear probe trả về accuracy thấp không có nghĩa thông tin vắng mặt, chỉ có nghĩa nó không tách được bằng một siêu phẳng. Nonlinear probing dùng một classifier có khả năng biểu diễn cao hơn — thường là MLP — để hỏi câu hỏi yếu hơn nhưng quan trọng: *thông tin đó có ở trong latent hay không, dưới bất kỳ dạng nào?* Đây là viên gạch thứ hai của Layer A (introspection), bổ sung chứ không thay thế linear probe.

---

## **1. Trực giác / Định nghĩa**

Quay lại ẩn dụ người thủ thư ở [note linear probing](01-linear-probing.md): linear probe là thủ thư *bị trói tay*, chỉ được vẽ một vách ngăn thẳng. Nonlinear probe **cởi trói** cho thủ thư — giờ họ được vẽ vách ngăn cong, uốn lượn tùy ý để gom sách cùng loại. Hệ quả kép:

- **Mặt lợi:** nếu thông tin tồn tại dưới *bất kỳ* hình dạng nào (kể cả XOR, vòng tròn đồng tâm, cụm rời rạc), thủ thư khéo tay sẽ tìm ra. Accuracy của nonlinear probe vì thế là **chặn trên** (upper bound) cho lượng thông tin mà linear probe có thể đạt: nó luôn ≥ linear probe.
- **Mặt hại:** thủ thư quá khéo có thể *tự sắp xếp lại* sách trong đầu — tức probe học task từ representation nghèo nàn, làm ta nhầm tưởng latent giàu thông tin. Ranh giới giữa "đọc thông tin có sẵn" và "tự tính ra thông tin" bị xóa nhòa.

Câu hỏi mà hai loại probe trả lời khác nhau về bản chất:

| | Linear probe | Nonlinear (MLP) probe |
|---|---|---|
| Câu hỏi | Thông tin có *khả truy cập tuyến tính*? | Thông tin có *tồn tại* (bất kể dạng)? |
| Vai trò | Test **dạng hình học** của mã hóa | **Upper bound** lượng thông tin |
| Selectivity | Cao | Thấp — dễ tự giải task |
| Rủi ro diễn giải | Underestimate (bỏ sót info phi tuyến) | Overestimate (probe tự tính) |
| Khi nào tin được | Gần như luôn | Cần kiểm soát bằng control task / MDL |

---

## **2. Cơ chế / Công thức**

### 2.1 Mô hình probe

Cho latent đóng băng $\{z_i\} \subset \mathbb{R}^d$ và nhãn $\{y_i\}$. Nonlinear probe là một MLP:

$$ \hat{y} = g_\theta(z) = \text{softmax}\big(W_2\,\sigma(W_1 z + b_1) + b_2\big) $$

trong đó $\sigma$ là hàm phi tuyến (ReLU/tanh), $W_1 \in \mathbb{R}^{h \times d}$, $W_2 \in \mathbb{R}^{C \times h}$ là trọng số của lớp ẩn $h$ chiều và lớp output $C$ lớp; $\theta = \{W_1, b_1, W_2, b_2\}$ là toàn bộ tham số probe. So với linear probe ($\hat{y}=\text{softmax}(Wz+b)$), điểm khác duy nhất là **lớp ẩn phi tuyến** $\sigma(W_1 z + b_1)$ — nó cho phép probe vẽ ranh giới quyết định cong. Vẫn tối thiểu hóa cross-entropy, vẫn giữ $z$ **đóng băng** (gradient không chảy vào model gốc).

**Diễn giải:** vì lớp tuyến tính là trường hợp đặc biệt của MLP (đặt $\sigma = $ identity, $h$ đủ lớn), nên $\text{Acc}_{\text{MLP}} \ge \text{Acc}_{\text{linear}}$ luôn đúng trên cùng dữ liệu (bỏ qua nhiễu tối ưu hóa). Khoảng cách $\text{Acc}_{\text{MLP}} - \text{Acc}_{\text{linear}}$ chính là lượng thông tin được mã hóa *phi tuyến* — tín hiệu cho biết latent space có cấu trúc cong tới mức nào với thuộc tính đó.

### 2.2 Diagnostic: chênh lệch linear–nonlinear nói lên điều gì

| $\text{Acc}_{\text{linear}}$ | $\text{Acc}_{\text{MLP}}$ | Kết luận về latent |
|---|---|---|
| Cao | Cao | Thông tin có, **mã hóa tuyến tính** — lý tưởng cho arithmetic/steering |
| Thấp | Cao | Thông tin có nhưng **gói phi tuyến** — cần decode phi tuyến, không steer thẳng được |
| Thấp | Thấp | Thông tin **không có** (hoặc latent quá nghèo) — đổi layer hoặc đổi adapter |
| Cao | Thấp | Hầu như không xảy ra (mâu thuẫn với upper-bound) → nghi ngờ lỗi train probe |

Đây là lý do hai probe luôn đi cặp: linear xác định *dạng*, nonlinear xác định *trần*. Đọc một mình loại nào cũng cho kết luận lệch.

### 2.3 Vấn đề cốt lõi: probe complexity và selectivity

Càng tăng capacity của probe, accuracy càng cao — nhưng một phần là vì probe *tự ghi nhớ/tự tính*, không phải vì latent giàu hơn. Hewitt & Liang (2019) đo điều này bằng **control task** (gán nhãn ngẫu nhiên cố định theo *type*) và **selectivity**:

$$ \text{Selectivity} = \text{Acc}_{\text{task thật}} - \text{Acc}_{\text{control task}} $$

trong đó số hạng sau là accuracy trên task nhãn-ngẫu-nhiên (chỉ giải được bằng khả năng ghi nhớ của probe). Kết quả thực nghiệm của họ: chuyển từ linear sang MLP probe cho POS-tagging chỉ tăng accuracy *một chút* nhưng **mất selectivity đáng kể** — phần accuracy tăng thêm đó không phản ánh trung thực representation. Đây là cảnh báo trung tâm khi dùng nonlinear probe.

---

## **3. Hai trường phái xử lý complexity**

Cộng đồng chia làm hai cách trả lời câu hỏi "probe nên mạnh đến đâu":

### 3.1 Trường phái selectivity — chọn probe đơn giản nhất (Hewitt & Liang)

Quan điểm: probe càng đơn giản, accuracy của nó càng *trung thực* phản ánh representation. Nên ưu tiên linear probe; chỉ dùng nonlinear khi có lý do, và luôn báo cáo selectivity kèm theo. Một probe mạnh với selectivity thấp là probe không đáng tin.

### 3.2 Trường phái information-theoretic — chọn probe mạnh nhất, đổi thước đo

Pimentel et al. (2020) lập luận ngược: nếu mục tiêu là ước lượng **mutual information** $I(Z; Y)$ giữa representation và nhãn, thì theo data-processing inequality, probe mạnh hơn luôn cho ước lượng *chặt hơn* và lộ ra nhiều thông tin hơn — vậy **nên dùng probe mạnh nhất có thể**, kể cả phức tạp. Nghịch lý "probe tự tính" được giải quyết không bằng cách giới hạn probe, mà bằng cách đổi đại lượng báo cáo.

Voita & Titov (2020) cụ thể hóa bằng **Minimum Description Length (MDL)**: thay vì đo accuracy, đo *độ dài mô tả* của nhãn khi đã biết representation — tức chi phí để "truyền" nhãn $Y$ qua $Z$. MDL gộp cả chất lượng probe *và* "công sức" cần để đạt nó. Ước lượng bằng **variational coding** hoặc **online coding** (huấn luyện probe trên các khối dữ liệu lớn dần, cộng dồn log-loss). Ưu điểm: MDL phân biệt rõ pretrained vs random representation (điều mà accuracy thường không làm được), ổn định hơn, và không cần dò tay control task.

| | Selectivity (Hewitt & Liang) | MDL / MI (Pimentel; Voita & Titov) |
|---|---|---|
| Triết lý | Probe đơn giản → trung thực | Probe mạnh → ước lượng MI chặt |
| Thước đo | Accuracy + selectivity | Description length / mutual info |
| Xử lý "probe tự tính" | Giới hạn capacity probe | Tính cả chi phí học vào thước đo |
| Phân biệt pretrained vs random | Qua control task | Trực tiếp, ổn định hơn |

Hai trường phái không loại trừ nhau: selectivity là kiểm tra sạch và rẻ; MDL là thước đo nguyên lý hơn khi cần so sánh representation định lượng.

---

## **4. Giới hạn / Khi nào thất bại**

**Mất selectivity → accuracy khó diễn giải.** Đây là giới hạn nền tảng: MLP probe có thể đạt accuracy cao trên cả task thật lẫn control task, nên một con số accuracy đơn lẻ không cho biết thông tin nằm trong latent hay do probe tự tính. Không bao giờ báo cáo nonlinear-probe accuracy trần — luôn kèm selectivity hoặc MDL.

**Nhạy với capacity và siêu tham số.** Kết quả phụ thuộc mạnh vào số lớp ẩn, $h$, regularization, learning rate, số epoch. Một MLP probe quá lớn sẽ overfit và thổi phồng accuracy; quá nhỏ thì không tận dụng được khả năng phi tuyến. Cần dò cẩn thận — chính sự "cần dò tay" này là một lý do Voita & Titov đề xuất MDL.

**Vẫn chỉ chứng minh *decodability*, không chứng minh *causality*.** Giống linear probe, nonlinear probe cao chỉ nói thông tin *có thể giải mã được*, không nói model *thực sự dùng* nó cho output. Cần **causal intervention (mục sau, cùng tầng)** như activation patching để khẳng định nhân quả.

**Không cho ra "concept direction" dùng được.** Linear probe tặng kèm một vector $W$ làm trục steering/arithmetic. Nonlinear probe chỉ cho một hàm $g_\theta$ — không có một hướng tuyến tính duy nhất để di chuyển latent, nên kết quả của nó *chẩn đoán* tốt nhưng khó *can thiệp* trực tiếp. Đây là lý do thực tiễn để vẫn ưu tiên linear probe cho Layer B.

**Upper bound chỉ tương đối.** "Upper bound" của MLP probe bị chặn bởi capacity hữu hạn và khó khăn tối ưu hóa — một thông tin tồn tại nhưng cần phi tuyến cực phức tạp vẫn có thể vượt khả năng của MLP probe thực tế. Accuracy thấp ở cả hai probe không tuyệt đối chứng minh thông tin vắng mặt.

---

## **5. Liên hệ với Latent-Anything**

Nonlinear probing là method introspection thứ hai của **Layer A**, định hình `Method` interface cùng với linear probe:

- **`LatentSpace.nonlinear_probe(labels, layer=ℓ, hidden=h)`** → fit MLP probe, trả về `ProbeResult{accuracy, selectivity, mdl}`. Việc trả `mdl` (hoặc ít nhất selectivity) là quyết định thiết kế ép người dùng không đọc accuracy trần — đúng bài học từ Hewitt & Liang.
- **Diagnostic linear-vs-nonlinear**: chạy cả hai probe trên cùng layer, báo cáo cặp $(\text{Acc}_{\text{linear}}, \text{Acc}_{\text{MLP}})$. Bảng ở mục 2.2 trở thành một *quy tắc quyết định* trong API: nếu linear thấp & nonlinear cao → cảnh báo người dùng rằng thuộc tính này **không steer tuyến tính được**, đừng dùng [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) cho nó.
- **Chọn layer để introspect**: như linear probe theo độ sâu, nonlinear probe theo layer cho biết layer nào *chứa* thông tin (upper bound), trong khi linear probe cho biết layer nào *trình bày tuyến tính* nó. Hai đường cong cùng nhau định vị layer tốt nhất cho cả introspection lẫn manipulation.
- **Cảnh báo cho manipulation**: vì nonlinear probe không cho concept direction, nó không feed thẳng vào Layer B. Vai trò của nó là *gác cổng*: xác nhận thông tin tồn tại trước khi tốn công tìm hướng tuyến tính bằng linear probe hoặc **steering vector (mục sau, cùng tầng)**.

Bộ đôi linear + nonlinear probe là *bài kiểm tra thực nghiệm* hoàn chỉnh cho [giả thuyết hướng tuyến tính](../../03-geometry-structure/research/01-linear-structure.md): khoảng cách giữa hai accuracy đo trực tiếp mức độ một thuộc tính lệch khỏi mã hóa tuyến tính lý tưởng.

---

## Liên quan

- [Linear probing (mục 01 — tầng này)](01-linear-probing.md) — probe đơn giản, selective, tặng kèm concept direction; nonlinear probe là phần bù cho nó (upper bound).
- [Cấu trúc tuyến tính trong latent](../../03-geometry-structure/research/01-linear-structure.md) — chênh lệch linear–nonlinear đo mức độ thuộc tính lệch khỏi mã hóa tuyến tính.
- [Latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) — chỉ dùng được khi linear probe đã cao; nonlinear-cao-mà-linear-thấp là tín hiệu *không* nên arithmetic.

## Tham khảo

- J. Hewitt, P. Liang, *Designing and Interpreting Probes with Control Tasks* (EMNLP 2019, arXiv:1909.03368). — Control task và selectivity; chứng minh chuyển sang MLP probe làm sụt selectivity dù tăng nhẹ accuracy.
- T. Pimentel, J. Valvoda, R. Hall Maudslay, R. Zmigrod, A. Williams, R. Cotterell, *Information-Theoretic Probing for Linguistic Structure* (ACL 2020, arXiv:2004.03061). — Lập luận nên dùng probe mạnh nhất để ước lượng mutual information $I(Z;Y)$ chặt hơn.
- E. Voita, I. Titov, *Information-Theoretic Probing with Minimum Description Length* (EMNLP 2020, arXiv:2003.12298). — MDL probing (variational + online coding); ổn định hơn accuracy và phân biệt rõ pretrained vs random representation.
- G. Alain, Y. Bengio, *Understanding Intermediate Layers Using Linear Classifier Probes* (ICLR Workshop 2017, arXiv:1610.01644). — Nền tảng khái niệm probe; cơ sở để so sánh linear vs nonlinear theo độ sâu.
