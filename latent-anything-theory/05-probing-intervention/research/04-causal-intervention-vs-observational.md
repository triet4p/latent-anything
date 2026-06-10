# Causal Intervention vs Observational Study

> **TL;DR.** Một probe đo *tương quan* — $P(Y\mid Z)$, "đọc được khái niệm từ latent" — nhưng tương quan không phải nhân quả: nó không nói model có **dùng** khái niệm đó để ra quyết định hay không. Câu hỏi nhân quả cần phân phối **can thiệp** $P(Y\mid do(Z=z))$, đo bằng cách *trực tiếp chỉnh sửa* representation rồi xem output đổi ra sao, thay vì chỉ quan sát. Caveat chính: can thiệp trong latent dễ đẩy điểm ra ngoài manifold dữ liệu và vướng đánh đổi completeness–selectivity, nên "intervention" không tự động đáng tin.

Cả ba note trước trong tầng này — [linear probing](01-linear-probing.md), [nonlinear probing](02-nonlinear-probing.md), [TCAV](03-tcav.md) — đều kết thúc bằng cùng một cảnh báo: probe chứng minh thông tin *giải mã được* (decodable), không chứng minh model *sử dụng* nó. Đó chính là khoảng cách giữa **observational** (quan sát) và **interventional** (can thiệp). Note này hình thức hóa khoảng cách đó bằng ngôn ngữ nhân quả của Pearl, rồi chỉ ra cách "can thiệp" được thực thi trong latent space — nền tảng lý thuyết cho **activation patching (mục sau, cùng tầng)** và mọi method manipulation của Layer B.

---

## **1. Trực giác / Định nghĩa**

Ẩn dụ kinh điển: số vụ chết đuối và doanh số kem tương quan dương rất mạnh. Quan sát ($P(\text{đuối}\mid \text{kem cao})$) gợi ý kem gây chết đuối. Nhưng cả hai đều do một **biến gây nhiễu** (confounder) chung — *mùa hè nóng*. Nếu ta **can thiệp** ép tăng doanh số kem giữa mùa đông ($do(\text{kem}=\text{cao})$), số chết đuối không đổi. Tương quan biến mất khi cắt đứt confounder.

Trong latent space, kịch bản y hệt: một probe phát hiện chiều $z_c$ tương quan với khái niệm "giới tính" và khái niệm này tương quan với output "nghề nghiệp". Nhưng model có thực sự *dùng* $z_c$ để dự đoán nghề nghiệp, hay cả hai chỉ cùng bị lái bởi một feature thứ ba? Cách duy nhất để biết: **ép** $z_c$ về một giá trị (xóa hoặc đảo nó), giữ nguyên phần còn lại, rồi xem output có đổi không. Nếu output đổi → model dùng $z_c$ một cách nhân quả. Nếu không → probe chỉ bắt được tương quan ăn theo.

Khác biệt cốt lõi:

- **Observational** $P(Y\mid Z=z)$: "trong các mẫu mà $Z$ *tình cờ* bằng $z$, $Y$ phân phối thế nào". Bao gồm cả ảnh hưởng của confounder.
- **Interventional** $P(Y\mid do(Z=z))$: "nếu ta *cưỡng bức* $Z=z$ bất kể nguyên nhân tự nhiên của nó, $Y$ phân phối thế nào". Đã cắt confounder.

**Confounding** chính là độ chênh giữa hai phân phối này.

---

## **2. Cơ chế / Công thức**

### 2.1 Structural Causal Model và toán tử do

Một **Structural Causal Model (SCM)** mô tả mỗi biến bằng một phương trình cấu trúc theo cha (parents) và nhiễu ngoại sinh:

$$ Z := f_Z(\text{PA}_Z, U_Z), \qquad Y := f_Y(\text{PA}_Y, U_Y) $$

trong đó $\text{PA}_Z$ là tập biến cha (nguyên nhân trực tiếp) của $Z$, $U_Z$ là nhiễu độc lập. Các phương trình này quy định một đồ thị nhân quả có hướng (DAG).

**Toán tử $do(Z=z)$** thực hiện *phẫu thuật đồ thị* (graph surgery): thay phương trình $Z := f_Z(\cdots)$ bằng hằng số $Z := z$, tức **xóa mọi mũi tên đi vào $Z$**. Điều này cắt đứt mọi đường confounding chạy qua các cha của $Z$:

$$ P(Y\mid do(Z=z)) \neq P(Y\mid Z=z) \quad\text{(nói chung)} $$

Vế trái đo hiệu ứng nhân quả thuần (đã cắt confounder); vế phải là quan sát thường (còn nhiễm confounder). Hai vế chỉ bằng nhau khi không có confounder giữa $Z$ và $Y$.

### 2.2 Backdoor adjustment — ước lượng can thiệp từ dữ liệu quan sát

Nếu ta xác định được một tập biến $W$ *chặn mọi đường cửa sau* (backdoor path) giữa $Z$ và $Y$ (tiêu chuẩn backdoor của Pearl), thì có thể tính phân phối can thiệp chỉ từ dữ liệu quan sát:

$$ P(Y\mid do(Z=z)) = \sum_{w} P(Y\mid Z=z, W=w)\,P(W=w) $$

trong đó $W$ là tập điều chỉnh (adjustment set) chặn confounding, tổng lấy trên mọi giá trị của $W$. Công thức này nói: nếu *biết* và *điều kiện hóa* đúng các confounder $W$, hiệu ứng nhân quả trở nên ước lượng được mà không cần thí nghiệm thực. Vấn đề thực tế: trong deep learning ta hiếm khi biết đầy đủ $W$.

### 2.3 Can thiệp *trực tiếp* trong latent space

Điểm mạnh đặc biệt của latent space: ta không cần backdoor adjustment vì có thể thực thi $do(\cdot)$ **trực tiếp**. Model cho ta toàn quyền ghi đè representation, điều bất khả thi với biến vật lý ngoài đời:

$$ z' = z - (w_c^\top z)\,w_c + v\,w_c, \qquad \hat{Y}_{\text{do}} = h(z') $$

trong đó $w_c$ là hướng khái niệm (concept direction, ví dụ từ probe/CAV), $w_c^\top z$ là thành phần hiện tại của $z$ theo hướng đó, $v$ là giá trị ta *cưỡng bức*, và $h$ là phần model từ latent tới output. Đặt $v=0$ tức **xóa** khái niệm (amnesic), đổi dấu $v$ tức **đảo** nó (counterfactual). So sánh $h(z')$ với $h(z)$ cho ta ảnh hưởng nhân quả của khái niệm lên output — đây chính là $do(Z_c=v)$ thực thi bằng phẫu thuật trên vector.

### 2.4 Hai họ can thiệp trong latent

| | Amnesic / erasure | Counterfactual / steering |
|---|---|---|
| Thao tác | Xóa thông tin khái niệm khỏi $z$ | Ép khái niệm sang giá trị khác |
| Câu hỏi | "Nếu model *không biết* khái niệm thì sao?" | "Nếu khái niệm *ngược lại* thì sao?" |
| Kỹ thuật | INLP (nullspace projection), LEACE | Thêm $\alpha w_c$, swap activation |
| Đo | Sụt accuracy task → mức độ phụ thuộc | Đổi output → hướng ảnh hưởng |

**Amnesic Probing** (Elazar et al., 2021) thực thi họ thứ nhất: xóa khái niệm tuyến tính khỏi representation bằng INLP rồi đo task chính sụt bao nhiêu — nếu sụt mạnh, khái niệm *được dùng*; nếu không đổi, probe chỉ bắt tương quan. Đây là phép kiểm nhân quả mà probe thường không có.

---

## **3. So sánh observational và interventional probing**

| | Observational probe | Interventional probe |
|---|---|---|
| Đại lượng | $P(Y\mid Z)$ — decodability | $P(Y\mid do(Z))$ — causal effect |
| Trả lời | Thông tin có *trong* latent? | Model có *dùng* thông tin? |
| Nhiễu confounder | Có — không tách được | Cắt bằng phẫu thuật trên $z$ |
| Ví dụ method | [linear probe](01-linear-probing.md), [TCAV](03-tcav.md) | Amnesic probing, activation patching |
| Rủi ro | Overclaim ("model hiểu X") | Off-manifold, completeness–selectivity |
| Bằng chứng | Yếu (tương quan) | Mạnh hơn (nhưng không tuyệt đối) |

TCAV nằm ở giữa: đạo hàm có hướng đo *phản ứng cục bộ* của logit — gần với can thiệp hơn probe thuần, nhưng vẫn là xấp xỉ tuyến tính địa phương, không phải phẫu thuật $do(\cdot)$ đầy đủ. Bậc thang chặt chẽ tăng dần: probe → TCAV → can thiệp trực tiếp.

---

## **4. Giới hạn / Khi nào thất bại**

**Can thiệp off-manifold.** Ghi đè $z_c := v$ có thể đẩy $z'$ ra vùng mà encoder thật không bao giờ sinh ra — *ngoài manifold dữ liệu*. Khi đó $h(z')$ chạy trên input phi thực tế, và thay đổi output có thể là *artifact* của vùng latent vô nghĩa chứ không phải hiệu ứng nhân quả thật. Đây là phiên bản latent của "intervention làm hỏng cấu trúc" — liên quan trực tiếp tới [giả thuyết manifold](../../01-space-representation/research/03-manifold-hypothesis.md).

**Xóa tuyến tính không hoàn toàn.** Một khái niệm thường được mã hóa *dư thừa* trên nhiều hướng (đã thấy ở [linear probing](01-linear-probing.md) và [subspace projection](../../04-latent-computation/research/04-subspace-projection.md)). Xóa một hướng (hoặc cả không gian con tuyến tính bằng INLP/LEACE) vẫn có thể chừa lại thông tin ở dạng phi tuyến — can thiệp *chưa hoàn chỉnh*, kết luận nhân quả bị thiên lệch.

**Đánh đổi completeness–selectivity.** Canby et al. (2024) chỉ ra mọi causal probing method đều vướng đánh đổi giữa **completeness** (xóa/đổi khái niệm mục tiêu *triệt để* đến đâu) và **selectivity** (ít làm hỏng khái niệm *khác* đến đâu); họ định nghĩa **reliability** là trung bình điều hòa của hai đại lượng này. Can thiệp mạnh tay (completeness cao) thường kéo theo tổn hại lan sang feature khác (selectivity thấp). Đáng chú ý: họ thấy **can thiệp phi tuyến thường đáng tin hơn can thiệp tuyến tính** — ngược với trực giác "đơn giản thì sạch".

**Confounder ẩn không quan sát được.** Backdoor adjustment chỉ đúng khi tập điều chỉnh $W$ chặn *mọi* đường cửa sau. Trong deep net, các nhân tố sinh dữ liệu phần lớn không quan sát được, nên đảm bảo này gần như không bao giờ thỏa hoàn toàn — ngay cả can thiệp trực tiếp cũng có thể vô tình đổi một feature tương quan mà ta không biết (concept entanglement).

**Hiệu ứng cục bộ không tổng quát.** Một can thiệp tại một điểm $z$ chỉ nói về lân cận điểm đó; ảnh hưởng nhân quả có thể khác ở vùng latent khác. Một TCAV/intervention score gộp toàn class che giấu sự không đồng nhất này.

---

## **5. Liên hệ với Latent-Anything**

Note này là *bản lề lý thuyết* giữa Layer A (introspection) và Layer B (manipulation): nó giải thích vì sao introspection không thể dừng ở probe, và vì sao mọi thao tác manipulation thực ra là một phép $do(\cdot)$ trên latent.

- **`LatentSpace.intervene(z, direction, value)`** → thực thi $z' = z - (w_c^\top z)w_c + v\,w_c$, trả về latent đã can thiệp. Đây là primitive chung cho cả erasure ($v=0$) và steering ($v\neq 0$), và là hiện thân trực tiếp của toán tử $do$.
- **`LatentSpace.causal_effect(z, direction, target)`** → đo $\lVert h(z') - h(z)\rVert$ (hoặc đổi logit/label) làm ước lượng hiệu ứng nhân quả. Trả kèm cảnh báo *off-manifold* (đo bằng [Mahalanobis distance](../../04-latent-computation/research/05-mahalanobis-distance.md) tới phân phối training) — nếu $z'$ quá xa manifold, kết quả không tin được.
- **Erasure pipeline** (amnesic): `LatentSpace.erase(concept, method="inlp"|"leace")` → xóa khái niệm tuyến tính; đo task sụt để xác nhận model có dùng nó. Trực tiếp dùng lại cơ chế [subspace projection](../../04-latent-computation/research/04-subspace-projection.md).
- **Báo cáo reliability**, không chỉ effect: theo Canby et al., `causal_effect` nên kèm `completeness` và `selectivity` để người dùng biết can thiệp có "sạch" không — đúng tinh thần "không đọc một con số trần" đã thiết lập từ [linear probing](01-linear-probing.md).

Quan trọng nhất: note này nâng cấp toàn bộ pipeline introspection từ *mô tả* (probe nói thông tin nằm đâu) lên *giải thích nhân quả* (can thiệp nói thông tin nào lái hành vi) — điều kiện cần để Layer B chỉnh sửa latent một cách *có cơ sở*, không phải đoán mò. Method tiếp theo, **activation patching**, là một dạng can thiệp đặc thù: thay vì chỉnh theo một hướng, nó *ghép* activation từ một run khác vào, hiện thực hóa counterfactual ở mức từng thành phần.

---

## Liên quan

- [Linear probing (mục 01 — tầng này)](01-linear-probing.md) — observational baseline; can thiệp là câu trả lời cho cảnh báo "probe không chứng minh model dùng thông tin".
- [TCAV (mục 03 — tầng này)](03-tcav.md) — đạo hàm có hướng là xấp xỉ can thiệp cục bộ, nằm giữa observational và interventional.
- [Subspace projection](../../04-latent-computation/research/04-subspace-projection.md) — cơ chế kỹ thuật của erasure/amnesic: xóa khái niệm = chiếu bỏ không gian con.
- [Giả thuyết manifold](../../01-space-representation/research/03-manifold-hypothesis.md) — lý do can thiệp off-manifold cho kết quả không tin được.
- [Mahalanobis distance (tầng 4)](../../04-latent-computation/research/05-mahalanobis-distance.md) — đo độ xa manifold để gác cổng can thiệp.

## Tham khảo

- J. Pearl, *Causality: Models, Reasoning, and Inference* (Cambridge University Press, 2nd ed. 2009). — Nguồn gốc toán tử $do$, SCM, tiêu chuẩn backdoor và do-calculus.
- S. Ravfogel, Y. Elazar, H. Gonen, M. Twiton, Y. Goldberg, *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection* (ACL 2020, arXiv:2004.07667). — INLP: xóa thông tin tuyến tính bằng chiếu nullspace lặp; nền tảng cho can thiệp erasure.
- Y. Elazar, S. Ravfogel, A. Jacovi, Y. Goldberg, *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals* (TACL 2021, arXiv:2006.00995). — Đo *việc sử dụng* thông tin bằng cách xóa nó rồi quan sát behavior; phân biệt rõ decodability với usage.
- N. Belrose, D. Schneider-Joseph, S. Ravfogel, R. Cotterell, E. Raff, S. Biderman, *LEACE: Perfect Linear Concept Erasure in Closed Form* (NeurIPS 2023, arXiv:2306.03819). — Xóa khái niệm tuyến tính tối ưu dạng đóng, ít làm hỏng phần còn lại hơn INLP.
- M. Canby, A. Davies, C. Rastogi, J. Hockenmaier, *How Reliable are Causal Probing Interventions?* (arXiv:2408.15510, 2024). — Đánh đổi completeness–selectivity; reliability là trung bình điều hòa; can thiệp phi tuyến thường đáng tin hơn tuyến tính.
