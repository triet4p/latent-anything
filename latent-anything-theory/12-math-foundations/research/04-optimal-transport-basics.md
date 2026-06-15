# Optimal Transport cơ bản

> **TL;DR.** Optimal transport (OT) đo khoảng cách giữa hai phân phối bằng chi phí nhỏ nhất để "chuyển khối lượng" từ phân phối này sang phân phối kia, thay vì chỉ so mean, covariance, hay mật độ tại từng điểm. Ý tưởng cốt lõi đi từ bài toán Monge với ánh xạ vận chuyển $T$, rồi được thư giãn thành bài toán Kantorovich trên **transport plan** $\pi$, từ đó sinh ra Wasserstein distance. Caveat là exact OT thường đắt tính toán; Sinkhorn regularization giúp scale tốt hơn nhưng làm transport plan bị "mờ" và distance bị entropic bias.

Trong roadmap của Latent-Anything, đây là mảnh toán cơ sở đứng sau [Optimal Transport trong Latent Space](../../04-latent-computation/research/07-optimal-transport-in-latent.md). Tầng 4 dùng OT như một công cụ phân tích và so khớp phân phối latent; mục hiện tại lùi lại một bước để trả lời câu hỏi nền tảng hơn: bài toán vận chuyển tối ưu thực sự là gì, vì sao Wasserstein khác KL, và Sinkhorn đã làm bài toán đó khả thi trong machine learning như thế nào.

---

## 1. Trực giác / Định nghĩa

Trực giác kinh điển của OT là bài toán "đống cát": có một đống cát ban đầu và một hình đích có cùng tổng khối lượng, mục tiêu là di chuyển cát sao cho tổng công vận chuyển là nhỏ nhất. Theo Peyré và Cuturi, đó là một cách nhìn rất tự nhiên để so sánh **hai phân phối xác suất** thay vì hai điểm riêng lẻ.

Điểm khác biệt then chốt giữa OT với các divergence quen thuộc là OT không chỉ hỏi "xác suất ở mỗi chỗ khác nhau bao nhiêu", mà còn hỏi "nếu khác nhau, phải **di chuyển bao xa** để biến phân phối này thành phân phối kia". Vì thế OT nhúng trực tiếp hình học của không gian nền vào khái niệm khoảng cách.

Một bảng so sánh trực giác:

| Thước đo | So sánh cái gì? | Khi nào mạnh? | Điểm yếu chính |
|---|---|---|---|
| **L2 / Mahalanobis** | Hai điểm hoặc hai vector hữu hạn chiều | So mẫu đơn lẻ, nearest-neighbor, retrieval | Không phải khoảng cách giữa *hai phân phối* |
| **KL divergence** | Chênh lệch mật độ tại cùng support | Khi đã có density model trơn và support chồng nhau | Có thể rất xấu hoặc vô hạn khi support lệch nhau |
| **Wasserstein / OT** | Chi phí biến cả phân phối này thành phân phối kia | Khi cần giữ hình học của không gian | Exact solve thường đắt |

Vì vậy OT đặc biệt hấp dẫn trong machine learning: nếu hai phân phối chỉ lệch nhau do một phép tịnh tiến nhỏ trong không gian đặc trưng, Wasserstein phản ánh điều đó khá tự nhiên, trong khi nhiều divergence khác chỉ thấy "không chồng support" và phạt rất mạnh.

---

## 2. Cơ chế / Công thức

### 2.1. Từ Monge sang Kantorovich

Dạng cổ điển của bài toán là **Monge transport**:

$$
\inf_{T : T_{\#}\mu = \nu} \int c(x, T(x)) \, d\mu(x).
$$

trong đó $\mu$ là phân phối nguồn, $\nu$ là phân phối đích, $T$ là ánh xạ vận chuyển, $T_{\#}\mu = \nu$ nghĩa là khi đẩy $\mu$ qua ánh xạ $T$ thì thu được $\nu$, và $c(x,T(x))$ là chi phí chuyển khối lượng từ vị trí $x$ sang vị trí mới $T(x)$. Công thức này nói rằng mỗi điểm nguồn phải được gán sang đúng một vị trí đích.

Bài toán Monge đẹp về trực giác nhưng khá cứng: không phải lúc nào cũng tồn tại một ánh xạ tất định phù hợp. Kantorovich thư giãn bài toán đó bằng cách cho phép "xé khối lượng" ra thành nhiều phần và chỉ yêu cầu một **transport plan** $\pi$ có đúng hai marginal:

$$
\operatorname{OT}_c(\mu,\nu)
=
\inf_{\pi \in \Pi(\mu,\nu)}
\int c(x,y)\, d\pi(x,y).
$$

trong đó $\Pi(\mu,\nu)$ là tập mọi coupling có marginal lần lượt là $\mu$ và $\nu$, còn $\pi(x,y)$ diễn tả bao nhiêu khối lượng được chuyển từ vùng quanh $x$ sang vùng quanh $y$. Đây là bước chuyển cực quan trọng: từ ánh xạ cứng sang một phân phối chung mềm hơn, giúp OT trở thành một bài toán tối ưu lồi trong nhiều trường hợp rời rạc.

### 2.2. Wasserstein distance

Khi chọn chi phí là lũy thừa của khoảng cách nền,

$$
c(x,y) = \|x-y\|^p,
$$

ta thu được Wasserstein distance bậc $p$:

$$
W_p(\mu,\nu)
=
\left(
\inf_{\pi \in \Pi(\mu,\nu)}
\int \|x-y\|^p\, d\pi(x,y)
\right)^{1/p}.
$$

trong đó $p \ge 1$ quyết định loại khoảng cách; $x,y$ là điểm trong không gian nền; còn biểu thức trong ngoặc là chi phí vận chuyển tối ưu. Ý nghĩa hình học là rõ ràng: OT không chỉ quan tâm có bao nhiêu mass sai khác, mà còn tính luôn nó phải đi bao xa.

Trong bài Cuturi 2013, chi phí tối ưu rời rạc được viết dưới dạng ma trận:

$$
\min_{P \in U(r,c)} \langle P, M \rangle,
$$

trong đó $r$ và $c$ là hai histogram trong simplex, $M$ là cost matrix, $P$ là transportation table, và $U(r,c)$ là tập các ma trận không âm có tổng hàng và tổng cột lần lượt bằng $r$ và $c$. Đây chính là phiên bản hữu hạn chiều của bài toán Kantorovich.

### 2.3. Dạng rời rạc trên histogram hoặc minibatch

Trong machine learning, hai phân phối thường được thấy qua mẫu:

$$
\hat\mu = \sum_{i=1}^{n} a_i \delta_{x_i},
\qquad
\hat\nu = \sum_{j=1}^{m} b_j \delta_{y_j}.
$$

trong đó $\delta_{x_i}$ và $\delta_{y_j}$ là Dirac masses tại các mẫu, còn $a_i, b_j$ là trọng số không âm có tổng bằng 1. Khi đó transport plan trở thành một ma trận $P \in \mathbb{R}_{+}^{n \times m}$, với mỗi phần tử $P_{ij}$ cho biết bao nhiêu khối lượng đi từ $x_i$ sang $y_j$.

Điều này làm OT rất trực quan trong notebook: có thể vẽ trực tiếp ma trận plan và xem cặp nào "nhận khối lượng" từ cặp nào, thay vì chỉ nhận một con số cuối cùng.

### 2.4. Entropic regularization và Sinkhorn

Exact OT là một linear program, nhưng Cuturi chỉ ra rằng có thể làm nó mượt hơn bằng cách thêm entropy:

$$
\operatorname{OT}_{\varepsilon}(r,c)
=
\min_{P \in U(r,c)}
\Big[
\langle P, M \rangle
 \varepsilon \sum_{i,j} P_{ij}(\log P_{ij} - 1)
\Big].
$$

trong đó $\varepsilon > 0$ là hệ số regularization; $\langle P,M \rangle$ là chi phí vận chuyển gốc; còn hạng entropy khuyến khích plan có nhiều phần tử dương hơn thay vì cực kỳ sparse. Bài Cuturi 2013 nhấn mạnh rằng regularization này biến bài toán thành một bài toán lồi chặt hơn và giải được rất nhanh bằng **Sinkhorn-Knopp matrix scaling**.

Đổi lại:

- $\varepsilon$ lớn: solve ổn định, nhanh, plan mượt hơn nhưng bị bias mạnh;
- $\varepsilon$ nhỏ: gần exact OT hơn nhưng khó tối ưu và dễ kém ổn định số học hơn.

Đó là trade-off thực tế quan trọng nhất khi dùng OT trong ML hiện đại.

---

## 3. Biến thể / Trường hợp

| Biến thể | Ý tưởng | Khi nào dùng | Trade-off |
|---|---|---|---|
| **Monge OT** | Tìm một ánh xạ tất định $T$ | Lý thuyết hình học, khi map thật sự tồn tại | Khá cứng, khó tồn tại |
| **Kantorovich OT** | Tìm transport plan $\pi$ | Dạng chuẩn cho OT hiện đại | Exact solve là linear program |
| **Wasserstein distance** | Dùng cost theo khoảng cách nền | Cần một khoảng cách phân phối có ý nghĩa hình học | Phụ thuộc mạnh vào ground cost |
| **Sinkhorn / entropic OT** | Thêm entropy để solve nhanh | ML, minibatch, differentiable loss | Entropic bias, plan bị loang |

Một cách nhớ ngắn:

- **Monge** hỏi: "mỗi điểm nguồn đi về đâu?"
- **Kantorovich** hỏi: "khối lượng được chia và ghép như thế nào?"
- **Sinkhorn** hỏi: "nếu chấp nhận plan mềm hơn, ta có thể giải nhanh đến mức dùng được trong ML không?"

---

## 4. Giới hạn / Khi nào thất bại

**Exact OT đắt.** Bài Cuturi 2013 nhấn mạnh rằng bài toán OT rời rạc là một linear program với chi phí tệ trên histogram lớn. Đây là lý do Sinkhorn trở nên quan trọng đến vậy.

**Ground cost quyết định ý nghĩa của kết quả.** Wasserstein không có ý nghĩa trừu tượng tách rời khỏi cost $c(x,y)$. Nếu metric nền không phản ánh semantic move cost, thì khoảng cách OT có thể rất chính xác về mặt số học nhưng vô nghĩa về mặt bài toán.

**Sinkhorn có bias.** Regularization entropy giúp tính nhanh nhưng làm transport plan bớt sắc nét. Khi cần correspondence thật sự rõ, plan Sinkhorn có thể "quá mượt".

**Không phải mọi bài toán đều cần OT.** Nếu chỉ so hai vector mean hoặc cần một divergence density-model đơn giản, dùng OT có thể là quá nặng và khó diễn giải hơn mức cần thiết.

---

## 5. Liên hệ với Latent-Anything

Optimal transport là nền toán học cho nhiều primitive phía trên:

- **Layer A — Introspection**: so hai cloud latent hoặc hai occupancy distribution bằng Wasserstein/Sinkhorn thay vì chỉ so centroid.
- **Trajectory analysis**: khi muốn so phân phối state mà hai trajectory đi qua, OT là một ứng viên tự nhiên trước khi xét metric có trật tự thời gian.
- **Layer B — Manipulation**: sau latent editing, có thể xem phân phối latent đã dịch chuyển bao nhiêu theo nghĩa hình học.
- **Runtime / Method registry**: exact OT phù hợp cho notebook nhỏ và inspection; Sinkhorn phù hợp hơn cho batch lớn, web UI, hoặc training loop.

Ở mức kiến trúc, mục này nối trực tiếp sang [Optimal Transport trong Latent Space](../../04-latent-computation/research/07-optimal-transport-in-latent.md): note hiện tại giải thích toán cơ bản, còn note kia giải thích vì sao OT đáng giá trong latent computation, trajectory comparison, và distribution alignment.

---

## Liên quan

- [Optimal Transport trong Latent Space](../../04-latent-computation/research/07-optimal-transport-in-latent.md) — bản ứng dụng trực tiếp của OT vào phân phối latent và trajectory.
- [Mahalanobis distance](../../04-latent-computation/research/05-mahalanobis-distance.md) — một ground cost tốt đôi khi quan trọng không kém chính solver OT.
- [Density estimation](../../04-latent-computation/research/06-density-estimation.md) — đối chiếu cách nhìn "mật độ" với cách nhìn "vận chuyển".
- [Trajectory similarity metrics](../../06-latent-temporal/research/08-trajectory-similarity-metrics.md) — OT so phân phối state tốt, nhưng không giữ thứ tự thời gian.

## Tham khảo

- G. Peyré, M. Cuturi, *Computational Optimal Transport* (Foundations and Trends in Machine Learning 2019, arXiv:1803.00567).
- M. Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances* (NeurIPS 2013, arXiv:1306.0895).
