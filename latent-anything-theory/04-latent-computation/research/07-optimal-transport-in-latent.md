# Optimal Transport trong Latent Space

> **TL;DR.** Optimal transport (OT) so sánh hai phân phối latent bằng chi phí nhỏ nhất để "chuyển khối lượng" từ phân phối này sang phân phối kia, thay vì chỉ so mean, covariance, hay likelihood tại từng điểm. Công thức cốt lõi là Wasserstein distance $W_p(\mu, \nu) = \left(\inf_{\pi \in \Pi(\mu,\nu)} \int \|z - z'\|^p \, d\pi(z, z')\right)^{1/p}$, trong đó $\pi$ là kế hoạch ghép cặp giữa hai phân phối. Caveat quan trọng: exact OT đắt cả về tính toán lẫn sample complexity trong latent cao chiều; Sinkhorn giúp scale tốt hơn nhưng đổi lại tạo entropic bias và làm correspondence bị "mờ".

Optimal transport là bước kế tiếp tự nhiên sau [Mahalanobis distance](05-mahalanobis-distance.md) và [Density estimation](06-density-estimation.md). Mahalanobis trả lời câu hỏi "một điểm $z$ có lạ không nếu latent gần Gaussian?", density estimation trả lời "điểm $z$ nằm ở vùng mật độ nào?", còn OT trả lời "cả *một phân phối latent* khác một phân phối latent khác bao nhiêu, và sự khác biệt đó nằm ở đâu?" Đây là thứ cần khi so sánh hai minibatch latent, hai rollout của world model, hoặc aggregate posterior với prior.

---

## **1. Trực giác / Định nghĩa**

Hình dung hai đám mây điểm trong latent space: một đám là trạng thái latent thu được từ dữ liệu thật, đám còn lại là trạng thái latent do model rollout ra. Nếu chỉ so mean, hai đám có thể trùng nhau dù hình dạng hoàn toàn khác. Nếu dùng KL divergence, bài toán có thể trở nên tệ khi support của hai phân phối không chồng nhau hoặc chỉ được quan sát qua mẫu hữu hạn. OT thay cách nhìn: mỗi điểm trong phân phối thứ nhất mang một lượng "khối lượng"; mục tiêu là chuyển toàn bộ khối lượng đó sang phân phối thứ hai với tổng chi phí nhỏ nhất.

Điểm mạnh của OT là **giữ được hình học của latent space**. Nếu hai cụm latent chỉ lệch nhau một khoảng nhỏ, Wasserstein distance cũng nhỏ; nếu phải di chuyển khối lượng rất xa, khoảng cách tăng đúng theo hình học đó. Vì vậy OT phù hợp hơn KL trong các tình huống support rời nhau nhưng vẫn "gần" về mặt hình học.

| Thước đo | So sánh cái gì? | Khi nào hữu ích trong latent? | Điểm yếu chính |
|---|---|---|---|
| **Euclidean / Mahalanobis** | Hai điểm riêng lẻ | Retrieval, nearest-neighbor, OOD cho từng latent | Không so sánh được *hai phân phối* |
| **KL divergence** | Mật độ xác suất tại cùng support | Khi đã có density model trơn và support chồng nhau | Có thể vô hạn hoặc rất không ổn định khi support lệch nhau |
| **Wasserstein / OT** | Chi phí vận chuyển giữa hai phân phối | So minibatch latent, trajectory distribution, prior/posterior alignment | Đắt tính toán, sample-hungry ở chiều cao |

Một cách đọc trực giác khác: nếu interpolation ở Tầng 4 trước đó là "đi từ điểm $z_1$ sang điểm $z_2$ thế nào", thì OT là "đi từ *một tập điểm* sang *một tập điểm* thế nào".

---

## **2. Cơ chế / Công thức**

### 2.1 Bài toán Kantorovich và Wasserstein distance

Cho hai phân phối xác suất $\mu$ và $\nu$ trên latent space $\mathcal{Z} \subseteq \mathbb{R}^d$, OT giải bài toán:

$$
W_p(\mu, \nu) = \left(\inf_{\pi \in \Pi(\mu,\nu)} \int_{\mathcal{Z} \times \mathcal{Z}} \|z - z'\|^p \, d\pi(z, z')\right)^{1/p}.
$$

trong đó $\Pi(\mu,\nu)$ là tập các *coupling* có marginal lần lượt là $\mu$ và $\nu$; $z, z' \in \mathcal{Z}$ là hai latent variables; $\|z-z'\|^p$ là cost để chuyển một đơn vị khối lượng từ $z$ sang $z'$; và $p \ge 1$ quyết định loại Wasserstein distance. Kết quả $W_p(\mu,\nu)$ là chi phí vận chuyển tối ưu sau khi đã lấy căn bậc $p$.

Nếu dùng cost tổng quát $c(z,z')$ thay cho $\|z-z'\|^p$, ta có dạng Kantorovich tổng quát:

$$
\operatorname{OT}_c(\mu,\nu) = \inf_{\pi \in \Pi(\mu,\nu)} \int c(z, z') \, d\pi(z, z').
$$

trong đó $c(z,z')$ có thể là Euclidean cost, Mahalanobis cost, hoặc một cost học được từ task. Ý nghĩa của công thức là: OT không chỉ cần biết *bao nhiêu khối lượng* ở mỗi vùng, mà còn cần biết *vùng đó cách nhau bao xa*.

### 2.2 Dạng rời rạc trên minibatch latent

Trong machine learning, ta hiếm khi có $\mu$ và $\nu$ dưới dạng analytic. Thay vào đó, mỗi phân phối được quan sát qua các mẫu latent:

$$
\hat\mu = \sum_{i=1}^{n} a_i \, \delta_{z_i},
\qquad
\hat\nu = \sum_{j=1}^{m} b_j \, \delta_{\tilde z_j}.
$$

trong đó $\delta_{z_i}$ là Dirac mass tại latent point $z_i$; $a_i, b_j \ge 0$ là trọng số; $\sum_i a_i = \sum_j b_j = 1$. Nghĩa là mỗi minibatch được xem như một empirical measure.

Khi đó bài toán OT trở thành:

$$
\min_{P \in U(a,b)} \langle C, P \rangle
=
\min_{P \in U(a,b)} \sum_{i=1}^{n}\sum_{j=1}^{m} C_{ij} P_{ij},
$$

trong đó $C_{ij} = c(z_i, \tilde z_j)$ là cost matrix; $P_{ij}$ là lượng khối lượng chuyển từ $z_i$ sang $\tilde z_j$; và
$U(a,b) = \{P \in \mathbb{R}_{+}^{n \times m} \mid P\mathbf{1}_m = a,\; P^T\mathbf{1}_n = b\}$ là tập transport plan hợp lệ. Kết quả tối ưu cho biết cả khoảng cách lẫn correspondence giữa hai cloud latent.

Với trường hợp cân bằng đơn giản: $n = m$ và $a_i = b_j = 1/n$, exact OT gần với bài toán assignment. Điều này làm OT rất trực quan trong notebook: mỗi latent ở batch A được "ghép" sang một latent ở batch B sao cho tổng chi phí nhỏ nhất.

### 2.3 Entropic regularization và Sinkhorn

Exact OT có thể quá đắt, nên thực tế thường dùng entropic regularization:

$$
\operatorname{OT}_{\varepsilon}(\mu,\nu)
=
\min_{P \in U(a,b)}
\left[
\langle C, P \rangle
+ \varepsilon \sum_{i,j} P_{ij}(\log P_{ij} - 1)
\right].
$$

trong đó $\varepsilon > 0$ là hệ số regularization; hạng entropy khuyến khích $P$ "mềm" hơn thay vì cực sắc. Ý nghĩa thực tế: bài toán trở nên differentiable, solve được bằng các phép nhân ma trận lặp kiểu Sinkhorn-Knopp, và scale tốt hơn nhiều cho minibatch latent.

Nhược điểm là regularization làm kết quả bị bias: plan tối ưu trở nên "loang" hơn mức cần thiết. Feydy et al. đề xuất **Sinkhorn divergence** để loại bớt self-bias đó:

$$
S_{\varepsilon}(\mu,\nu)
=
\operatorname{OT}_{\varepsilon}(\mu,\nu)
- \frac{1}{2}\operatorname{OT}_{\varepsilon}(\mu,\mu)
- \frac{1}{2}\operatorname{OT}_{\varepsilon}(\nu,\nu).
$$

trong đó hai hạng tự-so sánh $\operatorname{OT}_{\varepsilon}(\mu,\mu)$ và $\operatorname{OT}_{\varepsilon}(\nu,\nu)$ trừ đi phần entropic bias vốn có ngay cả khi so một phân phối với chính nó. Kết quả là một divergence thực dụng hơn cho training loss và monitoring.

### 2.4 Vì sao OT đặc biệt phù hợp với latent space

**Support không chồng nhau vẫn đo được.** Hai distribution latent của hai trajectory có thể nằm trên hai manifold gần nhau nhưng không overlap đúng theo mẫu hữu hạn. KL divergence khi đó không ổn định, còn Wasserstein vẫn hữu hạn vì nó nhìn theo cost di chuyển.

**Có thể thay ground cost theo hình học latent.** Nếu latent anisotropic, dùng trực tiếp Euclidean cost là nguy hiểm. Có thể thay $c(z,z') = (z-z')^T \Sigma^{-1}(z-z')$ hoặc whiten trước bằng pipeline của [Mahalanobis distance](05-mahalanobis-distance.md), rồi mới chạy OT.

**Tương thích với matching latent prior/posterior.** Wasserstein Auto-Encoder (WAE) dùng Wasserstein distance để ép aggregate posterior trong latent khớp với prior, thay vì dùng KL pointwise như VAE. Đây là ví dụ rất trực tiếp cho việc OT hoạt động như một cơ chế *latent distribution alignment*.

---

## **3. Biến thể / Trường hợp**

| Biến thể | Ý tưởng | Khi nào nên dùng trong latent? | Trade-off |
|---|---|---|---|
| **Exact OT** | Giải bài toán vận chuyển gốc | Batch nhỏ, cần correspondence sắc nét | Đắt nhất về thời gian và bộ nhớ |
| **Sinkhorn OT** | Thêm entropy để solve nhanh bằng matrix scaling | Training differentiable loss, monitoring giữa minibatch | Entropic bias, transport plan bị blur |
| **Sinkhorn divergence** | Debias Sinkhorn bằng cách trừ self-cost | Khi cần loss ổn định hơn cho optimization | Vẫn cần tính thêm self-terms |
| **Projection/Subspace-robust OT** | Chỉ đo OT trên subspace/projection quan trọng | Latent rất cao chiều hoặc nhiều noise directions | Có thể bỏ sót hình học ngoài subspace |

**Projection-robust intuition.** Trong latent 1024 chiều, phần lớn chiều có thể chỉ là nuisance. Paty & Cuturi, và sau đó Lin et al., chỉ ra rằng dùng projection/subspace robust OT có thể cải thiện cả robustness lẫn statistical behavior trong high dimension. Đây là ý tưởng rất phù hợp với pipeline Tầng 4: **project trước**, rồi mới so distribution.

**WAE như một application chứ không chỉ là theory.** Với [VAE](../../02-representation-learning/research/03-vae.md), regularizer chuẩn là KL giữa posterior từng-sample và prior. Với WAE, target đổi thành match *aggregate* encoded distribution với prior bằng một OT-flavored objective. Điều này quan trọng vì trong latent computation ta thường quan tâm tới cấu trúc *toàn phân phối*, không chỉ từng điểm.

---

## **4. Giới hạn / Khi nào thất bại**

**Chi phí bộ nhớ và thời gian vẫn lớn.** Với $n$ và $m$ sample latent, cost matrix $C \in \mathbb{R}^{n \times m}$ đã tốn $O(nm)$ bộ nhớ. Exact OT trên batch vài nghìn điểm có thể rất đắt; Sinkhorn giảm độ khó solve nhưng vẫn cần thao tác lặp trên ma trận dày cỡ $n \times m$.

**Curse of dimensionality ở mức phân phối.** Wasserstein distance giữa các empirical measures hội tụ chậm trong high dimension. Nói cách khác, chỉ vì latent có 512 chiều không có nghĩa minibatch 128 điểm đã đủ để ước lượng OT đáng tin. Nếu không giảm chiều hoặc chọn subspace tốt, noise directions sẽ làm distance vừa đắt vừa nhiễu. Đây là lý do projection-robust OT hấp dẫn trong latent cao chiều.

**Entropic bias của Sinkhorn.** Chọn $\varepsilon$ lớn giúp tối ưu dễ hơn nhưng làm transport plan bị loang, che mất correspondence thật sự. Chọn $\varepsilon$ quá nhỏ thì số vòng lặp tăng mạnh và dễ bất ổn số học. Sinkhorn divergence sửa được một phần bias của *distance*, nhưng không làm plan thôi bị blur.

**OT tĩnh không hiểu thứ tự thời gian.** Nếu coi một trajectory là empirical measure của các state $\{z_t\}_{t=0}^{T}$, ta đã vứt bỏ thứ tự thời gian. Hai trajectory đi qua cùng tập state nhưng theo thứ tự ngược nhau có thể có Wasserstein rất nhỏ, dù về mặt dynamics chúng khác hẳn. Vì vậy OT chỉ là *trajectory distribution metric*, không phải trajectory metric hoàn chỉnh; muốn giữ thứ tự cần kết hợp với DTW, Fréchet distance, hoặc causal / windowed OT.

**Ground cost sai kéo theo distance sai.** Nếu latent anisotropic hoặc manifold cong, cost Euclidean trong ambient space có thể không phản ánh semantic move cost. Chạy OT trên metric sai cho ra con số rất "đẹp" nhưng vô nghĩa. Pipeline đúng thường là: whiten hoặc project trước, rồi mới lập cost matrix.

**Mass mismatch trong thực tế.** Hai batch latent có thể có trọng số khác nhau, hoặc một rollout bị truncate nên tổng mass không nên buộc match cứng 100%. Balanced OT khi đó tạo ra correspondence giả. Trong production, thường phải dùng weighting, trimming, hoặc biến thể unbalanced OT.

---

## **5. Liên hệ với Latent-Anything**

Optimal transport là primitive ở mức **distribution-level** cho Layer A và Layer B:

- **So distribution của minibatch latent:** `LatentDistribution.distance(batch_a, batch_b, metric="sinkhorn")` để đo domain shift hoặc quality gap giữa latent thật và latent rollout.
- **So trajectory distribution:** `TrajectorySet.compare(mode="wasserstein")` khi muốn biết hai policy tạo ra *state occupancy* khác nhau bao nhiêu, không cần align từng timestamp một.
- **Barycenter trong latent:** từ nhiều trajectory hoặc nhiều cluster latent, có thể tính Wasserstein barycenter làm "prototype distribution" cho summarization hoặc curriculum sampling.
- **Prior/posterior alignment:** encoder plugin có thể expose `match_prior(method="sinkhorn")` theo tinh thần WAE, thay vì chỉ KL-regularize từng sample.

Pipeline thực dụng cho latent anisotropic:
1. Whiten hoặc đổi cost theo [Mahalanobis distance](05-mahalanobis-distance.md).
2. Nếu $d$ lớn, project xuống subspace tốt bằng [Subspace projection](04-subspace-projection.md) hoặc PCA.
3. Dựng empirical measures từ minibatch / trajectory window.
4. Dùng Sinkhorn divergence cho monitoring hoặc loss; chỉ dùng exact OT khi batch nhỏ và cần inspect plan.

Điều quan trọng là OT không thay thế mọi metric khác. Nó lấp đúng chỗ trống giữa point-metric và density-model: khi cần so **cấu trúc phân phối latent** mà vẫn tôn trọng hình học của không gian.

---

## Liên quan

- [Mahalanobis distance (mục 05 — tầng này)](05-mahalanobis-distance.md) — OT thường cần ground cost tốt; với latent anisotropic nên whiten hoặc dùng Mahalanobis cost trước khi vận chuyển.
- [Density estimation (mục 06 — tầng này)](06-density-estimation.md) — density estimation hỏi "điểm này nằm ở đâu trong phân phối?", còn OT hỏi "hai phân phối lệch nhau bao nhiêu và lệch theo hướng nào?".
- [Subspace projection (mục 04 — tầng này)](04-subspace-projection.md) — projection trước khi chạy OT là cách thực dụng để chống curse of dimensionality trong latent cao chiều.
- [Curse of dimensionality](../../01-space-representation/research/04-curse-of-dimensionality.md) — lý do sample complexity của empirical Wasserstein trở nên khó chịu khi latent dimension tăng.
- [Geodesic](../../01-space-representation/research/05-geodesic.md) — OT tạo ra notion "geodesic giữa hai phân phối" trong không gian xác suất, khác với geodesic giữa hai điểm latent riêng lẻ.
- [VAE](../../02-representation-learning/research/03-vae.md) — WAE là biến thể thay KL-style matching bằng Wasserstein-style distribution matching trong latent.

## Tham khảo

- G. Peyré, M. Cuturi, *Computational Optimal Transport* (Foundations and Trends in Machine Learning, 2019, arXiv:1803.00567). — Tài liệu chuẩn cho định nghĩa Kantorovich, Wasserstein distance, entropic OT, và góc nhìn tính toán.
- M. Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport* (NeurIPS 2013). — Paper mở đường cho entropic regularization + Sinkhorn algorithm trong machine learning.
- J. Feydy, T. Séjourné, F.-X. Vialard, S.-I. Amari, A. Trouvé, G. Peyré, *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences* (AISTATS 2019, arXiv:1810.08278). — Định nghĩa Sinkhorn divergence để loại entropic bias và dùng làm loss thực dụng hơn.
- I. Tolstikhin, O. Bousquet, S. Gelly, B. Schölkopf, M. Simon-Gabriel, *Wasserstein Auto-Encoders* (ICLR 2018, arXiv:1711.01558). — Ví dụ trực tiếp cho latent distribution alignment bằng OT trong generative modeling.
- F.-P. Paty, M. Cuturi, *Subspace Robust Wasserstein Distances* (ICML 2019). — Đề xuất đo OT trên subspace quan trọng để cải thiện robustness và tính toán trong high dimension.
- T. Lin, Z. Zheng, E. Chen, M. Cuturi, M. Jordan, *On Projection Robust Optimal Transport: Sample Complexity and Model Misspecification* (AISTATS 2021). — Phân tích vì sao projection-robust OT tốt hơn OT gốc trong nhiều bài toán high-dimensional inference.
