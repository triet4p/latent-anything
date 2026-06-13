# Trajectory Similarity Metrics

> **TL;DR.** So sánh trajectory không chỉ là trừ hai tensor: cần quyết định cách ghép các thời điểm, local metric trong latent space và cách tổng hợp sai khác dọc đường. Dynamic Time Warping (DTW) tối thiểu hóa **tổng** local cost sau time warping, còn Fréchet distance tối thiểu hóa **sai khác lớn nhất** dưới reparameterization đơn điệu. Caveat chính là metric có thể xóa đúng tín hiệu cần giữ: DTW quá linh hoạt che khác biệt tốc độ, Fréchet nhạy với một outlier, và mọi kết quả đều phụ thuộc mạnh vào geometry của latent representation.

Hai trajectory có thể biểu diễn cùng chuyển động nhưng khác số mẫu, sampling rate hoặc tốc độ thực hiện. Ngược lại, hai sequence có cùng tập latent points nhưng đi theo thứ tự khác nhau không nên được coi là cùng trajectory.

[Latent Trajectory](06-latent-trajectory.md) đã định nghĩa dữ liệu cần giữ: state, timestamp, mask, source và uncertainty. Mục này tập trung vào câu hỏi tiếp theo:

> Một trajectory cách trajectory khác bao xa, và phép đo đó đang cố giữ bất biến điều gì?

Không tồn tại một metric tốt nhất cho mọi bài toán. Chọn metric là chọn semantics:

- có bỏ qua tốc độ hay không;
- có phạt time shift hay không;
- có giữ thứ tự và hướng đi hay không;
- quan tâm tổng sai số hay worst-case deviation;
- có cho phép bỏ outlier, prefix hoặc suffix hay không;
- khoảng cách giữa hai latent states được định nghĩa thế nào.

---

## **1. Bài toán và ba lớp quyết định**

Cho hai trajectory:

$$
\mathcal{X}
=
(x_1,\ldots,x_n),
\qquad
\mathcal{Y}
=
(y_1,\ldots,y_m),
$$

trong đó $x_i,y_j\in\mathcal{M}$ là latent states và $n,m$ có thể khác nhau. Nếu có timestamp, trajectory đầy đủ là $\{(t_i,x_i)\}$ và $\{(s_j,y_j)\}$.

Một phép so sánh thường có ba lớp:

1. **Local metric** $c(i,j)=d(x_i,y_j)$: hai state riêng lẻ khác nhau bao nhiêu.
2. **Alignment policy** $\Pi$: state nào của $\mathcal{X}$ được phép ghép với state nào của $\mathcal{Y}$.
3. **Aggregation**: tổng, trung bình, maximum, số match hoặc edit cost dọc alignment.

Có thể viết khung tổng quát:

$$
D(\mathcal{X},\mathcal{Y})
=
\operatorname*{opt}_{\pi\in\Pi}
\operatorname{Agg}
\left(
\{d(x_i,y_j):(i,j)\in\pi\}
\right).
$$

Trong đó $\pi$ là một alignment khả dĩ. DTW dùng `min + sum`, Fréchet dùng `min + max`, còn LCSS tối đa hóa số cặp đủ gần.

### Metric, distance và discrepancy

Một metric toán học $D$ phải thỏa:

$$
D(X,Y)\ge 0,
\qquad
D(X,Y)=0\Leftrightarrow X=Y,
$$

$$
D(X,Y)=D(Y,X),
\qquad
D(X,Z)\le D(X,Y)+D(Y,Z).
$$

Trong đó bốn điều kiện lần lượt là không âm, identity of indiscernibles, đối xứng và bất đẳng thức tam giác.

Nhiều đại lượng gọi thông thường là “distance” không thỏa đủ bốn điều kiện. DTW chuẩn có thể cho khoảng cách zero giữa hai sequence khác nhau chỉ vì một state được lặp lại, và nhìn chung không thỏa triangle inequality. Trong API nên dùng tên trung tính `TrajectoryMeasure` hoặc ghi rõ `is_metric`.

---

## **2. Baseline pointwise và resampling**

Nếu hai trajectory có cùng length và cùng temporal grid:

$$
D_{\text{point}}
(\mathcal{X},\mathcal{Y})
=
\frac{1}{n}
\sum_{i=1}^{n}
d(x_i,y_i).
$$

Trong đó state tại cùng index được ghép trực tiếp. Metric này giữ timing rất chặt nhưng phạt mạnh phase shift hoặc speed variation.

Với timestamps không đều, có thể resample cả hai lên grid chung:

$$
\bar t_k=t_{\min}+k\Delta t,
\qquad
\bar x_k\approx x(\bar t_k),
\qquad
\bar y_k\approx y(\bar t_k).
$$

Trong đó $\Delta t$ là interval mục tiêu. Pointwise distance sau resampling trả lời “hai hệ ở cùng thời điểm vật lý có gần nhau không”, không phải “hai hệ có đi qua cùng path không”.

### Khi pointwise là lựa chọn đúng

- timing là tín hiệu, như collision xảy ra sớm hay muộn;
- trajectories đã được đồng bộ bằng clock/event;
- control policy cần state correspondence tại đúng timestamp;
- latency hoặc phase error phải bị phạt.

### Khi pointwise gây hiểu nhầm

Hai trajectory đi cùng path nhưng một trajectory chậm hơn sẽ lệch index gần như toàn bộ. Resampling cũng không giải quyết nonlinear speed variation nếu không biết correspondence đúng.

---

## **3. Dynamic Time Warping**

DTW cho phép time axis co giãn bằng cách ghép một state với một hoặc nhiều states ở trajectory kia. Đây là elastic alignment, không phải sửa latent states.

### Cost matrix

Định nghĩa local cost:

$$
C_{ij}
=
d(x_i,y_j).
$$

Trong đó $C\in\mathbb{R}^{n\times m}$ chứa mọi pairwise state cost. Metric $d$ có thể là Euclidean, cosine, Mahalanobis, geodesic hoặc task-aware.

### Warping path

Một warping path:

$$
\pi
=
\bigl(
(i_1,j_1),\ldots,(i_L,j_L)
\bigr)
$$

thường thỏa:

- **boundary**: $(i_1,j_1)=(1,1)$ và $(i_L,j_L)=(n,m)$;
- **monotonicity**: indices không giảm;
- **continuity**: mỗi bước thuộc $\{(1,0),(0,1),(1,1)\}$.

Monotonicity giữ thứ tự. Horizontal hoặc vertical step cho phép một state ghép với nhiều states.

### Dynamic programming

Accumulated cost:

$$
A_{ij}
=
C_{ij}
+
\min
\left\{
A_{i-1,j},
A_{i,j-1},
A_{i-1,j-1}
\right\}.
$$

Trong đó ba predecessor tương ứng kéo dài trajectory thứ nhất, kéo dài trajectory thứ hai hoặc tiến đồng thời. Boundary được khởi tạo $A_{11}=C_{11}$ và các cell ngoài bảng bằng $+\infty$.

Khoảng cách:

$$
D_{\mathrm{DTW}}
(\mathcal{X},\mathcal{Y})
=
A_{nm}
=
\min_{\pi\in\Pi}
\sum_{(i,j)\in\pi}
d(x_i,y_j).
$$

Trong đó DTW tối ưu **tổng** cost dọc alignment. Backtracking từ $(n,m)$ qua predecessor tối ưu trả warping path.

### Raw cost và normalized cost

Raw DTW tăng theo số cặp trong path. Một normalization thường dùng:

$$
\bar D_{\mathrm{DTW}}
=
\frac{1}{|\pi^\star|}
\sum_{(i,j)\in\pi^\star}
d(x_i,y_j).
$$

Trong đó $|\pi^\star|$ là độ dài optimal path. Normalization giúp so các cặp length khác nhau nhưng thay đổi objective diễn giải; không nên gọi raw và normalized DTW bằng cùng một tên trong benchmark.

### DTW bỏ qua gì?

Nếu hai motion có cùng shape nhưng tốc độ khác nhau, DTW có thể ghép đúng phase và cho cost thấp. Đây là ưu điểm khi tốc độ là nuisance, nhưng là lỗi khi tốc độ là behavior:

- phanh gấp so với phanh từ từ;
- thao tác an toàn so với thao tác quá nhanh;
- reward phụ thuộc completion time;
- cùng path nhưng timing contact khác nhau.

---

## **4. Ràng buộc warping**

Unconstrained DTW có thể tạo alignment cực đoan: một state bị kéo dài qua đoạn lớn ở trajectory kia.

### Sakoe–Chiba band

Chỉ cho phép cell gần đường chéo:

$$
|i-j|
\le
w.
$$

Trong đó $w$ là band radius theo index. Với length khác nhau, nên dùng normalized coordinates hoặc band quanh đường nối $(1,1)$ tới $(n,m)$ thay vì raw $|i-j|$.

Band nhỏ:

- giảm compute;
- giữ timing tốt hơn;
- tránh pathological alignment;
- có thể loại correspondence đúng nếu speed variation lớn.

### Slope và step pattern

Có thể đổi allowed steps hoặc thêm penalty:

$$
A_{ij}
=
C_{ij}
+
\min
\left\{
A_{i-1,j}+\lambda,
A_{i,j-1}+\lambda,
A_{i-1,j-1}
\right\}.
$$

Trong đó $\lambda>0$ phạt horizontal/vertical steps. Penalty biến “được phép warp bao nhiêu” thành trade-off mềm thay vì hard window.

### Timestamp-aware constraint

Với timestamps thật:

$$
|t_i-s_j|
\le
\Delta_{\max}.
$$

Trong đó $\Delta_{\max}$ là temporal tolerance theo đơn vị vật lý. Ràng buộc này rõ nghĩa hơn index band khi sampling rate khác nhau.

### Open begin/end và subsequence DTW

Boundary chuẩn ép ghép toàn bộ hai trajectories. Retrieval motif hoặc partial overlap cần:

- open-begin;
- open-end;
- subsequence DTW;
- prefix/suffix penalty;
- explicit unmatched cost.

Nếu không khai báo boundary mode, hai implementation cùng tên DTW có thể trả kết quả khác đáng kể.

---

## **5. Fréchet distance**

Fréchet distance xem trajectory như curve có hướng. Trực giác cổ điển là hai đối tượng đi dọc hai curve, mỗi bên được thay đổi tốc độ nhưng không được đi lùi; khoảng cách là dây ngắn nhất đủ để nối chúng trong toàn bộ hành trình.

### Continuous Fréchet

Cho hai curve liên tục:

$$
P,Q:[0,1]\to\mathcal{M}.
$$

Trong đó $P$ và $Q$ parameterize hai paths.

Fréchet distance:

$$
d_F(P,Q)
=
\inf_{\alpha,\beta}
\max_{\tau\in[0,1]}
d\!\left(
P(\alpha(\tau)),
Q(\beta(\tau))
\right).
$$

Trong đó $\alpha,\beta:[0,1]\to[0,1]$ là các reparameterization liên tục, không giảm và phủ toàn interval. Hai bên được đổi tốc độ, nhưng order và orientation được giữ.

Khác DTW, aggregation ở đây là maximum. Một đoạn lệch xa quyết định toàn distance dù phần còn lại rất gần.

### Free-space diagram

Với threshold $\varepsilon$, free space là:

$$
\mathcal{F}_{\varepsilon}
=
\left\{
(u,v)\in[0,1]^2:
d(P(u),Q(v))\le\varepsilon
\right\}.
$$

Trong đó $d_F(P,Q)\le\varepsilon$ khi tồn tại path liên tục, đơn điệu từ góc dưới-trái tới góc trên-phải nằm hoàn toàn trong free space.

Alt và Godau đưa bài toán polygonal curves vào computational geometry bằng free-space diagram. Thuật toán cổ điển cho hai polygonal curves có $n,m$ segments chạy gần quadratic, cụ thể $O(nm\log(nm))$ trong formulation của họ.

### Discrete Fréchet

Với sampled trajectories, recurrence:

$$
F_{ij}
=
\max
\left(
d(x_i,y_j),
\min
\left\{
F_{i-1,j},
F_{i-1,j-1},
F_{i,j-1}
\right\}
\right).
$$

Trong đó predecessor chọn coupling đơn điệu tốt nhất, còn `max` giữ bottleneck cost lớn nhất đã gặp.

Kết quả:

$$
D_{\mathrm{dF}}
(\mathcal{X},\mathcal{Y})
=
F_{nm}
=
\min_{\pi\in\Pi}
\max_{(i,j)\in\pi}
d(x_i,y_j).
$$

Trong đó discrete Fréchet dùng cùng kiểu monotone path như DTW nhưng objective là min-max thay vì min-sum.

### Continuous và discrete không đồng nhất

Discrete Fréchet chỉ đặt endpoints của “dây” tại sampled vertices. Continuous Fréchet cho phép điểm nằm trong interior của polygonal edges. Vì vậy discrete result phụ thuộc sampling density và thường là upper approximation của continuous result.

---

## **6. DTW và Fréchet trả lời hai câu hỏi khác nhau**

| Thuộc tính | DTW | Fréchet |
|---|---|---|
| Aggregation | Tổng/mean local cost | Maximum local cost |
| Trực giác | Tổng mismatch sau alignment | Dây dài nhất cần dùng |
| Nhạy với | Sai lệch nhỏ kéo dài | Một bottleneck/outlier lớn |
| Output tự nhiên | Cost + warping path | Bottleneck + coupling/free-space |
| Dạng phổ biến cho tensor | Discrete DP | Discrete DP |
| Curve liên tục | Không phải formulation chuẩn | Continuous Fréchet |
| Metric property | Thường không phải metric | Metric trên lớp curve modulo reparameterization phù hợp |

Ví dụ:

- Sai lệch $0.2$ kéo dài 100 steps có DTW cost lớn nhưng Fréchet khoảng $0.2$.
- Một spike lệch $5$ ở một step có thể quyết định Fréchet bằng $5$, trong khi normalized DTW pha loãng spike đó.

Không nên dùng hai metric thay thế lẫn nhau chỉ vì chúng cùng có dynamic programming table.

---

## **7. Hausdorff distance và vai trò của thứ tự**

Hausdorff distance xem mỗi trajectory như một point set:

$$
d_H(X,Y)
=
\max
\left\{
\sup_{x\in X}\inf_{y\in Y}d(x,y),
\sup_{y\in Y}\inf_{x\in X}d(x,y)
\right\}.
$$

Trong đó mỗi point chỉ cần có một point gần ở set kia. Thứ tự, hướng và multiplicity bị bỏ.

Nếu:

$$
\mathcal{Y}
=
(x_n,x_{n-1},\ldots,x_1),
$$

thì hai trajectories có cùng point set nên Hausdorff distance bằng zero, dù một trajectory đi ngược chiều. DTW và Fréchet giữ monotonic order nên thường phân biệt được.

Hausdorff phù hợp khi occupancy quan trọng hơn traversal:

- vùng latent đã đi qua;
- coverage;
- set of visited states;
- shape không định hướng.

Nó không phù hợp khi causal order hoặc event sequence quan trọng.

---

## **8. Robust và timestamp-aware alternatives**

### LCSS

Longest Common Subsequence cho trajectory thường coi hai states match nếu:

$$
d(x_i,y_j)\le\varepsilon
\quad\text{và có thể}\quad
|t_i-s_j|\le\Delta.
$$

Trong đó $\varepsilon$ là spatial/latent tolerance và $\Delta$ là temporal tolerance. Similarity có thể normalize:

$$
S_{\mathrm{LCSS}}
=
\frac{\operatorname{LCSS}(\mathcal{X},\mathcal{Y})}
{\min(n,m)}.
$$

Trong đó unmatched outliers không cộng một cost lớn như DTW. Đổi lại, kết quả nhạy với thresholds và mất thông tin độ lệch bên trong vùng match.

### EDR

Edit Distance on Real sequence dùng substitution cost thresholded:

$$
\operatorname{sub}(x_i,y_j)
=
\mathbf{1}
\left[
d(x_i,y_j)>\varepsilon
\right].
$$

Trong đó edit operations cho phép insert/delete/substitute. EDR robust với noise/outlier hơn raw accumulated distance, nhưng threshold $\varepsilon$ tạo discontinuity.

### ERP

Edit distance with Real Penalty dùng gap reference $g$:

$$
\operatorname{cost}(x_i,\text{gap})
=
d(x_i,g).
$$

Trong đó insert/delete có geometric cost thay vì constant. Với điều kiện phù hợp trên local metric, ERP có thể giữ metric properties hữu ích cho indexing, nhưng kết quả phụ thuộc lựa chọn gap state.

### TWED

Time Warp Edit Distance đưa timestamps trực tiếp vào cost và có stiffness parameter:

$$
\operatorname{cost}_{\text{time}}
\propto
\nu |t_i-s_j|,
$$

trong đó $\nu\ge 0$ phạt temporal displacement. Edit penalty $\lambda$ và stiffness $\nu$ điều khiển mức elastic; TWED được thiết kế để vẫn là metric dưới các điều kiện của formulation.

### Chọn robust alternative

| Nhu cầu | Candidate |
|---|---|
| Bỏ qua speed variation trơn | DTW |
| Kiểm soát worst-case deviation | Fréchet |
| Bỏ outlier và partial match | LCSS / EDR |
| Cần metric cho indexing | ERP / TWED / Fréchet phù hợp |
| Timestamp không đều có semantics | TWED hoặc timestamp-constrained DTW |
| So occupancy, không quan tâm order | Hausdorff |

---

## **9. Soft-DTW và learning objective**

Hard `min` trong DTW làm objective không khả vi tại nơi optimal path đổi. Soft-DTW thay `min` bằng soft minimum:

$$
\operatorname{softmin}_{\gamma}
(a_1,\ldots,a_k)
=
-\gamma
\log
\sum_{\ell=1}^{k}
\exp
\left(
-\frac{a_\ell}{\gamma}
\right).
$$

Trong đó $\gamma>0$ là temperature. Khi $\gamma\to 0^+$, soft minimum tiến về minimum.

Recurrence:

$$
A^{(\gamma)}_{ij}
=
C_{ij}
+
\operatorname{softmin}_{\gamma}
\left(
A^{(\gamma)}_{i-1,j},
A^{(\gamma)}_{i,j-1},
A^{(\gamma)}_{i-1,j-1}
\right).
$$

Trong đó mọi alignment đóng góp với trọng số khác nhau thay vì chỉ optimal path. Cuturi và Blondel chỉ ra value và gradient có thể tính bằng dynamic programming với quadratic time/space.

### Bias của soft-DTW

Raw soft-DTW không tự là một divergence: entropy reward từ nhiều alignment có thể làm giá trị âm và minimum không nhất thiết đạt tại hai sequence giống nhau.

Soft-DTW divergence hiệu chỉnh self-similarity:

$$
D_{\gamma}^{\text{sDTW}}
(X,Y)
=
\operatorname{sDTW}_{\gamma}(X,Y)
-
\frac{1}{2}
\operatorname{sDTW}_{\gamma}(X,X)
-
\frac{1}{2}
\operatorname{sDTW}_{\gamma}(Y,Y).
$$

Trong đó hai self terms loại phần bias do entropic smoothing. Dưới điều kiện phù hợp trên ground cost, divergence này không âm và được tối thiểu khi hai time series bằng nhau.

### Khi dùng làm loss

Soft-DTW hữu ích khi prediction đúng shape nhưng lệch phase nhẹ không nên chịu MSE lớn. Tuy nhiên nó có thể khuyến khích model “đúng event nhưng sai thời điểm”. Một objective forecasting có thể cần tách shape loss và temporal distortion loss thay vì làm timing hoàn toàn invariant.

---

## **10. Local metric trong latent space**

Alignment tốt không cứu được local metric sai.

### Euclidean

$$
d_2(x,y)
=
\|x-y\|_2.
$$

Trong đó mọi coordinate được cân như nhau. Euclidean chỉ hợp lý khi latent đã được scale và geometry gần isotropic.

### Cosine

$$
d_{\cos}(x,y)
=
1-
\frac{x^\top y}
{\|x\|_2\|y\|_2}.
$$

Trong đó magnitude bị bỏ qua. Metric này hữu ích khi direction mang semantics, nhưng không phù hợp nếu norm mã hóa confidence, intensity hoặc time-to-event.

### Mahalanobis

$$
d_M(x,y)
=
\sqrt{
(x-y)^\top
\Sigma^{-1}
(x-y)
}.
$$

Trong đó $\Sigma$ là covariance reference. Các direction variance lớn bị down-weight, như phân tích trong [Mahalanobis Distance](../../04-latent-computation/research/05-mahalanobis-distance.md).

### Geodesic hoặc pullback

Nếu latent manifold cong, local distance có thể dùng geodesic hoặc decoder-induced metric:

$$
d_G(x,y)
=
\inf_{z(0)=x,z(1)=y}
\int_0^1
\sqrt{
\dot z(\tau)^\top
G(z(\tau))
\dot z(\tau)
}
\,d\tau.
$$

Trong đó $G(z)$ là metric tensor. Cách này gần semantics manifold hơn nhưng làm cost matrix đắt.

### Task-aware metric

Có thể so qua reward/value/event heads:

$$
d_{\text{task}}(x,y)
=
\alpha\|x-y\|
+
\beta|V(x)-V(y)|
+
\gamma\,D(p(e\mid x),p(e\mid y)).
$$

Trong đó $V$ là value, $e$ là event và $\alpha,\beta,\gamma$ là weights. Metric task-aware có thể tốt cho planning nhưng không còn là measure thuần representation.

### Version và coordinate contract

Hai trajectories từ encoder checkpoints khác nhau không nên so trực tiếp. Trước khi chạy DTW/Fréchet cần kiểm tra:

- model/adapter identity;
- latent normalization;
- coordinate alignment;
- state schema;
- dimension mask;
- posterior/prior source;
- precision và quantization.

---

## **11. Timestamps, speed và dynamics**

DTW/Fréchet chuẩn invariant với monotone reparameterization ở mức nào đó. Điều này cố ý bỏ speed.

### Tách shape và timing

Có thể báo hai scores:

$$
D_{\text{shape}}
=
\frac{1}{|\pi|}
\sum_{(i,j)\in\pi}
d(x_i,y_j),
$$

$$
D_{\text{time}}
=
\frac{1}{|\pi|}
\sum_{(i,j)\in\pi}
\left|
\frac{t_i-t_1}{t_n-t_1}
-
\frac{s_j-s_1}{s_m-s_1}
\right|.
$$

Trong đó shape score dùng aligned state cost, còn timing score đo độ lệch normalized phase dọc cùng path. Hai trajectory có cùng path nhưng speed profile khác sẽ có shape score thấp và timing score cao.

### So velocity

Nếu dynamics quan trọng, augment state:

$$
\tilde x_i
=
\begin{bmatrix}
x_i\\
\eta v_i
\end{bmatrix},
\qquad
v_i
=
\frac{x_{i+1}-x_i}{t_{i+1}-t_i}.
$$

Trong đó $\eta$ cân state và velocity. DTW trên augmented state không còn bỏ hoàn toàn speed/direction change.

### Action-conditioned comparison

Hai latent paths giống nhau dưới action sequence khác nhau có thể không tương đương. Local cost mở rộng:

$$
c(i,j)
=
d_z(x_i,y_j)
+
\lambda_a d_a(a_i,b_j)
+
\lambda_r |r_i-r_j|.
$$

Trong đó action và reward được ghép cùng transition. Phải xử lý rõ state-aligned so với edge-aligned fields để tránh off-by-one.

---

## **12. Uncertainty và trajectory distributions**

Một stochastic rollout không phải một curve duy nhất mà là distribution trên curves.

### Expected pairwise distance

Với samples $\mathcal{X}^{(p)}$ và $\mathcal{Y}^{(q)}$:

$$
\widehat D_{\text{pair}}
=
\frac{1}{PQ}
\sum_{p=1}^{P}
\sum_{q=1}^{Q}
D(
\mathcal{X}^{(p)},
\mathcal{Y}^{(q)}
).
$$

Trong đó measure nền $D$ có thể là DTW hoặc Fréchet. Pairwise average dễ tính nhưng có cost $O(PQnm)$.

### Distance giữa matched particles

Nếu particle identity có correspondence:

$$
\widehat D_{\text{matched}}
=
\frac{1}{P}
\sum_{p=1}^{P}
D(
\mathcal{X}^{(p)},
\mathcal{Y}^{(p)}
).
$$

Trong đó cách ghép particle phải có semantics; index ngẫu nhiên không tạo correspondence thật.

### Distributional trajectory distance

Có thể xem mỗi trajectory là empirical measure trên state-time:

$$
\mu_{\mathcal{X}}
=
\sum_i w_i\,
\delta_{(t_i,x_i)}.
$$

Trong đó $w_i$ là temporal weights. Optimal transport so visitation distributions nhưng có thể mất causal order nếu ground cost không mã hóa time/order. [Optimal Transport trong Latent](../../04-latent-computation/research/07-optimal-transport-in-latent.md) cung cấp nền tảng cho trường hợp này.

Không được nhầm **Fréchet curve distance** với Fréchet distance giữa Gaussian distributions dùng trong FID. Hai khái niệm có tên chung nhưng object và công thức khác nhau.

---

## **13. Độ phức tạp và scaling**

Với lengths $n,m$:

| Phương pháp | Time phổ biến | Memory phổ biến |
|---|---:|---:|
| Pointwise sau resampling | $O(Ld)$ | $O(1)$ ngoài input |
| DTW | $O(nm)$ | $O(nm)$; $O(\min(n,m))$ nếu chỉ cần value |
| Discrete Fréchet | $O(nm)$ | $O(nm)$; có thể rolling rows |
| Continuous Fréchet cổ điển | $O(nm\log(nm))$ | phụ thuộc implementation |
| LCSS / EDR / ERP / TWED | thường $O(nm)$ | thường $O(nm)$ hoặc rolling rows |
| Pairwise particles | nhân thêm $PQ$ | phụ thuộc batching |

Trong đó lấy alignment path cần lưu predecessor hoặc recompute; rolling rows chỉ đủ cho scalar distance.

### Windowing

Nếu chỉ xét $K$ admissible cells:

$$
C_{\text{DP}}
=
O(K).
$$

Trong đó Sakoe–Chiba band có thể giảm $K$ gần $O(w\max(n,m))$. Đây là acceleration bằng assumption; path thật ngoài band sẽ bị mất.

### Lower bounds

Các kết quả conditional lower bound cho thấy strongly subquadratic exact algorithms cho DTW và Fréchet tổng quát khó tồn tại nếu Strong Exponential Time Hypothesis đúng. Vì vậy scaling thực tế thường dựa trên:

- constraints/windows;
- lower bounds và pruning;
- early abandoning;
- multiscale approximation;
- trajectory simplification;
- coarse descriptor retrieval rồi rerank;
- GPU wavefront DP;
- cache pairwise local costs.

### Batching không hoàn toàn song song theo DP

Cost matrix có thể tính vectorized, nhưng recurrence có dependency theo anti-diagonal. Batch tốt trên nhiều trajectory pairs; bên trong một pair cần wavefront hoặc specialized kernel.

---

## **14. Đánh giá một trajectory metric**

Không nên đánh giá metric chỉ bằng một vài plots đẹp.

### Invariance tests

Tạo controlled transforms:

- resampling density;
- global time shift;
- nonlinear speed warp;
- additive latent noise;
- one-step outlier;
- reversed order;
- missing segment;
- duplicated states;
- rigid transform hoặc coordinate scaling;
- encoder checkpoint drift.

Mỗi transform phải có expected behavior rõ. Ví dụ DTW nên ổn với speed warp vừa phải nhưng không nên coi reversed path là giống.

### Retrieval evaluation

Với labeled behaviors:

- Recall@k;
- mean average precision;
- nearest-neighbor classification;
- rank stability qua noise/sampling;
- false matches giữa behaviors có cùng occupancy nhưng khác order.

### Alignment evaluation

Nếu có event correspondences:

$$
E_{\text{align}}
=
\frac{1}{K}
\sum_{k=1}^{K}
|j_k-\hat j_k|.
$$

Trong đó $j_k$ là correspondence thật và $\hat j_k$ đến từ warping path. Distance tốt nhưng alignment sai vẫn nguy hiểm cho interpolation hoặc event transfer.

### Calibration theo task

Khoảng cách nên tương quan với đại lượng cần bảo toàn:

- difference in return;
- action disagreement;
- event timing;
- constraint violation;
- decoded perceptual difference;
- human similarity judgment.

Một metric có retrieval accuracy tốt cho action class chưa chắc phù hợp để audit rollout safety.

---

## **15. Giới hạn / Khi nào thất bại**

**Sai local geometry.** Euclidean distance trên latent anisotropic làm mọi alignment kế thừa scale bias.

**Warping che lỗi timing.** DTW có thể coi event đúng shape nhưng trễ nguy hiểm là gần.

**Pathological many-to-one match.** Unconstrained DTW kéo dài một state qua một interval lớn và tạo cost thấp giả.

**Length bias.** Raw accumulated cost tăng theo path length; normalization lại có thể pha loãng một đoạn lỗi dài.

**Outlier domination.** Fréchet là bottleneck metric nên một corrupted point có thể quyết định toàn distance.

**Sampling dependence.** Discrete Fréchet và DTW thay đổi khi duplicate, downsample hoặc densify points dù underlying continuous path giống nhau.

**Không phân biệt sequence expansion.** DTW có thể cho zero cost giữa sequence khác nhau do lặp state, nên không phải metric chuẩn.

**Order bị xóa.** Hausdorff hoặc occupancy OT coi forward và reverse traversal giống nhau nếu visited set giống.

**Threshold brittleness.** LCSS/EDR đổi đột ngột khi distance đi qua $\varepsilon$.

**Soft-DTW bias.** Raw soft-DTW có thể âm và không tối thiểu đúng tại self-match; cần dùng đúng objective.

**Endpoint semantics sai.** Boundary chuẩn ép mọi prefix/suffix ghép nhau; open-end tùy tiện lại có thể bỏ phần thất bại quan trọng.

**Timestamp bị bỏ.** Index-based window không có cùng nghĩa khi sampling rates khác nhau.

**Projection trước distance.** Tính metric sau PCA/UMAP có thể đo distortion của projection thay vì latent trajectory gốc.

**Representation drift.** Encoder version khác nhau làm distance phản ánh coordinate change, không phải behavior change.

**Stochastic collapse.** So mean trajectories bỏ multimodality; average distance giữa particles có thể che rare catastrophic branch.

**Quadratic cost.** Long trajectories, retrieval corpus lớn và particle ensembles làm exact pairwise DP quá đắt.

**Metric hacking.** Nếu training trực tiếp tối ưu một trajectory loss, model có thể học representation thuận lợi cho metric nhưng bỏ semantics ngoài loss.

---

## **16. Quy trình chọn metric**

1. **Xác định object:** sampled sequence, polygonal curve, state-action path hay distribution of paths.
2. **Xác định invariance:** speed, shift, sampling density, rotation, latent scaling.
3. **Chọn local metric:** Euclidean chỉ sau khi xác nhận geometry/normalization.
4. **Chọn alignment:** fixed, monotone elastic, edit-based, partial hoặc set-based.
5. **Chọn aggregation:** sum, mean, max, match count hoặc risk statistic.
6. **Khai báo constraints:** endpoints, window, step pattern, timestamp tolerance.
7. **Kiểm thử counterexamples:** reverse, duplicate, outlier, missing segment, speed warp.
8. **Đo task correlation:** retrieval, return, event timing hoặc safety.
9. **Báo compute:** lengths, admissible cells, approximation và memory mode.

Một default hợp lý cho latent trajectories chưa có domain-specific requirement:

- standardized/Mahalanobis local metric;
- normalized DTW với moderate timestamp-aware band;
- báo thêm discrete Fréchet để thấy worst-case deviation;
- giữ pointwise timing score riêng;
- audit reverse/outlier/duplicate counterexamples.

Default này là điểm bắt đầu, không phải universal answer.

---

## **17. Liên hệ với Latent-Anything**

Trajectory similarity nên là một family có cấu hình, không phải một hàm `distance(a, b)` mơ hồ:

```python
measure = DTWMeasure(
    local_metric=MahalanobisMetric(covariance=reference_cov),
    window=TimestampWindow(max_delta=0.25),
    normalize="path_length",
    boundary="closed",
)

result = measure.compare(query, candidate, return_alignment=True)
```

Kết quả nên chứa:

```python
TrajectoryComparison(
    value=...,
    alignment=...,
    local_costs=...,
    timing_distortion=...,
    bottleneck_pair=...,
    diagnostics=...,
)
```

### Contract bắt buộc

- measure name và version;
- local metric + parameters;
- state fields được so;
- normalization;
- boundary/step/window policy;
- timestamp units;
- mask handling;
- source/model compatibility checks;
- approximation/pruning mode;
- scalar value có phải metric hay không.

### Layer A — Introspection

Layer A có thể:

- heatmap cost matrix và warping path;
- tìm nearest trajectories;
- cluster behavior;
- so posterior trajectory với rollout;
- plot DTW accumulated error và Fréchet bottleneck;
- phát hiện đoạn gây mismatch;
- audit timing distortion riêng khỏi shape distortion;
- so ensemble branches và uncertainty.

### Layer B — Manipulation

Alignment tạo correspondence để:

- transfer event labels;
- blend trajectories theo phase;
- splice matching segments;
- retime motion;
- build trajectory barycenter;
- align trước interpolation;
- chọn exemplar gần nhất cho editing.

Alignment path không tự bảo đảm transition feasibility. Sau manipulation vẫn cần [Latent Transition Model](02-latent-transition-model.md) hoặc [Rollout / Latent Imagination](07-rollout-latent-imagination.md) kiểm tra dynamics.

### Layer C — Runtime

Runtime cần:

- mask-aware batched cost matrix;
- anti-diagonal DP kernel;
- rolling-memory scalar mode;
- optional predecessor storage;
- Sakoe–Chiba/timestamp bands;
- early abandon theo current top-k threshold;
- coarse descriptor index;
- CPU/GPU dispatch;
- deterministic tie-breaking;
- cache keyed bởi trajectory version và metric config.

### API tách local metric khỏi sequence measure

```python
local = LatentMetric.from_space(space)
dtw = DynamicTimeWarping(local=local, constraints=...)
frechet = DiscreteFrechet(local=local)
hausdorff = Hausdorff(local=local)
```

Thiết kế này cho phép cùng alignment algorithm chạy trên Euclidean latent, Gaussian belief, spherical state hoặc decoder-induced geometry mà không hard-code coordinate assumptions.

### Checkpoint của Tầng 6

Sau mục này, primitive `Trajectory` có đủ nền lý thuyết cho:

- lưu và resample;
- smooth/segment;
- rollout/branch;
- compare/alignment/retrieval;
- interpolation theo correspondence;
- uncertainty-aware diagnostics.

Tầng 7 có thể dùng các phép đo này để đánh giá candidate rollouts, diversity, model error và consistency giữa imagined với realized trajectories.

---

## Liên quan

- [Latent Trajectory](06-latent-trajectory.md) — định nghĩa schema, timestamp, resampling và alignment problem.
- [Rollout và Latent Imagination](07-rollout-latent-imagination.md) — tạo các trajectories cần so trong world-model planning.
- [Latent Transition Model](02-latent-transition-model.md) — kiểm tra alignment hoặc interpolation có phù hợp dynamics hay không.
- [Stochastic Transition](03-stochastic-transition.md) — mở rộng từ một curve sang particles và distribution of trajectories.
- [Mahalanobis Distance](../../04-latent-computation/research/05-mahalanobis-distance.md) — local metric cho latent anisotropic.
- [Optimal Transport trong Latent](../../04-latent-computation/research/07-optimal-transport-in-latent.md) — so distribution của states hoặc trajectory ensembles.
- [Geodesic](../../01-space-representation/research/05-geodesic.md) — local/curve distance khi latent nằm trên manifold cong.
- [Hình học Riemannian](../../03-geometry-structure/research/04-riemannian-geometry.md) — metric tensor và intrinsic path geometry.

## Tham khảo

- H. Sakoe, S. Chiba, *Dynamic Programming Algorithm Optimization for Spoken Word Recognition* (IEEE Transactions on Acoustics, Speech, and Signal Processing 1978).
- H. Alt, M. Godau, *Computing the Fréchet Distance Between Two Polygonal Curves* (International Journal of Computational Geometry & Applications 1995).
- T. Eiter, H. Mannila, *Computing Discrete Fréchet Distance* (Technical Report CD-TR 94/64, TU Vienna 1994).
- M. Vlachos, G. Kollios, D. Gunopulos, *Discovering Similar Multidimensional Trajectories* (ICDE 2002).
- L. Chen, R. Ng, *On the Marriage of Lp-norms and Edit Distance* (VLDB 2004).
- L. Chen, M. T. Özsu, V. Oria, *Robust and Fast Similarity Search for Moving Object Trajectories* (SIGMOD 2005).
- P.-F. Marteau, *Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching* (IEEE TPAMI 2009, arXiv:cs/0703033).
- M. Cuturi, M. Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series* (ICML 2017, arXiv:1703.01541).
- M. Blondel, A. Mensch, J.-P. Vert, *Differentiable Divergences Between Time Series* (AISTATS 2021, arXiv:2010.08354).
- K. Bringmann, *Why Walking the Dog Takes Time: Fréchet Distance Has No Strongly Subquadratic Algorithms Unless SETH Fails* (FOCS 2014, arXiv:1404.1448).
- K. Bringmann, M. Künnemann, *Quadratic Conditional Lower Bounds for String Problems and Dynamic Time Warping* (FOCS 2015, arXiv:1502.01063).
