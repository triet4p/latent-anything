# Kalman Filter và các biến thể

> **TL;DR.** Kalman filter duy trì belief Gaussian bằng hai pha: dynamics dự báo state và covariance, sau đó measurement dùng innovation để hiệu chỉnh với Kalman gain. Với hệ tuyến tính-Gaussian, cập nhật này cho posterior Bayes chính xác; rộng hơn, nó là bộ ước lượng tuyến tính mean-square tối ưu dưới các giả thiết moment thích hợp. Giới hạn chính nằm ở model mismatch, phi tuyến mạnh, noise không được mô hình hóa đúng và sai số số học của covariance.

[Markov Property và State Space](01-markov-property-state-space.md) đặt state làm biến đủ để dự báo tương lai. Kalman filter bổ sung một câu hỏi thực dụng: nếu state không quan sát trực tiếp, dynamics và sensor đều có noise, belief về state phải được cập nhật như thế nào sau mỗi measurement?

Câu trả lời là một vòng lặp **predict–correct**. Prediction truyền cả mean lẫn uncertainty qua dynamics. Correction so sánh measurement thật với measurement dự báo, rồi phân bổ mức tin cậy giữa model và sensor bằng Kalman gain.

Kalman filter là trường hợp giải tích quan trọng nhất của stochastic state estimation. Nó làm rõ nhiều khái niệm xuất hiện lại trong [RSSM](04-rssm-recurrent-state-space-model.md): prior, posterior, innovation, observation correction, uncertainty propagation và khoảng cách giữa model prediction với evidence mới.

---

## **1. Bài toán state estimation**

Xét linear state-space model rời rạc:

$$
x_k = F_k x_{k-1} + B_k u_k + w_k,
$$

trong đó $x_k\in\mathbb{R}^{n}$ là state tại bước $k$, $F_k$ là ma trận transition, $u_k$ là control input, $B_k$ ánh xạ control vào state và $w_k$ là process noise. Phương trình nói rằng state mới gồm dynamics dự báo được cộng với nhiễu mô hình.

Sensor tạo measurement:

$$
y_k = H_k x_k + v_k,
$$

trong đó $y_k\in\mathbb{R}^{m}$ là measurement, $H_k$ là observation matrix và $v_k$ là measurement noise. Sensor chỉ nhìn một phép chiếu có noise của state, không nhất thiết quan sát đủ mọi chiều.

Giả thiết moment chuẩn là:

$$
\mathbb{E}[w_k]=0,
\qquad
\mathbb{E}[v_k]=0,
\qquad
\operatorname{Cov}(w_k)=Q_k,
\qquad
\operatorname{Cov}(v_k)=R_k.
$$

Trong đó $Q_k$ là process-noise covariance và $R_k$ là measurement-noise covariance. Hai ma trận này mô tả mức bất định được thêm bởi dynamics và sensor.

Thông thường còn giả thiết process noise, measurement noise và initial-state error không tương quan chéo:

$$
\mathbb{E}[w_k v_j^\top]=0
\quad\text{với mọi }k,j.
$$

Trong đó kỳ vọng bằng zero loại bỏ correlation chưa được model hóa giữa process và measurement noise. Nếu correlation tồn tại, công thức gain chuẩn phải được sửa.

### Gaussian cần ở đâu?

Nếu initial state và noise đều Gaussian, mọi phép biến đổi đều affine nên:

$$
p(x_k\mid y_{1:k})
=
\mathcal{N}(\hat x_{k\mid k},P_{k\mid k}).
$$

Trong đó $\hat x_{k\mid k}$ và $P_{k\mid k}$ là posterior mean và covariance sau khi thấy measurement tới bước $k$. Khi đó Gaussian belief là posterior đầy đủ, không chỉ là xấp xỉ moment.

Nếu noise không Gaussian nhưng có finite second moments và thỏa các giả thiết cần thiết, cùng recursion vẫn cho linear minimum mean-square error estimator. Tuy nhiên mean và covariance không còn mô tả toàn bộ posterior; tail, skewness và multimodality có thể bị mất.

| Cách diễn giải | Điều kiện | Kalman filter tối ưu theo nghĩa nào? |
|---|---|---|
| **Bayesian Gaussian filter** | model tuyến tính, prior và noise Gaussian | posterior mean/covariance chính xác |
| **Linear least-squares estimator** | model tuyến tính, moment bậc hai tồn tại và correlation được mô hình hóa | tối ưu trong lớp estimator tuyến tính |
| **Heuristic moment filter** | assumptions bị vi phạm đáng kể | không có bảo đảm tối ưu tổng quát |

---

## **2. Hai pha predict–correct**

Ký hiệu:

- $\hat x_{k\mid k-1}$: prior mean tại $k$, trước measurement $y_k$;
- $P_{k\mid k-1}$: prior covariance;
- $\hat x_{k\mid k}$: posterior mean sau measurement;
- $P_{k\mid k}$: posterior covariance.

### 2.1 Prediction

State mean được truyền qua dynamics:

$$
\hat x_{k\mid k-1}
=
F_k\hat x_{k-1\mid k-1}
+
B_k u_k.
$$

Trong đó posterior mean ở bước trước trở thành đầu vào cho transition. Đây là state dự báo trước khi sensor hiện tại được đọc.

Covariance được truyền bằng:

$$
P_{k\mid k-1}
=
F_kP_{k-1\mid k-1}F_k^\top
+
Q_k.
$$

Trong đó phép nhân $F_kP F_k^\top$ biến đổi uncertainty cũ qua dynamics, còn $Q_k$ thêm uncertainty do model không hoàn hảo. Prediction thường làm covariance tăng.

### 2.2 Innovation

Measurement dự báo là:

$$
\hat y_{k\mid k-1}
=
H_k\hat x_{k\mid k-1}.
$$

Trong đó $\hat y_{k\mid k-1}$ là measurement mà model kỳ vọng từ prior state.

Innovation hoặc residual:

$$
\tilde y_k
=
y_k-\hat y_{k\mid k-1}.
$$

Trong đó $\tilde y_k$ đo phần evidence mới chưa được prior giải thích. Innovation gần zero nghĩa là sensor phù hợp với prediction, không có nghĩa state estimate chắc chắn đúng.

Innovation covariance:

$$
S_k
=
H_kP_{k\mid k-1}H_k^\top
+
R_k.
$$

Trong đó $S_k$ cộng uncertainty của state khi chiếu sang measurement space với noise sensor. Nó đặt scale để đánh giá innovation lớn hay nhỏ.

### 2.3 Kalman gain

Gain chuẩn:

$$
K_k
=
P_{k\mid k-1}H_k^\top S_k^{-1}.
$$

Trong đó $K_k\in\mathbb{R}^{n\times m}$ ánh xạ innovation từ measurement space về state space. Gain lớn theo một hướng nghĩa là measurement có nhiều thông tin so với prior uncertainty ở hướng đó.

Posterior mean:

$$
\hat x_{k\mid k}
=
\hat x_{k\mid k-1}
+
K_k\tilde y_k.
$$

Trong đó correction $K_k\tilde y_k$ dịch prior mean theo phần measurement chưa được giải thích. Đây là Bayesian observation correction trong trường hợp Gaussian.

Covariance dạng rút gọn:

$$
P_{k\mid k}
=
(I-K_kH_k)P_{k\mid k-1}.
$$

Trong đó $I-K_kH_k$ loại uncertainty trên các hướng measurement quan sát được. Dạng này gọn nhưng có thể mất đối xứng hoặc positive semidefiniteness do floating-point error.

Joseph form ổn định số hơn:

$$
P_{k\mid k}
=
(I-K_kH_k)P_{k\mid k-1}(I-K_kH_k)^\top
+
K_kR_kK_k^\top.
$$

Trong đó số hạng đầu truyền prior error qua correction, còn số hạng thứ hai giữ lại uncertainty do measurement noise. Với gain chính xác, Joseph form tương đương đại số với dạng rút gọn nhưng bảo toàn cấu trúc covariance tốt hơn trong tính toán hữu hạn độ chính xác.

Trong code không nên tạo $S_k^{-1}$ tường minh. Nên giải linear system hoặc dùng Cholesky factorization:

```python
gain = solve(innovation_covariance.T, cross_covariance.T).T
```

Phép `solve` ổn định và hiệu quả hơn nhân với inverse được materialize.

---

## **3. Trực giác của Kalman gain**

Trong trường hợp scalar với prior variance $P^-$ và measurement variance $R$:

$$
K
=
\frac{P^-}{P^-+R}.
$$

Trong đó $K\in[0,1]$ khi các variance dương. Gain tiến tới 1 khi prior rất bất định so với sensor, và tiến tới 0 khi sensor rất noisy so với prior.

Posterior mean là weighted average:

$$
\hat x^+
=
(1-K)\hat x^-
+
Ky.
$$

Trong đó $\hat x^-$ là prior mean, $y$ là measurement và $\hat x^+$ là posterior mean. Hai trọng số phụ thuộc precision tương đối chứ không được chọn thủ công.

Posterior variance:

$$
P^+
=
(1-K)P^-
=
\left(
\frac{1}{P^-}
+
\frac{1}{R}
\right)^{-1}.
$$

Trong đó precision posterior bằng tổng precision của prior và measurement. Hai nguồn thông tin độc lập làm uncertainty giảm.

### Không phải mọi gain đều nằm giữa 0 và 1

Trong hệ nhiều chiều, entry của $K_k$ có thể âm hoặc lớn hơn 1 vì covariance chéo ánh xạ một measurement sang nhiều state dimensions. Chỉ scalar gain đơn giản mới có trực giác convex combination trực tiếp.

---

## **4. $Q$, $R$ và ý nghĩa của uncertainty**

$Q_k$ và $R_k$ không chỉ là hyperparameter smoothing. Chúng định nghĩa cách filter hiểu lỗi:

- $Q_k$ lớn: dynamics không đáng tin, prediction covariance tăng nhanh, filter nghe sensor nhiều hơn;
- $R_k$ lớn: sensor không đáng tin, gain giảm, estimate bám dynamics nhiều hơn;
- cả hai quá nhỏ: filter quá tự tin, innovation thực tế lớn hơn mức dự báo;
- cả hai quá lớn: estimate uncertainty cao và correction có thể chậm.

Scale tương đối thường quan trọng hơn scale tuyệt đối đối với mean update, nhưng scale tuyệt đối quyết định calibration của covariance.

### Process noise không đồng nghĩa state noise vật lý

$Q_k$ thường hấp thụ:

- disturbance thật chưa đo;
- discretization error;
- parameter drift;
- dynamics phi tuyến bị tuyến tính hóa;
- input hoặc actuator uncertainty;
- state dimensions bị bỏ khỏi model.

Vì vậy một $Q_k$ hợp lý có thể lớn hơn noise vật lý đo trực tiếp. Nó biểu diễn toàn bộ model error được quy về state transition.

### Covariance phải có semantics rõ

Một covariance matrix hợp lệ cần đối xứng và positive semidefinite:

$$
P_k=P_k^\top,
\qquad
a^\top P_k a\ge 0
\quad\text{với mọi }a.
$$

Trong đó điều kiện thứ hai bảo đảm variance theo mọi hướng $a$ không âm. Covariance mất PSD thường là lỗi số học, sai Jacobian hoặc update không nhất quán.

---

## **5. Innovation diagnostics và consistency**

Innovation sequence không chỉ phục vụ update; nó là tín hiệu audit model.

Normalized Innovation Squared:

$$
\operatorname{NIS}_k
=
\tilde y_k^\top S_k^{-1}\tilde y_k.
$$

Trong đó NIS là squared Mahalanobis distance của innovation dưới covariance dự báo. Nếu model tuyến tính-Gaussian đúng và measurement có $m$ chiều, NIS tuân theo phân phối $\chi^2_m$.

NIS liên tục quá lớn thường báo:

- $Q$ hoặc $R$ bị đánh giá thấp;
- dynamics hoặc observation model sai;
- outlier hoặc regime change;
- timestamp, frame hoặc unit mismatch.

NIS liên tục quá nhỏ có thể báo covariance bị phóng đại hoặc measurement bị dùng lặp lại như evidence độc lập.

Nếu ground-truth state có sẵn, Normalized Estimation Error Squared:

$$
\operatorname{NEES}_k
=
(x_k-\hat x_{k\mid k})^\top
P_{k\mid k}^{-1}
(x_k-\hat x_{k\mid k}).
$$

Trong đó NEES đo state error theo uncertainty filter tự báo. Với state dimension $n$ và assumptions đúng, NEES tuân theo $\chi^2_n$.

[Mahalanobis Distance](../../04-latent-computation/research/05-mahalanobis-distance.md) cung cấp hình học đằng sau NIS và NEES: residual phải được chuẩn hóa theo hướng uncertainty, không chỉ lấy Euclidean norm.

### Whiteness của innovation

Với filter nhất quán, innovation sau khi normalize nên gần zero-mean, đúng covariance và ít autocorrelation. Innovation còn cấu trúc theo thời gian nghĩa là state hoặc dynamics chưa hấp thụ hết dependency có thể dự báo.

---

## **6. Observability và missing measurements**

Một state dimension chỉ được estimate nếu measurement theo thời gian chứa đủ thông tin về nó. Với hệ time-invariant, observability matrix là:

$$
\mathcal{O}
=
\begin{bmatrix}
H\\
HF\\
HF^2\\
\vdots\\
HF^{n-1}
\end{bmatrix}.
$$

Trong đó $F$ là transition matrix, $H$ là observation matrix và $n$ là state dimension. Hệ observable khi $\operatorname{rank}(\mathcal{O})=n$.

Ví dụ sensor chỉ đo position vẫn có thể estimate velocity nếu dynamics liên kết position và velocity qua thời gian. Ngược lại, một latent dimension không ảnh hưởng measurement hiện tại hoặc tương lai không thể được measurement correction xác định.

Khi measurement bị thiếu, filter chỉ chạy prediction:

$$
\hat x_{k\mid k}=\hat x_{k\mid k-1},
\qquad
P_{k\mid k}=P_{k\mid k-1}.
$$

Trong đó không có innovation hay gain update. Covariance tăng qua mỗi bước do dynamics và $Q_k$, rồi giảm khi sensor quay lại nếu measurement có thông tin.

---

## **7. Extended Kalman Filter**

Với dynamics và observation phi tuyến:

$$
x_k=f(x_{k-1},u_k)+w_k,
\qquad
y_k=h(x_k)+v_k.
$$

Trong đó $f$ và $h$ là hàm phi tuyến. Posterior sau phép biến đổi thường không còn Gaussian.

Extended Kalman Filter (EKF) tuyến tính hóa cục bộ quanh estimate hiện tại:

$$
F_k
=
\left.
\frac{\partial f}{\partial x}
\right|_{\hat x_{k-1\mid k-1},u_k},
\qquad
H_k
=
\left.
\frac{\partial h}{\partial x}
\right|_{\hat x_{k\mid k-1}}.
$$

Trong đó $F_k$ và $H_k$ là Jacobian của dynamics và observation tại linearization points. EKF dùng chúng trong covariance recursion giống Kalman filter tuyến tính.

Mean prediction vẫn đi qua hàm phi tuyến:

$$
\hat x_{k\mid k-1}
=
f(\hat x_{k-1\mid k-1},u_k).
$$

Trong đó EKF biến đổi mean đúng theo hàm $f$, nhưng covariance chỉ dùng xấp xỉ bậc nhất. Sai số lớn khi curvature mạnh hoặc uncertainty trải rộng.

Measurement prediction:

$$
\hat y_{k\mid k-1}
=
h(\hat x_{k\mid k-1}).
$$

Trong đó innovation được tính quanh measurement của prior mean. Nếu Jacobian tại mean bằng zero dù distribution sau biến đổi có variance lớn, EKF có thể kết luận sai rằng measurement không chứa thông tin.

### Điểm yếu của EKF

- cần Jacobian đúng và nhất quán về coordinate frame;
- linearization error không được phản ánh đầy đủ trong $P$;
- estimate xa state thật có thể làm Jacobian sai và filter divergence;
- góc, quaternion và manifold state cần error-state hoặc manifold-aware formulation;
- discontinuity và multimodality không phù hợp với Gaussian local approximation.

---

## **8. Unscented Kalman Filter**

Unscented Kalman Filter (UKF) không tuyến tính hóa hàm bằng Jacobian. Nó chọn một tập sigma points có cùng mean và covariance với Gaussian belief, truyền từng point qua hàm phi tuyến, rồi khôi phục moment đầu ra.

Với state dimension $n$, một construction phổ biến dùng $2n+1$ sigma points:

$$
\chi_0=\mu,
\qquad
\chi_i=\mu+\left[\sqrt{(n+\lambda)P}\right]_i,
\qquad
\chi_{i+n}=\mu-\left[\sqrt{(n+\lambda)P}\right]_i.
$$

Trong đó $\mu$ và $P$ là input mean/covariance, $\lambda$ điều khiển spread và cột thứ $i$ của matrix square root tạo cặp sigma points đối xứng. Tập điểm tái tạo đúng hai moment đầu trước biến đổi.

Mỗi sigma point đi qua hàm:

$$
\mathcal{Y}_i=g(\chi_i).
$$

Trong đó $g$ có thể là dynamics $f$ hoặc observation $h$. UKF đánh giá hàm tại nhiều điểm thay vì chỉ đạo hàm tại mean.

Output mean và covariance được xấp xỉ:

$$
\bar y
=
\sum_i W_i^{(m)}\mathcal{Y}_i,
\qquad
P_y
=
\sum_i W_i^{(c)}
(\mathcal{Y}_i-\bar y)
(\mathcal{Y}_i-\bar y)^\top.
$$

Trong đó $W_i^{(m)}$ và $W_i^{(c)}$ là weights cho mean và covariance. Unscented transform bắt được một phần curvature mà first-order Taylor expansion bỏ qua.

### UKF không phải Monte Carlo

Sigma points được chọn có cấu trúc để khớp moment, không phải random samples. Vì số điểm tăng tuyến tính theo state dimension, UKF thường rẻ hơn particle filter nhưng đắt hơn EKF khi Jacobian dễ tính.

### Điểm yếu của UKF

- cần matrix square root ổn định;
- scaling parameters ảnh hưởng spread và numerical behavior;
- vẫn ép posterior thành một Gaussian;
- chi phí $O(n^3)$ do covariance factorization và matrix operations;
- sigma points có thể đi vào vùng không hợp lệ của state space;
- high-dimensional latent làm số evaluations $2n+1$ trở nên đáng kể.

---

## **9. Các biến thể quan trọng**

| Biến thể | Representation của belief | Cơ chế chính | Phù hợp khi |
|---|---|---|---|
| **Kalman filter** | mean + full covariance | exact linear update | hệ tuyến tính, Gaussian hoặc cần LMMSE |
| **EKF** | Gaussian cục bộ | Jacobian linearization | phi tuyến nhẹ, Jacobian sẵn có |
| **UKF** | Gaussian qua sigma points | deterministic moment transform | phi tuyến vừa, Jacobian khó hoặc dễ sai |
| **Square-root filter** | Cholesky factor của covariance | factor updates | cần ổn định số và PSD |
| **Information filter** | precision + information vector | cộng thông tin measurement | sensor fusion phân tán, sparse precision |
| **Ensemble Kalman filter** | ensemble samples | sample covariance + Kalman-style correction | state rất lớn, covariance đầy đủ quá đắt |
| **RTS smoother** | filtered beliefs toàn chuỗi | backward correction | offline estimation dùng cả future measurements |
| **Particle filter** | weighted particles | sampling + reweighting | posterior phi Gaussian hoặc đa mode mạnh |

### Square-root filter

Thay vì lưu $P$, square-root filter lưu factor $L$:

$$
P=LL^\top.
$$

Trong đó $L$ thường là Cholesky factor. Update trực tiếp trên factor giúp duy trì positive semidefiniteness và cải thiện conditioning.

### Information filter

Information form dùng:

$$
\Lambda=P^{-1},
\qquad
\eta=P^{-1}\hat x.
$$

Trong đó $\Lambda$ là precision matrix và $\eta$ là information vector. Independent measurements đóng góp các information terms có thể cộng trực tiếp, nhưng prediction thường phức tạp hơn covariance form.

### Ensemble Kalman Filter

EnKF biểu diễn belief bằng ensemble $\{x_k^{(i)}\}_{i=1}^{N}$ và dùng sample covariance:

$$
\hat P_k
=
\frac{1}{N-1}
\sum_{i=1}^{N}
(x_k^{(i)}-\bar x_k)
(x_k^{(i)}-\bar x_k)^\top.
$$

Trong đó $N$ là ensemble size và $\bar x_k$ là ensemble mean. EnKF tránh materialize covariance dynamics chính xác ở state dimension rất lớn, nhưng sample covariance hạng thấp và có spurious long-range correlations khi ensemble nhỏ.

### Rauch–Tung–Striebel smoother

Filtering chỉ dùng measurements tới hiện tại:

$$
p(x_k\mid y_{1:k}).
$$

Trong đó belief phù hợp cho online inference.

Smoothing dùng cả measurements tương lai:

$$
p(x_k\mid y_{1:T}),
\qquad T>k.
$$

Trong đó future evidence có thể sửa state estimate trong quá khứ. RTS smoother chạy backward pass trên các filtered beliefs của hệ tuyến tính-Gaussian.

Particle filter không phải biến thể Kalman theo nghĩa chặt, nhưng là mốc so sánh quan trọng: nó có thể giữ multimodality mà mọi Gaussian filter ở trên đều làm mất.

---

## **10. So sánh EKF và UKF**

| | EKF | UKF |
|---|---|---|
| Xấp xỉ | Taylor bậc nhất | deterministic moment matching |
| Cần Jacobian | có | không |
| Số lần gọi hàm | thường một lần mỗi update | $2n+1$ hoặc tương tự |
| Phi tuyến vừa | có thể tốt | thường bắt curvature tốt hơn |
| State dimension lớn | thuận lợi hơn nếu Jacobian sparse | sigma-point cost tăng |
| Numerical issue | Jacobian sai, covariance inconsistency | square root, negative weights, invalid sigma points |
| Multimodality | không giữ | không giữ |

UKF không mặc định tốt hơn EKF. Một EKF có error-state formulation đúng, Jacobian giải tích và manifold handling phù hợp có thể nhanh và chính xác hơn. UKF hữu ích khi moment transform phi tuyến là nguồn sai số chính và số chiều đủ nhỏ.

---

## **11. Quan hệ với RSSM**

Kalman filter và RSSM có cùng skeleton:

| Kalman filter | RSSM |
|---|---|
| prior $\hat x_{k\mid k-1},P_{k\mid k-1}$ | dynamics prior $p_\theta(z_t\mid h_t)$ |
| measurement model $H_k$ | encoder/representation model |
| innovation $y_k-H_k\hat x^-$ | evidence observation chưa được prior giải thích |
| posterior $\hat x_{k\mid k},P_{k\mid k}$ | posterior $q_\phi(z_t\mid h_t,o_t)$ |
| open-loop prediction | latent imagination |

Khác biệt cốt lõi:

- Kalman filter có transition và observation semantics tường minh;
- posterior update được suy ra giải tích trong linear-Gaussian case;
- RSSM học amortized posterior và nonlinear dynamics từ data;
- RSSM thường dùng latent state không có coordinate semantics cố định;
- RSSM có deterministic memory $h_t$ ngoài stochastic state $z_t$.

Kalman filter vì vậy là baseline audit tốt. Nếu một learned stochastic transition không vượt được KF/EKF trên bài toán có structure gần tuyến tính, độ phức tạp thêm chưa được biện minh.

---

## **12. Giới hạn / Khi nào thất bại**

**Model mismatch.** Filter chỉ nhất quán nếu $F$, $H$, $Q$, $R$ và noise correlations phản ánh hệ thật đủ tốt. Covariance có thể rất nhỏ trong khi estimate sai nếu model tự tin sai.

**Phi tuyến mạnh.** EKF và UKF đều nén posterior về một Gaussian. Folding, saturation, discontinuity hoặc ambiguous measurement có thể tạo posterior đa mode mà mean/covariance không biểu diễn được.

**Unobservable state.** Covariance update không tạo thông tin từ hư không. Một latent dimension không tác động measurement theo thời gian sẽ không được sensor xác định.

**Outlier và heavy tail.** Gaussian likelihood làm squared residual lớn chi phối update. Gating, robust likelihood, mixture model hoặc particle methods có thể cần thiết.

**Noise có bias hoặc correlation.** Zero-mean independent-noise assumption bị phá bởi sensor bias, colored noise, shared calibration error hoặc asynchronous measurements. Cần augment state hoặc mô hình correlation.

**Sai unit, frame và timestamp.** Đây là failure mode triển khai phổ biến hơn lỗi công thức. Innovation lớn do meter–millimeter, world–camera frame hoặc latency mismatch dễ bị hiểu sai thành process noise.

**Covariance collapse.** $Q$ quá nhỏ, repeated measurement hoặc double-counting evidence làm filter quá tự tin. Gain giảm và filter không còn sửa được khi model drift.

**Numerical instability.** Trừ hai ma trận gần nhau, inverse tường minh và covariance ill-conditioned có thể phá đối xứng/PSD. Joseph form, square-root filtering và symmetrization có kiểm soát giúp giảm rủi ro.

**Dimension scaling.** Full covariance cần $O(n^2)$ memory và thường $O(n^3)$ factorization. Với latent hàng nghìn hoặc hàng triệu chiều, cần sparse, diagonal, low-rank, ensemble hoặc structured covariance.

**Tuning không chuyển domain.** $Q$ và $R$ hiệu quả ở một tốc độ, sensor hoặc regime không chắc còn calibrated sau domain shift.

**Smoother không dùng được online.** RTS dùng future measurements nên không hợp với runtime cần quyết định tức thời. Không được so filtered online estimate với smoothed offline estimate như hai hệ có cùng thông tin.

---

## **13. Liên hệ với Latent-Anything**

Kalman-family adapter cần biểu diễn belief, không chỉ một latent tensor:

```python
belief = GaussianBelief(
    mean=state_mean,
    covariance=state_covariance,
    timestamp=timestamp,
    source="posterior",
)
```

Metadata tối thiểu gồm:

- state coordinate semantics và unit;
- covariance parameterization: full, diagonal, low-rank hay square-root;
- process và measurement noise models;
- source: prior, filtered posterior hay smoothed posterior;
- timestamp, frame và control interval;
- innovation, innovation covariance, NIS và gating decision;
- numerical factorization và jitter đã dùng.

### Layer A — Introspection

Layer A có thể:

- plot confidence ellipse/ellipsoid theo thời gian;
- kiểm tra covariance eigenvalues và condition number;
- audit innovation mean, autocorrelation và NIS;
- phát hiện missing measurements, outlier và regime change;
- so predicted covariance với empirical error;
- kiểm tra observability cục bộ hoặc theo trajectory.

### Layer B — Manipulation

Layer B có thể:

- fuse measurement mới;
- inflate covariance khi domain shift;
- marginalize hoặc condition Gaussian belief;
- chuyển full covariance sang diagonal/low-rank approximation;
- chạy counterfactual với $Q$, $R$ hoặc sensor subset khác;
- smooth trajectory offline.

Manipulation cần giữ consistency. Sửa mean mà không sửa covariance, hoặc đổi coordinate frame nhưng không biến đổi covariance, tạo belief không còn semantics xác suất đúng.

### Layer C — Runtime

Layer C cần:

- dùng linear solve/factorization thay inverse;
- batch independent filters khi shape cho phép;
- cache transition và observation structure;
- xử lý measurement mask, asynchronous timestamps và reset;
- chọn square-root hoặc low-rank representation theo condition/dimension;
- tách online filtering khỏi offline smoothing;
- không materialize full covariance nếu adapter chỉ hứa diagonal uncertainty.

### API gợi ý

```python
prior = filter.predict(posterior, control, delta_time)
posterior, diagnostics = filter.update(prior, measurement)
smoothed = smoother.smooth(filtered_trajectory)
```

`predict`, `update` và `smooth` phải là operation riêng. Gộp chúng vào một `step` duy nhất làm mờ nguồn thông tin và khiến benchmark dễ dùng future evidence hoặc measurement correction ngoài ý muốn.

Với learned latent, Kalman update có thể hoạt động trên local linear dynamics hoặc structured Gaussian state. Tuy nhiên covariance trong latent coordinate chỉ có ý nghĩa nếu adapter khai báo coordinate transform và metric; một reparameterization tùy ý có thể thay đổi covariance components dù underlying uncertainty không đổi.

---

## Liên quan

- [Markov Property và State Space](01-markov-property-state-space.md) — đặt điều kiện để state hiện tại tóm tắt history cho prediction.
- [Latent Transition Model](02-latent-transition-model.md) — cung cấp transition dynamics mà Kalman prediction mở rộng bằng covariance propagation.
- [Stochastic Transition](03-stochastic-transition.md) — phân biệt process noise, predictive distribution và uncertainty semantics.
- [RSSM — Recurrent State Space Model](04-rssm-recurrent-state-space-model.md) — learned nonlinear filter với prior/posterior và observation correction.
- [Mahalanobis Distance](../../04-latent-computation/research/05-mahalanobis-distance.md) — nền tảng hình học cho NIS, NEES và innovation gating.
- [VAE](../../02-representation-learning/research/03-vae.md) — một cách khác học Gaussian latent posterior bằng variational inference.

## Tham khảo

- R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems* (Journal of Basic Engineering 1960).
- R. E. Kalman, R. S. Bucy, *New Results in Linear Filtering and Prediction Theory* (Journal of Basic Engineering 1961).
- H. E. Rauch, F. Tung, C. T. Striebel, *Maximum Likelihood Estimates of Linear Dynamic Systems* (AIAA Journal 1965).
- S. J. Julier, J. K. Uhlmann, *A New Extension of the Kalman Filter to Nonlinear Systems* (AeroSense 1997).
- S. J. Julier, J. K. Uhlmann, *A New Method for the Nonlinear Transformation of Means and Covariances in Filters and Estimators* (IEEE Transactions on Automatic Control 2000).
- S. J. Julier, J. K. Uhlmann, *Unscented Filtering and Nonlinear Estimation* (Proceedings of the IEEE 2004).
- G. Evensen, *Sequential Data Assimilation with a Nonlinear Quasi-Geostrophic Model Using Monte Carlo Methods to Forecast Error Statistics* (Journal of Geophysical Research 1994).
- G. Evensen, *The Ensemble Kalman Filter: Theoretical Formulation and Practical Implementation* (Ocean Dynamics 2003).
- J. L. Crassidis, J. L. Junkins, *Optimal Estimation of Dynamic Systems* (CRC Press 2004).
- J. J. LaViola Jr., *A Comparison of Unscented and Extended Kalman Filtering for Estimating Quaternion Motion* (American Control Conference 2003).
- M. B. Bordonaro, *The Linear Kalman Filter Does Not Require Gaussian Assumptions* (arXiv 2024, arXiv:2405.00058).
