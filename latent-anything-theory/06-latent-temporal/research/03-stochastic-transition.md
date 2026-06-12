# Stochastic Transition

> **TL;DR.** Stochastic transition dự báo một phân phối $p_\theta(z_{t+1}\mid z_t,a_t)$ thay vì một điểm duy nhất, nhờ đó có thể biểu diễn process noise và nhiều tương lai khả dĩ khi rollout trong latent. Baseline liên tục phổ biến là Gaussian có mean và variance phụ thuộc input, được học bằng negative log-likelihood và lấy mẫu qua reparameterization. Caveat chính là Gaussian đơn chỉ mô hình hóa uncertainty aleatoric đơn mode; nó không tự giải quyết partial observability, multimodality hay epistemic uncertainty do thiếu dữ liệu.

Trong [Latent Transition Model](02-latent-transition-model.md), dynamics tất định ánh xạ mỗi cặp $(z_t,a_t)$ tới một state kế tiếp $\hat z_{t+1}$. Cách này phù hợp khi latent gần Markov, môi trường gần tất định và mục tiêu chủ yếu là dự báo trung bình. Khi cùng state và action có thể dẫn tới nhiều kết quả hợp lệ, một điểm dự báo duy nhất làm mất cấu trúc của tương lai.

Stochastic transition thay đổi output contract từ một tensor state thành một **phân phối có điều kiện**. Model không chỉ trả lời "state tiếp theo là gì?" mà còn trả lời "những state nào có thể xảy ra, với xác suất tương đối ra sao?".

---

## **1. Từ hàm chuyển trạng thái đến transition kernel**

Transition tất định có dạng:

$$
z_{t+1}=f_\theta(z_t,a_t).
$$

Trong đó $z_t$ là latent state hiện tại, $a_t$ là action và $f_\theta$ là hàm dynamics. Với mỗi input, output chỉ có một giá trị.

Dưới góc nhìn xác suất, đây là một phân phối suy biến:

$$
p_\theta(z_{t+1}\mid z_t,a_t)
=
\delta\!\left(z_{t+1}-f_\theta(z_t,a_t)\right),
$$

trong đó $\delta$ là Dirac delta tập trung toàn bộ xác suất tại $f_\theta(z_t,a_t)$. Phương trình cho thấy deterministic transition là trường hợp riêng có variance bằng không của một transition kernel tổng quát.

Stochastic transition thay delta distribution bằng một phân phối có độ rộng hữu hạn:

$$
z_{t+1}\sim p_\theta(z_{t+1}\mid z_t,a_t).
$$

Trong đó $p_\theta$ là transition kernel do model tham số hóa. Cùng $(z_t,a_t)$ có thể sinh ra các mẫu $z_{t+1}$ khác nhau, nhưng phân phối của chúng phải phản ánh đúng xác suất chuyển trạng thái của dữ liệu.

### Khi nào cần stochastic transition?

Các nguồn ngẫu nhiên thường gặp gồm:

- contact dynamics nhạy với sai số nhỏ;
- tác động của agent khác chưa được điều khiển;
- process noise trong actuator hoặc môi trường;
- biến ẩn chưa quan sát được nhưng không thể suy luận chính xác;
- các sự kiện rời rạc như object trượt hay không trượt, va chạm theo nhánh nào, hoặc đối thủ chọn chiến lược nào.

Không phải mọi residual khó dự báo đều là stochasticity nội tại. Nếu latent thiếu velocity, friction hay object bị occlude, nguyên nhân gốc là state chưa đủ theo [Markov Property và State Space](01-markov-property-state-space.md). Thêm noise vào transition có thể che lỗi representation mà không khôi phục thông tin đã mất.

---

## **2. Gaussian transition**

Với latent liên tục $z_t\in\mathbb{R}^d$, baseline phổ biến là Gaussian đường chéo:

$$
p_\theta(z_{t+1}\mid z_t,a_t)
=
\mathcal{N}\!\left(
z_{t+1};
\mu_\theta(z_t,a_t),
\operatorname{diag}\!\left(\sigma_\theta^2(z_t,a_t)\right)
\right).
$$

Trong đó $\mu_\theta\in\mathbb{R}^d$ là mean dự báo, $\sigma_\theta^2\in\mathbb{R}_{>0}^d$ là variance theo từng chiều và $\theta$ là tham số network. Vì cả mean lẫn variance phụ thuộc $(z_t,a_t)$, model biểu diễn **heteroscedastic uncertainty**: độ bất định thay đổi theo vùng state-action.

Network thường output log-variance thay vì variance trực tiếp:

$$
\mu_t,\ell_t=f_\theta(z_t,a_t),
\qquad
\sigma_t^2=\exp(\ell_t).
$$

Trong đó $\ell_t=\log\sigma_t^2$ không bị ràng buộc trên trục thực, còn phép mũ bảo đảm variance dương. Trong implementation, $\ell_t$ thường được clamp vào một khoảng hữu hạn hoặc chuyển qua `softplus` để tránh variance bằng không và overflow.

### Residual Gaussian transition

Khi timestep nhỏ, mean có thể dùng residual parameterization:

$$
p_\theta(z_{t+1}\mid z_t,a_t)
=
\mathcal{N}\!\left(
z_t+\Delta_\theta(z_t,a_t),
\operatorname{diag}(\sigma_\theta^2(z_t,a_t))
\right).
$$

Trong đó $\Delta_\theta$ là mean increment của latent và $\sigma_\theta^2$ mô tả độ phân tán quanh increment đó. Cách viết này tách xu hướng chuyển động trung bình khỏi process noise cục bộ.

### Diagonal và full covariance

| Covariance | Số tham số | Giữ tương quan giữa chiều? | Phù hợp khi |
|---|---:|---|---|
| **Isotropic** $\sigma^2I$ | $1$ | không | baseline rất rẻ, các chiều cùng scale |
| **Diagonal** $\operatorname{diag}(\sigma^2)$ | $d$ | không | latent dimension vừa/lớn, batching quan trọng |
| **Full** $\Sigma$ | $d(d+1)/2$ | có | dimension nhỏ, correlation có ý nghĩa rõ |
| **Low-rank + diagonal** $UU^\top+D$ | $O(dr)$ | một phần | cần correlation nhưng full covariance quá đắt |

Full covariance cần được parameterize sao cho positive definite:

$$
\Sigma_\theta(z_t,a_t)
=
L_\theta(z_t,a_t)L_\theta(z_t,a_t)^\top+\epsilon I.
$$

Trong đó $L_\theta$ là ma trận tam giác dưới, $\epsilon>0$ là jitter số học và $I$ là ma trận đơn vị. Phép nhân $LL^\top$ bảo đảm covariance nửa xác định dương, còn jitter tránh singularity khi tính inverse hoặc log-determinant.

Diagonal Gaussian rẻ nhưng giả định các latent coordinate độc lập có điều kiện. Giả định này đặc biệt yếu khi latent anisotropic, các chiều biểu diễn pose liên kết, hoặc state nằm trên manifold cong.

---

## **3. Học mean và variance bằng likelihood**

Cho một transition quan sát được $(z_t,a_t,z_{t+1})$, Gaussian negative log-likelihood đường chéo, bỏ hằng số không phụ thuộc tham số, là:

$$
\mathcal{L}_{\text{NLL}}
=
\frac{1}{2}
\sum_{j=1}^{d}
\left[
\frac{(z_{t+1,j}-\mu_{t,j})^2}{\sigma_{t,j}^2}
+
\log\sigma_{t,j}^2
\right].
$$

Trong đó $j$ đánh chỉ số latent dimension, $\mu_{t,j}$ và $\sigma_{t,j}^2$ là mean và variance model dự báo. Số hạng đầu phạt prediction error theo precision, còn số hạng log-variance ngăn model tăng variance vô hạn để né lỗi.

NLL khác MSE ở chỗ mỗi sample và mỗi chiều tự học scale của residual. Với variance cố định $\sigma^2$, tối thiểu hóa Gaussian NLL tương đương tối thiểu hóa MSE sau khi nhân một hằng số. Khi variance được học, model phải cân bằng hai quyết định:

- mean nên giải thích phần dynamics có thể dự báo;
- variance nên giải thích phần phân tán còn lại.

Sự cân bằng này không luôn ổn định. Heteroscedastic NLL có thể tạo nghiệm mean kém nhưng variance lớn tại sample khó, vì gradient của mean bị chia cho $\sigma^2$. Seitzer và cộng sự chỉ ra failure mode này và đề xuất $\beta$-NLL để giảm việc sample có variance lớn bị bỏ qua trong tối ưu.

### Reparameterization để lấy mẫu khả vi

Một sample Gaussian được viết lại thành:

$$
\epsilon_t\sim\mathcal{N}(0,I),
\qquad
z_{t+1}
=
\mu_t+\sigma_t\odot\epsilon_t.
$$

Trong đó $\epsilon_t$ là noise không phụ thuộc tham số, $\odot$ là nhân từng phần tử và $\sigma_t$ là standard deviation. Randomness được tách khỏi $\mu_t,\sigma_t$, nên gradient có thể backpropagate qua sample tới transition network như trong [VAE](../../02-representation-learning/research/03-vae.md).

Reparameterization hữu ích khi reward, decoder hoặc multi-step loss nằm sau sample. Nó không làm sampling trở nên tất định; nó chỉ tạo một estimator gradient có variance thấp hơn so với score-function estimator trong trường hợp Gaussian.

---

## **4. Một Gaussian không đủ cho tương lai đa mode**

Giả sử một object sau va chạm có thể bật sang trái hoặc sang phải với xác suất gần bằng nhau. Single Gaussian phải đặt mean ở giữa hai nhánh:

$$
p_\theta(z_{t+1}\mid z_t,a_t)
=
\mathcal{N}(\mu_\theta,\Sigma_\theta).
$$

Trong đó $\mu_\theta$ nằm gần conditional mean của hai outcome. Nếu vùng giữa hai nhánh không tương ứng với state hợp lệ, model vẫn tạo probability density ở một tương lai không thể xảy ra.

Mixture density transition giữ nhiều mode tường minh:

$$
p_\theta(z_{t+1}\mid z_t,a_t)
=
\sum_{k=1}^{K}
\pi_{\theta,k}(z_t,a_t)\,
\mathcal{N}\!\left(
z_{t+1};
\mu_{\theta,k}(z_t,a_t),
\Sigma_{\theta,k}(z_t,a_t)
\right).
$$

Trong đó $K$ là số component, $\pi_{\theta,k}\ge 0$ là mixture weight với $\sum_k\pi_{\theta,k}=1$, còn $(\mu_{\theta,k},\Sigma_{\theta,k})$ là Gaussian thứ $k$. Sampling gồm hai bước: chọn component theo categorical distribution $\pi$, rồi lấy mẫu Gaussian trong component đó.

World Models dùng mixture-density RNN để dự báo phân phối latent kế tiếp. Thí nghiệm của Ha và Schmidhuber cho thấy giảm sampling temperature quá thấp có thể làm dynamics gần tất định, bỏ mất event hiếm và tạo một simulator dễ bị controller khai thác. Temperature ở đây là tham số điều chỉnh độ ngẫu nhiên khi sample, không phải bằng chứng rằng xác suất đã được calibration.

| Distribution family | Điểm mạnh | Giới hạn |
|---|---|---|
| **Single Gaussian** | đơn giản, NLL và KL thường có closed form | chỉ một mode, ellipse-shaped density |
| **Gaussian mixture** | biểu diễn nhánh rời rạc rõ | component collapse, cost tăng theo $K$ |
| **Normalizing flow** | density linh hoạt, sampling khả vi | invertibility và Jacobian làm model phức tạp |
| **Implicit generator** | có thể sinh phân phối rất phức tạp | thường không có tractable likelihood |
| **Discrete latent transition** | tự nhiên cho regime/event rời rạc | cần estimator hoặc objective phù hợp cho biến rời rạc |

[Normalizing Flows](../../03-geometry-structure/research/06-normalizing-flows.md) mở rộng một base distribution thành density phi Gaussian bằng phép biến đổi khả nghịch. Tuy nhiên, distribution family giàu hơn không sửa được state aliasing hoặc thiếu coverage trong dữ liệu.

---

## **5. Ba nguồn uncertainty không nên trộn lẫn**

| Nguồn | Ý nghĩa | Có giảm bằng thêm dữ liệu cùng loại? | Cơ chế phù hợp |
|---|---|---|---|
| **Aleatoric / process uncertainty** | dynamics thật có randomness | không, nếu state đã đầy đủ | output distribution, process noise |
| **Partial observability / state uncertainty** | không biết state thật vì observation thiếu hoặc nhiễu | có thể giảm bằng history, sensor hoặc inference tốt hơn | posterior/belief state, recurrent memory |
| **Epistemic / model uncertainty** | chưa biết dynamics vì dữ liệu hữu hạn hoặc OOD | thường có | Bayesian model, ensemble, posterior trên tham số |

### Aleatoric uncertainty

Aleatoric uncertainty là độ ngẫu nhiên còn lại ngay cả khi biết đúng dynamics và có vô hạn dữ liệu. Trong transition model, nó tương ứng với process noise:

$$
z_{t+1}=f_\theta(z_t,a_t)+\varepsilon_t,
\qquad
\varepsilon_t\sim p_\theta(\varepsilon\mid z_t,a_t).
$$

Trong đó $\varepsilon_t$ là nhiễu quá trình có thể phụ thuộc state và action. Gaussian output với input-dependent variance là một cách mô hình hóa aleatoric uncertainty đơn mode.

### Partial observability không đồng nghĩa process noise

Nếu encoder không biết object đang đi sang trái hay phải, posterior của state hiện tại đã bất định trước khi transition diễn ra:

$$
q_\phi(z_t\mid o_{\le t},a_{<t}).
$$

Trong đó $q_\phi$ là posterior xấp xỉ được suy ra từ observation-action history. Uncertainty này nên được cập nhật khi có observation mới; đẩy toàn bộ nó vào $p_\theta(z_{t+1}\mid z_t,a_t)$ làm transition gánh lỗi của state estimator.

### Epistemic uncertainty

Một probabilistic neural network output Gaussian vẫn chỉ là một hàm có bộ trọng số cố định. Ngoài vùng dữ liệu, variance nó output có thể tùy ý; distribution head không tự biết network chưa học dynamics ở đó.

Ensemble dùng $M$ model độc lập:

$$
p(z_{t+1}\mid z_t,a_t,\mathcal{D})
\approx
\frac{1}{M}
\sum_{m=1}^{M}
p_{\theta_m}(z_{t+1}\mid z_t,a_t).
$$

Trong đó $\mathcal{D}$ là training data và $\theta_m$ là tham số của ensemble member thứ $m$. Variance bên trong mỗi member biểu diễn aleatoric uncertainty, còn disagreement giữa các member là proxy cho epistemic uncertainty.

PETS kết hợp probabilistic networks với bootstrapped ensembles chính vì hai nguồn uncertainty có ý nghĩa khác nhau cho planning. Process noise không biến mất khi thu thập thêm dữ liệu, còn model disagreement có thể chỉ ra vùng cần exploration.

---

## **6. Prior transition và posterior inference**

Trong latent state-space model, transition tạo **prior** cho state kế tiếp:

$$
p_\theta(z_t\mid z_{t-1},a_{t-1}).
$$

Trong đó prior chỉ dùng state và action quá khứ, nên đây là phân phối có thể rollout khi không có observation tương lai.

Khi observation $o_t$ xuất hiện, encoder hoặc filter tạo posterior:

$$
q_\phi(z_t\mid z_{t-1},a_{t-1},o_t).
$$

Trong đó posterior được "nhìn" observation hiện tại và thường hẹp hoặc chính xác hơn prior. Khoảng cách giữa hai phân phối đo mức prediction surprise của transition trước dữ liệu mới.

Một variational state-space objective cơ bản là:

$$
\mathcal{L}_{\text{ELBO}}
=
\sum_{t=1}^{T}
\mathbb{E}_{q_\phi}
\left[
\log p_\theta(o_t\mid z_t)
\right]
-
\sum_{t=1}^{T}
D_{\mathrm{KL}}
\left(
q_\phi(z_t\mid z_{t-1},a_{t-1},o_t)
\;\|\;
p_\theta(z_t\mid z_{t-1},a_{t-1})
\right).
$$

Trong đó số hạng log-likelihood ép latent giải thích observation, còn KL kéo transition prior về posterior đã được observation hiệu chỉnh. Tối đa hóa ELBO đồng thời học encoder, stochastic transition và decoder.

DVBF dùng variational inference để học latent Markov state-space model từ raw data và cho gradient đi xuyên transition. PlaNet sau đó dùng cả deterministic lẫn stochastic dynamics, đồng thời đề xuất latent overshooting để prior rollout nhiều bước khớp posterior tương lai thay vì chỉ tối ưu one-step KL.

Đây là cầu nối tới **RSSM (mục tiếp theo)**: deterministic recurrent path giữ memory, stochastic state giữ uncertainty và information bottleneck. Note hiện tại tập trung vào semantics của phân phối transition; RSSM tập trung vào cách tổ chức hai path đó trong một architecture.

---

## **7. Rollout một phân phối**

Với action sequence $a_{t:t+H-1}$, predictive distribution sau $k$ bước phải tích phân qua các state trung gian:

$$
p(z_{t+k}\mid z_t,a_{t:t+k-1})
=
\int
\prod_{i=0}^{k-1}
p_\theta(z_{t+i+1}\mid z_{t+i},a_{t+i})
\;dz_{t+1:t+k-1}.
$$

Trong đó $z_{t+1:t+k-1}$ là các latent trung gian bị marginalize. Với neural transition phi tuyến, tích phân này hiếm khi có closed form.

### Mean propagation

Cách rẻ nhất là dùng mean ở mỗi bước:

$$
\hat z_{t+i+1}
=
\mu_\theta(\hat z_{t+i},a_{t+i}).
$$

Trong đó toàn bộ variance bị bỏ qua sau mỗi transition. Vì $\mathbb{E}[f(Z)]\ne f(\mathbb{E}[Z])$ với $f$ phi tuyến, mean rollout nhìn chung không bằng expected trajectory và có thể đi qua vùng giữa các mode.

### Particle rollout

Monte Carlo giữ $N$ trajectory particles:

$$
z_{t+i+1}^{(n)}
\sim
p_\theta\!\left(
z_{t+i+1}\mid z_{t+i}^{(n)},a_{t+i}
\right),
\qquad n=1,\ldots,N.
$$

Trong đó mỗi particle $n$ là một tương lai khả dĩ. Reward expectation, risk hoặc event probability được xấp xỉ bằng thống kê trên particle population.

Khi dùng ensemble, epistemic hypothesis nên thường được giữ nhất quán dọc một particle rollout: chọn model member $m_n$ cho particle $n$ rồi giữ member đó qua horizon. Nếu đổi member ở mỗi bước, rollout đang mô phỏng một dynamics function thay đổi theo thời gian và có thể làm epistemic uncertainty bị trộn sai.

### Moment matching

Moment matching nén particle distribution về Gaussian sau mỗi bước:

$$
\mu_{t+1}
=
\frac{1}{N}\sum_{n=1}^{N}z_{t+1}^{(n)},
\qquad
\Sigma_{t+1}
=
\frac{1}{N-1}
\sum_{n=1}^{N}
(z_{t+1}^{(n)}-\mu_{t+1})
(z_{t+1}^{(n)}-\mu_{t+1})^\top.
$$

Trong đó $(\mu_{t+1},\Sigma_{t+1})$ là Gaussian xấp xỉ particle cloud. Cách này tiết kiệm memory nhưng làm mất multimodality; hai nhánh trajectory có thể bị ép thành một ellipse nằm giữa chúng.

---

## **8. Đánh giá stochastic transition**

### Likelihood

Negative log-likelihood đo model gán bao nhiêu probability mass cho transition thật. NLL đánh giá cả mean lẫn dispersion, nhưng có thể bị chi phối bởi distribution family sai hoặc variance inflation.

### Calibration và sharpness

Một predictive interval mức $1-\alpha$ được calibration nếu transition thật rơi vào interval với tần suất gần $1-\alpha$ trên nhiều sample. Calibration tốt chưa đủ: interval cực rộng luôn dễ đạt coverage, nên phải báo cáo cùng **sharpness**, tức độ hẹp của distribution khi vẫn giữ coverage đúng.

### Probability integral transform và rank

Với output một chiều liên tục, giá trị CDF tại target thật nên gần uniform nếu distribution được calibration. Với particle rollout nhiều chiều, có thể dùng rank histogram, coverage theo projection hoặc task-specific event probability.

### Multi-step distribution quality

Đánh giá phải đi xa hơn one-step NLL:

- coverage và NLL theo horizon;
- mode coverage của trajectory distribution;
- calibration của reward hoặc termination probability;
- xác suất event hiếm;
- divergence giữa predicted state occupancy và encoded trajectory;
- performance khi planner dùng expectation, quantile hoặc risk-sensitive objective.

Một model có one-step NLL tốt vẫn có thể làm variance phình quá nhanh, collapse về một mode, hoặc drift khỏi latent manifold khi rollout.

### Phân rã uncertainty

Với ensemble probabilistic, cần báo cáo riêng:

- average within-member variance cho aleatoric uncertainty;
- variance giữa member means cho epistemic uncertainty;
- total predictive variance của mixture.

Nếu chỉ báo total variance, Layer A không biết uncertainty nào có thể giảm bằng thêm dữ liệu và uncertainty nào là process noise phải được planner chấp nhận.

---

## **9. Chọn distribution family**

| Tình huống | Baseline hợp lý | Dấu hiệu cần nâng cấp |
|---|---|---|
| Dynamics liên tục, gần tất định | deterministic residual model | residual có variance thay đổi rõ theo state |
| Noise liên tục, một mode | diagonal Gaussian | correlation mạnh hoặc support cong |
| Latent nhỏ, correlation quan trọng | full/low-rank Gaussian | distribution có nhiều nhánh tách biệt |
| Event có vài outcome rời rạc | Gaussian mixture hoặc discrete mode + continuous residual | số mode thay đổi, density quá phức tạp |
| Planning ngoài vùng dữ liệu | probabilistic ensemble | ensemble vẫn đồng thuận sai do shared bias |
| Partial observability | posterior state + recurrent memory | posterior còn đa mode hoặc filter collapse |

Quy tắc thực dụng là bắt đầu bằng deterministic baseline, kiểm tra residual theo state-action, rồi thêm stochasticity ở nơi dữ liệu chứng minh cần thiết. Distribution giàu hơn làm training và rollout đắt hơn; nó chỉ đáng giá nếu cải thiện likelihood, calibration hoặc decision quality ở horizon quan tâm.

---

## **10. Giới hạn / Khi nào thất bại**

**Variance trở thành thùng chứa lỗi.** Mean underfit có thể được che bằng variance lớn. NLL giảm nhưng prediction không sắc và planner nhận uncertainty phình không đúng nguyên nhân.

**Variance collapse và lỗi số học.** Khi target gần mean, optimizer có thể đẩy variance về rất nhỏ, làm NLL và gradient mất ổn định. Cần variance floor, log-variance bounds hoặc parameterization ổn định.

**Single Gaussian average các mode.** Có output variance không đồng nghĩa đã mô hình hóa multimodality. Một ellipse rộng vẫn đặt density vào vùng không có state hợp lệ.

**Mixture component collapse.** Một vài component có thể chiếm toàn bộ mixture weight, component khác chết; hoán vị component cũng làm việc theo dõi mode qua thời gian khó khăn.

**Partial observability bị gán nhầm cho process noise.** Transition học variance lớn vì state không đủ. Khi thêm observation hoặc history, uncertainty đáng lẽ phải giảm, nhưng một process-noise head không có cơ chế correction.

**Probabilistic output không phải epistemic uncertainty.** Gaussian variance của một network có thể tự tin sai ngoài phân phối. Planner dễ khai thác chính vùng model chưa biết nếu không có ensemble, Bayesian approximation hoặc support constraint.

**Distribution family sai.** Diagonal Gaussian trên coordinate tùy ý có thể không tôn trọng manifold, periodic variable, quaternion, positive scale hay cấu trúc set. Sampling hợp lệ về số học chưa chắc hợp lệ về geometry.

**Sampling cost tăng theo horizon và particle count.** Nhiều particle giúp giữ mode nhưng làm planning đắt. Quá ít particle bỏ event hiếm; resampling không cẩn thận gây particle impoverishment.

**Uncertainty compounding sai.** Independent noise ở mỗi bước có thể làm variance tăng quá nhanh, trong khi correlated disturbance hoặc persistent latent regime cần một biến ẩn dùng chung qua nhiều bước.

**Calibration không bền dưới action shift.** Variance học trên behavior policy không đáng tin ở action do planner tối ưu nhưng dữ liệu ít cover. Calibration phải được đo theo vùng state-action, không chỉ trung bình toàn validation set.

**Temperature không thay thế calibration.** Tăng hoặc giảm sampling temperature có thể làm simulator khó hay dễ hơn, nhưng không bảo đảm distribution khớp tần suất thật.

---

## **11. Liên hệ với Latent-Anything**

Stochastic transition yêu cầu interface giàu hơn `next_state = transition.step(state, action)`. Một plugin nên trả về một predictive object có semantics tường minh:

```python
prediction = transition.predict(state, action)
next_state = prediction.sample(generator=rng)
log_prob = prediction.log_prob(observed_next_state)
```

Predictive object cần khai báo:

- distribution family và event shape;
- mean, covariance hoặc distribution parameters;
- `sample`, `rsample` và `log_prob` có được support hay không;
- uncertainty là aleatoric, epistemic hay hỗn hợp;
- covariance parameterization và geometry constraint;
- random seed, sampling temperature và numerical bounds;
- ensemble-member semantics khi rollout;
- training và calibration domain theo state-action.

`Trajectory` không nên chỉ lưu một mean path. Nó cần support:

- particle trajectories và weights;
- mean/covariance summary theo thời gian;
- ensemble member hoặc model hypothesis của từng particle;
- posterior state từ observation và prior state từ transition;
- random seed để tái lập stochastic rollout.

Layer A có thể cung cấp calibration plot, interval coverage, decomposition aleatoric/epistemic, mode occupancy và uncertainty growth theo horizon. Layer B dùng distribution để planning theo expected return, lower confidence bound, quantile hoặc probability of constraint violation. Layer C chịu trách nhiệm vectorize particle rollout, giữ RNG state và tránh materialize decoder output cho mọi particle.

### Hai covariance khác nhau trong Gaussian-centric world model

Với [Gaussian Parameters là Latent Variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md), mỗi 3D Gaussian đã có covariance hình học $\Sigma_i^{\text{shape}}$ mô tả extent và orientation của primitive. Stochastic transition có thể đồng thời dự báo uncertainty trên chính các parameter đó:

$$
p\!\left(
\mu_{t+1,i},
\Sigma_{t+1,i}^{\text{shape}},
\alpha_{t+1,i},
c_{t+1,i}
\mid
\mathcal{Z}_t,a_t
\right).
$$

Trong đó $\mu_i$, $\Sigma_i^{\text{shape}}$, $\alpha_i$ và $c_i$ là position, shape covariance, opacity và appearance của primitive. Predictive covariance của distribution trên các parameter này **không phải** $\Sigma_i^{\text{shape}}$: một cái biểu diễn model/process uncertainty, cái kia biểu diễn hình dạng không gian của primitive.

Schema và tên field phải giữ hai tầng covariance tách biệt, ví dụ `shape_covariance` và `predictive_covariance`. Nếu gộp chúng, Layer A không thể biết Gaussian đang có kích thước lớn hay model chỉ không chắc vị trí và hình dạng của nó.

---

## Liên quan

- [Markov Property và State Space](01-markov-property-state-space.md) — phân biệt process stochasticity với uncertainty do state chưa đủ hoặc observation bị che khuất.
- [Latent Transition Model](02-latent-transition-model.md) — baseline tất định, residual dynamics và multi-step rollout mà stochastic transition mở rộng.
- [VAE](../../02-representation-learning/research/03-vae.md) — cung cấp variational inference, KL và reparameterization dùng để học stochastic latent state.
- [Normalizing Flows](../../03-geometry-structure/research/06-normalizing-flows.md) — distribution family linh hoạt hơn Gaussian cho conditional density phức tạp.
- [Density Estimation trong Latent](../../04-latent-computation/research/06-density-estimation.md) — support và density diagnostics bổ sung cho predictive uncertainty khi rollout.
- [Gaussian Parameters là Latent Variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) — trường hợp cần phân biệt covariance hình học của primitive với covariance dự báo của transition.
- [Causal Intervention vs Observational Study](../../05-probing-intervention/research/04-causal-intervention-vs-observational.md) — uncertainty phải được đánh giá dưới action intervention, không chỉ trên behavior-policy trajectories.

## Tham khảo

- C. M. Bishop, *Mixture Density Networks* (Aston University Technical Report NCRG/94/004, 1994).
- M. Karl, M. Soelch, J. Bayer, P. van der Smagt, *Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data* (ICLR 2017, arXiv:1605.06432).
- B. Lakshminarayanan, A. Pritzel, C. Blundell, *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (NeurIPS 2017, arXiv:1612.01474).
- A. Kendall, Y. Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* (NeurIPS 2017, arXiv:1703.04977).
- D. Ha, J. Schmidhuber, *World Models* (arXiv 2018, arXiv:1803.10122).
- K. Chua, R. Calandra, R. McAllister, S. Levine, *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* (NeurIPS 2018, arXiv:1805.12114).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (ICML 2019, arXiv:1811.04551).
- M. Seitzer, A. Tavakoli, D. Antic, G. Martius, *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks* (ICLR 2022, arXiv:2203.09168).
