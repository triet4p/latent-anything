# Latent Trajectory

> **TL;DR.** Latent trajectory là một đường đi có thứ tự theo thời gian $\mathcal{Z}=\{(t_i,z_i)\}_{i=0}^{T}$, không chỉ là tập các latent points. Các operation như resampling, smoothing, segmentation, alignment và interpolation phải giữ rõ timestamp, geometry, uncertainty và nguồn state. Caveat chính là latent distance và đường nối giữa hai state không tự có ý nghĩa vật lý: smoothing hoặc interpolation đẹp trong embedding vẫn có thể rời data manifold, vi phạm dynamics hoặc xóa event quan trọng.

Một latent point $z_t$ mô tả state tại một thời điểm. Một latent trajectory mô tả **state thay đổi như thế nào**: đi qua vùng nào, nhanh hay chậm, khi nào đổi regime, có quay lại hay phân nhánh, và uncertainty tích lũy ra sao.

[Latent Transition Model](02-latent-transition-model.md) định nghĩa một bước $z_t\to z_{t+1}$. [RSSM](04-rssm-recurrent-state-space-model.md) tạo posterior trajectory khi có observation và prior trajectory khi imagination. Latent trajectory gom các state đó thành một primitive có temporal semantics rõ để visualization, segmentation, retrieval, manipulation và runtime rollout dùng chung.

---

## **1. Định nghĩa rời rạc và liên tục**

### Trajectory rời rạc

Một trajectory có timestamp:

$$
\mathcal{Z}
=
\{(t_i,z_i)\}_{i=0}^{T},
\qquad
t_0<t_1<\cdots<t_T.
$$

Trong đó $z_i\in\mathcal{M}$ là latent state trên không gian $\mathcal{M}$ tại timestamp $t_i$. Thứ tự và khoảng thời gian giữa các điểm là một phần của dữ liệu, không phải metadata tùy chọn.

Nếu sampling đều với interval $\Delta t$, có thể viết ngắn:

$$
\mathcal{Z}
=
(z_0,z_1,\ldots,z_T),
\qquad
t_i=t_0+i\Delta t.
$$

Trong đó $\Delta t$ phải được lưu hoặc quy ước rõ. Hai trajectory có cùng latent points nhưng $\Delta t$ khác nhau có cùng đường hình học nhưng dynamics khác nhau.

### Trajectory liên tục

Một curve liên tục là ánh xạ:

$$
z:[t_0,t_1]\to\mathcal{M}.
$$

Trong đó $z(t)$ trả state ở mọi thời điểm trong interval. Dữ liệu rời rạc chỉ sample curve này tại hữu hạn timestamps.

Latent ODE mô hình hóa continuous-time dynamics:

$$
\frac{dz(t)}{dt}
=
f_\theta(z(t),t).
$$

Trong đó $f_\theta$ là vector field học được và nghiệm ODE tạo continuous latent trajectory. Cách biểu diễn này xử lý time gaps không đều tự nhiên hơn recursion giả định một bước có duration cố định.

### Trajectory không đồng nghĩa set

Hai sequence có cùng tập điểm nhưng thứ tự đảo ngược là hai trajectory khác nhau:

$$
(z_0,z_1,z_2)
\neq
(z_2,z_1,z_0).
$$

Trong đó khác biệt nằm ở orientation theo thời gian. Set-based metric hoặc pooling có thể xóa hoàn toàn thông tin này.

---

## **2. Schema của một trajectory**

Một representation tối thiểu không chỉ có tensor shape $(T+1,d)$:

```python
trajectory = Trajectory(
    states=z,
    timestamps=t,
    mask=valid,
    source="posterior",
)
```

Trong thực tế cần thêm:

- `actions`: control áp dụng giữa hai states;
- `observations`: reference hoặc embedding của evidence;
- `rewards`, `continuations`, `events`;
- `uncertainty`: covariance, distribution parameters, particles hoặc ensemble;
- `source`: observed, posterior, prior, imagined, smoothed hay edited;
- `episode_id`, `reset_mask`, terminal semantics;
- model/adapter version, coordinate frame và unit;
- sampling policy và interpolation method.

### State-aligned và edge-aligned fields

State thường nằm tại node $i$, còn action mô tả interval $[t_i,t_{i+1})$:

$$
z_i=z(t_i),
\qquad
a_i=a([t_i,t_{i+1})).
$$

Trong đó trajectory có $T+1$ states nhưng thường chỉ có $T$ actions. Reward và continuation phải khai báo gắn với state trước, state sau hay transition edge; off-by-one ở đây làm hỏng training target và return.

### Dense, padded và ragged

| Representation | Ưu điểm | Rủi ro |
|---|---|---|
| **Dense fixed-length** | batch nhanh, shape đơn giản | phải crop/pad, dễ nhầm padding là state thật |
| **Padded + mask** | batch được sequence khác length | mọi operation phải tôn trọng mask |
| **Ragged/list** | giữ length tự nhiên | khó vectorize và compile |
| **Chunked stream** | hợp trajectory dài | boundary state/cache phải nhất quán |

Padding value không được dùng để suy ra mask. Một latent zero có thể là state hợp lệ.

---

## **3. Đại lượng hình học và động học**

### Displacement

Displacement giữa hai thời điểm:

$$
\Delta z_i
=
z_{i+1}-z_i.
$$

Trong đó phép trừ giả định latent space có vector structure. Với manifold, displacement nên được thay bằng logarithmic map trong tangent space.

### Velocity với timestamp không đều

Finite-difference velocity:

$$
v_i
=
\frac{z_{i+1}-z_i}{t_{i+1}-t_i}.
$$

Trong đó denominator giữ semantics tốc độ. Dùng $z_{i+1}-z_i$ trực tiếp khi time gaps không đều sẽ trộn chuyển động nhanh với interval dài.

### Acceleration

Một xấp xỉ acceleration tại interval kế tiếp:

$$
a_i
=
\frac{v_{i+1}-v_i}
{\tfrac{1}{2}\left[(t_{i+2}-t_{i+1})+(t_{i+1}-t_i)\right]}.
$$

Trong đó mẫu số là duration trung bình giữa hai velocity estimates. Acceleration lớn có thể chỉ event, regime switch hoặc encoder jitter.

### Path length

Với metric $d$:

$$
L(\mathcal{Z})
=
\sum_{i=0}^{T-1}
d(z_i,z_{i+1}).
$$

Trong đó path length phụ thuộc metric latent. Euclidean length chỉ hợp lý khi coordinate scale và geometry hỗ trợ Euclidean distance.

Với continuous curve và Riemannian metric tensor $G(z)$:

$$
L[z]
=
\int_{t_0}^{t_1}
\sqrt{
\dot z(t)^\top
G(z(t))
\dot z(t)
}
\,dt.
$$

Trong đó $\dot z(t)$ là tangent velocity và $G(z)$ quy định local geometry. [Hình học Riemannian](../../03-geometry-structure/research/04-riemannian-geometry.md) giải thích vì sao cùng coordinate displacement có thể có độ dài nội tại khác nhau.

### Tortuosity

Một measure đơn giản cho độ vòng vèo:

$$
\tau(\mathcal{Z})
=
\frac{L(\mathcal{Z})}
{d(z_0,z_T)}.
$$

Trong đó $\tau\ge 1$ với metric thỏa triangle inequality và endpoint khác nhau. Giá trị lớn nghĩa là trajectory đi đường dài so với displacement đầu-cuối; nếu endpoints trùng nhau, ratio không xác định.

---

## **4. Sampling và resampling**

Trajectory từ sensor, simulator hoặc event stream thường có timestamps không đều. Trước khi dùng convolution, finite difference hoặc pointwise comparison, cần quyết định grid thời gian.

Grid đều:

$$
\bar t_j
=
t_0+j\Delta t,
\qquad
j=0,\ldots,N.
$$

Trong đó $\Delta t$ là sampling interval mục tiêu. Resampling cần tạo $\bar z_j\approx z(\bar t_j)$.

### Các lựa chọn resampling

| Phương pháp | Giả định | Điểm yếu |
|---|---|---|
| **Nearest neighbor** | state piecewise constant | tạo jump và duplicate |
| **Linear interpolation** | local path gần đoạn thẳng | cắt qua vùng unsupported |
| **Spline** | curve đủ trơn | overshoot, không causal |
| **Model rollout** | transition model đúng | compounding error |
| **Posterior interpolation/smoothing** | có evidence quanh gap | dùng future information |

Resampling không tạo thông tin thật. Một gap dài được điền bằng curve trơn vẫn phải giữ mask hoặc uncertainty để caller biết đoạn nào là inferred.

### Aliasing

Nếu sampling rate quá thấp so với dynamics, nhiều trajectory liên tục khác nhau có thể tạo cùng samples. Interpolation sau đó không thể phục hồi event đã mất. Downsampling cần anti-alias filtering nếu latent chứa thành phần tần số cao có ý nghĩa.

---

## **5. Smoothing**

Smoothing nhằm giảm noise hoặc encoder jitter nhưng giữ trend và event cần thiết.

### Moving average

Với window bán kính $r$:

$$
\tilde z_i
=
\frac{1}{2r+1}
\sum_{j=-r}^{r}z_{i+j}.
$$

Trong đó $\tilde z_i$ là smoothed state. Phương pháp đơn giản nhưng làm bẹt peak, gây lag nếu dùng causal window và không tôn trọng manifold geometry.

### Savitzky–Golay

Trong mỗi local window, fit polynomial bậc $p$ bằng least squares:

$$
\hat\beta
=
\arg\min_{\beta}
\sum_{j=-r}^{r}
\left\|
z_{i+j}
-
\sum_{q=0}^{p}\beta_q j^q
\right\|_2^2.
$$

Trong đó $\beta_q$ là coefficient polynomial cục bộ. Smoothed value là $\beta_0$, còn derivative có thể lấy từ các coefficient bậc cao; local polynomial thường giữ peak và slope tốt hơn moving average.

### Regularized trajectory smoothing

Có thể tối ưu toàn trajectory:

$$
\tilde{\mathcal{Z}}
=
\arg\min_{\tilde z_{0:T}}
\sum_{i=0}^{T}
\|z_i-\tilde z_i\|_2^2
+
\lambda
\sum_{i=1}^{T-1}
\|\tilde z_{i+1}-2\tilde z_i+\tilde z_{i-1}\|_2^2.
$$

Trong đó số hạng đầu giữ fidelity với data, số hạng thứ hai phạt discrete curvature/acceleration và $\lambda$ điều chỉnh độ trơn. $\lambda$ lớn tạo curve mượt nhưng có thể xóa regime change.

### Model-based smoothing

[Kalman Filter và các biến thể](05-kalman-filter-variants.md) phân biệt filtering và smoothing:

$$
p(z_i\mid o_{0:i})
\quad\text{so với}\quad
p(z_i\mid o_{0:T}).
$$

Trong đó filtering chỉ dùng evidence tới hiện tại, còn smoothing dùng cả future observations. Smoothed trajectory thường chính xác hơn offline nhưng không causal.

### Smoothing theo geometry

Coordinate-wise smoothing không invariant dưới reparameterization latent. Nếu decoder induce metric hoặc state nằm trên sphere/group, smoothing nên dùng tangent-space average, geodesic regression hoặc model-consistent objective.

---

## **6. Segmentation**

Segmentation chia trajectory thành các interval có dynamics, statistics hoặc semantics tương đối đồng nhất:

$$
0=\tau_0<\tau_1<\cdots<\tau_K=T.
$$

Trong đó $\tau_k$ là changepoints và segment $k$ chứa states từ $\tau_{k-1}$ tới $\tau_k$.

### Feature-based boundaries

Boundary score có thể dựa trên:

- speed hoặc acceleration spike;
- turning angle/curvature;
- reconstruction hoặc prediction error;
- prior–posterior KL trong RSSM;
- reward, contact, terminal hoặc discrete event;
- change trong covariance/entropy;
- cluster/state label transition.

Một score kết hợp:

$$
s_i
=
\alpha\|v_i-v_{i-1}\|
+
\beta D_{\mathrm{KL}}(q_i\|p_i)
+
\gamma e_i,
$$

trong đó $v_i$ là latent velocity, $D_{\mathrm{KL}}$ đo model surprise, $e_i$ là task-specific event/error và $\alpha,\beta,\gamma$ là weights. Threshold trên $s_i$ dễ triển khai nhưng nhạy với scale.

### Penalized changepoint objective

Một formulation tổng quát:

$$
\min_{\tau_{1:K-1}}
\sum_{k=1}^{K}
C\!\left(
z_{\tau_{k-1}:\tau_k}
\right)
+
\beta K.
$$

Trong đó $C$ đo within-segment cost và $\beta$ phạt số segment. Penalty nhỏ gây over-segmentation; penalty lớn gộp mất event.

PELT của Killick, Fearnhead và Eckley tìm nghiệm tối ưu cho lớp penalized cost này với pruning; dưới các điều kiện nhất định, expected computational cost tăng tuyến tính theo số observations.

### Segmentation khác simplification

Segmentation tìm **boundary semantics**. Curve simplification như Ramer–Douglas–Peucker chọn subset points để giữ hình dạng trong tolerance. Một corner hình học có thể là segment boundary, nhưng hai bài toán không đồng nhất.

---

## **7. Similarity và temporal alignment**

Pointwise distance cho hai trajectory cùng grid:

$$
D_{\text{point}}
(\mathcal{Z},\mathcal{Y})
=
\frac{1}{T+1}
\sum_{i=0}^{T}
d(z_i,y_i).
$$

Trong đó metric này so state tại cùng index. Nó phạt mạnh hai trajectory cùng path nhưng chạy khác tốc độ.

### Alignment trước comparison

Một monotone alignment path:

$$
\pi
=
\bigl((i_1,j_1),\ldots,(i_L,j_L)\bigr)
$$

trong đó indices tăng đơn điệu và mỗi pair ghép state của hai trajectory. Dynamic Time Warping tìm path có accumulated local cost thấp; Fréchet distance kiểm soát maximum separation dưới monotone reparameterization.

**Trajectory similarity metrics (mục sau)** sẽ đi sâu vào DTW và Fréchet. Ở đây nguyên tắc quan trọng là phải khai báo invariance mong muốn:

- có cho time shift không;
- có cho speed variation không;
- có giữ direction không;
- có cho skip/outlier không;
- so geometry, dynamics hay task outcome;
- dùng Euclidean, cosine, Mahalanobis hay learned metric.

### Descriptor-level similarity

Có thể so trajectory qua summary:

- endpoint/displacement;
- path length và tortuosity;
- mean/max speed;
- occupancy histogram hoặc state visitation distribution;
- event sequence;
- spectral features;
- pooled encoder representation.

Summary rẻ và dễ index nhưng có thể coi hai paths khác topology là giống nhau.

---

## **8. Interpolation trong và giữa trajectory**

### Interpolation theo thời gian

Giữa hai samples:

$$
z(t)
=
(1-\alpha)z_i+\alpha z_{i+1},
\qquad
\alpha
=
\frac{t-t_i}{t_{i+1}-t_i}.
$$

Trong đó $\alpha\in[0,1]$ là normalized time. Đây là [Lerp](../../04-latent-computation/research/01-lerp.md) theo coordinate Euclidean.

Nếu latent nằm gần hypersphere, [Slerp](../../04-latent-computation/research/02-slerp.md) giữ angular path và norm tốt hơn:

$$
\operatorname{slerp}(z_0,z_1;\alpha)
=
\frac{\sin((1-\alpha)\theta)}{\sin\theta}z_0
+
\frac{\sin(\alpha\theta)}{\sin\theta}z_1.
$$

Trong đó $\theta$ là angle giữa normalized endpoints. Công thức suy biến khi endpoints gần trùng hoặc đối nhau và cần nhánh xử lý số.

### Geodesic interpolation

Với decoder-induced metric:

$$
z^\star
=
\arg\min_{z(\cdot)}
\int_0^1
\sqrt{
\dot z(\alpha)^\top
G(z(\alpha))
\dot z(\alpha)
}
\,d\alpha.
$$

Trong đó endpoints được cố định và $G$ là pullback metric. Geodesic tránh một số shortcut qua vùng decoder distortion cao, như phân tích trong [Geodesic](../../01-space-representation/research/05-geodesic.md) và [Pullback Metric](../../01-space-representation/research/06-pullback-metric.md).

### Dynamics-consistent interpolation

Interpolation hình học chỉ yêu cầu đường nối endpoints. Một bridge hợp dynamics còn phải phù hợp transition và actions:

$$
\min_{z_{i:j},a_{i:j-1}}
\sum_{k=i}^{j-1}
\|z_{k+1}-f(z_k,a_k)\|^2
+
\lambda\,\mathcal{R}(z_{i:j},a_{i:j-1}).
$$

Trong đó số hạng đầu phạt vi phạm dynamics và $\mathcal{R}$ mã hóa smoothness, control cost hoặc constraints. Đường này có thể dài hơn geodesic nhưng khả thi theo hệ.

### Interpolation giữa hai trajectory

Trước khi blend hai trajectories, cần alignment:

$$
\bar z_\ell
=
(1-\alpha)z_{i_\ell}
+
\alpha y_{j_\ell},
\qquad
(i_\ell,j_\ell)\in\pi.
$$

Trong đó $\pi$ ghép các phase tương ứng. Blend cùng raw index khi speed khác nhau trộn sai event phase.

---

## **9. Uncertainty dọc trajectory**

Trajectory stochastic có thể chứa belief tại mỗi bước:

$$
z_i
\sim
\mathcal{N}(\mu_i,\Sigma_i).
$$

Trong đó $\mu_i$ là mean path và $\Sigma_i$ tạo uncertainty tube biến đổi theo thời gian. Chỉ lưu mean làm mất calibration và branching futures.

Cross-time covariance cũng quan trọng:

$$
\operatorname{Cov}(z_i,z_j)
\neq 0.
$$

Trong đó states chia sẻ process noise và history nên không độc lập theo thời gian. Danh sách marginal covariances $\Sigma_i$ không đủ để tính uncertainty của path integral, average hoặc event time chính xác.

Các representation phổ biến:

| Representation | Giữ được | Mất gì |
|---|---|---|
| Mean path | trend | uncertainty, modes |
| Mean + marginal covariance | local uncertainty | temporal correlation, multimodality |
| Particles | branches và nonlinear distribution | tốn memory/compute |
| Ensemble trajectories | epistemic disagreement | phụ thuộc ensemble diversity |
| Mixture/branch graph | explicit modes | matching và growth phức tạp |

---

## **10. Visualization**

Với $d>3$, projection trajectory xuống 2D/3D cần cẩn trọng.

### Fit projection một lần

Nếu PCA/UMAP được fit riêng từng trajectory hoặc từng frame, coordinate axes thay đổi và motion nhìn thấy có thể do projection chứ không do latent dynamics. Projection nên fit trên reference set chung rồi transform mọi points bằng cùng mapping khi method hỗ trợ.

### Mã hóa thời gian

Trajectory plot nên thể hiện:

- màu theo timestamp;
- marker cho start/end;
- arrows hoặc ordered line;
- changepoints/events;
- uncertainty ellipse/tube;
- observed và imagined segments bằng style khác nhau.

### Plot metric bên cạnh projection

Một 2D path có thể che distortion. Nên đi kèm speed, path length, prediction error, KL hoặc density theo time để kiểm tra pattern có tồn tại trong original latent metric.

---

## **11. Compression và indexing**

Trajectory dài cần representation gọn cho storage và retrieval.

### Keyframe selection

Giữ points có:

- curvature hoặc error cao;
- event/changepoint;
- uncertainty thay đổi mạnh;
- reward/contact;
- reconstruction contribution lớn.

Ramer–Douglas–Peucker giữ endpoints rồi đệ quy giữ point xa segment xấp xỉ nhất nếu distance vượt tolerance $\epsilon$. Nó bảo toàn polyline geometry theo metric đã chọn, không bảo toàn dynamics hoặc event semantics.

### Segment descriptor

Mỗi segment có thể lưu:

```python
SegmentSummary(
    start_state=...,
    end_state=...,
    duration=...,
    path_length=...,
    mean_velocity=...,
    event=...,
)
```

Descriptor hỗ trợ index nhanh; full trajectory chỉ được load sau khi coarse retrieval trả candidates.

### Tokenization

Discrete latent hoặc codebook có thể chuyển trajectory thành token sequence. Sequence compression và suffix/index structures trở nên dễ hơn, nhưng quantization có thể xóa motion nhỏ hoặc tạo token jitter.

---

## **12. Giới hạn / Khi nào thất bại**

**Latent coordinate drift.** Encoder retraining, checkpoint khác hoặc normalization đổi làm cùng observation có latent khác. Hai trajectory không so trực tiếp được nếu không có alignment/version contract.

**Distance không có semantics.** Euclidean proximity có thể không tương ứng observation, action feasibility hay task equivalence. Mọi operation dựa distance kế thừa lỗi metric.

**Sampling không đều bị bỏ qua.** Index-space smoothing và derivatives sai khi $\Delta t$ thay đổi. Resampling tùy tiện có thể tạo dynamics giả.

**Smoothing xóa event.** Contact, collision, mode switch và terminal transition thường là discontinuity có ý nghĩa. Regularizer mạnh biến event thành curve trơn sai vật lý.

**Non-causal leakage.** Centered smoothing, spline và offline smoother dùng future data. Dùng kết quả này làm online feature hoặc benchmark gây leakage.

**Interpolation rời support.** Lerp/Slerp/geodesic theo metric sai có thể đi qua latent ít density hoặc decode không hợp lệ.

**Trajectory mean không phải trajectory hợp lệ.** Average hai paths sau alignment có thể cắt qua obstacle hoặc trộn mutually exclusive modes.

**Segmentation phụ thuộc scale.** Threshold speed/KL không chuyển ổn giữa model, dimension hoặc normalization khác.

**Projection tạo ảo giác.** Self-intersection, cluster crossing hoặc smoothness trong 2D có thể không tồn tại ở không gian gốc.

**Padding và reset leak.** Window vượt episode boundary tạo transition giả từ terminal state sang initial state khác.

**Uncertainty theo thời gian bị factorize.** Marginal covariance mỗi bước không giữ correlation, nên risk của toàn path bị tính sai.

**Branching bị ép thành một curve.** Trong dynamics đa mode, một sequence mean làm mất futures riêng biệt và có thể nằm giữa các mode không khả thi.

**Interpolation không phải rollout.** Đường nối endpoints không dự báo state tương lai và không bảo đảm có action sequence thực hiện được.

---

## **13. Liên hệ với Latent-Anything**

`Trajectory` nên là core primitive thay vì tensor phụ:

```python
trajectory = Trajectory(
    states=states,
    timestamps=timestamps,
    actions=actions,
    mask=mask,
    uncertainty=uncertainty,
    source="posterior",
    frame="model_latent_v3",
)
```

### Invariants cần kiểm tra

- timestamps tăng nghiêm ngặt trên valid region;
- `len(states) = len(timestamps)`;
- edge fields như actions có length tương thích;
- mask không vượt episode boundary;
- state shape/dtype/device thống nhất;
- covariance PSD và khớp state dimension;
- source/model/frame không bị trộn ngầm;
- interpolation/resampling history được ghi lại.

### Layer A — Introspection

Layer A có thể:

- tính speed, acceleration, path length và tortuosity;
- visualize trajectory với time/event/uncertainty;
- phát hiện changepoints và anomalous segments;
- so posterior với prior/imagination;
- audit density/support và projection distortion;
- tìm nearest trajectories hoặc reusable segments.

### Layer B — Manipulation

Layer B có thể:

- slice, concatenate và resample;
- smooth với event-aware constraints;
- align và interpolate trajectories;
- replace/warp một segment;
- condition endpoints hoặc events;
- edit state rồi reproject về dynamics/support hợp lệ.

Mọi edit cần trả provenance. Một trajectory đã smooth, aligned hoặc manually edited không nên vẫn mang `source="observed"`.

### Layer C — Runtime

Layer C cần:

- append theo stream mà không copy toàn buffer;
- chunk sequence nhưng giữ recurrent boundary state;
- batch ragged/padded trajectories an toàn;
- cache descriptors và projections;
- lưu particles/ensembles theo memory budget;
- decode lazy, chỉ materialize observation khi cần;
- giữ timestamp và reset semantics qua device/runtime boundaries.

### API gợi ý

```python
segment = trajectory.slice(t_start, t_end)
uniform = trajectory.resample(dt=0.05, method="linear")
smooth = trajectory.smooth(method="savitzky_golay", causal=False)
parts = trajectory.segment(method="changepoint")
aligned = trajectory.align(reference, method="dtw")
bridge = trajectory.interpolate(other, alignment=aligned, method="geodesic")
```

API cần buộc caller chọn method và semantics thay vì âm thầm dùng Euclidean/index-space defaults.

Mục tiếp theo, **Rollout / latent imagination**, sẽ tập trung vào cách transition model sinh trajectory tương lai và compounding error. **Trajectory similarity metrics** sau đó đi sâu vào DTW và Fréchet cho sequence khác tốc độ hoặc độ dài.

---

## Liên quan

- [Markov Property và State Space](01-markov-property-state-space.md) — định nghĩa state được xâu chuỗi thành trajectory.
- [Latent Transition Model](02-latent-transition-model.md) — sinh từng transition và cung cấp diagnostics cho path.
- [Stochastic Transition](03-stochastic-transition.md) — mở rộng point trajectory thành trajectory phân phối hoặc particles.
- [RSSM — Recurrent State Space Model](04-rssm-recurrent-state-space-model.md) — phân biệt posterior trajectory với prior imagination.
- [Kalman Filter và các biến thể](05-kalman-filter-variants.md) — cung cấp filtering, covariance propagation và offline smoothing.
- [Lerp](../../04-latent-computation/research/01-lerp.md) — linear interpolation giữa latent points.
- [Slerp](../../04-latent-computation/research/02-slerp.md) — interpolation trên hypersphere.
- [Hình học Riemannian](../../03-geometry-structure/research/04-riemannian-geometry.md) — metric, path length và geodesic trên manifold.
- [Density Estimation](../../04-latent-computation/research/06-density-estimation.md) — kiểm tra trajectory có rời vùng latent support hay không.

## Tham khảo

- A. Savitzky, M. J. E. Golay, *Smoothing and Differentiation of Data by Simplified Least Squares Procedures* (Analytical Chemistry 1964).
- H. Sakoe, S. Chiba, *Dynamic Programming Algorithm Optimization for Spoken Word Recognition* (IEEE Transactions on Acoustics, Speech, and Signal Processing 1978).
- H. Alt, M. Godau, *Computing the Fréchet Distance Between Two Polygonal Curves* (International Journal of Computational Geometry & Applications 1995).
- R. Killick, P. Fearnhead, I. A. Eckley, *Optimal Detection of Changepoints with a Linear Computational Cost* (Journal of the American Statistical Association 2012, arXiv:1101.1438).
- U. Ramer, *An Iterative Procedure for the Polygonal Approximation of Plane Curves* (Computer Graphics and Image Processing 1972).
- D. H. Douglas, T. K. Peucker, *Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line or Its Caricature* (The Canadian Cartographer 1973).
- G. Arvanitidis, L. K. Hansen, S. Hauberg, *Latent Space Oddity: on the Curvature of Deep Generative Models* (ICLR 2018, arXiv:1710.11379).
- Y. Rubanova, R. T. Q. Chen, D. Duvenaud, *Latent ODEs for Irregularly-Sampled Time Series* (NeurIPS 2019, arXiv:1907.03907).
- S. Tonekaboni, D. Eytan, A. Goldenberg, *Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding* (ICLR 2021, arXiv:2106.00750).
