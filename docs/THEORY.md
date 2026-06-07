# Latent Anything — Theory Index

> Danh mục lý thuyết cần nắm để build Latent-Anything
> Phiên bản: 0.1 — May 2026
>
> Cách đọc: mỗi mục có trạng thái [ ] chưa học / [~] đang học / [x] đã nắm.
> Checkbox phản ánh mức độ đã được cover trong `latent-anything-theory`; một mục có thể nằm trong note/notebook gộp, không nhất thiết map 1-1 với một file riêng. Theo workflow hiện tại: [~] = đã có research note nhưng notebook coverage chưa hoàn tất, [x] = đã có cả research và notebook coverage.
> Thứ tự trong mỗi tầng là thứ tự đọc đề xuất.

---

## Tầng 1 — Không gian và biểu diễn

Nền tảng toán học trước khi đụng vào bất kỳ model nào.

- [x] **Metric space và vector space** — norm, inner product, cosine similarity, Mahalanobis distance. Câu hỏi cốt lõi: "gần nhau" trong latent space nghĩa là gì?
- [x] **Manifold hypothesis** — tại sao dữ liệu thực nằm trên submanifold có chiều thấp. Local vs global structure, tangent space.
- [x] **Geodesic** — đường ngắn nhất trên manifold khác đường thẳng trong ambient space. Hệ quả cho interpolation.
- [x] **Curse of dimensionality** — volume tập trung ở rìa, nearest neighbor mất ý nghĩa ở chiều cao. Lý do cần compress.
- [x] **Intrinsic vs extrinsic dimension** — chiều thật của manifold vs chiều của ambient space chứa nó.

---

## Tầng 2 — Học biểu diễn

Cách model học map observation xuống latent.

- [x] **Information bottleneck principle** — maximize I(Z;Y), minimize I(Z;X). Tại sao ép model bỏ thông tin không cần thiết lại cho biểu diễn tốt hơn.
- [x] **Autoencoder** — reconstruction objective, bottleneck architecture. Vấn đề: latent không có cấu trúc probabilistic.
- [x] **Variational inference** — approximate posterior q(Z|X), ELBO derivation, tại sao không optimize log p(X) trực tiếp.
- [x] **KL divergence** — đo khoảng cách giữa hai phân phối. Role trong ELBO: regularize posterior về prior.
- [x] **VAE (Kingma & Welling, 2013)** — ELBO = reconstruction term − KL term. Reparameterization trick: z = μ + σ·ε.
- [x] **β-VAE (Higgins et al., 2017)** — penalty KL mạnh hơn để ép disentanglement. Trade-off reconstruction vs regularization.
- [x] **VQ-VAE (Oord et al., 2017)** — discrete latent space, vector quantization, straight-through estimator, codebook.
- [x] **VQGAN (Esser et al., 2021)** — VQ-VAE + perceptual loss + adversarial loss. Cách latent được dùng trong generative model lớn.

---

## Tầng 3 — Cấu trúc hình học của latent space

Latent space không phải hộp đen — nó có hình dạng.

- [x] **Linear structure trong latent** — tại sao direction trong latent tương ứng với factor biến thiên. Hệ quả từ cách model học.
- [x] **Disentanglement** — lý tưởng mỗi chiều control một factor độc lập. Metric: mutual information, DCI score, intervention effect.
- [x] **Isotropy vs anisotropy** — latent có phân phối đều theo mọi hướng không? Tại sao hầu hết model anisotropic và hệ quả.
- [x] **Riemannian geometry cơ bản** — curvature, parallel transport. Khi nào manifold trong latent cong đủ để lerp fail.
- [x] **Slerp (spherical linear interpolation)** — đi theo geodesic trên hypersphere thay vì đường thẳng. Khi nào dùng thay lerp.
- [x] **Normalizing flows** — học bijective mapping giữa simple distribution và complex latent distribution. Density estimation chính xác trong latent.

---

## Tầng 3B — 3D Representation trong Latent Space

*Song song với tầng 3. Prerequisite cho LeWM adapter và world model 3D.*

- [x] **Neural implicit representation** — biểu diễn 3D bằng function f(x,y,z) → (density, color) thay vì explicit mesh/voxel.
- [x] **NeRF (Mildenhall et al., 2020)** — MLP nhận tọa độ 3D + viewing direction, output color và density. Train từ 2D image supervision.
- [x] **Volume rendering và ray marching** — integrate density dọc theo ray để render ảnh 2D. Alpha compositing. Tại sao differentiable.
- [x] **Positional encoding trong NeRF** — Fourier features để MLP capture high-frequency detail.
- [x] **Instant-NGP (Müller et al., 2022)** — hash encoding thay MLP thuần. Nhanh hơn NeRF nhiều bậc. Hiểu để biết bottleneck của NeRF là gì.
- [x] **3D Gaussian Splatting (Kerbl et al., 2023)** — biểu diễn scene bằng tập 3D Gaussian {μ, Σ, α, color}. Render bằng projection + alpha compositing, không cần ray marching.
- [x] **Covariance matrix trong 3DGS** — Σ = R·S·Sᵀ·Rᵀ, decompose thành rotation và scale. Tại sao parameterize thế này thay vì trực tiếp.
- [x] **Spherical harmonics** — basis function để encode view-dependent color compact. SH degree 0, 1, 2, 3 và trade-off.
- [x] **Gaussian rasterization** — project 3D Gaussian lên 2D, sort by depth, tile-based rendering. Tại sao nhanh hơn NeRF.
- [x] **Gaussian parameters là latent variable** — encoder map observation → Gaussian set. Transition model predict Gaussian set mới. Decoder là rasterizer deterministic. Đây là hướng tổ chức latent ngày càng rõ trong các world model Gaussian-centric.
- [x] **Dynamic 3DGS** — extend 3DGS cho scene có motion. Gaussian deformation field, temporal consistency.
- [ ] **Gaussian set operations** — add, remove, merge Gaussian. Tại sao explicit representation amenable với world model hơn implicit NeRF.

---

## Tầng 4 — Tính toán trong latent space

Các operation cơ bản trực tiếp implement vào Layer B của Latent-Anything.

- [ ] **Lerp (linear interpolation)** — z = (1−t)·z₁ + t·z₂. Khi nào đủ tốt, khi nào fail.
- [ ] **Slerp** — chi tiết hơn tầng 3: implementation, tại sao tốt hơn lerp cho unit-norm latent.
- [ ] **Latent arithmetic** — điều kiện để z_a − z_b + z_c có nghĩa. Tại sao cần cùng coordinate system.
- [ ] **Subspace projection** — project lên PCA direction, concept direction. Decompose z = z_concept + z_residual.
- [ ] **Mahalanobis distance** — normalize theo covariance. Khi nào tốt hơn Euclidean.
- [ ] **Density estimation trong latent** — Gaussian mixture model, normalizing flow trên latent. Dùng để in/out-of-distribution detection.
- [ ] **Optimal transport trong latent** — Wasserstein distance giữa hai distribution latent. Dùng để compare trajectory distribution.

---

## Tầng 5 — Probe và can thiệp

Nền tảng của Layer A (introspection) và một phần Layer B (manipulation).

- [ ] **Linear probing** — train linear classifier trên frozen latent. Test cho "feature này có encode tuyến tính không".
- [ ] **Nonlinear probing** — MLP probe. Upper bound cho thông tin có trong latent, không phân biệt tuyến tính hay không.
- [ ] **Concept Activation Vectors — TCAV (Kim et al., 2018)** — tìm direction trong latent tương ứng với human-defined concept. TCAV score đo sensitivity của output với direction đó.
- [ ] **Causal intervention vs observational study** — correlation trong latent khác causation. Do-calculus cơ bản áp dụng cho latent.
- [ ] **Activation patching** — intervene vào forward pass: patch activation từ run A sang run B. Identify đâu trong model một piece of information được xử lý.
- [ ] **Superposition hypothesis** — model encode nhiều feature hơn số neuron bằng superposition. Tại sao PCA không đủ để decompose.
- [ ] **Sparse autoencoder — SAE (Anthropic, 2023)** — tìm overcomplete sparse basis. Mỗi observation activate sparse subset. Mono-semantic feature hơn PCA direction.
- [ ] **Dictionary learning** — general framework cho sparse decomposition. SAE là một instance. K-SVD, ISTA.
- [ ] **Steering vectors (Zou et al., 2023 — Representation Engineering)** — mean difference giữa có/không có concept. Apply vào latent để "inject" concept. Đơn giản nhưng work mạnh.
- [ ] **Logit lens / tuned lens** — inspect latent ở giữa chừng transformer bằng cách decode sớm. Hiểu information flow qua layers.

---

## Tầng 6 — Latent space qua thời gian

Từ đây latent trở thành *state*, không chỉ là *embedding*.

- [ ] **Markov property và state space** — tương lai chỉ phụ thuộc state hiện tại. Khi nào assumption này valid trong thực tế.
- [ ] **Latent transition model** — f(z_t, a_t) → z_{t+1}. Deterministic trước, stochastic sau.
- [ ] **Stochastic transition** — f(z_t, a_t) → p(z_{t+1}). Gaussian output. Khi nào cần stochastic.
- [ ] **RSSM — Recurrent State Space Model (Dreamer)** — tách deterministic component (GRU) và stochastic component (Gaussian). Tại sao split này giúp stability và long-horizon prediction.
- [ ] **Kalman filter và variants** — linear Gaussian state space model. Nền tảng lý thuyết cho stochastic transition. Extended KF, Unscented KF cho nonlinear.
- [ ] **Latent trajectory** — sequence z_0..T trong latent space. Operations: smoothing, segmentation, similarity, interpolation giữa trajectory.
- [ ] **Rollout / latent imagination** — lặp z_{t+1} = f(z_t, a_t) nhiều step. Cost O(k·d) vs O(k·H·W·C) trong pixel space.
- [ ] **Trajectory similarity metrics** — DTW (Dynamic Time Warping), Fréchet distance. So sánh trajectory có length khác nhau.

---

## Tầng 7 — Planning trong latent space

Dùng latent world model để plan, không chỉ để represent.

- [ ] **Model-based vs model-free RL** — trade-off sample efficiency vs computational cost. Tại sao latent world model là sweet spot.
- [ ] **Reward model trong latent** — r(z_t) → scalar. Train trên imagined trajectory.
- [ ] **Value function trong latent** — V(z_t) → expected return. Bellman equation trong latent space.
- [ ] **Model Predictive Control (MPC)** — rollout nhiều action sequence, chọn tốt nhất, thực thi bước đầu, lặp lại. Receding horizon.
- [ ] **Cross-Entropy Method (CEM)** — population-based optimizer cho MPC. Sample action sequence, keep elite, refit distribution.
- [ ] **MPPI (Model Predictive Path Integral)** — smooth MPC bằng importance-weighted average thay vì hard selection. Tốt hơn CEM cho continuous control.
- [ ] **Policy gradient trên imagined trajectory (Dreamer)** — backprop qua differentiable transition model. Actor-critic trong latent.
- [ ] **Value equivalence (MuZero)** — latent không cần reconstruct observation, chỉ cần predict value/policy/reward đúng. Latent tồn tại chỉ để serve planning.
- [ ] **MCTS trong latent** — Monte Carlo Tree Search với world model làm simulator. AlphaZero-style planning trong learned latent space.
- [ ] **Latent imagination horizon** — trade-off giữa rollout dài (more signal) và compound error (model drift). Truncated backprop through time.

---

## Tầng 8 — Predict trong latent, không decode

Từ đây không cần decoder nữa — reasoning thuần latent.

- [ ] **Masked Autoencoder — MAE (He et al., 2021)** — predict pixel bị mask. Baseline để so sánh với latent prediction.
- [ ] **Representation collapse** — tại sao predict trong latent naively dẫn đến collapse. Tất cả encode thành cùng một điểm.
- [ ] **Stop-gradient và asymmetric architecture** — BYOL, SimSiam. Fix collapse bằng cách ngăn gradient chạy qua một nhánh.
- [ ] **Contrastive learning** — SimCLR, MoCo. Fix collapse bằng cách push negative pair ra xa. InfoNCE loss.
- [ ] **EMA target encoder** — DINO, JEPA. Target encoder update chậm bằng EMA của online encoder. Stable training.
- [ ] **JEPA — Joint Embedding Predictive Architecture (LeCun, 2022)** — predict latent của target từ latent của context. Không có decoder. Không predict pixel.
- [ ] **I-JEPA (Assran et al., 2023)** — JEPA cho image. Predict latent của masked region từ visible context.
- [ ] **V-JEPA (Bardes et al., 2024)** — JEPA cho video. Predict latent của masked frames.
- [ ] **Tại sao latent prediction tốt hơn pixel prediction** — pixel prediction ép model capture noise và texture. Latent prediction ép model capture semantic structure.

---

## Tầng 9 — Discrete latent space

Unify latent space với language model architecture.

- [ ] **Vector quantization** — nearest neighbor lookup trong codebook. Forward: hard assignment. Backward: straight-through estimator.
- [ ] **Commitment loss** — ép encoder output gần codebook entry. Phần của VQ-VAE loss.
- [ ] **EMA codebook update** — update codebook bằng exponential moving average thay vì gradient. Stable hơn.
- [ ] **Codebook collapse** — dead codes không được dùng. Detection và fix: random restart, usage tracking.
- [ ] **Residual VQ (SoundStream, EnCodec)** — stack nhiều VQ layer, mỗi layer quantize residual của layer trước. Tốt hơn single VQ cho audio và continuous signal.
- [ ] **Finite Scalar Quantization — FSQ** — discrete latent không cần codebook. Simpler alternative to VQ.
- [ ] **Tokenized world model** — encode observation → discrete token sequence → Transformer model dynamics. Unify với LM architecture.
- [ ] **GAIA-1 (Wayve, 2023)** — large-scale tokenized world model cho autonomous driving. Đọc để hiểu cách scale.
- [ ] **Genie (Google, 2024)** — interactive world model từ video, controllable bằng latent action.

---

## Tầng bổ sung — Large-scale World Models & VLA

*Đọc sau khi hoàn thành tầng 1–9. Đây là frontier application, không phải lý thuyết nền.*

- [ ] **DreamerV1 (Hafner et al., 2019)** — RSSM + latent imagination + actor-critic. First complete latent world model for RL.
- [ ] **DreamerV2 (Hafner et al., 2020)** — discrete latent trong Dreamer. Stability improvement.
- [ ] **DreamerV3 (Hafner et al., 2023)** — scale lên nhiều domain. KL balancing, symlog transform.
- [ ] **TD-MPC2 (Hansen et al., 2023)** — world model cho continuous control robotics. Temporal difference learning trong latent. Liên quan trực tiếp đến embodied use case.
- [ ] **MuZero (Schrittwieser et al., 2019)** — value equivalence. Không cần reconstruct observation.
- [ ] **OpenVLA** — open-source VLA. Đọc để hiểu latent action space và cách thiết kế ModelAdapter.
- [ ] **π0 — Physical Intelligence (Black et al., 2024)** — VLA với flow matching action head. Latent action representation.
- [ ] **LeWM (2026)** — large-scale embodied world model với 3DGS-based latent. Anchor model cho framework. Đọc architecture paper kỹ nhất trong nhóm này.
- [ ] **UniSim (Yang et al., 2023)** — universal simulator từ diverse data. Cách handle multi-domain latent.

---

## Tầng bổ sung — Interpretability & Analysis Tools

*Đọc song song với tầng 5. Directly inform Layer A của Latent-Anything.*

- [ ] **Mechanistic interpretability overview** — circuit finding, superposition, feature geometry. Survey của Anthropic/Neel Nanda.
- [ ] **Towards Monosemanticity (Anthropic, 2023)** — SAE on MLP neurons. Đọc kỹ phần method.
- [ ] **Scaling Monosemanticity (Anthropic, 2024)** — scale SAE lên Claude 3 Sonnet. Cách evaluate feature quality.
- [ ] **Probing classifiers survey (Belinkov, 2022)** — tổng quan các probing method. Khi nào dùng gì.
- [ ] **UMAP theory (McInnes et al., 2018)** — fuzzy topological structure. Tại sao UMAP preserve global structure tốt hơn t-SNE.
- [ ] **PaCMAP** — alternative UMAP, balance local và global structure tốt hơn.

---

## Tầng bổ sung — Toán học cần thiết

*Review nếu chưa chắc. Không cần học mới nếu đã có ML/DL base.*

- [ ] **Multivariate Gaussian** — parameterization, sampling, KL divergence giữa hai Gaussian (closed form). Dùng liên tục trong VAE và RSSM.
- [ ] **SVD và PCA** — connection giữa SVD, eigendecomposition, và PCA. Tại sao PCA là linear projection optimal.
- [ ] **Riemannian geometry cơ bản** — metric tensor, geodesic, exponential map, logarithmic map. Cần cho slerp và latent space geometry tầng 3.
- [ ] **Optimal transport cơ bản** — Wasserstein distance, Sinkhorn algorithm. Cần cho trajectory comparison và latent distribution alignment.
- [ ] **Lie groups và Lie algebra** — rotation group SO(3), SE(3). Cần cho 3DGS covariance parameterization và robotics.
- [ ] **Spherical harmonics** — basis functions trên sphere, SH coefficients, addition theorem. Cần cho 3DGS color encoding.
- [ ] **Information theory cơ bản** — entropy, mutual information, KL divergence, ELBO. Cần cho tầng 2 và information bottleneck.

---

## Thứ tự đọc đề xuất

```
Toán học cần thiết (nếu cần review)
    ↓
Tầng 1 → Tầng 2 → Tầng 3
                      ↓
                   Tầng 3B (song song, cần trước tầng 6)
                      ↓
Tầng 4 → Tầng 5 → Tầng 6 → Tầng 7 → Tầng 8 → Tầng 9
                      ↓
              Interpretability tools (song song tầng 5)
                                                    ↓
                                          Large-scale WM & VLA
```

**Điểm checkpoint quan trọng với Latent-Anything:**

Sau tầng 3 → có thể thiết kế `LatentSpace` primitive.
Sau tầng 3B → có thể thiết kế `ModelAdapter` cho LeWM.
Sau tầng 5 → có thể implement Layer A đầu tiên, rút ra `Method` interface.
Sau tầng 6 → có thể implement `Trajectory` với đủ operation.
Sau tầng 7 → có thể implement Layer B rollout và planning method.
Sau tầng 9 → có thể implement adapter cho tokenized world model.

---

*Tài liệu này cập nhật khi có paper mới liên quan hoặc khi implementation reveal gap lý thuyết mới.*
