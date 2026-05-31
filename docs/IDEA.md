# Lumen — Idea

> Tài liệu định hướng và lý do tồn tại của dự án
> Phiên bản: 0.1 — May 2026

---

## 1. Tên dự án

**Lumen** — *Latent Understanding, Manipulation & Execution Network*

Tên gợi ý "ánh sáng soi vào không gian tiềm ẩn", phản ánh ba trụ cột của dự án: hiểu (introspection), tác động (manipulation), và thực thi (execution). Tên ngắn, dễ phát âm, chưa bị chiếm dụng trong ecosystem ML lớn, và không gắn chặt với bất kỳ model architecture nào — phù hợp với định hướng layer ngang phục vụ nhiều thế hệ model.

*Lưu ý*: tên có thể thay đổi trước 1.0. Verify package name availability trên PyPI và crates.io khi gần publish lần đầu.

---

## 2. Mô tả ngắn

Lumen là một framework Python (sẽ có Rust core sau) coi **latent space như một đối tượng first-class**: load latent từ bất kỳ model nào (VAE, VLA, world model, diffusion, LLM hidden state), inspect chúng, thao tác trên chúng, và execute pipeline hiệu quả. Kiến trúc plugin-first cho phép tích hợp method từ paper mới như cách LeRobot tích hợp policy. Mục tiêu là trở thành tooling layer chung cho mọi người làm việc với representation learning, world model, và embodied AI.

**Thesis:** *Latent space xứng đáng có một abstraction layer riêng, độc lập với model architecture. Mọi thứ làm việc với latent — visualization, probing, editing, rollout — nên build trên cùng primitive thay vì mỗi lab tự reinvent.*

---

## 3. Vấn đề đang tồn tại

LLM và VLA hiện nay về bản chất là imitation learning ở quy mô lớn — chúng học phân phối hành vi từ demonstration của con người. Điều này có hai giới hạn cơ bản: agent không thể vượt qua người dạy, và reasoning xảy ra trong token space (text hoặc action token) vốn tốn kém và dễ vỡ.

World model như LeWM đại diện cho hướng khác: học một latent space mà ở đó dynamics của thế giới có thể được mô phỏng trực tiếp. Rollout xảy ra trong latent, decode ra observation chỉ khi cần. Đây là cách JEPA, Dreamer series, và LeWM đang đi.

Vấn đề là **không có tooling layer nào tốt cho việc làm việc với latent space**. Hiện tại mỗi lab tự viết code visualize, probe, và manipulate latent của riêng họ — PCA rồi matplotlib, script ad-hoc, không reusable. Không có cái gì giống "framework cho latent space operation" tồn tại. Đây là khoảng trống Lumen muốn lấp.

---

## 4. Tại sao bây giờ

Khoảng trống này đang widening vì ba lý do hội tụ:

**VLA và world model đủ tốt để build agent thật trên đó.** LeWM, π0, OpenVLA — tất cả release trong 2025-2026 — là frontier model mà người ta muốn inspect, debug, và compose. Nhu cầu tooling có thật và đang tăng.

**Mechanistic interpretability đang scale từ LLM sang mọi loại model.** Sparse autoencoder, steering vector, activation patching — những technique này đang được áp dụng ra ngoài LLM. Chưa có framework nào unify chúng cho latent space tổng quát.

**Rust ecosystem cho ML đang mature.** PyO3, burn, candle — có thể build Rust core cho hot path mà vẫn có Python ergonomics. Đây là window opportunity trước khi Python frameworks kịp bắt kịp về performance.

---

## 5. Định vị

Lumen không cạnh tranh với các framework hiện tại — nó ngồi ở khoảng trống giữa chúng:

**Agent frameworks (LangChain, AutoGen, CrewAI)** giả định agent sống trong máy tính: tool là API call, state là text, latency không quan trọng. Toàn bộ stack được shaped bởi assumption đó.

**Robotics frameworks (ROS 2, LeRobot, Isaac)** tốt về sensor, motor, và message passing, nhưng yếu về modern agent pattern. Chúng predate skills, VLA as first-class citizen, và world-model-in-the-loop planning.

**Interpretability tools (TensorBoard projector, Neuroscope, SAE libraries)** là point tools, không phải framework. Không composable, không có unified interface, không có runtime layer.

Lumen là **layer ngang** phục vụ tất cả những thứ trên: bất kỳ ai làm việc với latent representation đều là user tiềm năng.

---

## 6. Ba trụ cột

### A — Introspection

Hiểu cái gì đang xảy ra bên trong latent space: visualization, probing, clustering, sparse decomposition, trajectory analysis. User chính là researcher đang debug model, người đang explore pretrained model, interpretability work.

### B — Manipulation

Tác động lên latent space: interpolation, arithmetic, steering, activation patching, composition, constrained editing. User chính là người build application trên pretrained model, agent developer cần compose skill trong latent, researcher thử nghiệm causal intervention.

### C — Runtime

Chạy pipeline hiệu quả: batching, caching, async execution, streaming, profiling. User chính là người deploy framework trong production, người chạy large-scale experiment, người build embodied agent cần real-time latent rollout.

---

## 7. Triết lý thiết kế

**Plugin-first, nhưng plugin surface tối thiểu.** Ba interface công khai ban đầu: `ModelAdapter`, `Method`, `Pipeline`. Mọi thứ khác là internal, có thể refactor tự do trong giai đoạn 0.x. Mở rộng plugin surface chỉ khi có evidence từ ít nhất hai use case thực tế.

**Incremental interface design dưới 1.0.** Interface đúng được *discovered*, không được *designed*. Sau khi tích hợp 5–7 method khác nhau về triết lý cho mỗi layer, pattern đúng sẽ tự lộ ra. API ở 0.x cho đến khi không còn muốn refactor — đó là tín hiệu sẵn sàng 1.0, không phải timeline.

**Incremental không phải cẩu thả.** Mỗi lần sửa interface, migrate tất cả integration cũ cùng lúc. Dưới 1.0 là license để break, không phải license để bỏ qua consistency.

**Quality bar production từ đầu.** Đây không phải MVP hay prototype. Type system chặt, test coverage thật, documentation song hành với code, reproducibility là first-class. Thứ duy nhất incremental là *thứ tự* implement.

**Cross-language plugin từ thiết kế.** Interface định nghĩa qua data structure (pydantic model, dict, array) thay vì Python-specific object. Rust plugin sau này plug vào được mà không cần redesign.

**Stateful và stateless plugin đều first-class.** Fitted PCA, trained sparse autoencoder cần state. Interpolation, projection stateless. Framework handle cả hai với serialization và reproducibility hook rõ ràng.

**No vector DB cho metadata routing.** Plugin discovery qua registry + config (giống LeRobot), không qua semantic search. Just-in-time loading khi pipeline yêu cầu.

---

## 8. Nền tảng lý thuyết

Lumen build trên một chuỗi lý thuyết liên kết chặt chẽ. Hiểu chuỗi này là prerequisite để thiết kế primitive đúng. Dưới đây là cây kiến thức theo dòng chảy khái niệm — mỗi tầng build trực tiếp lên tầng trước.

---

### Tầng 1 — Không gian và biểu diễn

**Không gian vector và metric.** Latent space trước hết là một không gian vector có metric — có cách đo khoảng cách. Euclidean là default nhưng không phải lúc nào cũng đúng. Cần hiểu: norm, inner product, cosine similarity, Mahalanobis distance. Câu hỏi nền: hai điểm "gần nhau" trong latent space nghĩa là gì?

**Manifold hypothesis.** Dữ liệu thực (ảnh, audio, trajectory robot) tuy có dimension cao nhưng thực ra nằm trên một manifold có dimension thấp hơn nhiều. Latent space là cách parameterize cái manifold đó. Cần hiểu: local vs global structure, tangent space, geodesic (đường ngắn nhất trên manifold khác đường thẳng trong ambient space).

**Chiều và curse of dimensionality.** Tại sao không gian cao chiều khó làm việc: volume tập trung ở rìa, nearest neighbor mất ý nghĩa, sampling trở nên thưa. Đây là lý do cần compress xuống latent space.

---

### Tầng 2 — Học biểu diễn

**Information bottleneck.** Muốn học biểu diễn tốt, ép model giữ lại chỉ thông tin *cần thiết* để reconstruct hoặc predict, bỏ đi noise. Đây là nguyên lý nền tảng của mọi latent learning. Formal: maximize I(Z; Y) trong khi minimize I(Z; X).

**Reconstruction objective.** Cách đơn giản nhất để học latent: encode X → Z, decode Z → X̂, minimize ||X - X̂||. Autoencoder. Vấn đề: latent space không có cấu trúc probabilistic, không thể sample, không interpolate được tốt.

**Probabilistic latent space.** Thay vì map X → z (một điểm), map X → q(Z|X) (một phân phối). Đây là bước nhảy quan trọng nhất của tầng này. Latent bây giờ là một phân phối, không phải điểm. Kéo theo: cần KL divergence để regularize phân phối đó về prior p(Z).

**ELBO.** log p(X) = ELBO + KL(q||p). Không thể maximize log p(X) trực tiếp vì intractable, thay vào đó maximize ELBO = reconstruction term − KL term. Đây là objective của VAE. Mọi thứ sau đều build trên cái này.

**Reparameterization trick.** Sample từ Gaussian không differentiable. Reparameterize: z = μ + σ·ε, ε ~ N(0,I). Bây giờ gradient chạy qua μ và σ được. Cho phép train toàn bộ network end-to-end.

---

### Tầng 3 — Cấu trúc hình học của latent space

**Linear structure.** Sau khi học, latent space thường có cấu trúc tuyến tính có nghĩa: direction trong latent tương ứng với factor biến thiên trong data. Đây không phải ngẫu nhiên — là hệ quả của cách model được train để dùng chiều latent hiệu quả.

**Disentanglement.** Lý tưởng: mỗi chiều latent control một factor độc lập. Thực tế: các factor entangle với nhau. β-VAE penalty KL mạnh hơn để ép orthogonality giữa các chiều. Metric: mutual information giữa chiều latent và factor, intervention effect.

**Isotropy vs anisotropy.** Latent space có phân phối đều theo mọi hướng không (isotropic) hay tập trung theo một số hướng (anisotropic)? Thực tế hầu hết là anisotropic — một số direction active, số còn lại gần như không dùng. Hệ quả cho interpolation và sampling.

**Curvature.** Manifold trong latent space có thể cong — geodesic khác đường thẳng. Khi interpolate bằng lerp, bạn có thể đi qua vùng low-density, decode ra observation vô nghĩa. Slerp là partial fix. Riemannian metric là full fix.

---

### Tầng 3B — 3D Representation trong Latent Space

*Tầng này song song với tầng 3, chuyên về biểu diễn không gian 3D. Prerequisite trực tiếp cho các world model như LeWM. Cần học trước tầng 6.*

**Bài toán biểu diễn 3D.** Mesh, point cloud, voxel grid là các cách explicit truyền thống — mỗi cái có trade-off về topology, continuity, và memory. Neural representation học cách biểu diễn 3D structure ngầm định trong weight của network.

**Neural implicit representation và NeRF.** Thay vì lưu 3D structure explicit, học một function f(x,y,z) → (density, color). NeRF là ví dụ: MLP nhận tọa độ 3D, output color và density. Không gian 3D trở thành *latent của một neural network*. Đây là bước nối 3D representation với latent space thinking.

**Volume rendering và differentiability.** Để render ra ảnh 2D từ neural implicit field, integrate density dọc theo ray (ray marching, alpha compositing). Quan trọng: quá trình này differentiable — cho phép train NeRF từ 2D image supervision mà không cần 3D ground truth.

**Vấn đề của NeRF: tốc độ.** NeRF cực chậm vì phải query MLP hàng trăm lần per ray per pixel. Mọi cải tiến sau đó (Instant-NGP, TensoRF, Mip-NeRF) đều là cách làm neural implicit field nhanh hơn hoặc chính xác hơn.

**3D Gaussian Splatting — paradigm shift.** Thay vì implicit, trở về explicit theo cách mới: biểu diễn scene bằng một tập hữu hạn các 3D Gaussian. Mỗi Gaussian có: position (μ), covariance matrix (Σ — hình dạng và hướng), color (spherical harmonics), opacity (α). Render bằng cách project Gaussian lên 2D, sắp xếp theo depth, alpha composite. Không cần ray marching — nhanh hơn NeRF nhiều bậc.

**Spherical harmonics cho view-dependent color.** Mỗi Gaussian encode color bằng spherical harmonics basis function — màu thay đổi theo góc nhìn một cách smooth và compact. Cần hiểu SH basis và tại sao nó hiệu quả hơn MLP cho color encoding.

**Gaussian parameters là latent variable.** Đây là điểm kết nối với Lumen. Trong context world model, các parameter (μ, Σ, α, SH coefficients) là latent state của thế giới 3D. Encoder học map observation → tập Gaussian. Transition model predict tập Gaussian mới từ tập cũ và action. Decoder là Gaussian rasterizer — deterministic, không cần learn. Đây là cách LeWM tổ chức latent space.

**Tại sao 3DGS tốt hơn NeRF cho world model.** NeRF latent khó manipulate vì implicit — không có "object handle". 3DGS explicit: mỗi Gaussian là một entity, có thể move, rotate, add, remove. World model có thể learn transition trên *tập Gaussian* — cấu trúc này amenable hơn nhiều với transition model và planning.

---

### Tầng 4 — Tính toán trong latent space

**Interpolation.** Lerp: z = (1−t)·z₁ + t·z₂. Slerp: đi theo geodesic trên hypersphere. Slerp tốt hơn cho Gaussian latent vì tôn trọng norm. Khi nào lerp đủ tốt, khi nào cần slerp.

**Latent arithmetic.** Cộng trừ vector: z_result = z_a − z_b + z_c. Work vì linear structure. Điều kiện: các vector phải được encode bởi cùng một model trong cùng một coordinate system.

**Projection.** Project latent lên subspace (PCA direction, concept direction). Decompose: z = z_concept + z_residual. Dùng để isolate hoặc remove một factor cụ thể.

**Distance và similarity.** Euclidean trong latent không luôn meaningful vì anisotropy. Mahalanobis distance normalize theo covariance của latent distribution, tốt hơn cho downstream task.

**Density estimation trong latent.** Fit distribution (Gaussian mixture, normalizing flow) lên latent space để biết vùng nào in-distribution. Dùng để detect out-of-distribution input — observation nào cho latent nằm ở vùng thưa thì model không chắc chắn.

---

### Tầng 5 — Probe và can thiệp

**Linear probing.** Train linear classifier trên latent để predict label. Nếu work tốt → feature được encode tuyến tính. Test cho "what does the model know". Nonlinear probe (MLP) cho upper bound.

**Concept direction.** Tìm direction trong latent tương ứng với concept: mean latent của samples có concept trừ mean không có concept. Direction này là concept activation vector. SVM thay vì mean difference cho kết quả tốt hơn.

**Intervention vs observation.** Probe chỉ quan sát. Intervention tác động — "nếu tôi thay đổi biến này trong latent thì output thay đổi thế nào?" Đây là causal reasoning trong latent space. Activation patching là implementation của intervention.

**Superposition hypothesis.** Model có thể encode nhiều feature hơn số neuron bằng cách superpose — mỗi neuron participate vào nhiều feature, mỗi feature activate sparse subset of neurons. Đây là lý do PCA không đủ để decompose latent thành interpretable feature.

**Sparse decomposition.** Sparse autoencoder tìm K direction (K >> N) trong đó mỗi observation chỉ activate một vài direction. Sparse decomposition cho interpretable feature hơn PCA vì mỗi direction mono-semantic hơn. State-of-the-art cho latent decomposition.

---

### Tầng 6 — Latent space qua thời gian

**State space.** Latent bây giờ là *state* của một hệ thống, không chỉ là embedding của một observation. State có Markov property: tương lai chỉ phụ thuộc vào state hiện tại, không phải toàn bộ history.

**Transition model.** Function f(z_t, a_t) → z_{t+1}: từ state và action, predict state tiếp theo. Học transition model trong latent rẻ hơn trong observation space vì latent compact hơn.

**Stochastic vs deterministic transition.** Transition thực tế có noise. RSSM tách thành deterministic component (GRU hidden state — capture history chắc chắn) và stochastic component (Gaussian sample — capture uncertainty). Đây là core architecture của Dreamer series.

**Latent trajectory.** Sequence z_0, z_1, ..., z_T là trajectory trong latent space. Operation trên trajectory: smoothing, segmentation, similarity, interpolation giữa hai trajectory. Đây là `Trajectory` primitive của Lumen.

**Rollout và imagination.** Có transition model, rollout k step trong latent mà không cần observation thật: z_{t+1} = f(z_t, a_t), lặp lại. Agent simulate tương lai trong latent — "imagination". Rollout trong latent O(k·d) rẻ hơn nhiều so với pixel space O(k·H·W·C).

---

### Tầng 7 — Planning trong latent space

**Objective trong latent.** Cần learn thêm: reward model r(z_t) → scalar và value model V(z_t) → expected future return. Cả hai train trên imagined trajectory.

**Model Predictive Control trong latent.** Tại mỗi step, rollout nhiều action sequence trong latent, evaluate bằng reward model, chọn tốt nhất, thực thi action đầu tiên, lặp lại. CEM (Cross-Entropy Method) là optimizer phổ biến. Không cần train policy explicit.

**Policy gradient trên imagined trajectory.** Train policy neural network bằng cách backprop qua imagined trajectory trong latent. Được vì transition model differentiable. Đây là Dreamer approach.

**Value equivalence.** MuZero extreme case: latent không cần reconstruct observation, chỉ cần *value equivalent* — đủ để predict value, policy, reward đúng. Latent tồn tại chỉ để serve planning.

---

### Tầng 8 — Predict trong latent, không decode

**Self-supervised learning trong latent.** Thay vì predict pixel bị mask (MAE), predict *latent representation* của phần bị mask. Pixel prediction ép model capture low-level texture. Latent prediction ép model capture semantic structure.

**Collapse problem.** Nếu predict trong latent không cẩn thận, model collapse — encode mọi thứ thành cùng một điểm. Fix: stop-gradient (BYOL, SimSiam), contrastive loss (SimCLR), EMA target encoder (DINO, JEPA).

**Joint embedding.** Cả context và target đều được encode vào latent trước khi predict. Predictor học map latent của context sang latent của target. Không có decoder. Đây là JEPA architecture.

**Implication cho Lumen.** Nếu model học được world dynamics trong latent mà không cần decode, thì decode chỉ là một output head khi cần. Reasoning, planning, prediction xảy ra trong latent. Đây là thesis của Lumen: latent space là môi trường suy luận chính, không phải token space.

---

### Tầng 9 — Discrete latent space

**Tại sao discrete.** Continuous latent phù hợp cho smooth data. Discrete latent stable hơn khi scale, dễ dùng với Transformer architecture, và phù hợp tự nhiên với action space rời rạc.

**Vector quantization.** Encoder output discrete code bằng cách tìm nearest neighbor trong learned codebook. Forward pass: nearest neighbor lookup. Backward pass: straight-through estimator — copy gradient từ decoder về encoder bỏ qua lookup.

**Codebook collapse.** Nhiều code không được dùng. Fix: commitment loss, EMA codebook update, random restart của dead codes.

**Tokenized world model.** Encode observation thành sequence of discrete token, dùng Transformer để model p(token_{t+1} | token_1..t, action). Unify world model với language model architecture. Hướng của GAIA, Genie, và large-scale world model hiện nay.

---

### Tóm tắt dòng chảy lý thuyết

```
Không gian & manifold
    → Học biểu diễn (reconstruction, ELBO, reparameterization)
        → Cấu trúc hình học (linear, disentangle, curvature)
            → [Song song] 3D representation (NeRF → 3DGS → Gaussian as latent)
        → Tính toán (interpolation, arithmetic, projection, density)
            → Probe & can thiệp (probing, concept, superposition, SAE)
                → Latent qua thời gian (state space, RSSM, trajectory)
                    → Planning (MPC, policy gradient, value equivalence)
                        → Predict trong latent (collapse, JEPA, no decode)
                            → Discrete latent (VQ, codebook, tokenized WM)
```

Lumen cần người build hiểu đến tầng 8 để thiết kế primitive đúng, và tầng 9 + tầng 3B để support world model adapter hiện đại (LeWM, GAIA, Genie).

---

## 9. Rủi ro đã nhận diện

**Scope creep.** "Latent space từ A–Z" dễ trở thành "everything for everyone". Mitigation: kỷ luật về plugin surface area, mỗi giai đoạn có tiêu chí hoàn thành rõ ràng.

**Field moving fast.** VLA, world model architecture có thể thay đổi mạnh. Mitigation: abstraction trên latent space (concept bền), không trên architecture cụ thể (thay đổi nhanh).

**Premature optimization.** Cám dỗ build Rust core trước. Mitigation: Python trước, Rust khi profiling chỉ ra bottleneck thật.

**Plugin interface design sai.** Risk lớn vì interface là contract với plugin author. Mitigation: extract từ working code, verify với method thứ ba khác triết lý.

**Adoption.** Framework cần community. Mitigation: minimal "hello world" plugin, documentation tốt từ đầu, showcase use case mạnh sớm.

---

## 10. Câu hỏi mở

- *Wire protocol cho cross-language plugin*: Arrow IPC vs gRPC vs ZeroMQ. Quyết khi đến giai đoạn 7, dựa trên benchmark.
- *Tên chính thức*: "Lumen" là placeholder. Verify PyPI + crates.io khi gần publish.
- *Cache hosting*: HuggingFace Hub là tự nhiên nhưng cần đánh giá lock-in.
- *License*: Apache 2.0 hoặc MIT là default reasonable.
- *Real-time guarantee level*: soft hay hard real-time, jitter budget — quyết khi có integration partner thật.

---

*Tài liệu này là định hướng và lý do tồn tại của Lumen. Cập nhật khi thesis thay đổi hoặc khi có insight mới từ implementation.*