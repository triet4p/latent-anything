# LeWM — LeWorldModel (2026)

> **TL;DR.** LeWM (LeWorldModel) là **JEPA ổn định đầu tiên train end-to-end từ pixel**: một encoder map ảnh → embedding và một **predictor điều kiện-hành động** dự đoán embedding tương lai — *decoder-free*, không predict pixel. Chỉ **hai loss**: (1) next-embedding prediction và (2) một **regularizer ép latent về Gaussian đẳng hướng** để chống [representation collapse](../../08-latent-prediction/research/02-representation-collapse.md) — bỏ được EMA, pretrained encoder, contrastive mà các JEPA trước cần (giảm 6 hyperparameter loss xuống 1). ~15M tham số, train vài giờ trên 1 GPU, **plan nhanh tới 48× so với world model foundation**, và latent của nó *encode cấu trúc vật lý* probe được. Đây là **anchor model** của Latent-Anything. Caveat: nhỏ/competitive chứ chưa vượt mọi foundation model; JEPA chống collapse vẫn là vấn đề tinh tế.

> **Lưu ý cập nhật:** roadmap [THEORY.md](https://github.com/triet4p/latent-anything/blob/main/docs/THEORY.md) trước đây dự đoán LeWM dùng latent *3DGS-based*; bản phát hành thực tế (LeCun et al., Mila/AMI Labs, 2026) là **JEPA từ pixel**, không phải 3DGS. Mục này viết theo kiến trúc thực.

LeWM là *đỉnh* của [tầng 8](../../08-latent-prediction/research/06-jepa.md) (predict trong latent, không decode) áp dụng làm world model điều khiển được — và là lý do cả tầng 8 lẫn 3 (isotropy) tồn tại trong roadmap. Nó hợp nhất ba thứ framework cần: latent-first, decoder-free, và đủ nhỏ/ổn định để là một anchor thực dụng.

---

## 1. Trực giác: predict embedding, giữ latent "khỏe" bằng một regularizer

Mọi world model trước đối mặt một lựa chọn về *cách học latent*:

- **Reconstruction** ([Dreamer](01-dreamerv1.md)): dựng lại pixel — tốn capacity cho chi tiết vô nghĩa.
- **Value equivalence** ([MuZero](05-muzero.md)/[TD-MPC2](04-td-mpc2.md)): học latent qua reward/value — cần tín hiệu reward.
- **JEPA** ([I-JEPA](../../08-latent-prediction/research/07-i-jepa.md)/[V-JEPA](../../08-latent-prediction/research/08-v-jepa.md)): predict *embedding* của target từ context — không pixel, không cần reward. Nhưng JEPA dễ [collapse](../../08-latent-prediction/research/02-representation-collapse.md) (mọi thứ về một điểm), nên phải vá bằng [stop-gradient](../../08-latent-prediction/research/03-stop-gradient-asymmetric.md), [EMA target](../../08-latent-prediction/research/05-ema-target-encoder.md), hoặc [contrastive](../../08-latent-prediction/research/04-contrastive-learning.md) — đống hyperparameter mong manh.

LeWM chọn JEPA nhưng thay toàn bộ "đống vá" chống collapse bằng **một regularizer duy nhất**: ép phân phối embedding về **Gaussian đẳng hướng**. Trực giác: collapse = embedding co về một điểm/đường (phương sai sụp, [anisotropic](../../03-geometry-structure/research/03-isotropy-anisotropy.md)); ép latent trải đều theo mọi hướng (isotropic Gaussian) *trực tiếp ngăn* sự co đó. Kết quả: train ổn định end-to-end từ pixel, từ con số 0, không EMA/pretrained.

---

## 2. Cơ chế: hai khối, hai loss

### Kiến trúc

- **Encoder** $f_\theta$: ảnh quan sát $o_t \to$ embedding $z_t$.
- **Predictor** $g_\phi$ (điều kiện-hành động): $(z_t, a_t) \to \hat z_{t+1}$ — dự đoán embedding kế trong [latent](../../06-latent-temporal/research/02-latent-transition-model.md), *không* decode về pixel.

Đây là [latent transition model](../../06-latent-temporal/research/02-latent-transition-model.md) thuần embedding: cả representation lẫn dynamics học chung end-to-end.

### Hai loss

$$
\mathcal{L} = \underbrace{\big\lVert g_\phi(z_t, a_t) - \mathrm{sg}[z_{t+1}] \big\rVert^2}_{\text{next-embedding prediction}} \;+\; \lambda\,\underbrace{\mathcal{R}_{\text{Gauss}}(\{z\})}_{\text{isotropic-Gaussian regularizer}}.
$$

- **Next-embedding prediction**: predictor khớp embedding của observation kế (target embedding, thường có stop-gradient để định nghĩa mục tiêu). Đây là tín hiệu học động lực.
- **Gaussian regularizer** $\mathcal R_{\text{Gauss}}$: ép phân phối của các embedding $z$ về **đẳng hướng Gaussian** (trung bình 0, hiệp phương sai $\approx I$) — phương sai mỗi chiều không được sụp, các chiều không được dồn. Chính số hạng này thay vai trò của EMA/contrastive: nó *một tay* chống collapse.

Toàn bộ chỉ **một** hyperparameter loss ($\lambda$), so với *sáu* của giải pháp end-to-end JEPA trước đó.

### Planning và quy mô

LeWM dùng latent đã học để **plan** (rollout embedding + chọn action, kiểu [MPC](../../07-latent-planning/research/04-model-predictive-control.md)/[CEM](../../07-latent-planning/research/05-cross-entropy-method.md) trong latent) trên các task điều khiển 2D/3D. Vì latent nhỏ (~15M params toàn model) và rollout là embedding chiều thấp, nó **plan nhanh tới 48×** so với world model dựa foundation model — train vài giờ trên một GPU.

### Latent có cấu trúc vật lý

Đáng chú ý cho introspection: probe latent của LeWM *trích ra được các đại lượng vật lý*, và một "surprise score" (lỗi prediction cao) *phát hiện sự kiện phi vật lý*. Tức là JEPA-objective ép latent học cấu trúc thế giới thật — bằng chứng mạnh cho luận điểm *predict-để-hiểu* của [tầng 8](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md).

---

## 3. Vì sao LeWM là anchor cho Latent-Anything

| Tiêu chí framework | LeWM thỏa |
|---|---|
| Latent là first-class | latent *là* sản phẩm chính (decoder-free) |
| Decoder-free | predict embedding, không pixel |
| Ổn định, ít hyperparameter | 1 loss-hyperparameter, train từ 0 |
| Nhỏ, chạy được | ~15M params, 1 GPU, vài giờ |
| Latent probe được | probe ra đại lượng vật lý, surprise detection |
| Plan trong latent | rollout + plan nhanh 48× |

So với Dreamer (reconstruction) và TD-MPC2/MuZero (value-equivalence), LeWM đại diện *JEPA-equivalence*: latent học chỉ để **dự đoán chính nó qua thời gian**, regularize bằng hình học (Gaussian/isotropy). Đây là dạng latent sạch nhất, hợp nhất nhất với triết lý framework — nên roadmap chọn nó làm model để "đọc kỹ nhất" và làm chuẩn thiết kế adapter.

---

## 4. Giới hạn / Khi nào thất bại

**Competitive, chưa thống trị.** LeWM mạnh ở tỉ lệ hiệu năng/chi phí và tốc độ plan, nhưng "competitive across 2D/3D control" — không hứa vượt mọi foundation model lớn ở đỉnh hiệu năng tuyệt đối.

**Regularizer Gaussian là một giả định.** Ép isotropic Gaussian chống collapse hiệu quả nhưng áp một dạng phân phối cố định lên latent; với dữ liệu mà manifold thật xa Gaussian, đây có thể là inductive bias chưa lý tưởng.

**Collapse vẫn tinh tế.** Một regularizer đơn giản hơn cả đống vá cũ, nhưng anti-collapse trong JEPA vẫn là vấn đề nhạy; chọn $\lambda$ sai vẫn có thể collapse hoặc latent quá "trải" mất cấu trúc.

**Quy mô nhỏ.** 15M params là điểm mạnh (rẻ) nhưng cũng là trần dung lượng; task rất phức tạp/đa modal có thể cần lớn hơn.

**Plan vẫn compounding error.** Rollout embedding dài vẫn drift như mọi world model ([imagination horizon](../../07-latent-planning/research/10-latent-imagination-horizon.md)).

---

## 5. Liên hệ với Latent-Anything

LeWM là **anchor model** — adapter của nó là mẫu chuẩn cho cả ba pillar:

```python
class LeWMAdapter(Protocol):
    def encode(self, obs: np.ndarray) -> np.ndarray: ...               # f: obs -> z (isotropic Gaussian)
    def predict(self, z: np.ndarray, action: np.ndarray) -> np.ndarray: ...  # g: (z,a) -> z'
    def plan(self, z: np.ndarray, goal: np.ndarray) -> np.ndarray: ...  # MPC/CEM in latent
    def surprise(self, z: np.ndarray, z_next: np.ndarray) -> float: ... # prediction error
```

- **Layer A — Introspection**: LeWM là *ca lý tưởng* cho Layer A — latent decoder-free nhưng probe ra đại lượng vật lý ([linear probing](../../05-probing-intervention/research/01-linear-probing.md)), và "surprise score" là một detector out-of-distribution sẵn có. Kiểm tra latent có thực sự isotropic Gaussian là một audit trực tiếp về sức khỏe biểu diễn.
- **Layer B — Manipulation**: latent đẳng hướng, không collapse là *nền tốt nhất* cho [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md), interpolation, steering — vì hình học sạch (isotropic) làm các phép tuyến tính có nghĩa.
- **Layer C — Runtime**: nhỏ + rollout embedding chiều thấp → plan 48× nhanh; đúng loại model Layer C chạy hiệu quả, và là baseline để đo overhead của framework.

LeWM khép phần "anchor" của tầng. Mục cuối — **UniSim** — mở rộng câu hỏi sang đa-domain: làm sao một universal simulator handle latent từ dữ liệu cực kỳ đa dạng.

---

## Liên quan

- [JEPA](../../08-latent-prediction/research/06-jepa.md) — LeWM là JEPA ổn định đầu tiên train end-to-end từ pixel.
- [V-JEPA](../../08-latent-prediction/research/08-v-jepa.md) / [I-JEPA](../../08-latent-prediction/research/07-i-jepa.md) — tiền thân JEPA cho video/ảnh.
- [Representation Collapse](../../08-latent-prediction/research/02-representation-collapse.md) — vấn đề mà regularizer Gaussian của LeWM giải.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — isotropic Gaussian latent là cơ chế anti-collapse.
- [EMA Target Encoder](../../08-latent-prediction/research/05-ema-target-encoder.md) — thủ thuật cũ mà LeWM loại bỏ.
- [DreamerV1](01-dreamerv1.md) / [TD-MPC2](04-td-mpc2.md) — đối chiếu: reconstruction vs value-equivalence vs JEPA-equivalence.

## Tham khảo

- Y. LeCun et al. (Mila, AMI Labs), *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* (2026, arXiv:2603.19312).
- A. Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA, 2024, arXiv:2404.08471).
- M. Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA, CVPR 2023, arXiv:2301.08243).
- Y. LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — tầm nhìn JEPA.
