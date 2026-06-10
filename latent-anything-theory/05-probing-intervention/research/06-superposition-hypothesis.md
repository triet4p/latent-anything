# Superposition Hypothesis

> **TL;DR.** Superposition hypothesis (Elhage et al., 2022) nói model có thể mã hóa **nhiều feature hơn số chiều** bằng cách đặt chúng vào các hướng *gần trực giao* (almost-orthogonal) thay vì trực giao — khả thi vì trong không gian chiều cao có thể nhét theo cấp số mũ nhiều vector gần trực giao (bổ đề Johnson–Lindenstrauss), và vì feature *thưa* (sparse) nên hiếm khi cùng bật gây nhiễu. Cái giá là **interference**: mỗi feature bật trông như mọi feature khác hơi bật, được ReLU lọc đi. Hệ quả quan trọng nhất: **PCA không decompose được latent** — PCA tìm hướng trực giao, mà feature trong superposition lại không trực giao và chỉ phục hồi được nếu giả định sparsity (dẫn tới sparse autoencoder).

[Giả thuyết hướng tuyến tính](../../03-geometry-structure/research/01-linear-structure.md) nói mỗi factor là một hướng trong latent. Superposition là *bản tinh chỉnh* gây hậu quả lớn cho cả tầng 5: nếu model nhồi nhiều feature hơn số neuron vào các hướng không trực giao, thì một neuron đơn lẻ mang nhiều nghĩa (**polysemantic**), và mọi công cụ dựa trên hướng trực giao — PCA, thậm chí probe ngây thơ — đều decompose sai. Note này giải thích *tại sao* và *khi nào* superposition xảy ra, và vì sao nó là lý do tồn tại của **sparse autoencoder (mục sau, cùng tầng)**.

---

## **1. Trực giác / Định nghĩa**

Hai khái niệm cần tách bạch:

- **Feature**: một nhân tố ngữ nghĩa độc lập model muốn biểu diễn (ví dụ: "có sọc", "là tiếng Pháp", "đề cập DNA"). Số feature model muốn track thường *lớn hơn nhiều* số neuron.
- **Neuron / chiều**: một trục tọa độ của activation. Số chiều là cố định, nhỏ.

Nếu mỗi feature chiếm trọn một chiều, model chỉ track được tối đa $m$ feature trong không gian $m$ chiều. Nhưng model thực track nhiều hơn thế. **Superposition** là mánh: gán mỗi feature một *hướng* (vector) trong không gian $m$ chiều, và cho phép các hướng đó **gần trực giao** thay vì trực giao hoàn toàn. Trong không gian chiều cao, ta có thể nhét **theo cấp số mũ** nhiều vector mà mọi cặp đều gần vuông góc (bổ đề Johnson–Lindenstrauss) — nên $m$ chiều chứa được $\gg m$ feature.

Vì sao không vỡ trận vì nhiễu? Vì feature **thưa**: tại mỗi input chỉ một số ít feature bật. Hai feature gần trực giao chỉ gây nhiễu cho nhau khi *cùng bật*, mà điều đó hiếm. Phần nhiễu nhỏ còn lại được **ReLU + bias âm** lọc bỏ (đẩy các giá trị nhiễu nhỏ về 0). Đây là lý do superposition cần *phi tuyến* — một model thuần tuyến tính không lọc được nhiễu nên buộc phải làm PCA (chỉ giữ $m$ feature lớn nhất).

**Polysemanticity** (một neuron phản hồi với nhiều khái niệm không liên quan) chính là *triệu chứng quan sát được* của superposition: vì feature không nằm dọc trục neuron mà nằm chéo, mỗi neuron là tổ hợp của nhiều feature.

---

## **2. Cơ chế / Công thức**

### 2.1 Toy model của Elhage et al.

Mô hình đồ chơi: feature thật $x \in \mathbb{R}^n$ (thưa, mỗi chiều là một feature với *importance* khác nhau), nén xuống latent $m$ chiều rồi tái tạo:

$$ h = Wx \in \mathbb{R}^m, \qquad x' = \text{ReLU}(W^\top h + b) $$

trong đó $W \in \mathbb{R}^{m \times n}$ với $m < n$ là ma trận nén (mỗi *cột* $W_i \in \mathbb{R}^m$ là hướng biểu diễn của feature $i$), $b$ là bias, và $x'$ là tái tạo. Model tối thiểu hóa sai số tái tạo *có trọng số theo importance*:

$$ \mathcal{L} = \sum_i I_i\, \mathbb{E}_x\big[(x_i - x'_i)^2\big] $$

trong đó $I_i$ là độ quan trọng của feature $i$. Tích chập then chốt là ma trận **$W^\top W \in \mathbb{R}^{n \times n}$**: phần tử $(i,j)$ là $W_i^\top W_j$ — tích vô hướng giữa hai hướng feature.

- Nếu $W^\top W = I$ (các cột trực giao): không nhiễu, nhưng chỉ làm được khi $n \le m$.
- Khi $n > m$: *không thể* mọi cột trực giao. Phần tử off-diagonal $W_i^\top W_j \neq 0$ chính là **interference** — feature $j$ bật làm rò một lượng $W_i^\top W_j$ vào ước lượng feature $i$.

### 2.2 Khi nào superposition xảy ra — phase transition theo sparsity

Quyết định nằm ở **độ thưa** của feature. Gọi $S$ là xác suất một feature *bật* (sparsity = $1-S$ cao khi $S$ nhỏ):

| Chế độ | Hành vi của model |
|---|---|
| **Dense** ($S$ lớn, feature thường cùng bật) | Nhiễu thường xuyên → không đáng đánh đổi → model chỉ giữ $m$ feature quan trọng nhất theo hướng **trực giao** (giống PCA), bỏ phần còn lại. |
| **Sparse** ($S$ nhỏ, feature hiếm khi cùng bật) | Nhiễu hiếm → đáng nhồi thêm feature vào hướng **gần trực giao** → **superposition**, biểu diễn $> m$ feature. |

Elhage et al. cho thấy có **phase transition sắc nét**: khi tăng sparsity, model đột ngột chuyển từ "biểu diễn riêng từng feature" sang "superposition". Lượng feature nhồi được tăng theo sparsity.

### 2.3 Hình học: feature sắp thành đa diện đều

Khi vào superposition, các hướng feature *không* phân bố ngẫu nhiên mà tự tổ chức thành **cấu trúc hình học đều**: cặp đối xứng (digon), tam giác đều, ngũ giác, và các uniform polytope/tegum product — sắp xếp tối đa hóa góc giữa các hướng (giảm interference) cho một số feature cho trước. Đây là lý do superposition có cấu trúc *học được và lặp lại*, không phải mớ hỗn độn.

### 2.4 Vì sao PCA thất bại — và điều kiện để phục hồi

PCA tìm một hệ trục **trực giao** giải thích phương sai. Nhưng feature trong superposition nằm trên các hướng **không trực giao** và *nhiều hơn số chiều* — không tồn tại phép quay trực giao nào tách chúng ra. Do đó top-$k$ thành phần PCA trộn lẫn nhiều feature, không cho mono-semantic direction.

Điều kiện cứu vãn: nếu *biết* biểu diễn là **thưa**, ta có thể phục hồi feature gốc dù chiều thấp hơn — đây chính là bài toán **sparse coding / dictionary learning**. Ràng buộc sparsity thay cho ràng buộc trực giao của PCA. Đó là nền tảng lý thuyết của **sparse autoencoder (mục sau, cùng tầng)**: học một dictionary overcomplete ($> m$ atom) với activation thưa để "giải nén" superposition về feature mono-semantic.

---

## **3. So sánh: biểu diễn trực giao vs superposition**

| | Trực giao (PCA-like) | Superposition |
|---|---|---|
| Số feature tối đa | $\le m$ (số chiều) | $\gg m$ (cấp số mũ theo $m$) |
| Góc giữa feature | $90°$ (trực giao) | gần $90°$ (almost-orthogonal) |
| Nhiễu (interference) | Không | Có, nhỏ — bị ReLU lọc |
| Điều kiện | Feature dày (dense) | Feature thưa (sparse) |
| Neuron | Mono-semantic | **Poly**-semantic |
| Phục hồi feature | PCA / phép quay trực giao | Sparse coding (giả định thưa) |
| Phi tuyến cần? | Không | Có (lọc nhiễu) |

Điểm mấu chốt: superposition là *đánh đổi* — hi sinh độ chính xác (interference) để lấy dung lượng (nhiều feature hơn). Model chấp nhận khi feature đủ thưa để interference hiếm khi cắn.

---

## **4. Giới hạn / Khi nào thất bại**

**Superposition là giả thuyết, không phải định lý chứng minh cho mọi model.** Bằng chứng mạnh nhất đến từ *toy model* được dựng có chủ đích với feature thưa, importance phân bậc. Mức độ và dạng superposition trong model lớn thực tế vẫn là câu hỏi nghiên cứu mở, không phải kết luận chắc chắn.

**"Feature" được giả định là hướng tuyến tính.** Toàn bộ khung dựa trên giả định feature = hướng (linear representation hypothesis). Feature mã hóa phi tuyến hoặc đa chiều (multi-dimensional feature) không vừa khuôn này; superposition không mô tả chúng.

**Không phải mọi polysemanticity đều do superposition.** Một neuron đa nghĩa có thể vì lý do khác (ví dụ: hai khái niệm thực sự liên quan, hoặc do khởi tạo). Superposition là *một* cơ chế gây polysemanticity, không phải lời giải thích duy nhất — quy mọi polysemanticity về superposition là sai.

**Phục hồi bằng sparse coding không đảm bảo đúng.** Giả định sparsity giúp giải nén, nhưng SAE có thể học feature *bịa* (không tương ứng feature thật của model), bỏ sót feature, hoặc tách một feature thật thành nhiều atom (feature splitting). Decompose được *một* basis thưa không có nghĩa là basis đó *là* feature model dùng.

**Phụ thuộc định nghĩa sparsity và importance.** Phase transition và lượng feature nhồi được phụ thuộc mạnh vào giả định về phân phối sparsity và importance — đổi giả định có thể đổi kết luận định lượng.

---

## **5. Liên hệ với Latent-Anything**

Superposition là *lý do nền tảng* khiến introspection của Layer A không thể chỉ dựa vào trục neuron hay PCA — nó định hình cả thiết kế lẫn cảnh báo:

- **Cảnh báo cho mọi method dựa trên hướng.** [Linear probe](01-linear-probing.md) tìm một hướng decode được, nhưng nếu feature ở trong superposition, hướng đó *trộn* nhiều feature (interference) — probe accuracy cao vẫn có thể trỏ vào một tổ hợp poly-semantic. `LatentSpace.linear_probe` nên báo kèm cảnh báo khi nghi ngờ superposition (ví dụ: số feature ước lượng $\gg$ số chiều).
- **Bác bỏ PCA như công cụ decompose ngữ nghĩa.** `LatentSpace.pca_decompose` chỉ phù hợp ở chế độ dense; cho latent thưa, framework phải chuyển sang **sparse decomposition** — đặt nền cho `LatentSpace.dictionary_decompose` (SAE/dictionary learning) ở các mục sau.
- **Đo "độ superposition" của một adapter.** Một chẩn đoán Layer A hữu ích: ước lượng số feature mono-semantic recover được so với số chiều, và phân bố interference $W^\top W$. Adapter có superposition mạnh cần SAE để introspect; adapter dense có thể dùng PCA.
- **Hệ quả cho manipulation (Layer B).** Vì feature gần trực giao chứ không trực giao, steering theo một hướng feature sẽ *rò* sang feature khác (interference) — [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) và steering cần tính tới việc này, không giả định các trục độc lập hoàn toàn.

Quan trọng nhất: superposition là cầu nối từ "feature là hướng" (tầng 3) sang "làm sao *tìm* được các hướng đó khi chúng đông hơn số chiều và không trực giao" — câu hỏi mà **sparse autoencoder** và **dictionary learning** (hai mục kế tiếp của tầng 5) sinh ra để trả lời.

---

## Liên quan

- [Cấu trúc tuyến tính trong latent](../../03-geometry-structure/research/01-linear-structure.md) — superposition tinh chỉnh giả thuyết hướng tuyến tính: feature *là* hướng, nhưng đông hơn số chiều và gần trực giao.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — cách phân bố hướng trong latent; superposition đẩy về sắp xếp hình học đều để giảm interference.
- [Linear probing (mục 01 — tầng này)](01-linear-probing.md) — probe tìm hướng, nhưng superposition khiến hướng đó trộn nhiều feature.
- [Subspace projection](../../04-latent-computation/research/04-subspace-projection.md) — vì sao chiếu lên trục trực giao (PCA) không tách được feature trong superposition.

## Tham khảo

- N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, et al., *Toy Models of Superposition* (Transformer Circuits / Anthropic, 2022, arXiv:2209.10652). — Định nghĩa superposition; toy ReLU model; phase transition theo sparsity; hình học uniform polytope; liên hệ adversarial examples.
- W. B. Johnson, J. Lindenstrauss, *Extensions of Lipschitz mappings into a Hilbert space* (Contemporary Mathematics, 1984). — Bổ đề JL: tồn tại theo cấp số mũ nhiều vector gần trực giao trong không gian chiều cao — cơ sở toán cho khả năng nhồi $\gg m$ feature.
