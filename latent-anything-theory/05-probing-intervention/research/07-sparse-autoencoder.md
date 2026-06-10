# Sparse Autoencoder — SAE

> **TL;DR.** SAE là lời giải cho [superposition](06-superposition-hypothesis.md): học một **dictionary overcomplete** (số atom $\gg$ số chiều) với activation **thưa**, để "trải" các feature chồng chất trong latent ra thành các chiều **mono-semantic** — mỗi input chỉ bật vài feature, mỗi feature lý tưởng mang một nghĩa. Cốt lõi: $f = \text{ReLU}(W_\text{enc}(x-b_\text{dec})+b_\text{enc})$, $\hat{x}=W_\text{dec}f+b_\text{dec}$, tối thiểu hóa reconstruction + $\lambda\lVert f\rVert_1$. Caveat chính: L1 gây **activation shrinkage** và **dead features**, dictionary có thể **feature-splitting** và *không* đảm bảo tìm đúng feature "thật" của model (no canonical units).

[Superposition](06-superposition-hypothesis.md) kết luận: feature nằm trên các hướng *không trực giao*, đông hơn số chiều, nên PCA (trực giao) không tách được — chỉ phục hồi được *nếu giả định sparsity*. SAE chính là hiện thực hóa giả định đó: thay ràng buộc trực giao của PCA bằng ràng buộc **thưa**, và cho phép dictionary *overcomplete*. Đây là method introspection mạnh nhất của Layer A để decompose một latent chồng chất, và là nguồn sinh ra các **concept direction** mono-semantic cho Layer B.

---

## **1. Trực giác / Định nghĩa**

Hình dung latent $x$ là một *bản hòa âm*: nhiều nốt (feature) phát cùng lúc, chồng lên nhau thành một dạng sóng rối. Một neuron đơn lẻ nghe như nhiều nốt trộn (poly-semantic). SAE là *bộ tách nốt*: học một "bảng nốt" lớn (dictionary, nhiều atom hơn số chiều sóng), và với mỗi bản hòa âm, tìm **một tổ hợp thưa** vài nốt tái tạo lại nó. Vì tổ hợp thưa, mỗi atom được ép mang đúng *một* nốt (mono-semantic) thay vì một mớ.

Ba thành phần định nghĩa:

- **Overcomplete**: số atom $d_\text{hidden} \gg d_\text{model}$ (Bricken et al. dùng $512 \to 8192$, mở rộng $16\times$). Đông hơn số chiều để có chỗ cho mọi feature chồng chất — đúng điều superposition đòi hỏi.
- **Sparse**: mỗi input chỉ kích hoạt một số ít atom (phần lớn $f_i = 0$). Sparsity là ràng buộc thay cho trực giao.
- **Reconstruction**: tổ hợp thưa đó phải tái tạo lại $x$ chính xác — đảm bảo dictionary thực sự bắt được thông tin, không vứt bỏ.

Khác với [autoencoder thường](../../02-representation-learning/research/02-autoencoder.md) (nén xuống chiều *thấp*), SAE *mở rộng* lên chiều cao hơn nhưng ép thưa — mục tiêu không phải nén mà là **decompose** (phân rã thành feature).

---

## **2. Cơ chế / Công thức**

### 2.1 Kiến trúc và loss

Cho activation $x \in \mathbb{R}^{d_\text{model}}$. SAE encode thành feature thưa $f \in \mathbb{R}^{d_\text{hidden}}$ rồi decode lại:

$$ f = \text{ReLU}\big(W_\text{enc}(x - b_\text{dec}) + b_\text{enc}\big), \qquad \hat{x} = W_\text{dec}\,f + b_\text{dec} $$

trong đó $W_\text{enc} \in \mathbb{R}^{d_\text{hidden}\times d_\text{model}}$, $W_\text{dec} \in \mathbb{R}^{d_\text{model}\times d_\text{hidden}}$, $b_\text{enc}, b_\text{dec}$ là bias; mỗi **cột** của $W_\text{dec}$ là một **feature direction** (atom của dictionary). $f_i$ là độ kích hoạt của feature $i$. Loss kết hợp tái tạo và sparsity:

$$ \mathcal{L} = \underbrace{\lVert x - \hat{x}\rVert_2^2}_{\text{reconstruction}} + \lambda \underbrace{\sum_i f_i\,\lVert W_{\text{dec},i}\rVert_2}_{\text{L1 sparsity}} $$

trong đó số hạng đầu ép tái tạo chính xác, số hạng sau (chuẩn $\ell_1$ có trọng số theo norm cột decoder) ép phần lớn $f_i = 0$, và $\lambda$ điều chỉnh đánh đổi. L1 là chìa khóa: nó là *convex surrogate* của đếm số feature bật ($\ell_0$), đẩy nghiệm về thưa.

**Diễn giải:** tối thiểu hóa $\mathcal{L}$ buộc model giải thích mỗi $x$ bằng *ít* feature nhất có thể mà vẫn tái tạo tốt — chính là bài toán **sparse coding**. Dictionary $W_\text{dec}$ học được sẽ căn chỉnh các atom theo feature thật của model (nếu mọi thứ thuận lợi).

### 2.2 Vì sao thưa cho phép phục hồi superposition

Trong [superposition](06-superposition-hypothesis.md), $x = \sum_i s_i\, g_i$ với $g_i$ là hướng feature (gần trực giao, đông hơn $d_\text{model}$) và $s$ thưa. PCA thất bại vì tìm $d_\text{model}$ hướng trực giao. SAE thành công vì: với *giả định thưa*, hệ phương trình under-determined ($d_\text{hidden} > d_\text{model}$ ẩn) lại có nghiệm *duy nhất* — nguyên lý nền của compressed sensing. Dictionary overcomplete cung cấp đủ atom cho mọi $g_i$; ràng buộc thưa chọn đúng tổ hợp.

### 2.3 Biến thể activation — giải quyết shrinkage

L1 có khuyết tật cố hữu: nó phạt *cả độ lớn* của feature, nên ép các activation thật về nhỏ hơn giá trị đúng — **activation shrinkage** (Tibshirani 1996), làm tái tạo lệch. Các biến thể thay đổi *cách chọn feature nào bật*:

| Biến thể | Cơ chế chọn feature | Ưu điểm |
|---|---|---|
| **Vanilla (L1)** | ReLU + phạt $\ell_1$ | Đơn giản; nhưng shrinkage + dead features |
| **TopK** (Gao et al., 2024) | Giữ đúng $k$ feature lớn nhất, zero phần còn lại | Kiểm soát sparsity trực tiếp; tránh shrinkage; scale tới GPT-4 |
| **Gated** (Rajamanoharan et al., 2024) | Tách *chọn* (gate) khỏi *ước lượng độ lớn* | Tránh shrinkage; ước lượng magnitude tốt hơn |
| **JumpReLU** (Rajamanoharan et al., 2024) | ReLU có ngưỡng: zero nếu dưới ngưỡng | State-of-the-art fidelity; gating phi tuyến |

Mạch chung của các cải tiến: thay phạt magnitude liên tục bằng một **thao tác ngưỡng/gating** để quyết định feature nào tham gia, tách quyết-định-bật khỏi ước-lượng-độ-lớn.

---

## **3. So sánh: PCA vs SAE**

| | PCA | SAE |
|---|---|---|
| Số thành phần | $\le d_\text{model}$ | $\gg d_\text{model}$ (overcomplete) |
| Ràng buộc | Trực giao | Thưa (sparse) |
| Hướng | Trực giao, theo phương sai | Không trực giao, theo feature |
| Mỗi điểm dùng | Mọi thành phần (dense) | Vài feature (sparse) |
| Tách được superposition? | Không | Có (nếu đủ thưa) |
| Mono-semantic? | Không (top-PC trộn) | Thường có (~70% theo Bricken) |
| Tối ưu | Đóng (SVD) | Học bằng SGD, có dead features |

PCA và SAE giải hai bài toán khác nhau: PCA tìm hệ trục *giải thích phương sai*, SAE tìm dictionary *giải thích dữ liệu bằng ít feature*. Chính ràng buộc thưa (thay vì trực giao) cho phép SAE vượt rào superposition mà PCA không qua được.

---

## **4. Giới hạn / Khi nào thất bại**

**Activation shrinkage.** L1 phạt độ lớn nên kéo mọi activation về nhỏ hơn thật, gây thiên lệch tái tạo có hệ thống. Đây là động lực cho TopK/Gated/JumpReLU; SAE vanilla L1 không nên dùng cho phân tích định lượng độ lớn feature.

**Dead features.** Một phần lớn atom có thể *không bao giờ* kích hoạt sau khi train (dead latents), lãng phí dung lượng dictionary. Cần kỹ thuật như **resampling** (khởi tạo lại latent chết định kỳ) hoặc auxiliary loss (Gao et al.) — chi phí huấn luyện bổ sung.

**Feature splitting.** Khi tăng kích thước dictionary, một feature "thật" có thể bị **tách** thành nhiều atom gần giống nhau (ví dụ: "DNA" tách thành nhiều biến thể ngữ cảnh). Số atom là siêu tham số *tùy ý*, và không có kích thước "đúng" — feature ở độ phân giải nào phụ thuộc lựa chọn dictionary size.

**Không có canonical units.** Nghiêm trọng nhất: SAE *không đảm bảo* tìm đúng feature mà model thực sự dùng. Các SAE khác nhau (seed, kích thước, hàm activation khác) tìm ra *tập feature khác nhau*; nghiên cứu gần đây cho thấy SAE không hội tụ về một bộ "đơn vị chuẩn" duy nhất. Decompose được *một* basis thưa không chứng minh basis đó *là* feature của model.

**Đánh đổi reconstruction–sparsity.** Luôn tồn tại Pareto frontier giữa tái tạo tốt và thưa nhiều; chọn $\lambda$ (hoặc $k$) là một sự đánh đổi không có đáp án khách quan, và kết luận có thể đổi theo điểm vận hành.

**Interpretability do người chấm, có thể ảo.** "70% mono-semantic" dựa trên đánh giá của người — một feature *trông* mono-semantic vẫn có thể không tương ứng khái niệm nhân quả nào của model. Tính diễn giải không tự động kéo theo tính nhân quả: cần [can thiệp](04-causal-intervention-vs-observational.md) để xác nhận feature thực sự *lái* hành vi.

---

## **5. Liên hệ với Latent-Anything**

SAE là **công cụ decompose của Layer A** cho latent chồng chất, và là *nhà máy* sinh concept direction cho Layer B — nó đóng vòng tròn mà superposition đã mở ra:

- **`LatentSpace.train_sae(activations, expansion=16, sparsity="topk", k=...)`** → học dictionary từ tập activation của một adapter, trả về `Dictionary{features=W_dec, encode_fn}`. Mặc định dùng **TopK/JumpReLU** chứ không phải L1 vanilla — đúng bài học shrinkage, và để kiểm soát sparsity trực tiếp.
- **`Dictionary.decompose(x)`** → trả về activation thưa $f$ và danh sách feature đang bật cho một latent $x$ — thay thế `pca_decompose` ở chế độ superposed mà [note superposition](06-superposition-hypothesis.md) đã chỉ ra PCA bất lực.
- **Feature → Layer B**: mỗi cột $W_{\text{dec},i}$ là một concept direction mono-semantic, dùng được ngay cho [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) và **steering vector (mục sau, cùng tầng)** — clamp một feature bật/tắt để chỉnh hành vi một cách *có mục tiêu*, sạch hơn nhiều so với steering theo hướng poly-semantic.
- **Bắt buộc kèm chẩn đoán chất lượng**: `Dictionary` phải báo `dead_feature_rate`, điểm reconstruction–sparsity, và cảnh báo *không canonical* — người dùng không được coi feature SAE là "sự thật" của model mà chưa [can thiệp](04-causal-intervention-vs-observational.md) kiểm chứng.
- **Ràng buộc `ModelAdapter`**: SAE cần thu thập activation ở một layer (đọc tĩnh, như probe) — nhẹ hơn yêu cầu hook của [activation patching](05-activation-patching.md), nhưng tốn pha *huấn luyện riêng* một SAE cho mỗi (adapter, layer).

SAE là bản lề giữa "hiểu" (introspection) và "chỉnh" (manipulation): nó biến một latent rối thành tập feature mono-semantic, mỗi feature là một nút điều khiển. Nó cũng đặt nền cho **dictionary learning (mục sau)** — khung tổng quát mà SAE là một instance — và cho mọi pipeline steering dựa trên feature.

---

## Liên quan

- [Superposition hypothesis (mục 06 — tầng này)](06-superposition-hypothesis.md) — bài toán SAE sinh ra để giải: feature chồng chất, đông hơn số chiều, không trực giao.
- [Autoencoder (tầng 2)](../../02-representation-learning/research/02-autoencoder.md) — SAE là autoencoder *overcomplete + sparse*, decompose thay vì nén.
- [Subspace projection](../../04-latent-computation/research/04-subspace-projection.md) — vì sao chiếu trực giao (PCA) không tách được feature superposed; SAE thay bằng ràng buộc thưa.
- [Causal intervention vs observational (mục 04 — tầng này)](04-causal-intervention-vs-observational.md) — feature SAE *trông* mono-semantic cần can thiệp để xác nhận tính nhân quả.
- [Linear probing (mục 01 — tầng này)](01-linear-probing.md) — probe tìm một hướng; SAE tìm *cả dictionary* hướng mono-semantic.

## Tham khảo

- T. Bricken, A. Templeton, J. Batson, B. Chen, et al., *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (Transformer Circuits / Anthropic, 2023). — SAE trên transformer 1 lớp; mở rộng $512\to8192$; ~70% feature mono-semantic; mô tả feature splitting.
- H. Cunningham, A. Ewart, L. Riggs, R. Huben, L. Sharkey, *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (ICLR 2024, arXiv:2309.08600). — SAE giải superposition một cách unsupervised, scalable; feature mono-semantic hơn neuron.
- L. Gao, T. Dupré la Tour, et al., *Scaling and Evaluating Sparse Autoencoders* (OpenAI 2024, arXiv:2406.04093). — TopK SAE: kiểm soát sparsity trực tiếp, tránh shrinkage và dead latents, scaling laws.
- S. Rajamanoharan, T. Lieberum, et al., *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders* (2024, arXiv:2407.14435). — JumpReLU: gating có ngưỡng, fidelity SOTA; train bằng straight-through estimator.
- A. Templeton, T. Conerly, J. Marcus, et al., *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (Transformer Circuits / Anthropic, 2024). — Scale SAE lên model sản xuất; cách đánh giá chất lượng feature ở quy mô lớn.
