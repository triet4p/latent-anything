# Logit Lens / Tuned Lens

> **TL;DR.** Logit lens (nostalgebraist, 2020) "đọc sớm" latent ở giữa chừng transformer: áp **final layernorm + unembedding** lên hidden state của *từng layer* để ra một phân phối vocabulary tại mỗi tầng — cho thấy dự đoán *hình thành dần qua các layer* (iterative inference). Tuned lens (Belrose et al., 2023) sửa khuyết tật của nó bằng cách học một **affine translator** riêng cho mỗi layer trước khi decode: $\text{logits}_\ell = W_U\,\text{LN}_f(A_\ell h_\ell + b_\ell)$. Caveat chính: logit lens giả định mọi layer cùng *hệ tọa độ* với layer cuối — thường sai, gây thiên lệch và fail trên nhiều model (GPT-Neo, BLOOM, OPT); tuned lens cần *huấn luyện* và được train để *khớp output cuối*, nên có thể "đọc" nhiều hơn lượng thông tin layer thực sự mang.

Cả tầng 5 đã đọc latent ở *một* layer (probe, TCAV, SAE). Logit lens/tuned lens đọc latent *theo chiều sâu*: chiếu mỗi hidden state về không gian output để xem **prediction tiến hóa thế nào qua các layer**. Đây là method introspection đặc thù cho model có **unembedding/decoder head dùng chung** (transformer, một số world model), bổ sung chiều "khi nào/ở layer nào prediction hình thành" cho bộ công cụ Layer A, và khép lại Tầng 5.

---

## **1. Trực giác / Định nghĩa**

Transformer xây dự đoán *dần dần*: residual stream là một tổng đang lớn lên, mỗi block cộng thêm một lượng, và ở cuối, **final layernorm + ma trận unembedding** $W_U$ biến hidden state thành logits trên vocabulary. Câu hỏi tự nhiên: nếu áp *cùng* phép decode đó lên hidden state ở *giữa chừng*, ta thấy gì? Logit lens trả lời: một phân phối vocabulary "model đang nghĩ gì" tại layer đó. Quét mọi layer → một *quỹ đạo* phân phối, hội tụ gần đơn điệu về đáp án cuối. Ta "xem" model quyết định.

Vấn đề: phép decode cuối được *học cho layer cuối*. Hidden state giữa chừng có thể sống trong một hệ tọa độ *xoay/dịch* khác — áp $W_U$ thẳng vào chúng cho kết quả lệch. Logit lens chạy đẹp trên GPT-2 nhưng *thiên lệch có hệ thống* và *fail* trên GPT-Neo, BLOOM, OPT.

Tuned lens sửa đúng chỗ đó: trước khi decode, **dịch** hidden state của mỗi layer về hệ tọa độ của layer cuối bằng một phép **affine học được** (một "translator" riêng cho từng layer), huấn luyện để output decode khớp với output thật của model. Kết quả: lens chính xác hơn, ổn định hơn, ít thiên lệch, và chạy được trên nhiều model.

Điểm cốt lõi: cả hai là **lens** — một *cách đọc* latent ra không gian output, không thay đổi model. Khác biệt là logit lens dùng decode *cố định* (giả định cùng basis), tuned lens dùng decode *hiệu chỉnh từng layer* (học để khớp).

---

## **2. Cơ chế / Công thức**

### 2.1 Logit lens

Gọi $h_\ell \in \mathbb{R}^{d}$ là **residual stream** sau block $\ell$ (tại một vị trí token), $\text{LN}_f$ là final layernorm, $W_U \in \mathbb{R}^{|V| \times d}$ là ma trận unembedding ($|V|$ = kích thước vocabulary). Logit lens decode sớm:

$$ \text{logits}_\ell^{\text{logit-lens}} = W_U\,\text{LN}_f(h_\ell) $$

trong đó $h_\ell$ là hidden state *trung gian*, nhưng $\text{LN}_f$ và $W_U$ là *cùng* các tham số dùng ở layer cuối. Kết quả là một phân phối (sau softmax) trên vocabulary — "dự đoán nếu dừng tại layer $\ell$". Khi $\ell$ chạy từ nông tới sâu, phân phối thường hội tụ đơn điệu về $\text{logits}_L$ (output thật). *Giả định ngầm*: $h_\ell$ nằm cùng hệ tọa độ với $h_L$ — đây là điểm yếu.

### 2.2 Tuned lens — affine translator học được

Tuned lens chèn một phép affine $(A_\ell, b_\ell)$ riêng cho mỗi layer, *dịch* $h_\ell$ về hệ tọa độ layer cuối trước khi decode:

$$ \text{logits}_\ell^{\text{tuned-lens}} = W_U\,\text{LN}_f(A_\ell\, h_\ell + b_\ell) $$

trong đó $A_\ell \in \mathbb{R}^{d \times d}$ và $b_\ell \in \mathbb{R}^{d}$ là **translator** của layer $\ell$ (model và $W_U$ vẫn đóng băng). Huấn luyện $(A_\ell, b_\ell)$ để tối thiểu hóa **KL** giữa phân phối lens và phân phối *cuối* của model:

$$ \min_{A_\ell, b_\ell}\ \mathbb{E}_x\ D_{\text{KL}}\big(\text{softmax}(\text{logits}_L)\ \big\|\ \text{softmax}(\text{logits}_\ell^{\text{tuned-lens}})\big) $$

trong đó kỳ vọng lấy trên dữ liệu, $\text{logits}_L$ là output thật của model. Translator học cách "dự đoán output cuối *từ* hidden state layer $\ell$" — đây vừa là sức mạnh (chính xác, unbiased) vừa là cạm bẫy (nó được train để *nhìn về phía trước*, xem mục giới hạn). Belrose et al. xác nhận bằng causal experiment rằng tuned lens dùng *feature tương tự* model thật.

### 2.3 Quỹ đạo dự đoán và ứng dụng

Quét lens qua mọi layer cho một **quỹ đạo** phân phối $\{p_\ell\}_{\ell=1}^{L}$. Quỹ đạo này cho biết: layer nào prediction "chốt", thông tin nào xuất hiện sớm/muộn, và — theo Belrose et al. — **quỹ đạo bất thường giúp phát hiện input độc hại** với độ chính xác cao. Đây là một tín hiệu chẩn đoán mà chỉ đọc layer cuối không có.

---

## **3. So sánh: logit lens vs tuned lens**

| | Logit lens | Tuned lens |
|---|---|---|
| Phép decode | $W_U\,\text{LN}_f(h_\ell)$ — cố định | $W_U\,\text{LN}_f(A_\ell h_\ell + b_\ell)$ — affine học |
| Huấn luyện | Không (zero-shot) | Có: fit $(A_\ell,b_\ell)$ per layer (KL về output cuối) |
| Giả định | $h_\ell$ cùng basis với $h_L$ | Học phép dịch basis cho từng layer |
| Độ tin cậy | Tốt trên GPT-2, **fail** GPT-Neo/BLOOM/OPT | Reliable, unbiased, đa model |
| Chi phí | Rẻ tức thì | Cần train translator (nhẹ, model đóng băng) |
| Rủi ro diễn giải | Thiên lệch do basis mismatch | Train-để-khớp-output → có thể "nhìn trước" |

Hai cái cùng trả lời "model dự đoán gì ở layer này", nhưng logit lens *giả định* hệ tọa độ, tuned lens *học* nó. Đổi lại độ chính xác, tuned lens thêm một bước huấn luyện và một câu hỏi diễn giải tinh tế hơn.

---

## **4. Giới hạn / Khi nào thất bại**

**Logit lens: basis mismatch.** Giả định mọi layer cùng hệ tọa độ với layer cuối thường sai — representation trung gian bị xoay/dịch. Hệ quả là thiên lệch *có hệ thống*, và lens *fail* trên nhiều kiến trúc (GPT-Neo, BLOOM, OPT). Một phân phối logit-lens "vô nghĩa" ở layer giữa có thể chỉ là artifact của basis mismatch, không phải model "chưa biết gì".

**Tuned lens: train để khớp output cuối → có thể nhìn trước.** Translator được tối ưu để *dự đoán output cuối* từ $h_\ell$. Nếu $h_\ell$ chứa *manh mối tuyến tính* về đáp án mà bản thân model *chưa* dùng tới ở layer đó, translator vẫn khai thác được — nên lens có thể cho thấy prediction "chốt sớm hơn" thực tế model dùng. Đây là phiên bản lens của bẫy probe (xem [linear probing](01-linear-probing.md)): *decode được* không bằng *model dùng*. Belrose dùng causal check để giảm lo ngại này, nhưng nó không biến mất.

**Cả hai chỉ là correlational về thông tin, không về causation.** Lens cho thấy thông tin dự đoán *có mặt* ở layer $\ell$; nó không tự chứng minh model *dùng* dự đoán trung gian đó cho output. Khẳng định nhân quả cần [can thiệp](04-causal-intervention-vs-observational.md) như [activation patching](05-activation-patching.md).

**Giả định readout tuyến tính.** Cả hai decode bằng phép tuyến tính (unembedding, có thể thêm affine). Thông tin mã hóa *phi tuyến* ở layer trung gian sẽ bị bỏ sót — lens chỉ thấy phần *tuyến tính khả đọc* về vocabulary, giống giới hạn của [linear probe](01-linear-probing.md).

**Đặc thù kiến trúc.** Lens cần một **unembedding/decoder head dùng chung** và một residual stream nhất quán giữa các layer. Model không có cấu trúc này (encoder thuần, một số world model) không áp dụng trực tiếp; phải có một head ánh xạ latent → output space.

---

## **5. Liên hệ với Latent-Anything**

Logit lens/tuned lens là method introspection *theo chiều sâu* của Layer A — bổ sung trục "ở layer nào prediction hình thành" mà các method một-layer (probe, TCAV, SAE) không cho:

- **`LatentSpace.lens(z, layer=ℓ, method="logit"|"tuned")`** → decode hidden state của một layer về output space của adapter, trả về phân phối. `method="tuned"` dùng translator đã fit; `"logit"` decode thẳng (rẻ, nhưng cảnh báo basis-mismatch).
- **`Adapter.fit_tuned_lens(data)`** → học $(A_\ell, b_\ell)$ cho mỗi layer bằng KL về output cuối; là một bước calibration một lần cho mỗi adapter. Vì translator là affine và model đóng băng, chi phí nhẹ.
- **`LatentSpace.prediction_trajectory(z)`** → quỹ đạo phân phối qua mọi layer, dùng để: chẩn đoán layer nào "chốt" prediction, phát hiện input bất thường (theo Belrose), và debug khi tích hợp adapter mới. Đây là tín hiệu Layer A độc nhất từ lens.
- **Ràng buộc `ModelAdapter`**: lens đòi adapter expose hidden state *mọi layer* + một **decoder/unembedding head dùng chung**. Đây là yêu cầu chặt hơn probe (chỉ một layer) — chỉ áp dụng cho adapter kiểu transformer/decoder. API phải báo "lens không khả dụng" cho adapter thiếu cấu trúc này.
- **Cảnh báo diễn giải nhất quán với cả tầng**: `lens` trả kết quả kèm nhắc rằng decode-được ≠ model-dùng; muốn khẳng định nhân quả phải kết hợp [activation patching](05-activation-patching.md). Đây là sợi chỉ đỏ xuyên suốt Tầng 5.

Lens khép Tầng 5 bằng cách nối introspection (đọc latent) với hướng **predict trong latent** (Tầng 8): nếu mỗi layer đã mang một dự đoán đọc được, thì latent *là* một trạng thái dự báo qua thời gian-độ-sâu — tiền đề cho việc reasoning thuần latent mà không cần decode đầy đủ ở mỗi bước.

---

## Liên quan

- [Linear probing (mục 01 — tầng này)](01-linear-probing.md) — tuned lens về bản chất là một *affine probe ra vocabulary* cho từng layer; chia sẻ bẫy "decode được ≠ model dùng".
- [Activation patching (mục 05 — tầng này)](05-activation-patching.md) — lens cho thấy *thông tin gì* ở mỗi layer; patching khẳng định *nhân quả* và localize thành phần.
- [Causal intervention vs observational (mục 04 — tầng này)](04-causal-intervention-vs-observational.md) — lens là quan sát; muốn nói model *dùng* dự đoán trung gian cần can thiệp.
- [Superposition hypothesis (mục 06 — tầng này)](06-superposition-hypothesis.md) — lens decode tuyến tính nên bỏ sót thông tin phi tuyến/chồng chất ở layer trung gian.

## Tham khảo

- nostalgebraist, *interpreting GPT: the logit lens* (LessWrong, 2020). — Bài gốc giới thiệu logit lens: áp final layernorm + unembedding lên hidden state trung gian để decode sớm.
- N. Belrose, Z. Furman, L. Smith, D. Halawi, I. Ostrovsky, L. McKinney, S. Biderman, J. Steinhardt, *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023, arXiv:2303.08112). — Tuned lens: affine translator học per-layer (KL về output cuối); reliable/unbiased hơn logit lens, kèm causal check và phát hiện input độc hại.
