# Activation Patching

> **TL;DR.** Activation patching *định vị* nơi một mẩu thông tin được xử lý trong model: chạy hai lượt — một input **sạch** (clean) và một input **hỏng** (corrupted) — rồi **ghép** (patch) activation của một thành phần từ lượt này sang lượt kia, đo output phục hồi (hay sụp đổ) bao nhiêu. Đó là một dạng can thiệp nhân quả *interchange* — thay vì chỉnh latent theo một hướng, nó tráo activation thật từ một counterfactual run. Caveat chính: kết quả phụ thuộc cặp prompt cụ thể, bị **dư thừa (redundancy)** che giấu, và localization có thể là artifact của cách làm hỏng input (Zhang & Nanda).

[Causal intervention vs observational](04-causal-intervention-vs-observational.md) đặt nền: muốn biết model *dùng* thông tin nào, phải can thiệp chứ không chỉ quan sát. Activation patching là hiện thực hóa *cụ thể và sắc bén nhất* của ý tưởng đó — nó không hỏi "khái niệm có lái output không" ở mức toàn cục, mà hỏi "**thành phần nào, ở vị trí nào, layer nào** mang mẩu thông tin này". Đây là công cụ localization trung tâm của mechanistic interpretability và là method introspection thứ năm của Layer A.

---

## **1. Trực giác / Định nghĩa**

Hình dung một dây chuyền lắp ráp. Một sản phẩm *đạt chuẩn* (clean) và một sản phẩm *lỗi* (corrupted) đi qua cùng dây chuyền. Ta muốn biết *công đoạn nào* chịu trách nhiệm cho sự khác biệt. Cách làm: cho sản phẩm lỗi chạy bình thường, nhưng tại **đúng một công đoạn**, lén thay bán-thành-phẩm bằng bản từ dây chuyền sản phẩm đạt chuẩn. Nếu sản phẩm cuối *đột nhiên đạt chuẩn* → công đoạn đó mang thông tin quyết định. Nếu không đổi → công đoạn đó không liên quan.

Trong model: "công đoạn" là một activation cụ thể (output của một attention head, một MLP, một vị trí token, ở một layer). "Sản phẩm" là output (logit của đáp án đúng). Ta:

1. Chạy lượt **clean** $x_\text{clean}$, lưu lại *mọi* activation trung gian $a^\text{clean}_\ell$.
2. Chạy lượt **corrupted** $x_\text{corr}$ (ví dụ: đổi một từ khóa, hoặc thêm nhiễu vào embedding) — output sai.
3. Chạy lại lượt corrupted nhưng **ghi đè** activation tại thành phần $\ell$ bằng $a^\text{clean}_\ell$, giữ nguyên mọi thứ khác.
4. Đo output phục hồi về phía clean bao nhiêu → mức độ thành phần $\ell$ *gây ra* hành vi đúng.

Khác biệt cốt lõi với steering/erasure ở [mục 04](04-causal-intervention-vs-observational.md): ở đó ta chỉnh latent theo một *hướng* do mình chọn ($z' = z - (w^\top z)w + vw$); ở đây ta tráo *cả activation thật* từ một run khác — một **interchange intervention**. Giá trị patch đến từ phân phối thật của model, nên ít rủi ro off-manifold hơn, và độ phân giải là *từng thành phần* chứ không phải một trục khái niệm.

---

## **2. Cơ chế / Công thức**

### 2.1 Metric phục hồi (recovery)

Gọi $M(\cdot)$ là một metric đo độ "đúng" của output, thường là **logit difference** giữa đáp án đúng và đáp án sai, hoặc xác suất của đáp án đúng. Ba đại lượng:

- $M_\text{clean}$: metric khi chạy lượt clean (cao).
- $M_\text{corr}$: metric khi chạy lượt corrupted (thấp).
- $M_\text{patch}(\ell)$: metric khi chạy corrupted nhưng patch activation $a^\text{clean}_\ell$ vào thành phần $\ell$.

Hiệu ứng patching được chuẩn hóa thành tỉ lệ phục hồi:

$$ \text{Recovery}(\ell) = \frac{M_\text{patch}(\ell) - M_\text{corr}}{M_\text{clean} - M_\text{corr}} $$

trong đó tử số đo mức metric tăng nhờ patch, mẫu số là khoảng cách tối đa có thể phục hồi. $\text{Recovery}(\ell) \approx 1$ nghĩa là *chỉ riêng* thành phần $\ell$ đã đủ kéo output về đúng — thông tin tập trung ở đó; $\approx 0$ nghĩa là $\ell$ không liên quan. Quét $\ell$ trên mọi (layer × vị trí token) cho ta một **bản đồ nhiệt localization**.

### 2.2 Denoising vs noising — hai hướng patch, hai câu hỏi

| | Denoising (patch clean → corrupted) | Noising (patch corrupted → clean) |
|---|---|---|
| Base run | Corrupted (output sai) | Clean (output đúng) |
| Patch vào | Activation **clean** của thành phần $\ell$ | Activation **corrupted** của thành phần $\ell$ |
| Output đo | Có *phục hồi* về đúng không? | Có *sụp đổ* về sai không? |
| Trả lời | $\ell$ có **đủ** (sufficient) để khôi phục? | $\ell$ có **cần** (necessary) cho hành vi đúng? |

Hai hướng *không* tương đương: một thành phần có thể đủ mà không cần (do dư thừa), hoặc cần mà một mình không đủ. Heimersheim & Nanda (2024) nhấn mạnh phải nói rõ đang dùng hướng nào và diễn giải đúng theo sufficiency/necessity — lẫn lộn hai cái là lỗi diễn giải phổ biến.

### 2.3 Causal tracing (Meng et al., 2022)

Phiên bản nổi tiếng nhất: Meng et al. làm hỏng input bằng cách thêm **nhiễu Gaussian** $\mathcal{N}(0, 3\sigma)$ vào embedding của các token *chủ thể* (subject), rồi denoise từng hidden state để xem khôi phục được sự thật bao nhiêu. Kết quả: thông tin factual recall tập trung ở **MLP layer giữa, tại vị trí token cuối của chủ thể** — phát hiện này dẫn thẳng tới phương pháp edit ROME (chỉnh một hàng của ma trận MLP để sửa một sự thật). Đây là minh chứng kinh điển rằng patching không chỉ *giải thích* mà còn *chỉ đường can thiệp*.

### 2.4 Attribution patching — xấp xỉ gradient để mở rộng quy mô

Patch từng thành phần cần một forward pass cho *mỗi* thành phần — quá đắt với model lớn (hàng triệu thành phần). **Attribution patching** (Syed et al., 2023) xấp xỉ tuyến tính hiệu ứng patching bằng gradient, chỉ cần *hai forward + một backward*:

$$ \text{Recovery}(\ell) \approx \big(a^\text{clean}_\ell - a^\text{corr}_\ell\big)^\top \left.\frac{\partial M}{\partial a_\ell}\right|_{a^\text{corr}_\ell} $$

trong đó $(a^\text{clean}_\ell - a^\text{corr}_\ell)$ là độ chênh activation giữa hai lượt, và $\partial M/\partial a_\ell$ là gradient của metric theo activation đó (lấy tại lượt corrupted). Tích vô hướng cho ước lượng bậc nhất: patch theo hướng chênh lệch sẽ đổi metric bao nhiêu. Nhờ một backward pass tính được gradient cho *mọi* thành phần cùng lúc, attribution patching scale tới toàn bộ model — nhưng là **xấp xỉ**, dùng để *sàng lọc* giả thuyết rồi mới patch thật để xác nhận.

---

## **3. So sánh với các can thiệp khác**

| | Steering / erasure ([mục 04](04-causal-intervention-vs-observational.md)) | Activation patching |
|---|---|---|
| Cách can thiệp | Chỉnh latent theo *hướng* $w$ tự chọn | Tráo *activation thật* từ run khác |
| Nguồn giá trị | Nhân tạo ($vw$) | Phân phối thật của model |
| Rủi ro off-manifold | Cao | Thấp hơn (giá trị thật) |
| Độ phân giải | Một trục khái niệm | Từng thành phần × vị trí × layer |
| Mục tiêu | "Khái niệm có lái output?" | "**Ở đâu** thông tin được xử lý?" |
| Quy mô | Rẻ | Đắt (trừ attribution patching) |

Activation patching trả lời câu hỏi *localization* mà steering không trả lời được, và ngược lại steering cho ta một trục *thao tác* mà patching không trực tiếp cho. Chúng bổ sung nhau: patching tìm *nơi*, steering/erasure cung cấp *cách chỉnh*.

---

## **4. Giới hạn / Khi nào thất bại**

**Phụ thuộc phân phối prompt.** Patching luôn gắn với một *cặp* (clean, corrupted) cụ thể; kết luận chỉ đúng cho phân phối prompt đó, không tổng quát ra hành vi model ngoài phân phối ấy. Một mạch (circuit) tìm thấy trên một template có thể không xuất hiện ở template khác.

**Dư thừa (redundancy) che giấu hiệu ứng.** Heimersheim & Nanda chỉ ra: dư thừa *song song* thì patching xử lý ổn, nhưng dư thừa *nối tiếp* gây rối — nếu patch (ablate) head 1, head 2 có thể *nhảy vào thay thế* (backup head), khiến ta đo được hiệu ứng ~0 và kết luận sai rằng head 1 không quan trọng. Một thành phần thực sự quan trọng có thể "vô hình" dưới patching vì có bản sao dự phòng.

**Artifact của cách làm hỏng input.** Zhang & Nanda (2024) cho thấy kết quả localization nhạy với *phương pháp corruption* và *metric*: pattern tập trung ở MLP giữa mà ROME dựa vào có thể một phần là *artifact* của nhiễu Gaussian, không phải sự thật bất biến về model. Đổi cách corrupt (ví dụ thay token thay vì thêm nhiễu) có thể đổi bản đồ localization.

**Xấp xỉ attribution patching sai ở "activation lớn".** Linear approximation tốt khi patch activation *nhỏ* (head output) nhưng tệ khi patch activation *lớn* (residual stream); LayerNorm khiến đạo hàm triệt tiêu khi hướng patch trùng residual stream, tạo sai lệch giữa ước lượng attribution và patching thật. Vì vậy attribution patching chỉ là chế độ *thăm dò*, cần patching thật để *xác nhận*.

**Localization ≠ hiểu cơ chế.** Patching nói thành phần $\ell$ *quan trọng*, không nói nó *làm gì*. Biết "MLP layer 18 mang factual recall" chưa giải thích *cách* nó lưu và truy xuất — đó là bước phân tích tiếp theo, không phải kết luận của patching.

**Cần cặp clean/corrupted khớp.** Phương pháp đòi hai input chỉ khác nhau ở *đúng* mẩu thông tin cần dò, mọi thứ khác giữ nguyên. Thiết kế cặp prompt tồi (khác nhiều yếu tố) làm hiệu ứng patching trộn lẫn nhiều nguyên nhân, không localize được.

---

## **5. Liên hệ với Latent-Anything**

Activation patching là **công cụ localization của Layer A** — bổ sung chiều "ở đâu" cho bộ method introspection (probe tìm *thông tin gì*, TCAV tìm *khái niệm nào lái class*, patching tìm *thành phần nào xử lý*):

- **`adapter.run_with_patch(x_corr, donor_run, component)`** → chạy lượt corrupted nhưng ghi đè activation tại `component` bằng giá trị từ `donor_run` (lượt clean đã cache), trả về output. Đây là interchange intervention thuần.
- **`LatentSpace.patching_map(clean, corrupted, metric)`** → quét mọi (layer × vị trí), trả về `Recovery` heatmap để định vị nơi thông tin tập trung. Có chế độ `mode="denoise"|"noise"` ép người dùng phân biệt sufficiency/necessity — đúng cảnh báo của Heimersheim & Nanda.
- **`LatentSpace.attribution_patching(...)`** → chế độ thăm dò nhanh bằng gradient cho adapter lớn; trả kèm cờ cảnh báo "chỉ xấp xỉ, cần xác nhận" và không tin cậy khi patch residual-stream-scale activation.
- **Ràng buộc với `ModelAdapter`**: patching đòi adapter expose *forward có hook* — đọc *và ghi* activation tại layer trung gian, cache theo run. Đây là yêu cầu API mạnh hơn probe (chỉ cần đọc activation tĩnh) và mạnh hơn cả steering (chỉ cần ghi theo hướng); patching cần *cache + interchange* giữa nhiều run.

Quan trọng nhất: patching khép kín mạch lý luận của tầng 5. [Probe](01-linear-probing.md) nói thông tin *giải mã được*; [causal intervention](04-causal-intervention-vs-observational.md) nói khái niệm *được dùng*; activation patching chỉ ra *thành phần cụ thể* thực thi việc dùng đó — bước cuối để Layer B chỉnh sửa *đúng chỗ* thay vì chỉnh mù cả latent. Nó cũng nối thẳng sang [subspace projection](../../04-latent-computation/research/04-subspace-projection.md): có thể patch một *không gian con* của activation thay vì toàn bộ, để localize ở mức mịn hơn.

---

## Liên quan

- [Causal intervention vs observational (mục 04 — tầng này)](04-causal-intervention-vs-observational.md) — patching là hiện thực hóa interchange của can thiệp nhân quả, ở độ phân giải từng thành phần.
- [Linear probing (mục 01 — tầng này)](01-linear-probing.md) — probe tìm thông tin *giải mã được*; patching tìm *thành phần thực thi*.
- [TCAV (mục 03 — tầng này)](03-tcav.md) — TCAV localize *khái niệm*; patching localize *vị trí tính toán*.
- [Subspace projection](../../04-latent-computation/research/04-subspace-projection.md) — có thể patch một không gian con activation để localize mịn hơn.

## Tham khảo

- K. Meng, D. Bau, A. Andonian, Y. Belinkov, *Locating and Editing Factual Associations in GPT* (NeurIPS 2022, arXiv:2202.05262). — Causal tracing với nhiễu Gaussian; localize factual recall vào MLP layer giữa; dẫn tới ROME.
- F. Zhang, N. Nanda, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods* (ICLR 2024, arXiv:2309.16042). — Khảo sát ảnh hưởng của metric và corruption method; cảnh báo localization có thể là artifact.
- A. Syed, C. Rager, A. Conmy, *Attribution Patching Outperforms Automated Circuit Discovery* (BlackboxNLP 2024, arXiv:2310.10348). — Attribution patching: xấp xỉ gradient hai-forward-một-backward để scale patching.
- S. Heimersheim, N. Nanda, *How to Use and Interpret Activation Patching* (arXiv:2404.15255, 2024). — Hướng dẫn thực hành: denoising vs noising, sufficiency vs necessity, và bẫy dư thừa nối tiếp.
