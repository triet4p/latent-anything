# JiT — Để mô hình denoise đúng nghĩa denoise, và số phận của latent

> **TL;DR.** Li & He (2025) lập luận rằng diffusion nên **dự đoán ảnh sạch $x$** thay vì nhiễu $\epsilon$, vì dữ liệu sạch nằm trên một **đa tạp ít chiều** còn nhiễu thì lấp đầy không gian chiều cao — nên một mạng "thiếu dung lượng" mới chạy được trên pixel patch lớn (16, 32) mà không sụp đổ. Hệ quả của họ là **JiT**: một ViT thuần chạy thẳng trên pixel, *không VAE, không tokenizer, không pre-training, không loss phụ*. Bài này dùng nó như một phản-đề để làm rõ một điều cốt lõi với Latent-Anything: **"bỏ latent" mà JiT nói tới là bỏ VAE bottleneck (latent nghĩa hẹp), chứ không phải bỏ cấu trúc đa tạp ít chiều (latent nghĩa rộng) — và trong sinh mẫu 3D/robotics, latent nghĩa rộng vẫn không đi đâu cả.**

Mô hình diffusion hiện đại, dù gọi là "denoising", thực ra **hiếm khi xuất ra ảnh sạch**: chúng được huấn luyện để dự đoán nhiễu $\epsilon$, hoặc vận tốc $v$, hoặc score $\nabla \log p$. Paper *Back to Basics* đặt lại một câu hỏi tưởng đã ngã ngũ: mục tiêu dự đoán nên là gì? Câu trả lời của họ — *hãy để mạng dự đoán chính ảnh sạch* — nghe ngây thơ, nhưng kéo theo một hệ quả kiến trúc mạnh: có thể vứt bỏ toàn bộ tầng nén latent (VAE) mà các diffusion model lớn (Stable Diffusion, DiT) phụ thuộc vào. Đây là lý do note này nằm trong Latent-Anything: nó tấn công thẳng vào tiền đề "luôn có một latent space để load".

---

## **1. Trực giác / Định nghĩa**

Hình dung không gian pixel của một patch $32\times32\times3$ là một phòng **3072 chiều**. Ảnh tự nhiên không rải đều khắp phòng đó — chúng tụ lại trên một **tấm màng mỏng cong** (đa tạp $\mathcal{M}$) có số chiều thật $d$ nhỏ hơn rất nhiều so với $D = 3072$. Đây chính là [giả thuyết đa tạp](../../01-space-representation/research/03-manifold-hypothesis.md).

Bây giờ so sánh hai "đích đến" mà ta có thể bắt mạng nhắm tới:

- **Ảnh sạch $x$**: luôn nằm *trên* tấm màng $\mathcal{M}$. Tập ảnh ra (image) của hàm cần học là một tập ít chiều.
- **Nhiễu $\epsilon \sim \mathcal{N}(0, I)$**: đẳng hướng, lấp đầy *toàn bộ* phòng 3072 chiều, hoàn toàn nằm *ngoài* màng.

Trực giác của paper: một mạng nơ-ron với dung lượng hữu hạn hành xử như một ánh xạ **trơn, gần thấp hạng** — nó "thích" xuất ra những tập ít chiều. Bắt nó nhắm tới $x$ (ít chiều) là thuận; bắt nó tái tạo chính xác một vector Gaussian 3072 chiều đẳng hướng (nhiễu) là đòi hỏi nó trải đều năng lực ra mọi chiều với độ chính xác cao — điều một mạng "thiếu dung lượng" **không làm nổi**, và càng tệ khi patch càng lớn (chiều mỗi token càng cao). Đó là gốc rễ của "catastrophic failure".

## **2. Cơ chế / Công thức**

### 2.1. Ba mục tiêu dự đoán đều tương đương về đại số

Đặt theo dạng nội suy của flow matching / diffusion:

$$ x_t = a_t\, x + b_t\, \epsilon $$

trong đó $x$ là ảnh sạch (trên $\mathcal{M}$), $\epsilon \sim \mathcal{N}(0, I)$ là nhiễu, còn $a_t, b_t$ là hệ số lịch trình theo thời gian $t$ (ví dụ flow matching tuyến tính: $a_t = 1-t$, $b_t = t$). $x_t$ là điểm bị nhiễu mà mạng quan sát được.

Từ một $x_t$ cho trước, ba mục tiêu chỉ là **biến đổi tuyến tính của nhau**:

$$ x = \frac{x_t - b_t\,\epsilon}{a_t}, \qquad v \equiv \dot{x}_t = \dot{a}_t\, x + \dot{b}_t\, \epsilon $$

trong đó $v$ là **vận tốc** (mục tiêu của v-prediction), $\dot{a}_t, \dot{b}_t$ là đạo hàm theo $t$. Vì biết $x_t$ thì biết một cái là suy ra cả hai cái kia, nên **về mặt toán học, x-/$\epsilon$-/v-prediction là tương đương** — đây chính là điều khiến lựa chọn này bị xem nhẹ suốt nhiều năm.

### 2.2. Nhưng *độ khó học* thì không tương đương

Điểm sắc bén của paper: tương đương đại số **không kéo theo** tương đương về độ khó tối ưu, vì cái thay đổi là **số chiều nội tại của tập đích mà mạng phải xuất ra**.

- Hàm $f_\theta: x_t \mapsto \hat{x}$ có tập ảnh nằm gần $\mathcal{M}$ ($d$ chiều). Một mạng trơn, gần tuyến tính cục bộ, **dễ** khớp.
- Hàm $f_\theta: x_t \mapsto \hat{\epsilon}$ có tập ảnh là toàn bộ $\mathbb{R}^D$ đẳng hướng. Mạng phải tái dựng thành phần nhiễu **vuông góc với màng** trên cả $D - d$ chiều thừa — phần này gần như không có cấu trúc để bám.

Đặc biệt ở vùng $t$ lớn (nhiều nhiễu), $x_t \approx \epsilon$, nên dự đoán $\epsilon$ gần như là "tái tạo lại chính input nhiễu" với độ chính xác cao trên mọi chiều — vô vọng với patch lớn. Còn x-prediction ở vùng đó tương đương trả về **kỳ vọng có điều kiện** $\mathbb{E}[x \mid x_t]$, một điểm "trung tâm" trên màng — trơn và ít chiều. Khi $D$ mỗi token nhảy từ $16^2\cdot3 = 768$ (patch 16) lên $32^2\cdot3 = 3072$ (patch 32), khoảng cách độ khó này giãn ra tới mức $\epsilon$-prediction **phân kỳ** còn x-prediction vẫn ổn định — đó là bằng chứng thực nghiệm cho luận điểm đa tạp.

### 2.3. JiT — kiến trúc tối giản

JiT (*Just image Transformers*) khai thác trực tiếp kết luận trên:

- **Patchify pixel thô** thành token chiều $p^2\cdot 3$ (với $p \in \{16, 32\}$), đưa vào một **ViT thuần** — không U-Net, không tokenizer, không VAE, không pre-training, không loss phụ trợ.
- Mạng được huấn luyện với **mục tiêu x-prediction**.
- Các biến thể theo quy mô: **JiT-B / L / H**, mỗi cỡ có bản patch /16 và /32; thí nghiệm trên ImageNet $256^2$ và $512^2$ (kèm mẹo *noise scaling*: `noise_scale` $=1.0$ cho 256, $=2.0$ cho 512 để giữ tỉ lệ tín hiệu/nhiễu hợp lý khi $D$ lớn).
- Kết quả: **cạnh tranh với các pixel-space diffusion** trên ImageNet 256/512 *mà không cần latent space của VAE* — trong khi cùng cấu hình đó, $\epsilon$- và v-prediction ở patch 16/32 hỏng nặng.

Ý nghĩa: cái patch lớn đóng vai một dạng **nén không-học** (thay cho VAE), và x-prediction là thứ khiến nén thô bạo đó vẫn train được.

## **3. Hai nghĩa của "latent" — chỗ dễ nhầm nhất**

Phát biểu "JiT bỏ được latent space" chỉ đúng nếu nói rõ *nghĩa nào*:

| | **(a) Latent nghĩa hẹp** | **(b) Latent nghĩa rộng** |
|---|---|---|
| Là gì | Bottleneck **học được**, bắt buộc làm tiền xử lý | Cấu trúc **đa tạp ít chiều** nội tại của dữ liệu/biểu diễn |
| Ví dụ | VAE của Stable Diffusion, tokenizer VQGAN | Đa tạp $\mathcal{M}$; hidden states của ViT |
| JiT có bỏ? | **Có** — chạy thẳng trên pixel | **Không** — cả lập luận đặt nền trên nó |

JiT *xóa* (a) nhưng *dựa vào* (b) mạnh hơn bao giờ hết: chính vì $x$ nằm trên đa tạp ít chiều mà x-prediction mới khả thi. Và bản thân ViT của JiT vẫn có biểu diễn nội tại ở mỗi lớp — đó vẫn là latent theo nghĩa (b), đúng thứ mà Latent-Anything coi là *first-class object*. Nói cách khác, JiT không giết khái niệm latent; nó **dời định nghĩa** từ "VAE bottleneck" sang "bất kỳ biểu diễn có cấu trúc đa tạp nào".

## **4. Giới hạn / Khi nào thất bại — và vì sao chúng dựng nên lý do tồn tại của Latent-Anything**

Mỗi giới hạn dưới đây không chỉ là "chỗ JiT yếu", mà là **một miền nơi việc bỏ latent bất khả thi** — và đó chính xác là miền Latent-Anything nhắm tới. Đọc §4 như một danh sách các phản-ví dụ cho khẩu hiệu "bỏ được latent".

#### 4.1. Modality không có "patch": sinh mẫu 3D đã *hội tụ* về latent

Mánh của JiT phụ thuộc một đặc quyền của ảnh: pixel nằm trên **lưới đều**, nên gộp patch lớn vẫn giữ được cấu trúc cục bộ và một thứ tự token xác định. Dữ liệu 3D **không có đặc quyền đó**:

- Tập điểm / tập Gaussian của [3DGS](../../03b-3d-representation/research/06-3d-gaussian-splatting.md) là một **tập vô thứ tự, hoán vị bất biến, siêu thừa** (hàng trăm nghìn–triệu primitive mỗi scene) — không có "patch" hiển nhiên, không có lưới để cắt.
- Hệ quả thực nghiệm rất rõ: **toàn bộ dòng sinh mẫu 3D native hiện đại đều quay về latent**, không ai diffuse thẳng trên hình học thô. 3DShape2VecSet mã hoá shape thành một **vecset latent thưa** qua cross-attention rồi mới diffuse; CLAY mở rộng đúng pipeline đó bằng DiT; GaussianAnything học **VAE → latent point-cloud có cấu trúc → latent diffusion** để sinh Gaussian. Tức là khi chuyển từ *fit một scene* sang *sinh phân phối scene*, cộng đồng 3D đã **độc lập tái phát minh latent nghĩa (a)** vì không còn lựa chọn nào khác.

    → Đây là phản-ví dụ mạnh nhất: ở 3D, "bỏ latent" không phải lựa chọn bị bỏ qua, mà là lựa chọn **đã được thử và thua**. Một framework coi latent là first-class phải xử lý đúng lớp "tập vô thứ tự" này — không thể giả định mọi latent đều là tensor có lưới.

#### 4.2. Quỹ đạo & robotics/VLA: latent là *điều kiện sống còn của rollout*, không phải tuỳ chọn

Đây là chỗ khác biệt sâu nhất với ảnh tĩnh: trong điều khiển, latent không chỉ để *biểu diễn* một quan sát, mà để **lăn (rollout) qua thời gian**. Hai sức ép buộc phải có state nén:

- **Chi phí rollout.** Lăn $k$ bước trong latent tốn $O(k\cdot d)$; lăn trong pixel tốn $O(k\cdot H\cdot W\cdot C)$. Với planning kiểu MPC/imagination cần hàng nghìn quỹ đạo tưởng tượng, khoảng cách này là sự khác biệt giữa *khả thi* và *bất khả thi* — đúng lý do [latent imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) tồn tại.
- **Tích luỹ sai số (compounding error).** Dự đoán *thẳng trên pixel* qua chuỗi dài làm sai số dồn lại, future trôi dạt và mất nhất quán vật lý. Nén về state ngữ nghĩa ít chiều làm chậm sự trôi này — chính là lý do JEPA/value-equivalence chọn *không* tái dựng quan sát.

Còn VLA thì sao? Nhìn kỹ **π0** (flow-matching action head trên PaliGemma): nó *có* dùng tinh thần "denoise đại lượng sạch" — học một velocity field rồi tích phân ~10 bước để xuất **action liên tục**. Nhưng điều đó **không** đồng nghĩa bỏ latent:

- Action vốn đã **ít chiều** nên denoise thẳng trên action là hợp lý — JiT chỉ xác nhận thêm, không phát hiện mới.
- Cốt lõi của π0 nằm ở **điều kiện hoá bằng latent của VLM** (SigLIP + Gemma): toàn bộ tri thức ngữ nghĩa internet-scale đi vào policy *qua latent space*, không qua pixel thô. Bỏ latent ở đây là bỏ luôn khả năng tổng quát hoá.

    → Vậy bài học cho robotics đảo ngược JiT: **đích sinh (action) ít chiều thì denoise thẳng được, nhưng điều kiện và state-qua-thời-gian thì bắt buộc latent.** Latent-Anything định vị đúng vào khe này — nơi latent là *trục thời gian* của bài toán, không phải một embedding tĩnh.

#### 4.3. x-prediction không miễn phí ở mọi $t$

Ở vùng ít nhiễu ($t$ nhỏ, $x_t \approx x$), $\epsilon$-prediction lại có tín hiệu tốt hơn vì phần nhiễu nhỏ và có cấu trúc cục bộ; nhiều hệ thực tế vẫn cân bằng hoặc đổi mục tiêu theo $t$. Paper chọn x-prediction vì nó *cứu* được chế độ patch-lớn–nhiễu-nhiều, **không** phải vì nó thắng tuyệt đối ở mọi $t$.

#### 4.4. Chi phí tính toán và phạm vi đánh giá hẹp

Bỏ VAE nghĩa là Transformer phải nuốt thẳng độ phân giải pixel; lý do lịch sử người ta dùng latent diffusion chính là để **cắt FLOPs**. JiT chứng minh tính *khả thi* và *đúng nguyên lý*, chưa phải lựa chọn rẻ nhất ở mọi quy mô. Thêm nữa, kết quả tập trung ở ImageNet class-conditional $256$/$512$ — chưa phải bằng chứng cho text-to-image quy mô lớn, video, hay 3D/điều khiển.

**Tổng hợp:** JiT đẩy được latent nghĩa (a) ra khỏi *ảnh tĩnh* vì ảnh có lưới và đích sinh là chính nó. Nhưng ở **3D** (tập vô thứ tự) và **quỹ đạo/robotics** (state qua thời gian), cả hai đặc quyền đó biến mất — và latent quay lại không phải như di sản, mà như **điều kiện cần**. Đó chính là khoảng trống §5 chỉ ra rằng Latent-Anything được sinh ra để lấp.

## **5. Liên hệ với Latent-Anything**

**Câu hỏi gốc: có bỏ được latent space trong 3D (kiểu 3DGS đang chiếm lĩnh) / robotics không?** Phân tích theo hai nghĩa latent ở trên:

- **3D.** Có một song song tinh thần đẹp: [NeRF → 3DGS](../../03b-3d-representation/research/06-3d-gaussian-splatting.md) cũng là nước "bỏ trường ẩn (implicit MLP — latent nghĩa (a)) để quay về biểu diễn tường minh", giống JiT bỏ VAE. **Nhưng** 3DGS thắng vì *render tường minh nhanh + chỉnh sửa được*, không phải vì lý lẽ đa tạp. Và khi chuyển từ *fit một scene* sang *sinh* scene bằng diffusion, ta vấp ngay vào ba thứ mánh patch-lớn không xử lý được (tập vô thứ tự, siêu thừa, không có patch). Bằng chứng: GaussianAnything vẫn phải **học một VAE → latent point-cloud có cấu trúc → latent diffusion** mới sinh được Gaussian. Tức trong khâu generative, **latent nghĩa (a) vẫn sống**; ít nhất tới khi có ai tìm ra "patch của không gian 3D". Điều này khớp với [Gaussian parameters là latent variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) và với thông điệp ["3D là phương tiện, không phải mục đích"](../../13-practical-3d-reconstruction/research/04-vision-for-action.md).
- **Robotics.** Lập luận đa tạp cắt cả hai phía. Không gian **hành động** vốn đã ít chiều → denoise/flow thẳng trên action *đã* latent-free từ lâu; JiT chỉ xác nhận thêm. Nhưng **world model thị giác** (quan sát siêu cao chiều) vẫn cần latent vì *hiệu quả tính toán và dự đoán dài hạn*, không phải vì chọn sai mục tiêu — JiT không phủ nhận động cơ này.

**Hệ quả thiết kế cho framework** — đây là phần đáng ghi nhất:

1. **Định nghĩa "latent" theo nghĩa (b).** Layer A (introspection) phải coi latent là *bất kỳ biểu diễn có cấu trúc đa tạp nào*, không neo vào "VAE bottleneck". Nếu không, một mô hình kiểu JiT (no-explicit-latent) sẽ bị `ModelAdapter` xem là "không có latent để load" — sai.
2. **Adapter cần phân loại ít nhất ba lớp mô hình:** (i) có latent tường minh (VAE/VQGAN — [tier 2](../../02-representation-learning/research/05-vqgan.md)); (ii) no-explicit-latent, latent = hidden states (JiT, ViT); (iii) biểu diễn tường minh-không-latent ở ranh giới (3DGS như một điểm dữ liệu trêu ngươi).
3. **x-prediction ↔ triết lý "predict trong latent".** Việc nhắm tới *đại lượng sạch trên đa tạp* thay vì *nhiễu* chính là phiên bản pixel-space của bài học [latent vs pixel prediction](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md) và [JEPA](../../08-latent-prediction/research/06-jepa.md): nhắm vào cấu trúc semantic ít chiều, đừng nhắm vào thứ vô cấu trúc chiều cao. Anchor model [LeWM](../../10-world-models-vla/research/08-lewm.md) cũng nằm trên trục này — dự đoán embedding tương lai (đích "sạch", có cấu trúc), decoder-free.

Kết: paper *không* hủy tiền đề của Latent-Anything; nó **làm sắc** tiền đề đó — buộc ta phải tách bạch hai nghĩa của latent, và nhận ra rằng cái "first-class object" ta đang xây framework quanh nó là **đa tạp ít chiều**, chứ không phải cái hộp VAE cụ thể nào.

---

## Liên quan

- [Giả thuyết Đa tạp](../../01-space-representation/research/03-manifold-hypothesis.md) — nền tảng trực tiếp cho toàn bộ luận điểm "$x$ trên màng, $\epsilon$ thì không".
- [Lời nguyền chiều](../../01-space-representation/research/04-curse-of-dimensionality.md) — vì sao chiều $D$ mỗi token tăng làm $\epsilon$-prediction sụp đổ.
- [Latent vs Pixel Prediction](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md) — cùng một bài học (nhắm cấu trúc, tránh nhiễu) ở tầng biểu diễn.
- [JEPA](../../08-latent-prediction/research/06-jepa.md) — dự đoán đích "sạch" trong latent, không decode; song hành về tinh thần với x-prediction.
- [LeWM](../../10-world-models-vla/research/08-lewm.md) — anchor model decoder-free, dự đoán embedding tương lai làm đích sạch.
- [3D Gaussian Splatting](../../03b-3d-representation/research/06-3d-gaussian-splatting.md) và [Gaussian là latent variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) — ca thử cho câu hỏi "bỏ latent trong 3D".
- [VQGAN](../../02-representation-learning/research/05-vqgan.md) và [VAE](../../02-representation-learning/research/03-vae.md) — chính là "latent nghĩa (a)" mà JiT vứt bỏ.
- [CVPR 2026: Từ nhìn đến làm](../../13-practical-3d-reconstruction/research/04-vision-for-action.md) — cùng mạch "biểu diễn trung gian là phương tiện, không phải mục đích".

## Tham khảo

- Tianhong Li, Kaiming He, *Back to Basics: Let Denoising Generative Models Denoise* (CVPR 2026, arXiv:2511.13720).
- Yaron Lipman et al., *Flow Matching for Generative Modeling* (ICLR 2023, arXiv:2210.02747).
- Tim Salimans, Jonathan Ho, *Progressive Distillation for Fast Sampling of Diffusion Models* (ICLR 2022, arXiv:2202.00512) — nguồn v-prediction.
- Robin Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022, arXiv:2112.10752) — latent diffusion / VAE bottleneck mà JiT vứt bỏ.
- Bernhard Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023, arXiv:2308.04079).
- Yushi Lan et al., *GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation* (ICLR 2025, arXiv:2411.08033).
- Biao Zhang et al., *3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models* (SIGGRAPH 2023, arXiv:2301.11445) — latent vecset cho sinh mẫu 3D.
- Longwen Zhang et al., *CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets* (SIGGRAPH 2024, arXiv:2406.13897) — mở rộng vecset bằng DiT.
- Kevin Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control* (2024, arXiv:2410.24164) — flow-matching action head điều kiện hoá bằng latent VLM.
- Yann LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) — JEPA.
