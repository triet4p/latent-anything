# GAIA-1 (Wayve, 2023)

> **TL;DR.** GAIA-1 là [tokenized world model](07-tokenized-world-model.md) quy mô lớn (~9 tỷ tham số) cho lái xe tự động: nó nén video camera + text + hành động thành **token rời rạc**, dùng một Transformer tự hồi quy ~9B để **dự đoán token kế tiếp**, rồi dùng một **video diffusion decoder** để dựng token về video độ phân giải cao, mượt theo thời gian. Điểm đáng học: chính công thức next-token của LM, khi scale, làm *nổi lên* hiểu biết về hình học 3D, động lực cảnh và phản ứng với hành động — biến world model thành neural simulator sinh dữ liệu vô hạn cho AV. Caveat: tốn kém, tokenizer chặn chất lượng, và rollout dài vẫn drift.

[Tokenized world model](07-tokenized-world-model.md) cho ta mẫu hình; GAIA-1 (Hu et al., Wayve, 2023) cho thấy mẫu hình đó *scale tới đâu* trong một domain thực, an toàn-trọng yếu. Mục này đọc GAIA-1 như một bản thiết kế ba khối — tokenize, model, decode — và rút ra bài học về cách ráp các nguyên thủy tầng 9 thành một hệ thống.

---

## 1. Bức tranh lớn: world model như sequence model

GAIA-1 đặt bài toán world model thành **unsupervised sequence modeling**: mọi modal (ảnh, chữ, hành động) → token rời rạc → "đoán token kế tiếp". Đầu vào điều kiện gồm khung hình quá khứ, mô tả văn bản, và tín hiệu điều khiển xe; đầu ra là token của khung kế. Vì đây đúng là công thức của một language model, GAIA-1 thừa hưởng khả năng scale của LLM — và chính việc scale lên 9B là nơi các tính chất thú vị xuất hiện.

---

## 2. Ba khối kiến trúc

### Khối 1 — Tokenizers (đa modal → token)

- **Image tokenizer**: nén mỗi khung hình thành lưới token rời rạc, huấn luyện bằng reconstruction loss + quantization loss kiểu [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md), **cộng distillation từ DINO** để token mang inductive bias *ngữ nghĩa* (giảm token dành cho texture vô nghĩa, tăng token cho cấu trúc). Đây là một lựa chọn quan trọng: token ngữ nghĩa làm world model dễ học động lực hơn token thuần pixel.
- **Text tokenizer**: dùng T5 đã pretrain, sinh ~32 token mỗi bước thời gian để điều kiện hóa bằng ngôn ngữ ("xe rẽ trái", "trời mưa").
- **Action**: tín hiệu điều khiển xe được rời rạc/embedding thành token, xen vào chuỗi.

### Khối 2 — World model (Transformer tự hồi quy)

Một Transformer kiểu GPT (~9B tham số tổng thể) nhận chuỗi xen kẽ (token ảnh, text, hành động) và dự đoán token ảnh kế tiếp bằng mục tiêu cross-entropy next-token. Đây là trái tim "tưởng tượng": cho lịch sử và một hành động/điều kiện, nó sample tương lai trong không gian token — không đụng pixel.

### Khối 3 — Video diffusion decoder

Token dự đoán được đưa qua một **video diffusion model** (3D U-Net với attention tách rời không gian/thời gian) để dựng về video độ phân giải cao. Mô hình hóa decode như *khử nhiễu cả một chuỗi khung* cải thiện mạnh tính nhất quán thời gian (temporal consistency) — tránh hiện tượng nhấp nháy giữa các frame. Lúc train, decoder điều kiện trên token của ảnh thật; lúc inference, điều kiện trên token *do world model dự đoán*.

Lưu ý kiến trúc: GAIA-1 **tách tokenizer (rời rạc) khỏi decoder (diffusion)**. Đây là một mẫu lai đáng học — dùng token rời rạc cho phần *reasoning/động lực* (nơi cần sequence model và đa mode), nhưng dùng diffusion liên tục cho phần *render* (nơi cần chi tiết tần số cao). Cách phân vai này lặp lại ở nhiều world model lớn về sau.

---

## 3. Vì sao scale lại làm nổi lên hiểu biết

GAIA-1 báo cáo các **emergent properties** khi scale: hiểu hình học 3D (vật thể nhất quán khi góc nhìn đổi), động lực cảnh (xe khác di chuyển hợp lý), contextual awareness, và generalization sang tình huống hiếm. Không tính chất nào được supervise trực tiếp — chúng *xuất hiện* từ việc ép một model đủ lớn dự đoán đúng token kế tiếp trên dữ liệu lái xe đa dạng.

Đây là minh chứng mạnh cho luận điểm trung tâm của tầng 9 (và [latent prediction](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md)): *predict để hiểu*. Khác với MAE phải dựng từng pixel, world model token học cấu trúc thế giới vì đó là cách duy nhất để đoán token đúng — và ở quy mô lớn, cấu trúc đó bao gồm cả hình học [3D](../../03b-3d-representation/research/02-nerf.md) và nhân quả của hành động.

Ứng dụng then chốt: GAIA-1 là **neural simulator** — sinh vô hạn kịch bản lái xe (kể cả tình huống hiếm/nguy hiểm) để train và kiểm thử hệ thống AV, điều khiển được bằng text và hành động.

---

## 4. Giới hạn / Khi nào thất bại

**Chi phí khổng lồ.** 9B tham số + diffusion decoder = train và inference rất đắt; rollout video dài tốn kém, khó realtime.

**Tokenizer là trần chất lượng.** Thông tin mất ở image tokenizer chặn mọi thứ phía sau; [codebook collapse](04-codebook-collapse.md) hay token thiếu chi tiết quan trọng (biển báo nhỏ) trực tiếp hại độ tin cậy — vấn đề nghiêm trọng trong domain an toàn-trọng yếu.

**Compounding error.** Như mọi rollout tự hồi quy ([imagination horizon](../../07-latent-planning/research/10-latent-imagination-horizon.md)), video dài drift khỏi động lực thật.

**Không đảm bảo vật lý/an toàn.** Model học thống kê dữ liệu, không có ràng buộc vật lý cứng; có thể sinh cảnh "trông thật" nhưng vi phạm luật giao thông hay vật lý — rủi ro khi dùng làm simulator để chứng nhận an toàn.

**Phụ thuộc dữ liệu lái xe.** Emergent properties đến từ dữ liệu Wayve quy mô lớn; tái lập ở domain khác cần dữ liệu tương xứng.

---

## 5. Liên hệ với Latent-Anything

GAIA-1 là một *anchor model* điển hình cho framework: latent của nó là **token rời rạc đa modal**, và pipeline của nó (tokenize → model → decode) trải đúng ba pillar của Latent-Anything.

```python
class GAIAStyleAdapter(Protocol):
    def tokenize(self, video: np.ndarray, text: str | None,
                 action: np.ndarray | None) -> np.ndarray: ...   # -> token stream
    def imagine(self, tokens: np.ndarray, action: np.ndarray) -> np.ndarray: ...  # next tokens
    def render(self, tokens: np.ndarray) -> np.ndarray: ...       # diffusion decode -> video
```

- **Layer A — Introspection**: token đa modal cho phép soi *điều kiện hóa chéo* — text token nào ảnh hưởng token ảnh nào (attention), entropy dự đoán đo độ bất định của cảnh tương lai. Các emergent property (hình học 3D) là thứ Layer A có thể *kiểm tra* bằng probe trên token.
- **Layer B — Manipulation**: điều khiển bằng text/action chính là manipulation latent có ngữ nghĩa cao — chèn token điều kiện để lái kịch bản ("thêm người đi bộ"), đúng tinh thần steering nhưng ở mức world model.
- **Layer C — Runtime**: kiến trúc lai (sequence model rời rạc + diffusion decoder) là một runtime pattern — reasoning rẻ trong token, render đắt chỉ khi cần xem; Layer C có thể lập lịch decode lười (lazy) để tiết kiệm.

GAIA-1 cho thấy mẫu hình token world model scale tới hệ thống thực, đa modal, điều khiển bằng ngôn ngữ. Mục cuối tầng — **Genie** — đẩy ý tưởng "điều khiển được" đi xa hơn: học *latent action* trực tiếp từ video không nhãn, sinh world model tương tác chơi được.

---

## Liên quan

- [Tokenized World Model](07-tokenized-world-model.md) — GAIA-1 là hiện thân quy mô lớn của mẫu hình này.
- [Vector Quantization](01-vector-quantization.md) — image tokenizer của GAIA-1 dựa trên VQ + DINO distillation.
- [Tại sao latent prediction tốt hơn pixel prediction](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md) — emergent understanding từ next-token, không từ dựng pixel.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — imagination trong token là cách GAIA-1 sinh kịch bản.
- [NeRF](../../03b-3d-representation/research/02-nerf.md) — hiểu hình học 3D nổi lên trong GAIA-1, liên hệ biểu diễn 3D.

## Tham khảo

- A. Hu et al., *GAIA-1: A Generative World Model for Autonomous Driving* (Wayve, 2023, arXiv:2309.17080).
- J. Ho et al., *Video Diffusion Models* (NeurIPS 2022, arXiv:2204.03458) — nền tảng cho decoder.
- M. Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO, ICCV 2021, arXiv:2104.14294) — nguồn distillation cho tokenizer.
- C. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, JMLR 2020, arXiv:1910.10683) — text encoder.
