# VQGAN (Vector Quantized Generative Adversarial Network)

VQGAN được đề xuất bởi Esser và các cộng sự vào năm 2021 trong bài báo *"Taming Transformers for High-Resolution Image Synthesis"*, là một bước đột phá trong việc xử lý hình ảnh độ phân giải cao bằng cách kết hợp sức mạnh biểu diễn rời rạc của VQ-VAE với khả năng sinh ảnh sắc nét của GAN. 

Dưới đây là cơ chế hoạt động chi tiết của VQGAN và cách các "tô-ken ẩn" (latent tokens) của nó mở đường cho kỷ nguyên của các mô hình sinh (Generative Models) quy mô lớn.

### 1. VQGAN = VQ-VAE + Perceptual Loss + Adversarial Loss

Mặc dù VQ-VAE giải quyết được nhiều vấn đề của Autoencoder truyền thống, nhưng khi sử dụng hàm mất mát tái cấu trúc ở cấp độ pixel (chẳng hạn như L2/MSE), nó có xu hướng "trung bình hóa" các chi tiết hình học tần số cao, khiến hình ảnh được giải mã thường bị mờ (blurry) và thiếu chân thực. VQGAN khắc phục triệt để điểm yếu này bằng việc thay thế/bổ sung hàm mất mát bằng **Perceptual Loss** và **Adversarial Loss**:

*   **Tổn thất nhận thức (Perceptual Loss / LPIPS):** 
    Thay vì so sánh trực tiếp từng điểm ảnh (pixel-by-pixel), VQGAN sử dụng một mạng nơ-ron đã được huấn luyện trước (thường là VGG-16) để trích xuất các bản đồ đặc trưng (feature maps) của cả ảnh gốc và ảnh tái cấu trúc. Khoảng cách được tính toán trên không gian đặc trưng (latent space) của mạng VGG này. Cơ chế này mô phỏng cách hệ thống thị giác của con người nhận thức cấu trúc và hình học, giúp mô hình bắt được những chi tiết mà sai số pixel không thể đo lường được.
*   **Tổn thất đối nghịch (Adversarial Loss / PatchGAN):** 
    VQGAN tích hợp thêm một bộ phân biệt (Discriminator) hoạt động theo cơ chế PatchGAN. Thay vì đánh giá toàn bộ bức ảnh là thật hay giả bằng một quyết định duy nhất, PatchGAN đánh giá tính chân thực trên từng phân vùng nhỏ (patch) của bức ảnh. Việc ép bộ giải mã (decoder) phải lừa được bộ phân biệt ở cấp độ cục bộ giúp mô hình sinh ra các kết cấu bề mặt (textures) và chi tiết cực kỳ sắc nét, tự nhiên.
*   **Trọng số thích ứng $\lambda$ (Adaptive Loss Weight):**
    Để dung hòa lực kéo giữa việc giữ đúng cấu trúc tổng thể (Perceptual/Reconstruction Loss) và việc tạo chi tiết sắc nét (Adversarial Loss), VQGAN sử dụng một trọng số thích ứng $\lambda$ vô cùng thông minh. Trọng số này được tính toán động tại mỗi bước lặp dựa trên tỷ lệ độ lớn gradient của lớp cuối cùng trong bộ giải mã:
    $$\lambda = \frac{\nabla_{G_L}[\mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{perceptual}}]}{\nabla_{G_L}[\mathcal{L}_{\text{GAN}}] + \delta}$$
    Cơ chế này đảm bảo sự cân bằng gradient: khi mô hình đang bận học cấu trúc thô (gradient tái cấu trúc lớn), $\lambda$ sẽ tự động nhỏ lại để giảm ảnh hưởng của GAN. Khi cấu trúc đã ổn định, $\lambda$ tăng lên, cho phép GAN tinh chỉnh độ sắc nét của bức ảnh.

### 2. Cách Latent (Tô-ken rời rạc) được dùng trong các Generative Model lớn

Mục tiêu tối thượng của VQGAN không chỉ là tái cấu trúc ảnh, mà là tạo ra một **bảng mã (Codebook) và các chuỗi tô-ken** để phục vụ cho các mạng Transformer phía sau.

**Vấn đề của Transformer với hình ảnh:**
Mạng Transformer có độ phức tạp tính toán tăng theo bình phương (quadratic complexity) độ dài của chuỗi đầu vào. Do đó, việc đưa trực tiếp hàng triệu pixel của một bức ảnh độ phân giải cao vào Transformer là bất khả thi về mặt tài nguyên tính toán (ví dụ: ảnh 224x224 sẽ có chuỗi độ dài $224^2 \times 3$, vượt quá giới hạn bộ nhớ).

**Sự kết hợp hoàn hảo giữa VQGAN và Transformer:**

1.  **Nén thành chuỗi tô-ken (Tokenization):** VQGAN đóng vai trò là một "bộ từ vựng hóa" (tokenizer) cho hình ảnh. Nó nén một bức ảnh độ phân giải cao khổng lồ thành một lưới (grid) nhỏ hơn nhiều (ví dụ 16x16) gồm các vector ẩn. Thay vì lưu trữ vector liên tục, VQGAN ánh xạ từng vector vào một ID duy nhất trong Codebook.
2.  **Đồng nhất hóa hình ảnh và ngôn ngữ:** Lúc này, bức ảnh được biểu diễn bằng một chuỗi 1D các con số nguyên (ví dụ: `[45, 1023, 12, ...]`). Về mặt toán học, chuỗi tô-ken hình ảnh này hoàn toàn **giống hệt với một chuỗi từ khóa (text tokens)** trong ngôn ngữ tự nhiên. 
3.  **Học tự hồi quy (Autoregressive Modeling):** Các mô hình ngôn ngữ lớn hoặc Transformer giờ đây có thể coi việc sinh ra một bức ảnh giống hệt như tác vụ "dự đoán từ tiếp theo" (next-token prediction). Mô hình sẽ đọc các tô-ken văn bản (prompt) và các tô-ken ảnh đã sinh ra trước đó để dự đoán xác suất của tô-ken ảnh tiếp theo $p(s_i | s_{<i})$. Sau khi dự đoán xong toàn bộ chuỗi tô-ken, bộ giải mã của VQGAN sẽ nhận chuỗi này và render ngược lại thành một bức ảnh độ phân giải cao.

**Ý nghĩa đối với các mô hình Generative hiện đại:**
Nhờ việc số hóa dữ liệu liên tục (hình ảnh, video, âm thanh) thành các "tô-ken ẩn" (latent tokens) rời rạc như của VQGAN, các thế hệ AI đa phương tiện quy mô lớn hiện nay (như LLaMA, Muse, Chameleon, Parti, v.v.) có thể kết hợp chung dữ liệu văn bản và hình ảnh vào cùng một từ điển. Điều này mở ra kỷ nguyên cho phép duy nhất một cấu trúc Transformer tiêu chuẩn xử lý linh hoạt mọi loại dữ liệu (multimodal) mà không cần thay đổi kiến trúc nội bộ.

---

## Liên quan

- [VQ-VAE](04-vq-vae.md) — nền tảng codebook + lượng tử hóa mà VQGAN kế thừa.
- [VAE](03-vae.md) — gốc rễ của họ autoencoder sinh.

## Tham khảo

- Esser, Rombach, Ommer, *Taming Transformers for High-Resolution Image Synthesis* (CVPR 2021, arXiv:2012.09841) — VQGAN.
- Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (CVPR, 2018) — LPIPS / perceptual loss.
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks* (CVPR, 2017) — PatchGAN discriminator.