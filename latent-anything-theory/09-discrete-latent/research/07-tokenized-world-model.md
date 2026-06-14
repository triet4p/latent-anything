# Tokenized World Model

> **TL;DR.** Một tokenized world model gồm hai khối: (1) một **discrete autoencoder** ([VQ](01-vector-quantization.md) hoặc [FSQ](06-finite-scalar-quantization.md)) biến mỗi observation thành một chuỗi token rời rạc, và (2) một **Transformer tự hồi quy** học động lực bằng cách dự đoán *token kế tiếp* — token của frame sau, kèm reward và tín hiệu kết thúc. Nó biến học động lực thành **sequence modeling**, dùng nguyên xi bộ máy của language model. Caveat: lỗi tích lũy khi rollout dài, mỗi frame tốn nhiều token (chuỗi nở ra), và chất lượng bị chặn bởi tokenizer.

Cả tầng 9 đến đây đã xây dựng *cách rời rạc hóa* latent. Tokenized world model là nơi thu hoạch: một khi observation là chuỗi token, [latent transition model](../../06-latent-temporal/research/02-latent-transition-model.md) không còn là một mạng hồi quy đặc thù nữa — nó trở thành một bài toán "đoán từ tiếp theo" y hệt GPT. Đây chính là lời hứa unify của tầng: latent space và language model dùng *cùng một kiến trúc*.

---

## 1. Trực giác: động lực thế giới như một ngôn ngữ

Language model học $p(\text{từ}_t \mid \text{từ}_{<t})$. Tokenized world model học $p(\text{token}_t \mid \text{token}_{<t}, a_{<t})$ — phân phối token kế tiếp cho trước lịch sử token và hành động. Nếu tokenizer biến mỗi frame thành $K$ token từ từ vựng cỡ $N$, thì một đoạn video là một *câu* dài trong "ngôn ngữ của token ảnh", và Transformer học ngữ pháp của thế giới: cái gì hợp lệ theo sau cái gì.

Phép loại suy này không chỉ là ẩn dụ — nó cho phép **tái dùng toàn bộ** kỹ thuật LM: kiến trúc Transformer decoder, mục tiêu cross-entropy next-token, sampling (top-k, nucleus), scaling laws, KV-cache cho inference. Đây là lý do hướng này bùng nổ: mọi tiến bộ của LLM lập tức chảy sang world model.

---

## 2. Cơ chế: hai khối

### Khối 1 — Discrete autoencoder (tokenizer)

Encoder + quantizer biến observation $x_t$ thành $K$ token $(w_t^1, \dots, w_t^K)$, mỗi token thuộc từ vựng $N$. Decoder dựng lại $x_t$ từ token để huấn luyện tokenizer (reconstruction). Đây đúng là [VQ-VAE](../../02-representation-learning/research/04-vq-vae.md) / [VQGAN](../../02-representation-learning/research/05-vqgan.md), hoặc một FSQ tokenizer. Sau khi train, tokenizer bị *đóng băng* và chỉ còn là cầu giữa pixel và token.

### Khối 2 — Transformer động lực tự hồi quy

Một Transformer kiểu GPT nhận chuỗi token (xen kẽ token observation và token/embedding hành động) và dự đoán, theo thứ tự tự hồi quy:

$$
p_\theta\big(w_{t+1}^{1:K},\, r_t,\, d_t \mid w_{\le t}^{1:K},\, a_{\le t}\big),
$$

trong đó $w_{t+1}^{1:K}$ là $K$ token của frame kế, $r_t$ là reward, $d_t$ là cờ kết thúc (done), và $a$ là hành động. Mục tiêu huấn luyện là **cross-entropy next-token** trên token frame kế (cộng loss cho reward/done) — không có pixel loss ở khối này. IRIS (Micheli et al., 2023) là hiện thân kinh điển: nó học *thuần trong tưởng tượng* của world model này và lập SOTA trên Atari 100k cho nhóm không dùng tìm kiếm.

### Rollout / imagination trong không gian token

Sinh tương lai = sampling tự hồi quy: cho lịch sử, sample $K$ token của frame kế, nối vào chuỗi, lặp. Đây là [rollout / latent imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) phiên bản rời rạc — agent học và plan hoàn toàn trong token, chỉ decode về pixel khi cần xem.

---

## 3. Vì sao token rời rạc lại hợp cho động lực

| | Latent liên tục (RSSM/Dreamer) | Tokenized (IRIS-style) |
|---|---|---|
| Đầu ra transition | Gaussian/phân phối liên tục | **categorical** trên từ vựng token |
| Loss | KL + reconstruction | **cross-entropy** next-token |
| Kiến trúc | RNN/GRU + stochastic head | Transformer decoder (GPT) |
| Multi-modal tương lai | khó (Gaussian đơn mode) | dễ — categorical biểu diễn đa mode tự nhiên |
| Tái dùng từ LM | ít | **toàn bộ** stack LM |

Điểm mạnh quan trọng: phân phối categorical trên token **biểu diễn tương lai đa mode** một cách tự nhiên (nhiều token có xác suất cao) — tránh trung bình hóa làm mờ như Gaussian đơn mode. DreamerV2 phát hiện chính điều này khi chuyển sang discrete latent. Đồng thời, mục tiêu cross-entropy không ép model dựng lại từng pixel ở khối động lực — gần với tinh thần [value equivalence](../../07-latent-planning/research/08-value-equivalence-muzero.md) và [latent prediction](../../08-latent-prediction/research/09-latent-vs-pixel-prediction.md): chỉ cần đoán đúng *token*, không cần render.

---

## 4. Giới hạn / Khi nào thất bại

**Lỗi tích lũy (compounding error).** Token sai ở bước sớm làm lệch toàn bộ rollout sau — đúng vấn đề của [latent imagination horizon](../../07-latent-planning/research/10-latent-imagination-horizon.md). Rollout dài drift khỏi động lực thật.

**Chuỗi nở ra.** Mỗi frame là $K$ token; video $T$ frame → $T\cdot K$ token. Với ảnh độ phân giải cao, $K$ lớn khiến attention $O((TK)^2)$ tốn kém — nút thắt chính, thúc đẩy các tokenizer "ít token hơn" (Δ-IRIS mã hóa *delta* giữa frame, context-aware tokenization).

**Bị chặn bởi tokenizer.** Thông tin mất ở bước lượng tử là cận trên cho mọi thứ phía sau; [codebook collapse](04-codebook-collapse.md) ở tokenizer trực tiếp giết chất lượng world model. Tokenizer và dynamics thường train tách, nên lỗi tokenizer không được dynamics sửa.

**Token rời rạc mất cấu trúc metric.** Trừ FSQ, token VQ không có thứ tự nội tại — "gần nhau trong không gian token" không có nghĩa, làm một số thao tác hình học (interpolation) khó hơn latent liên tục.

**Phụ thuộc thứ tự token trong frame.** Sinh $K$ token của một frame theo thứ tự tự hồi quy áp một thứ tự nhân tạo lên không gian 2D; lựa chọn thứ tự (raster, v.v.) ảnh hưởng chất lượng.

---

## 5. Liên hệ với Latent-Anything

Tokenized world model là một *kiến trúc model* mà framework cần adapter riêng — nó phơi bày latent dưới dạng **chuỗi token**, không phải vector. Đây là loại model mà checkpoint "sau tầng 9" của roadmap nhắm tới:

```python
class TokenizedWorldModel(Protocol):
    def tokenize(self, obs: np.ndarray) -> np.ndarray: ...      # -> (K,) token ids
    def detokenize(self, tokens: np.ndarray) -> np.ndarray: ... # -> obs
    def predict_next(self, tokens: np.ndarray, action: np.ndarray
                     ) -> tuple[np.ndarray, float, bool]: ...   # next tokens, reward, done
    vocab_size: int
    tokens_per_frame: int   # K
```

- **Layer A — Introspection**: latent là token, nên introspection mượn được công cụ NLP — phân tích phân phối token kế tiếp (entropy = độ bất định của world model), attention map (token nào ảnh hưởng dự đoán nào), perplexity của dynamics. Đây là cầu để Layer A nói chung một ngôn ngữ với phân tích LLM.
- **Layer B — Manipulation**: can thiệp ở mức token — đổi một token observation rồi rollout để xem world model "tưởng tượng" gì — là một thí nghiệm phản thực (counterfactual) sạch, rời rạc, dễ kiểm soát.
- **Layer C — Runtime**: rollout token là sequence generation; Layer C tái dùng KV-cache, batched sampling, speculative decoding — hạ tầng inference của LLM dùng lại được nguyên khối.

Tokenized world model là *mẫu hình* mà tầng 9 hướng tới. Hai mục cuối là hai hiện thân quy mô lớn của nó: **GAIA-1** (lái xe tự động) và **Genie** (world model tương tác từ video), cho thấy mẫu hình này scale tới đâu.

---

## Liên quan

- [Vector Quantization](01-vector-quantization.md) — tokenizer thường là VQ; token là đầu ra của nó.
- [Finite Scalar Quantization (FSQ)](06-finite-scalar-quantization.md) — tokenizer thay thế, né collapse cho world model.
- [Latent Transition Model](../../06-latent-temporal/research/02-latent-transition-model.md) — tokenized WM là transition model với đầu ra categorical.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — rollout token là imagination rời rạc.
- [Value Equivalence (MuZero)](../../07-latent-planning/research/08-value-equivalence-muzero.md) — cùng tinh thần: không cần render, chỉ cần đoán đúng cái phục vụ planning.

## Tham khảo

- V. Micheli, E. Alonso, F. Fleuret, *Transformers are Sample-Efficient World Models* (IRIS, ICLR 2023, arXiv:2209.00588).
- V. Micheli, E. Alonso, F. Fleuret, *Efficient World Models with Context-Aware Tokenization* (Δ-IRIS, ICML 2024, arXiv:2406.19320).
- D. Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2, ICLR 2021, arXiv:2010.02193) — discrete latent cho world model.
- A. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — kiến trúc tự hồi quy được tái dùng.
