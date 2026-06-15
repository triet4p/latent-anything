# OpenVLA

> **TL;DR.** OpenVLA là Vision-Language-Action model mã nguồn mở 7B: một VLM (vision encoder kép **DINOv2 + SigLIP** → 256 visual token, backbone **Llama-2 7B**) được fine-tune để xuất **hành động robot dưới dạng token**. Mỗi chiều của action 7-DoF bị **rời rạc hóa thành 1 trong 256 bin** ([scalar quantization](../../09-discrete-latent/research/06-finite-scalar-quantization.md) per-dimension) rồi dự đoán tự hồi quy như token ngôn ngữ. Train trên 970k demo Open X-Embodiment, nó vượt RT-2-X (55B) dù nhỏ hơn nhiều. Caveat: action rời rạc thô có thể giới hạn điều khiển mịn; inference autoregressive chậm cho điều khiển realtime; 7B vẫn nặng cho robot trên thiết bị.

Các mục trước của tầng là *world model* — predict/plan trong latent. OpenVLA (Kim et al., 2024) đại diện hướng khác: **policy trực tiếp** map vision + language → action, tái dùng nguyên kiến trúc LLM. Điểm framework cần học (THEORY.md ghi rõ): *latent action space* và *cách thiết kế ModelAdapter* cho VLA — và mấu chốt là action được xử lý như **token rời rạc**, nối thẳng về tầng 9.

---

## 1. Trực giác: hành động cũng là token

Bài học lớn của tầng 9 là: rời rạc hóa cái gì thì dùng được bộ máy LM cho cái đó. OpenVLA áp đúng ý đó lên *hành động*. Một VLM đã giỏi map (ảnh, chữ) → token văn bản; OpenVLA chỉ thay "token văn bản" bằng "token hành động". Mỗi lệnh điều khiển robot trở thành một chuỗi ngắn token, và dự đoán hành động = **dự đoán token kế tiếp** — y hệt sinh văn bản.

Nhờ vậy, OpenVLA thừa hưởng pretrained VLM (kiến thức thị giác–ngôn ngữ web-scale) rồi chỉ cần fine-tune để "nói ngôn ngữ hành động" trên dữ liệu robot.

---

## 2. Cơ chế: kiến trúc và action tokenization

### Backbone VLM (Prismatic-7B)

- **Vision**: encoder *kép* — DINOv2 (đặc trưng không gian/hình học mạnh) + SigLIP (đặc trưng ngữ nghĩa CLIP-style), nối lại (~600M) rồi chiếu vào không gian token của LLM, cho **256 visual token** mỗi quan sát.
- **Language**: Llama-2 7B nhận xen kẽ visual token + lệnh ngôn ngữ, suy luận, và xuất token hành động.

### Action tokenization (mấu chốt)

Hành động robot 7-DoF (Δx, Δy, Δz, 3 góc xoay, gripper) là vector liên tục. OpenVLA **rời rạc hóa từng chiều** thành 256 bin (chia theo phân vị của dữ liệu), mỗi bin một token:

$$
a^{(i)} \;\xrightarrow{\text{bin}}\; k_i \in \{0,\dots,255\}, \qquad i = 1,\dots,7.
$$

Trong đó $a^{(i)}$ là chiều thứ $i$ của hành động, $k_i$ là chỉ số bin. Đây chính là **scalar quantization per-dimension** — họ hàng trực tiếp của [FSQ](../../09-discrete-latent/research/06-finite-scalar-quantization.md): không codebook học-được, chỉ lưới cố định trên mỗi trục. 7 token này được nhét vào từ vựng LLM (ghi đè các token ít dùng nhất), và model dự đoán chúng tự hồi quy bằng cross-entropy — không cần đầu ra hồi quy liên tục.

### Huấn luyện và hiệu quả

Fine-tune end-to-end trên 970k demo Open X-Embodiment (nhiều robot, nhiều task). Hỗ trợ **LoRA** (parameter-efficient fine-tuning) và quantization để chạy/tinh chỉnh rẻ. Kết quả: vượt RT-2-X (55B, đóng) và Diffusion Policy, dù chỉ 7B và mở.

---

## 3. Vì sao quan trọng cho framework

| | World model (Dreamer/MuZero) | VLA (OpenVLA) |
|---|---|---|
| Vai trò latent | state để predict/plan | cầu vision+language → **action** |
| Action | đầu ra của planner | **token rời rạc** dự đoán trực tiếp |
| Kiến trúc | RSSM/MCTS riêng | **tái dùng LLM** (Llama-2) nguyên khối |
| Học | reconstruction/value | imitation (next-action-token) |

OpenVLA là *anchor model* cho `ModelAdapter` kiểu VLA: nó cho thấy latent action có thể là **token rời rạc** trong từ vựng LLM. Đây là điểm thiết kế then chốt — adapter không cần coi action là vector liên tục bí ẩn; nó có thể phơi bày action như token, dùng chung công cụ introspection/manipulation với observation token. Cũng nối thẳng [Genie latent action](../../09-discrete-latent/research/09-genie.md): cả hai rời rạc hóa action, khác ở chỗ OpenVLA bin theo dữ liệu robot có nhãn còn Genie học latent action không giám sát.

---

## 4. Giới hạn / Khi nào thất bại

**Rời rạc hóa thô giới hạn điều khiển mịn.** 256 bin/chiều là lưới cố định; tác vụ cần độ chính xác dưới-bin (lắp ráp tinh vi) có thể bị giới hạn — động lực cho action head liên tục như flow matching của **π0 (mục sau)**.

**Inference autoregressive chậm.** Sinh 7 token tuần tự mỗi bước + backbone 7B → tần số điều khiển thấp; khó cho robot phản ứng nhanh (có các hướng tăng tốc: parallel decoding, action chunking).

**7B vẫn nặng.** Triển khai on-robot cần quantization/LoRA; không phải lúc nào cũng vừa phần cứng nhúng.

**Imitation, không planning.** OpenVLA là policy bắt chước, không có world model để tưởng tượng hệ quả; ngoài phân phối dữ liệu demo, nó không "suy nghĩ trước" như MuZero/Dreamer.

**Phụ thuộc phân bố Open X-Embodiment.** Tổng quát hóa sang embodiment/robot rất khác dữ liệu train vẫn cần fine-tune.

---

## 5. Liên hệ với Latent-Anything

OpenVLA định hình `ModelAdapter` cho VLA: cả observation lẫn action đều là **token**, nên adapter phơi bày một giao diện token thống nhất.

```python
class VLAAdapter(Protocol):
    def encode_obs(self, image: np.ndarray, instruction: str) -> np.ndarray: ...  # -> token sequence
    def predict_action_tokens(self, context: np.ndarray) -> np.ndarray: ...        # -> 7 action-bin ids
    def detokenize_action(self, tokens: np.ndarray) -> np.ndarray: ...             # bins -> continuous 7-DoF
    action_bins: int   # 256
    action_dim: int    # 7
```

- **Layer A — Introspection**: action-as-token cho phép soi *phân phối action* (entropy mỗi chiều, bin usage giống usage codebook tầng 9), và dùng attention để xem token ảnh/lệnh nào quyết định action — đúng loại phân tích Layer A nhắm tới, mượn thẳng từ NLP.
- **Layer B — Manipulation**: chỉnh một action token (đổi bin) là can thiệp điều khiển có ngữ nghĩa; nối lệnh ngôn ngữ là steering policy ở mức cao.
- **Layer C — Runtime**: sinh action token là sequence generation — KV-cache, action chunking, parallel decoding áp dụng được; Layer C tối ưu latency điều khiển bằng chính hạ tầng LLM.

OpenVLA cho thấy action rời rạc + LLM hoạt động ở quy mô mở. Mục kế tiếp — **π0** — đi hướng đối lập cho action: bỏ rời rạc hóa, dùng **flow matching** để xuất action liên tục mịn, một thiết kế latent action khác hẳn mà adapter cũng phải hỗ trợ.

---

## Liên quan

- [Finite Scalar Quantization (FSQ)](../../09-discrete-latent/research/06-finite-scalar-quantization.md) — action binning của OpenVLA là scalar quantization per-dimension.
- [Genie](../../09-discrete-latent/research/09-genie.md) — cũng rời rạc hóa action; Genie học latent action không giám sát, OpenVLA bin có nhãn.
- [Tokenized World Model](../../09-discrete-latent/research/07-tokenized-world-model.md) — cùng triết lý "mọi thứ là token", áp lên action.
- [Vector Quantization](../../09-discrete-latent/research/01-vector-quantization.md) — nền tảng rời rạc hóa latent.

## Tham khảo

- M. J. Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model* (CoRL 2024, arXiv:2406.09246).
- S. Karamcheti et al., *Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models* (ICML 2024, arXiv:2402.07865) — backbone VLM.
- A. Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (2023, arXiv:2307.15818) — tiền thân action-as-token.
- Open X-Embodiment Collaboration, *Open X-Embodiment: Robotic Learning Datasets and RT-X Models* (2023, arXiv:2310.08864).
