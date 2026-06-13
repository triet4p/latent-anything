# Masked Autoencoder (MAE)

> **TL;DR.** MAE học biểu diễn tự giám sát bằng cách *che ngẫu nhiên ~75% patch* của ảnh rồi **dựng lại pixel** đã che từ phần nhìn thấy, qua một kiến trúc **bất đối xứng**: encoder lớn chỉ chạy trên patch nhìn thấy, decoder nhẹ dựng lại toàn ảnh từ latent + mask token. Mask tỉ lệ cao biến reconstruction thành bài toán *không tầm thường*, ép model hiểu cấu trúc toàn cục thay vì nội suy texture lân cận. Caveat: mục tiêu là **pixel** — model phải tiêu tốn capacity cho chi tiết bề mặt (texture, nền) vô nghĩa với semantics; đây chính là baseline mà cả tầng 8 sẽ vượt qua bằng cách predict *trong latent*.

Tầng 8 đặt một câu hỏi triệt để: nếu mục tiêu là *reasoning* trong latent, có cần decode về observation không? MAE là **mốc khởi đầu** đúng nghĩa — nó là đỉnh cao của hướng *masked prediction ở mức pixel*: học biểu diễn mạnh, scalable, nhưng vẫn buộc model dựng lại từng pixel. Đọc MAE để thấy rõ cái gì là *generative reconstruction* trước khi các mục sau (collapse, BYOL/SimSiam, JEPA) lần lượt tháo bỏ decoder.

---

## **1. Trực giác / Định nghĩa**

MAE là một **denoising autoencoder** dạng đặc biệt: "nhiễu" ở đây là *che mất* phần lớn input, và nhiệm vụ là khôi phục phần bị che. Ảnh được chia thành patch không chồng lấp (như ViT); một tỉ lệ lớn patch (mặc định **75%**) bị bỏ ngẫu nhiên; model phải dự đoán lại pixel của các patch đó.

Vì sao tỉ lệ che phải *cao*? Ảnh có dư thừa không gian rất lớn — che 10–20% thì model chỉ cần nội suy từ pixel lân cận, một bài toán tầm thường không buộc hiểu nội dung. Che 75% phá vỡ dư thừa đó: để điền vào một mảng lớn bị mất, model phải nắm *cấu trúc toàn cục* của vật thể và cảnh. Đây là điểm khác biệt then chốt với masked language modeling (BERT che ~15%) — ngôn ngữ đặc thông tin hơn ảnh rất nhiều.

---

## **2. Cơ chế: kiến trúc bất đối xứng**

MAE gồm hai khối với chi phí lệch hẳn nhau:

$$
z = f_{\text{enc}}(\{x_i : i \in \mathcal{V}\}), \qquad \hat{x} = f_{\text{dec}}(z \cup \{m_j : j \in \mathcal{M}\}).
$$

Trong đó $\mathcal{V}$ là tập patch *nhìn thấy* (visible, ~25%), $\mathcal{M}$ là tập patch *bị che* (masked, ~75%), $m_j$ là một **mask token** học được (vector chung, có thêm positional embedding để biết vị trí cần điền), và $\hat x$ là ảnh dựng lại. Điểm cốt lõi: **encoder chỉ nhận patch nhìn thấy** — nó không hề thấy mask token. Nhờ vậy encoder chỉ xử lý 25% số token, giảm ~3–4× chi phí tính toán và bộ nhớ, cho phép scale lên ViT-Huge.

Decoder thì **nhẹ** (nông và hẹp hơn encoder nhiều) và chỉ tồn tại lúc pretrain; sau pretrain nó bị **vứt bỏ**, chỉ giữ encoder làm bộ trích đặc trưng. Đây là chỗ "autoencoder" của MAE khác autoencoder cổ điển: decoder không phải sản phẩm, chỉ là giàn giáo tạo tín hiệu học.

### Hàm mất mát: pixel-space MSE

MAE tối ưu sai số bình phương trên pixel của *riêng các patch bị che*:

$$
\mathcal{L} = \frac{1}{|\mathcal{M}|}\sum_{j \in \mathcal{M}} \lVert \hat{x}_j - x_j \rVert_2^2.
$$

Trong đó tổng chỉ chạy trên patch bị che ($\mathcal{M}$) — tính loss trên patch nhìn thấy làm giảm chất lượng. Một biến thể quan trọng: dùng **normalized pixel** (chuẩn hóa mean/variance trong mỗi patch) làm target, cho biểu diễn tốt hơn vì buộc model tập trung vào *cấu trúc tương phản cục bộ* thay vì độ sáng tuyệt đối. Dù vậy, target vẫn là **pixel** — đây là đặc điểm định nghĩa và cũng là giới hạn của MAE.

---

## **3. Vì sao MAE mạnh — và nó dạy gì cho tầng 8**

MAE đạt 87.8% trên ImageNet-1K với ViT-Huge (chỉ dùng dữ liệu ImageNet-1K), train nhanh hơn 3× nhờ encoder chỉ thấy 25% token. Hai bài học cho phần còn lại của tầng:

| | MAE (pixel prediction) | JEPA-style (latent prediction, mục sau) |
|---|---|---|
| Mục tiêu dự đoán | pixel của patch bị che | **latent** của patch bị che |
| Có decoder | có (nhẹ, vứt sau pretrain) | **không** decoder ảnh |
| Tín hiệu học từ | reconstruction error ở pixel | predictive error trong embedding space |
| Capacity tiêu vào | gồm cả texture / chi tiết tần số cao | chỉ phần *dự đoán được* về mặt semantic |
| Đánh giá tốt nhất qua | fine-tuning (linear probe yếu hơn) | linear probe mạnh hơn |

Quan sát then chốt: MAE phải dành sức mạnh model để dựng lại *texture và chi tiết tần số cao* — phần lớn vô nghĩa cho quyết định downstream. Linear-probe của MAE yếu hơn fine-tuning đáng kể, gợi ý rằng biểu diễn của nó *chưa* tách bạch semantic một cách tuyến tính như các phương pháp latent. Chính khoảng cách này là động lực cho **JEPA (mục 6)** và **Tại sao latent prediction tốt hơn pixel prediction (mục cuối tầng)**.

---

## **4. Giới hạn / Khi nào thất bại**

**Mục tiêu pixel ép học chi tiết thừa.** Reconstruction buộc model mô hình hóa mọi pixel, kể cả texture/noise không liên quan semantics — lãng phí capacity, đúng phê phán mà [value equivalence (MuZero)](../../07-latent-planning/research/08-value-equivalence-muzero.md) nêu cho world model.

**Linear separability yếu hơn.** Biểu diễn MAE cần fine-tuning để toả sáng; trên linear probe nó thua các phương pháp contrastive/self-distillation — dấu hiệu rằng tín hiệu pixel không sắp xếp latent theo trục semantic.

**Phụ thuộc mask ratio và patch size.** Tỉ lệ che, kích thước patch, độ sâu decoder đều là hyperparameter nhạy; quá thấp thì bài toán tầm thường, quá cao thì mất ngữ cảnh để suy luận.

**Inductive bias kém cho dữ liệu phi-ảnh.** MAE khai thác dư thừa không gian của ảnh tự nhiên; với modal ít dư thừa (text, tín hiệu đã nén) chiến lược che-rồi-dựng-pixel kém hiệu quả.

**Reconstruction ≠ understanding.** Dựng lại pixel hoàn hảo không đảm bảo biểu diễn nắm được yếu tố sinh dữ liệu; một model có thể "vẽ lại" tốt mà vẫn rối các yếu tố biến thiên — chính lý do tầng 8 chuyển sang predict trong latent.

---

## **5. Liên hệ với Latent-Anything**

MAE là **baseline đối chứng** quan trọng: nó định nghĩa rõ thế nào là *predict-then-decode* để các mục sau đo bằng cùng thước. Với `ModelAdapter`, MAE là loại model `decodable=True` có decoder reconstruction:

```python
class MAEAdapter(Protocol):
    def encode_visible(self, patches: np.ndarray, mask: np.ndarray) -> np.ndarray: ...  # f_enc trên 25% patch
    def decode(self, z: np.ndarray, mask: np.ndarray) -> np.ndarray: ...                # f_dec → pixel (chỉ pretrain)
    reconstruction_target: str  # "pixel" — đối lập với "latent" của JEPA
```

- **Layer A — Introspection**: vì MAE *có* decoder, Layer A audit được trực tiếp — decode patch bị che và so với ground truth, trực quan hóa cái model "tưởng tượng". Đây là tiện ích mà model decoder-free (JEPA, MuZero) không có; MAE là ca dễ nhất để soi.
- **Layer B — Manipulation**: che/điền là một phép biến đổi latent có cấu trúc — chọn patch nào để mask, nội suy latent của vùng bị che, là một dạng manipulation định hướng được.
- **Layer C — Runtime**: kiến trúc bất đối xứng (encoder chỉ chạy trên patch nhìn thấy) là một mẫu *runtime efficiency* — bỏ token để giảm chi phí — mà Layer C có thể tổng quát hóa cho mọi sparse inference.

MAE khép phần "predict ở pixel" và mở đường cho câu hỏi trung tâm của tầng: nếu bỏ decoder và predict thẳng trong latent, *cái gì hỏng*? Câu trả lời đầu tiên là **representation collapse (mục tiếp theo)**.

---

## Liên quan

- [Value Equivalence (MuZero)](../../07-latent-planning/research/08-value-equivalence-muzero.md) — cùng phê phán: pixel reconstruction tiêu capacity cho chi tiết vô nghĩa với mục tiêu.
- [Autoencoder](../../02-representation-learning/research/02-autoencoder.md) — MAE là một denoising autoencoder với "nhiễu" = masking tỉ lệ cao.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) — pixel target giữ lại thông tin thừa; latent prediction nén mạnh hơn về phía semantic.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — decoder-free audit, trade-off giữ vs bỏ decoder.

## Tham khảo

- K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, *Masked Autoencoders Are Scalable Vision Learners* (CVPR 2022, arXiv:2111.06377).
- J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (NAACL 2019, arXiv:1810.04805).
- A. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT, ICLR 2021, arXiv:2010.11929).
- P. Vincent, H. Larochelle, Y. Bengio, P.-A. Manzagol, *Extracting and Composing Robust Features with Denoising Autoencoders* (ICML 2008).
