# Representation Collapse

> **TL;DR.** Bỏ decoder và huấn luyện encoder để hai view của cùng một input có latent *giống nhau* — $\min \lVert f(x)-f(x')\rVert^2$ — có một nghiệm tầm thường: **mọi input map về cùng một điểm** ($f\equiv \text{const}$, loss $=0$). Đây là **representation collapse**. Hai dạng: *complete collapse* (output hằng số) và *dimensional collapse* (embedding chỉ trải trên một không gian con thấp chiều, một số phương sai/singular value tụt về 0). Caveat: collapse là *thất bại mặc định*, không phải ngoại lệ — mọi phương pháp predict-trong-latent của tầng 8 về bản chất là một *cơ chế chống collapse* khác nhau (negatives, stop-gradient, EMA, variance/covariance regularization).

[MAE](01-masked-autoencoder-mae.md) tránh được collapse "miễn phí" vì nó có **decoder + pixel target**: nếu encoder vứt hết thông tin, decoder không thể dựng lại pixel, loss tăng. Pixel là *mỏ neo thông tin*. Tầng 8 muốn bỏ decoder để reasoning thuần latent — nhưng vừa bỏ mỏ neo đó, bài toán "khớp latent của hai view" lập tức có lối thoát tầm thường. Hiểu collapse là hiểu *vì sao* phần còn lại của tầng tồn tại.

---

## **1. Trực giác / Định nghĩa**

Joint-embedding: lấy hai view $x, x'$ của cùng một ảnh (qua augmentation), encode thành $z=f(x), z'=f(x')$, và ép chúng *bất biến* — gần nhau trong latent. Mục tiêu thuần invariance:

$$
\mathcal{L}_{\text{inv}} = \mathbb{E}_{x}\big\lVert f(x) - f(x') \big\rVert_2^2.
$$

Trong đó $f$ là encoder, $x'$ là một augmentation của $x$. Loss này đo "hai view có cùng biểu diễn không". Vấn đề: nó **không có sàn thông tin**. Nghiệm tối ưu toàn cục là $f(x)=c$ với mọi $x$ (một hằng số $c$ bất kỳ) → loss $=0$ tuyệt đối. Encoder học cách *quên sạch* input để thoả mãn invariance một cách rẻ nhất. Đó là collapse.

Đối chiếu: autoencoder/MAE thêm số hạng *reconstruction* buộc $z$ giữ đủ thông tin để dựng lại $x$ — sàn thông tin đó chặn collapse. Contrastive thêm số hạng *đẩy* các ảnh khác nhau ra xa — cũng là một sàn. Invariance một mình thì không.

---

## **2. Hai dạng collapse**

| | Complete collapse | Dimensional collapse |
|---|---|---|
| Hiện tượng | $f(x)\approx c$ cho mọi $x$ | embedding trải trên không gian con $k' < k$ chiều |
| Covariance $\mathrm{Cov}(z)$ | $\approx 0$ (mọi phương sai $\to 0$) | hạng thấp: vài singular value $\to 0$ |
| Dễ phát hiện? | dễ (variance gần 0) | **khó** — nhìn qua thì latent vẫn "đa dạng" |
| Hệ quả | vô dụng hoàn toàn | mất nhiều chiều hữu ích; downstream yếu |

**Dimensional collapse** (Jing et al., 2022) tinh vi hơn nhiều: embedding không sụp về một điểm mà về một *mặt phẳng con* — ma trận hiệp phương sai $\mathrm{Cov}(z)$ trở nên **hạng thấp**, phổ singular value có một đuôi tụt về 0. Latent vẫn "động đậy" nên qua mắt thường trông ổn, nhưng phần lớn chiều biểu diễn chết. Đáng chú ý: Jing et al. chứng minh dimensional collapse xảy ra *cả* trong contrastive learning, do tương tác giữa strong augmentation và over-parameterization, chứ không riêng non-contrastive.

### Vì sao đo bằng phổ

Gọi $\sigma_1\ge\dots\ge\sigma_k$ là singular value của $\mathrm{Cov}(z)$. *Effective rank* / *effective dimension* tóm tắt "thực sự dùng bao nhiêu chiều":

$$
d_{\text{eff}} = \exp\!\Big(-\sum_i p_i \log p_i\Big),\qquad p_i = \frac{\sigma_i}{\sum_j \sigma_j}.
$$

Trong đó $p_i$ là tỉ trọng phương sai của chiều $i$, và $d_{\text{eff}}$ là entropy mũ của phổ — bằng $k$ khi phổ phẳng (dùng hết chiều), tụt về 1 khi một chiều áp đảo (collapse). Đây là *metric chẩn đoán* trung tâm: theo dõi $d_{\text{eff}}$ trong lúc train là cách bắt dimensional collapse trước khi nó phá hỏng downstream.

---

## **3. Bốn họ cơ chế chống collapse (bản đồ phần còn lại của tầng)**

Mọi phương pháp predict-trong-latent là một câu trả lời cho "lấy gì làm sàn thông tin thay decoder":

1. **Negative pairs (contrastive).** Đẩy embedding của *ảnh khác nhau* ra xa, tạo lực căng chống lại lực kéo invariance. InfoNCE — xem **Contrastive learning (mục 4)**.
2. **Asymmetry + stop-gradient.** Một nhánh predictor + chặn gradient ở nhánh target khiến nghiệm hằng số *không còn là điểm cân bằng* của động lực học huấn luyện — BYOL, SimSiam, xem **Stop-gradient (mục 3)**.
3. **EMA target encoder.** Target encoder cập nhật chậm bằng trung bình trượt, phá tính đối xứng đủ để tránh collapse và ổn định hoá — DINO, JEPA, xem **EMA target encoder (mục 5)**.
4. **Variance / covariance regularization.** Ép tường minh: mỗi chiều embedding phải có phương sai $\ge$ ngưỡng (chống complete collapse) và các chiều phải decorrelated (chống dimensional collapse). VICReg (variance–invariance–covariance), Barlow Twins (soft-whitening cross-correlation về identity), W-MSE (whitening).

VICReg cho công thức tường minh nhất — đáng viết ra vì nó *là* định nghĩa toán học của "không collapse":

$$
\mathcal{L}_{\text{VICReg}} = \lambda\,\underbrace{\mathcal{L}_{\text{inv}}}_{\text{kéo gần}} + \mu\sum_j \underbrace{\max(0,\,\gamma - \mathrm{std}(z_j))}_{\text{variance: chống shrink}} + \nu\sum_{i\ne j}\underbrace{\mathrm{Cov}(z)_{ij}^2}_{\text{covariance: chống redundancy}}.
$$

Trong đó số hạng variance giữ độ lệch chuẩn của *từng chiều* $z_j$ trên ngưỡng $\gamma$ (chặn complete collapse), số hạng covariance đẩy các phần tử ngoài đường chéo của hiệp phương sai về 0 (decorrelate, chặn dimensional collapse), và $\mathcal{L}_{\text{inv}}$ vẫn kéo hai view lại gần. Ba lực này là phiên bản tường minh của thứ mà BYOL/SimSiam/DINO đạt được *ngầm* qua kiến trúc.

---

## **4. Giới hạn / Khi nào thất bại**

**Dimensional collapse khó phát hiện.** Complete collapse lộ ngay (variance $\approx 0$); dimensional collapse ẩn — accuracy training có thể vẫn ổn trong khi $d_{\text{eff}}$ đang rơi. Không đo phổ thì không thấy.

**Chống collapse ≠ biểu diễn tốt.** Tránh collapse là *điều kiện cần*, không đủ. Một encoder có phổ phẳng hoàn hảo vẫn có thể mã hoá toàn nuisance (augmentation invariance bị lạm dụng) mà bỏ semantic — "anti-collapse" và "informative" là hai trục khác nhau.

**Regularization explicit cần tuning.** Ngưỡng variance $\gamma$, trọng số $\mu,\nu$ của VICReg, hay batch normalization trong BYOL đều nhạy; đặt sai thì hoặc vẫn collapse hoặc làm hỏng invariance.

**Phụ thuộc augmentation.** Cả họ joint-embedding giả định augmentation "giữ semantic, đổi nuisance". Augmentation quá mạnh có thể xoá chính tín hiệu cần giữ, đẩy về collapse một cách hợp lệ-trên-loss; quá yếu thì task tầm thường.

**Cơ chế ngầm khó diễn giải.** BYOL/SimSiam tránh collapse nhờ "implicit bias" của kiến trúc mà bản thân lý do còn thiếu giải thích sạch — một rủi ro vận hành: nó hoạt động cho tới khi đổi một chi tiết (bỏ batchnorm, đổi lr) thì collapse trở lại.

---

## **5. Liên hệ với Latent-Anything**

Collapse là một **chỉ số sức khoẻ của latent space** — đúng loại thứ Latent-Anything coi là first-class. Layer A nên đo nó như một diagnostic chuẩn cho *bất kỳ* embedding nào nạp vào, không chỉ model JEPA:

```python
def collapse_diagnostics(Z: np.ndarray) -> dict[str, float]:
    # Z: (n_samples, dim) embedding matrix
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / (len(Z) - 1)
    sv = np.linalg.svdvals(cov)
    p = sv / sv.sum()
    eff_dim = float(np.exp(-(p * np.log(p + 1e-12)).sum()))   # effective dimension
    return {
        "per_dim_std_min": float(Zc.std(axis=0).min()),       # ~0 => complete collapse
        "effective_dim": eff_dim,                              # << dim => dimensional collapse
        "rank_ratio": eff_dim / Z.shape[1],
    }
```

- **Layer A — Introspection**: theo dõi `effective_dim` / phổ singular value như health metric; cảnh báo dimensional collapse mà mắt thường không thấy; so phổ trước/sau một can thiệp.
- **Layer B — Manipulation**: khi chỉnh sửa latent, phải kiểm không vô tình đẩy phân phối về subspace thấp chiều; whitening/decorrelation là phép biến đổi Layer B có thể áp để "mở lại" chiều chết.
- **Layer C — Runtime**: chọn cơ chế chống collapse (contrastive cần batch lớn nhiều negatives; non-contrastive cần EMA/stop-grad) là một quyết định runtime với chi phí khác nhau — Layer C lộ trade-off đó.

Collapse định khung cả tầng: từ đây mỗi mục là một *cách neo thông tin khi không có decoder*. Mục tiếp theo — **stop-gradient và kiến trúc bất đối xứng (mục 3)** — là cơ chế tinh tế nhất: chống collapse *không* bằng negatives, *không* bằng regularization tường minh, mà chỉ bằng một dòng `stop_gradient`.

---

## Liên quan

- [Masked Autoencoder (MAE)](01-masked-autoencoder-mae.md) — pixel/decoder làm mỏ neo thông tin; bỏ nó đi là nguyên nhân gốc của collapse.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) — collapse là cực đoan của nén: vứt *toàn bộ* thông tin về input.
- [Autoencoder](../../02-representation-learning/research/02-autoencoder.md) — reconstruction loss là sàn thông tin mà invariance-only thiếu.
- [Value Equivalence (MuZero)](../../07-latent-planning/research/08-value-equivalence-muzero.md) — decoder-free khác: neo bằng value/reward thay vì anti-collapse augmentation.

## Tham khảo

- L. Jing, P. Vincent, Y. LeCun, Y. Tian, *Understanding Dimensional Collapse in Contrastive Self-Supervised Learning* (ICLR 2022, arXiv:2110.09348).
- A. Bardes, J. Ponce, Y. LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning* (ICLR 2022, arXiv:2105.04906).
- J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny, *Barlow Twins: Self-Supervised Learning via Redundancy Reduction* (ICML 2021, arXiv:2103.03230).
- A. Ermolov, A. Siarohin, E. Sangineto, N. Sebe, *Whitening for Self-Supervised Representation Learning* (W-MSE, ICML 2021, arXiv:2007.06346).
- O. Roy, M. Vetterli, *The Effective Rank: A Measure of Effective Dimensionality* (EUSIPCO 2007).
