# Dictionary Learning

> **TL;DR.** Dictionary learning là *khung tổng quát* cho phân rã thưa: học một **dictionary** overcomplete $D$ (số atom $> $ số chiều) sao cho mỗi mẫu $x \approx D a$ với mã $a$ **thưa**. Bài toán $\min_{D,\{a_i\}} \sum_i \lVert x_i - Da_i\rVert^2$ với ràng buộc $\lVert a_i\rVert_0$ nhỏ, giải bằng **alternating minimization**: cố định $D$ → giải sparse coding (ISTA/OMP); cố định mã → cập nhật atom (MOD/K-SVD). [SAE](07-sparse-autoencoder.md) chính là một *instance neural, amortized* của khung này. Caveat chính: mục tiêu phi lồi → chỉ đảm bảo local optimum, dictionary không định danh duy nhất (permutation/scale), và sparse coding cổ điển chậm (lặp từng mẫu).

[Sparse autoencoder](07-sparse-autoencoder.md) giải superposition bằng một dictionary overcomplete + activation thưa, học qua một encoder neural. Dictionary learning là *lý thuyết tổng quát* đứng sau nó: cùng bài toán phân rã thưa, nhưng có cả một họ thuật toán cổ điển (K-SVD, MOD, ISTA, OMP) giải nó *chính xác từng mẫu* thay vì amortize bằng encoder. Hiểu khung này làm rõ SAE *là gì* (một lựa chọn thiết kế trong không gian lớn hơn), nó đánh đổi gì (tốc độ vs độ chính xác của mã thưa), và đặt nền cho mọi method decompose của Layer A.

---

## **1. Trực giác / Định nghĩa**

Hình dung một bộ "bảng chữ cái" hình ảnh: vài trăm mảnh vá (atom) nhỏ. Bất kỳ mảnh ảnh tự nhiên nào cũng dựng lại được bằng cách **chồng vài** atom (không phải tất cả). Dictionary learning học đồng thời hai thứ từ dữ liệu: (1) **bảng chữ cái** $D$ — tập atom; (2) cách viết mỗi mẫu bằng **ít chữ** nhất — mã thưa $a$. Olshausen & Field (1996) phát hiện khi ép sparse reconstruction trên ảnh tự nhiên, các atom học được *giống hệt receptive field của simple cell* trong vỏ não thị giác — bằng chứng đầu tiên rằng sparse coding là nguyên lý biểu diễn tự nhiên.

Ba thành phần định nghĩa, giống hệt [SAE](07-sparse-autoencoder.md) nhưng tổng quát hơn:

- **Overcomplete dictionary** $D \in \mathbb{R}^{d \times m}$, $m > d$: nhiều atom hơn số chiều — chỗ cho mọi feature chồng chất.
- **Sparse code** $a \in \mathbb{R}^m$: mỗi mẫu chỉ dùng vài atom ($\lVert a\rVert_0 \ll m$).
- **Reconstruction** $x \approx Da$: tổ hợp thưa phải tái tạo chính xác.

Khác biệt cốt lõi với SAE nằm ở *cách tính mã thưa* $a$: SAE dùng một encoder ($a = \text{ReLU}(W_\text{enc}x + b)$) cho ra mã trong *một forward* (amortized); dictionary learning cổ điển *giải một bài tối ưu* riêng cho mỗi $x$ (ISTA/OMP, lặp đến hội tụ) — chính xác hơn nhưng chậm hơn.

---

## **2. Cơ chế / Công thức**

### 2.1 Bài toán

Cho tập mẫu $\{x_i\}_{i=1}^N \subset \mathbb{R}^d$, tìm dictionary $D$ và mã thưa $\{a_i\}$:

$$ \min_{D,\,\{a_i\}} \ \sum_{i=1}^{N} \lVert x_i - D a_i \rVert_2^2 \quad \text{s.t.} \quad \lVert a_i \rVert_0 \le k,\ \ \lVert D_j \rVert_2 = 1 $$

trong đó $D_j$ là atom (cột) thứ $j$ chuẩn hóa đơn vị, $\lVert a_i\rVert_0$ là số phần tử khác 0 của mã (đếm atom dùng), $k$ là ngân sách sparsity. Mục tiêu: tái tạo tốt với *ít* atom mỗi mẫu. Đây là bài toán **phi lồi** (tích $Da$ của hai ẩn cùng tối ưu), nên giải bằng *xen kẽ* hai bài con dễ hơn.

### 2.2 Alternating minimization — hai bước xen kẽ

**Bước 1 — Sparse coding (cố định $D$, giải $a_i$).** Với $D$ cố định, mỗi mã độc lập:

$$ a_i = \arg\min_{a} \lVert x_i - Da \rVert_2^2 + \lambda \lVert a \rVert_1 $$

trong đó $\ell_0$ NP-hard được *nới lỏng* thành $\ell_1$ lồi (LASSO), $\lambda$ điều chỉnh sparsity. Giải bằng **ISTA** (iterative shrinkage-thresholding): lặp một bước gradient rồi soft-threshold,

$$ a \leftarrow \mathcal{S}_{\lambda\eta}\big(a + \eta\, D^\top (x - Da)\big), \qquad \mathcal{S}_\tau(z) = \text{sign}(z)\max(|z|-\tau, 0) $$

trong đó $\eta$ là step size, $\mathcal{S}_\tau$ là toán tử soft-threshold (kéo mọi giá trị về 0 một lượng $\tau$, ép phần nhỏ thành 0). ISTA hội tụ tốc độ $O(1/t)$; **FISTA** (Beck & Teboulle, 2009) tăng lên $O(1/t^2)$. Phương án thay thế là **OMP** (Orthogonal Matching Pursuit) — greedy chọn từng atom khớp nhất với residual.

**Bước 2 — Dictionary update (cố định mã, cập nhật $D$).** Với mọi mã cố định, cập nhật atom để giảm residual:

- **MOD** (Method of Optimal Directions): nghiệm bình phương tối thiểu đóng $D = X A^\top (A A^\top)^{-1}$, với $X$ là ma trận mẫu, $A$ là ma trận mã.
- **K-SVD** (Aharon et al., 2006): cập nhật *từng* atom $D_j$ một, bằng SVD bậc-1 của ma trận residual của các mẫu dùng atom đó — đồng thời tinh chỉnh cả atom *và* các hệ số tương ứng. Tổng quát hóa K-means (mỗi mẫu gán nhiều atom thay vì một cluster).

Lặp Bước 1 ↔ Bước 2 đến khi hội tụ. Mỗi bước giảm loss, nhưng vì bài toán phi lồi, điểm hội tụ chỉ là *local optimum*.

### 2.3 SAE = dictionary learning amortized

Đặt cạnh nhau, SAE là một *lựa chọn* trong khung này:

| | Dictionary learning cổ điển | SAE (neural) |
|---|---|---|
| Sparse coding | Giải tối ưu từng mẫu (ISTA/OMP), lặp | Một forward: $a=\text{ReLU}(W_\text{enc}x+b)$ (amortized) |
| Dictionary | Cập nhật MOD/K-SVD | $W_\text{dec}$, học bằng SGD |
| Sparsity | $\ell_0$ (OMP) hoặc $\ell_1$ (ISTA) | $\ell_1$ / TopK / JumpReLU |
| Tốc độ infer | Chậm (lặp mỗi mẫu) | Nhanh (1 forward) |
| Độ chính xác mã | Cao (tối ưu thật) | Có **amortization gap** (encoder xấp xỉ) |
| Quy mô | Khó scale tới LLM | Scale tốt (Mairal online DL, SAE) |

SAE đánh đổi *độ chính xác của mã thưa* lấy *tốc độ*: encoder dự đoán mã trong một bước thay vì giải tối ưu, để lại một **amortization gap** (mã SAE không tối ưu bằng ISTA). Đây là lý do SAE thắng ở quy mô lớn nhưng dictionary learning cổ điển vẫn là chuẩn vàng về chất lượng mã trên dữ liệu nhỏ.

---

## **3. Biến thể sparse coding: $\ell_0$ vs $\ell_1$**

| | $\ell_0$ (OMP, K-SVD) | $\ell_1$ (ISTA/FISTA, LASSO) |
|---|---|---|
| Ràng buộc | Đếm atom trực tiếp ($\lVert a\rVert_0 \le k$) | Nới lỏng lồi ($\lambda\lVert a\rVert_1$) |
| Bài toán | NP-hard, giải greedy xấp xỉ | Lồi, nghiệm toàn cục cho bước con |
| Sparsity | Kiểm soát $k$ trực tiếp | Gián tiếp qua $\lambda$ |
| Shrinkage | Không | Có (kéo biên độ nhỏ lại — như [SAE L1](07-sparse-autoencoder.md)) |
| Tương ứng SAE | TopK | L1 vanilla |

Đáng chú ý: cùng cặp đánh đổi xuất hiện ở [SAE](07-sparse-autoencoder.md) — TopK (giống $\ell_0$, kiểm soát $k$, không shrinkage) vs L1 (giống ISTA, shrinkage). Đây không phải trùng hợp: SAE *kế thừa* trực tiếp các lựa chọn của dictionary learning.

---

## **4. Giới hạn / Khi nào thất bại**

**Phi lồi → local minima.** Mục tiêu là *lồi theo từng ẩn* nhưng *không lồi theo cả hai*. Alternating minimization chỉ đảm bảo tới một local optimum; dictionary học được có thể khác dictionary "thật", và phụ thuộc khởi tạo. Định danh đúng dictionary chỉ có bảo đảm dưới điều kiện chặt (incoherence, sparsity đủ thấp).

**Không định danh duy nhất.** Dictionary chỉ xác định tới *hoán vị và scale* atom — không có "bộ atom chuẩn". Hai lần chạy (khởi tạo khác, $m$ khác) cho dictionary khác nhau. Đây chính là vấn đề "no canonical units" đã thấy ở [SAE](07-sparse-autoencoder.md), nhưng là tính chất *nền tảng* của cả lớp bài toán, không riêng SAE.

**Chi phí inference của sparse coding.** Giải $a_i$ cho mỗi mẫu là *lặp* (ISTA cần nhiều bước; OMP greedy) — đắt khi $N$ lớn. Đây là động lực gốc cho amortization: Gregor & LeCun (LISTA) và sau đó SAE thay bộ giải lặp bằng một mạng feed-forward.

**Siêu tham số tùy ý.** Kích thước dictionary $m$, ngân sách $k$ (hoặc $\lambda$) đều phải chọn tay; không có giá trị "đúng", và kết luận đổi theo lựa chọn — như Pareto reconstruction–sparsity của SAE.

**Khó scale.** K-SVD/MOD cổ điển dựng cho dữ liệu vừa (ảnh patch). Scale tới activation của model lớn cần online/stochastic variant (Mairal et al., 2009) hoặc amortization neural — lý do thực tiễn khiến interpretability hiện đại dùng SAE chứ không K-SVD.

---

## **5. Liên hệ với Latent-Anything**

Dictionary learning là *khung lý thuyết hợp nhất* cho mọi method decompose của Layer A — nó định nghĩa không gian thiết kế mà SAE là một điểm trong đó:

- **`LatentSpace.dictionary_decompose(x, method=...)`** → trừu tượng hóa việc phân rã thưa thành một interface chung, với backend chọn được: `"sae"` (amortized, mặc định cho quy mô lớn), `"ksvd"` / `"ista"` (chính xác từng mẫu, cho tập nhỏ hoặc khi cần mã tối ưu không amortization gap). Cùng một output `Dictionary{atoms, codes}`, khác cơ chế.
- **Chọn backend theo đánh đổi**: với một adapter có ít activation nhưng cần mã thưa *chính xác* (ví dụ kiểm chứng một feature), ISTA/K-SVD cho mã tốt hơn SAE; với activation khổng lồ, SAE là lựa chọn duy nhất khả thi. Khung này cho Latent-Anything *ngôn ngữ* để nói về đánh đổi đó thay vì cứng nhắc một thuật toán.
- **Mã thưa $a$ là tọa độ feature**: bất kể backend, $a$ là biểu diễn của $x$ theo các concept direction (atom). Atom dùng được cho [latent arithmetic](../../04-latent-computation/research/03-latent-arithmetic.md) và **steering vector (mục sau, cùng tầng)** y như feature SAE.
- **Cảnh báo non-identifiability ở tầng framework**: vì *cả lớp* bài toán không định danh duy nhất, `Dictionary` phải coi atom là *một* phân rã khả dĩ, cần [can thiệp](04-causal-intervention-vs-observational.md) xác nhận — đây là ràng buộc kế thừa từ lý thuyết, không phải khuyết tật của một implementation.

Quan trọng nhất: hiểu SAE là *dictionary learning amortized* giúp Latent-Anything không thần thánh hóa SAE — nó chỉ là một điểm vận hành (nhanh, scale được, có amortization gap) trong một khung cổ điển đã được nghiên cứu kỹ. Khung này cũng nối [superposition](06-superposition-hypothesis.md) (vì sao cần overcomplete + sparse) với mọi pipeline steering dựa trên feature ở Layer B.

---

## Liên quan

- [Sparse autoencoder — SAE (mục 07 — tầng này)](07-sparse-autoencoder.md) — instance neural, amortized của dictionary learning; cùng bài toán, khác cách tính mã thưa.
- [Superposition hypothesis (mục 06 — tầng này)](06-superposition-hypothesis.md) — lý do cần dictionary overcomplete + mã thưa: feature đông hơn số chiều, không trực giao.
- [Subspace projection](../../04-latent-computation/research/04-subspace-projection.md) — phân rã trực giao (PCA) là trường hợp *dense, trực giao*; dictionary learning thay bằng overcomplete + thưa.
- [Causal intervention vs observational (mục 04 — tầng này)](04-causal-intervention-vs-observational.md) — atom của dictionary là phân rã khả dĩ, cần can thiệp để xác nhận tính nhân quả.

## Tham khảo

- B. A. Olshausen, D. J. Field, *Emergence of Simple-Cell Receptive Field Properties by Learning a Sparse Code for Natural Images* (Nature, 1996). — Sparse coding gốc; atom học được giống receptive field vỏ não thị giác.
- M. Aharon, M. Elad, A. Bruckstein, *K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation* (IEEE Transactions on Signal Processing 54(11), 2006). — Cập nhật atom bằng SVD bậc-1; tổng quát hóa K-means.
- I. Daubechies, M. Defrise, C. De Mol, *An Iterative Thresholding Algorithm for Linear Inverse Problems with a Sparsity Constraint* (Communications on Pure and Applied Mathematics, 2004). — ISTA: soft-threshold + gradient cho sparse coding.
- A. Beck, M. Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems (FISTA)* (SIAM Journal on Imaging Sciences 2(1), 2009). — Tăng tốc ISTA lên $O(1/t^2)$.
- J. Mairal, F. Bach, J. Ponce, G. Sapiro, *Online Dictionary Learning for Sparse Coding* (ICML 2009, arXiv:0908.0050). — Online/stochastic dictionary learning, scale tới dữ liệu lớn.
- K. Gregor, Y. LeCun, *Learning Fast Approximations of Sparse Coding (LISTA)* (ICML 2010). — Thay bộ giải ISTA bằng mạng feed-forward — tiền thân ý tưởng amortization của SAE.
