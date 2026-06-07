# Latent Arithmetic — Điều kiện để phép toán trong latent space có nghĩa

> **TL;DR.** Latent arithmetic là các phép cộng/trừ vector trong latent space để thao tác ngữ nghĩa: $z_d \approx z_c + (z_a - z_b)$ — ví dụ kinh điển là `king − man + woman ≈ queen` trong word2vec. Phép toán chỉ có nghĩa khi latent space có *linear structure* (quan hệ ngữ nghĩa được mã hóa dạng additive offset) VÀ tất cả vector cùng *coordinate system* (cùng model, cùng layer, cùng training run). Vi phạm bất kỳ điều kiện nào dẫn đến kết quả vô nghĩa.

Latent arithmetic xuất phát từ quan sát đáng ngạc nhiên trong word2vec (Mikolov et al., 2013): vector biểu diễn từ không chỉ mang thông tin về nghĩa, mà còn mã hóa *quan hệ* giữa các khái niệm theo dạng hướng (*direction*) trong không gian. Nếu hướng "nghề vua → người thường" là $z_{\text{man}} - z_{\text{king}}$, thì phép trừ đó bảo toàn được hướng "giới tính nam → giới tính nữ" đủ để cộng thêm $z_{\text{woman}}$ cho ra $z_{\text{queen}}$. Điều này cho thấy latent space, trong những trường hợp lý tưởng, có cấu trúc tuyến tính đủ mạnh để dùng đại số vector như một ngôn ngữ thao tác ngữ nghĩa.

---

## **1. Trực giác / Định nghĩa**

Hình dung một biểu đồ tọa độ mà mỗi trục tương ứng với một nhân tố ngữ nghĩa: trục X là "giới tính", trục Y là "địa vị xã hội". Nếu model encode các nhân tố này tuyến tính và cộng hợp (*additively*), thì:

- `king` nằm ở (Nam, Cao), `man` nằm ở (Nam, Thấp), hiệu là (0, Cao) → vector "địa vị cao".
- Cộng thêm `woman` ở (Nữ, Thấp) → (Nữ, Cao) ≈ `queen`.

Trực giác này *đúng* khi hai điều kiện đồng thời thỏa:

1. **Linear structure**: model thực sự phân biệt hai nhân tố trên hai hướng tuyến tính, độc lập.
2. **Cùng coordinate system**: `king`, `man`, `woman` cùng được encode bởi cùng một model, cùng một layer — tức cùng không gian tọa độ. Nếu `king` từ model A và `woman` từ model B, hai vector này sống trong hai hệ tọa độ khác nhau; phép trừ sẽ trả về nhiễu.

**Định nghĩa chính thức:** Cho latent space $\mathbb{R}^d$ với mã hóa $\phi: X \to \mathbb{R}^d$. Latent arithmetic *có nghĩa* khi tồn tại hướng $\delta \in \mathbb{R}^d$ sao cho:

$$\phi(b) - \phi(a) \approx \phi(d) - \phi(c)$$

với mọi cặp $(a, b)$ và $(c, d)$ cùng mang quan hệ $A:B$. Khi đó, để "chuyển" $c$ theo quan hệ $A:B$, ta dùng:

$$\hat\phi(d) = \phi(c) + \underbrace{(\phi(b) - \phi(a))}_{\delta}$$

---

## **2. Cơ chế / Công thức**

### 2.1 Analogy arithmetic

Dạng cơ bản nhất — *A is to B as C is to D*:

$$z_d \approx z_c + (z_b - z_a)$$

trong đó $z_a, z_b$ là cặp ví dụ của quan hệ cần "học" (ví dụ: `man`, `king`), $z_c$ là điểm nguồn cần biến đổi (`woman`), và $z_d$ là kết quả dự kiến (`queen`). Phép tính cho ra một điểm trong latent space; kết quả cuối lấy nearest-neighbor trong tập vocabulary (với word2vec) hoặc decode trực tiếp (với generative model).

Để ổn định, DCGAN (Radford et al., 2016) dùng *trung bình* của nhiều vector thay vì một vector đơn:

$$\delta = \bar z_{\text{with}} - \bar z_{\text{without}}, \quad \bar z = \frac{1}{n}\sum_{i=1}^n z_i$$

trong đó $\bar z_{\text{with}}$ và $\bar z_{\text{without}}$ là trung bình của $n$ mẫu *có* và *không có* thuộc tính cần thao tác (ví dụ: kính, nụ cười). Trung bình hóa triệt tiêu các thành phần nhiễu không liên quan và làm nổi bật hướng ngữ nghĩa chính.

### 2.2 Steering vector

Thay vì tạo điểm mới từ analogy, *steering* thêm/bớt một thuộc tính vào điểm hiện tại:

$$z_{\text{new}} = z + \alpha \cdot \delta$$

trong đó $\delta = \bar z_{\text{with}} - \bar z_{\text{without}}$ là hướng khái niệm (*concept direction*), $\alpha \in \mathbb{R}$ là cường độ can thiệp (âm = loại bỏ thuộc tính, dương = thêm thuộc tính), và $z$ là latent ban đầu. Kỹ thuật này được phổ biến rộng rãi dưới tên *representation engineering* và *steering vector*.

### 2.3 Điều kiện để phép toán có nghĩa

Có ba điều kiện cần kiểm tra trước khi dùng latent arithmetic:

| Điều kiện | Ký hiệu toán học | Ý nghĩa thực tế |
|---|---|---|
| Cùng coordinate system | $\phi_A = \phi_B$ | Cùng model, cùng layer, cùng checkpoint |
| Linear structure | $\phi(b) - \phi(a) \approx \phi(d) - \phi(c)$ | Quan hệ encode thành hướng song song |
| Concept orthogonality | $\langle \delta_1, \delta_2 \rangle \approx 0$ | Hai concept không interfere khi cộng đồng thời |

**Điều kiện 1 — cùng coordinate system** là bắt buộc và không thể bỏ qua. Latent vector là tọa độ trong không gian trừu tượng được định nghĩa bởi từng model cụ thể. Cùng một ảnh encode bởi hai model khác nhau sẽ ra hai vector hoàn toàn khác nhau — không có căn cứ nào để cộng hai vector đó. Ngay cả cùng kiến trúc nhưng khác random seed cũng cho hệ tọa độ khác nhau (xem mục 4).

**Điều kiện 2 — linear structure** — là giả định mạnh nhất. Một model *có thể* encode concept giới tính theo hướng tuyến tính nhất quán, hoặc *có thể* encode nó theo cách phi tuyến hoặc phụ thuộc vào context. Word2vec đáp ứng điều kiện này tốt cho nhiều quan hệ cú pháp và ngữ nghĩa. Generative model thường đáp ứng một phần — một số thuộc tính encode tuyến tính, số khác không.

**Điều kiện 3 — orthogonality** ảnh hưởng đến độ chính xác khi thêm nhiều concept cùng lúc. Nếu hướng "nụ cười" và hướng "giới tính" không vuông góc, thêm nụ cười sẽ vô tình làm thay đổi một chút về giới tính.

---

## **3. Biến thể**

### 3.1 Direction-based manipulation

Thay vì dùng cặp ví dụ cụ thể, *direction-based* method tìm hướng bằng cách tối ưu trực tiếp:

$$\delta = \arg\max_{\|\delta\| = 1} \left[ \bar z_{\text{with}} - \bar z_{\text{without}} \right]^T \delta$$

Đây là *mean difference* normalized — đơn giản nhất và thường hiệu quả nhất. Các phương pháp phức tạp hơn dùng linear SVM để tìm separating hyperplane, hoặc PCA trên $\{z_{\text{with}}\} \cup \{z_{\text{without}}\}$ để lấy principal direction.

### 3.2 Concept negation

Xóa thuộc tính bằng cách project $z$ ra khỏi hướng concept:

$$z_{\text{no-concept}} = z - \frac{z \cdot \hat\delta}{\|\hat\delta\|^2} \hat\delta$$

trong đó $\hat\delta = \delta / \|\delta\|$ là hướng concept đơn vị. Phép toán này là *null-space projection* — giữ lại tất cả thông tin trong $z$ ngoại trừ thành phần dọc theo $\delta$.

### 3.3 Arithmetic trên tập mẫu (DCGAN style)

Dùng $n$ mẫu thay vì một mẫu duy nhất để ổn định hướng concept:

$$z_{\text{result}} = z_{\text{neutral}} + (\bar z_{\text{with A}} - \bar z_{\text{neutral A}}) + (\bar z_{\text{with B}} - \bar z_{\text{neutral B}})$$

cho phép stack nhiều thuộc tính — thêm kính VÀ thêm nụ cười — bằng cách cộng hai vector hướng vào cùng một latent trung tính.

---

## **4. Giới hạn / Khi nào thất bại**

**Cross-model và cross-training-run.** Latent vector từ hai model khác nhau không thể cộng trừ trực tiếp. Thậm chí cùng kiến trúc, cùng dữ liệu nhưng khác random seed cũng tạo ra hai hệ tọa độ không tương đương — vector "nam giới" của model A và "nam giới" của model B hướng về hai hướng hoàn toàn khác nhau trong $\mathbb{R}^d$. Moschella et al. (2022) chứng minh điều này và đề xuất *relative representations* — thay tọa độ tuyệt đối bằng tọa độ tương đối so với tập anchor — để enable zero-shot communication giữa các model.

**Entangled features.** Nếu model encode "giới tính" và "tuổi tác" dọc theo hướng tương quan cao (không orthogonal), thao tác giới tính sẽ kéo theo thay đổi tuổi tác — một hiện tượng gọi là *concept leakage*. Điều này phổ biến trong model thiếu explicit disentanglement training.

**Non-linear encoding.** Không phải mọi thuộc tính đều được encode tuyến tính. Một model có thể encode "phong cách họa sĩ" theo cách phi tuyến — không tồn tại một hướng đơn lẻ nào tương ứng với "Van Gogh", mà thay vào đó là một tập manifold cong trong latent space. Với những thuộc tính như vậy, arithmetic tuyến tính sẽ cho kết quả vô nghĩa.

**Nonlinear interaction giữa các concept.** Ngay khi từng concept được encode tuyến tính riêng lẻ, *tổ hợp* của chúng có thể không tuyến tính. Ví dụ: "cười" + "trẻ em" có thể không bằng tổng hai hướng riêng lẻ nếu model học rằng "trẻ em cười" là một pattern đặc biệt trong dữ liệu training.

**Magnitude không bất biến.** Trong không gian anisotropic, $\|\delta\|$ có thể lớn theo một số chiều hơn chiều khác. Dùng cùng một $\alpha$ để steering theo hai concept khác nhau sẽ cho hiệu ứng can thiệp không cân bằng — concept nào có $\|\delta\|$ lớn hơn sẽ dominate. Cần normalize $\delta$ (hoặc dùng Mahalanobis-aware arithmetic).

**Out-of-distribution sau arithmetic.** Sau khi cộng $\alpha \cdot \delta$ với $\alpha$ lớn, kết quả có thể rời khỏi vùng latent có mật độ cao — decoder sẽ decode về vùng nó chưa thấy lúc train, cho kết quả nhòa hoặc artifact. Đây cùng vấn đề như norm dip của lerp: arithmetic không tự động đảm bảo kết quả nằm trên manifold.

---

## **5. Liên hệ với Latent-Anything**

Latent arithmetic là **Layer B (manipulation)** quan trọng nhất sau interpolation. Nó là nền tảng cho:

- **Concept injection**: `LatentSpace.steer(z, concept_direction, alpha)` — thêm/bớt thuộc tính ngữ nghĩa vào một latent state.
- **Attribute transfer**: trừ đi style của $z_a$, cộng thêm style của $z_b$ — dùng trong style transfer và persona adaptation.
- **Trajectory manipulation**: thêm concept direction vào toàn bộ trajectory để "recolor" sequence mà không thay đổi nội dung cốt lõi.
- **Diagnostic tool**: kiểm tra xem latent space có linear structure không bằng cách đo cos-similarity giữa các $\delta$ trên nhiều cặp ví dụ.

Framework sẽ expose `LatentSpace.concept_direction(pos_examples, neg_examples) → ConceptDirection` để tính $\delta = \bar z_{\text{pos}} - \bar z_{\text{neg}}$, và `LatentSpace.arithmetic(z_a, z_b, z_c) → z_d` với kiểm tra tự động:

1. Cảnh báo nếu $z_a, z_b, z_c$ không có cùng metadata (model ID, layer index).
2. Cảnh báo nếu $\|\delta\|$ quá lớn so với spread của phân phối latent (out-of-distribution risk).
3. Cảnh báo nếu kết quả $z_d$ vượt ngoài $k\sigma$ của prior latent distribution.

Hai mục tiếp theo trong tầng 4 — **Subspace projection** và **Mahalanobis distance** — cung cấp công cụ để làm cho arithmetic chính xác hơn trong không gian anisotropic.

---

## Liên quan

- [Lerp (mục 01 — tầng này)](01-lerp.md) — baseline interpolation; latent arithmetic dùng cùng kiểm tra norm để phát hiện out-of-distribution.
- [Slerp (mục 02 — tầng này)](02-slerp.md) — khi latent có unit-norm, arithmetic thực hiện trên hypersphere.
- [Cấu trúc tuyến tính](../../03-geometry-structure/research/01-linear-structure.md) — lý giải vì sao một số latent space có linear structure và có thể dùng arithmetic.
- [Tính tách biệt biểu diễn (disentanglement)](../../03-geometry-structure/research/02-disentanglement.md) — latent disentangled giúp các concept direction orthogonal và arithmetic sạch hơn.
- [Đẳng hướng & Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — không gian anisotropic làm arithmetic bị lệch scale; cần Mahalanobis-aware normalization.

## Tham khảo

- T. Mikolov, K. Chen, G. Corrado, J. Dean, *Efficient Estimation of Word Representations in Vector Space* (ICLR Workshop 2013, arXiv:1301.3781). — Giới thiệu word2vec và phép tính analogy `king − man + woman = queen`.
- T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, J. Dean, *Distributed Representations of Words and Phrases and their Compositionality* (NeurIPS 2013, arXiv:1310.4546). — Phân tích sâu hơn về analogy task và skip-gram với negative sampling.
- A. Radford, L. Metz, S. Chintala, *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (ICLR 2016, arXiv:1511.06434). — Chứng minh latent arithmetic hoạt động trong GAN visual latent space; giới thiệu mean-vector technique để ổn định hướng concept.
- L. Moschella, V. Maiorca, M. Fumero, A. Norelli, F. Locatello, E. Rodolà, *Relative Representations Enable Zero-Shot Latent Space Communication* (ICLR 2023, arXiv:2209.15430). — Phân tích vấn đề coordinate system; đề xuất relative representation để enable cross-model arithmetic.
