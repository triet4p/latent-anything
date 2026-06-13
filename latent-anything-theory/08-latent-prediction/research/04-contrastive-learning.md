# Contrastive Learning

> **TL;DR.** Contrastive learning học biểu diễn bằng cách kéo hai view của cùng một mẫu lại gần và đẩy các mẫu khác ra xa; InfoNCE biến việc này thành bài toán nhận diện một positive giữa một tập negatives. Negative term tạo áp lực phân tán embedding nên chặn nghiệm collapse, nhưng chất lượng phụ thuộc mạnh vào augmentation, số lượng và độ đúng ngữ nghĩa của negatives. SimCLR dùng negatives trong batch lớn, còn MoCo dùng queue và momentum encoder để duy trì một dictionary lớn, nhất quán.

[Stop-gradient và kiến trúc bất đối xứng](03-stop-gradient-asymmetric.md) tránh collapse mà không cần lực đẩy tường minh. Contrastive learning đi theo con đường trực tiếp hơn: mỗi anchor phải nhận ra view tương ứng của nó giữa nhiều ứng viên. Nếu mọi input cùng được encode thành một vector, positive không thể nổi bật hơn negatives và loss không thể đạt giá trị thấp. Chính nhiệm vụ phân biệt này buộc latent vừa **bất biến với augmentation hợp lệ**, vừa **giữ đủ thông tin để tách các instance khác nhau**.

---

## **1. Trực giác: kéo positive, đẩy negative**

Cho một mẫu $x$, lấy hai phép biến đổi ngẫu nhiên $t,t'\sim\mathcal{T}$ để tạo:

$$
\tilde{x}_i=t(x),\qquad \tilde{x}_j=t'(x).
$$

Trong đó $\mathcal{T}$ là phân phối augmentation và $(\tilde{x}_i,\tilde{x}_j)$ là một **positive pair** vì chúng được giả định giữ cùng nội dung ngữ nghĩa. Các view sinh từ mẫu khác được dùng làm **negative pairs**.

Encoder $f_\theta$ và projection head $g_\phi$ tạo biểu diễn:

$$
h_i=f_\theta(\tilde{x}_i),\qquad z_i=g_\phi(h_i),\qquad \bar z_i=\frac{z_i}{\lVert z_i\rVert_2}.
$$

Trong đó $h_i$ là representation dùng cho downstream task, $z_i$ là vector dùng riêng để tính contrastive loss, và $\bar z_i$ là vector đã chuẩn hoá về hypersphere đơn vị. Việc tách $h$ khỏi $z$ quan trọng: SimCLR cho thấy một projection head phi tuyến giúp loss loại bỏ thông tin nuisance trong $z$ mà không buộc representation $h$ phải mất thông tin hữu ích.

Contrastive learning cần giải đồng thời hai yêu cầu:

- **Alignment**: positive pairs phải gần nhau dù augmentation khác nhau.
- **Uniformity**: toàn bộ embedding chuẩn hoá phải trải tương đối đều trên hypersphere, thay vì dồn vào một điểm hoặc một vùng nhỏ.

Wang và Isola chứng minh contrastive loss tiệm cận tối ưu đúng hai tính chất này. Alignment tạo invariance; uniformity là lực chống collapse và duy trì khả năng phân biệt.

### Augmentation định nghĩa ngữ nghĩa

Không có nhãn lớp, positive pair được tạo bằng một giả định thiết kế: các phép biến đổi trong $\mathcal{T}$ không làm đổi nội dung cần học. Vì vậy augmentation không chỉ là regularization mà chính là **định nghĩa operational của invariance**.

Với ảnh tự nhiên, random crop, resize, color jitter và blur thường giữ identity của object. Với medical image, màu hoặc orientation có thể mang ý nghĩa chẩn đoán; dùng cùng augmentation có thể xoá tín hiệu cần thiết. Với trajectory robot, time shift nhỏ có thể hợp lệ nhưng đảo thứ tự thời gian thường không hợp lệ. Một contrastive objective đúng công thức vẫn có thể học representation sai nếu positive pair được định nghĩa sai.

---

## **2. InfoNCE: phân loại positive giữa negatives**

Với anchor $i$, positive $j$, và tập ứng viên $\mathcal{A}(i)$ chứa positive cùng các negatives, normalized temperature-scaled cross-entropy loss là:

$$
\ell_{i,j}
=
-\log
\frac{\exp\left(\operatorname{sim}(\bar z_i,\bar z_j)/\tau\right)}
{\sum\limits_{k\in\mathcal{A}(i)}
\exp\left(\operatorname{sim}(\bar z_i,\bar z_k)/\tau\right)}.
$$

Trong đó $\operatorname{sim}(\bar z_i,\bar z_k)=\bar z_i^\top\bar z_k$ là cosine similarity của hai vector chuẩn hoá, $\tau>0$ là temperature, và $\mathcal{A}(i)$ loại anchor khỏi mẫu số. Loss là cross-entropy cho câu hỏi: "ứng viên nào là positive của anchor $i$?"

Gradient có hai thành phần trực giác:

- tăng logit của positive, kéo $\bar z_i$ về $\bar z_j$;
- giảm logit của negatives theo trọng số softmax, trong đó hard negatives có similarity cao nhận lực đẩy mạnh nhất.

Nếu mọi embedding collapse thành cùng một vector thì mọi logit bằng nhau. Với $K=|\mathcal{A}(i)|$ ứng viên:

$$
\ell_{i,j}=\log K.
$$

Trong đó $K$ là số ứng viên gồm một positive và $K-1$ negatives. Đây không phải nghiệm tối ưu vì model có thể giảm loss bằng cách làm positive nổi bật hơn; negatives vì thế tạo lực chống complete collapse một cách tường minh.

### Temperature điều khiển độ sắc của bài toán

Đạo hàm theo logit đã chia temperature có dạng:

$$
\frac{\partial \ell_{i,j}}{\partial s_{ik}}
=
\frac{1}{\tau}\left(p_{ik}-\mathbb{1}[k=j]\right),
\qquad
p_{ik}
=
\frac{\exp(s_{ik}/\tau)}
{\sum_{a\in\mathcal{A}(i)}\exp(s_{ia}/\tau)}.
$$

Trong đó $s_{ik}=\operatorname{sim}(\bar z_i,\bar z_k)$, $p_{ik}$ là xác suất softmax của ứng viên $k$, và $\mathbb{1}[k=j]$ bằng 1 cho positive. Temperature nhỏ làm phân phối sắc hơn và khuếch đại gradient, tập trung học vào hard negatives; quá nhỏ khiến optimization nhiễu và rất nhạy với false negatives.

### Liên hệ với mutual information

Contrastive Predictive Coding giới thiệu InfoNCE để ước lượng density ratio:

$$
f(x,c)\propto\frac{p(x\mid c)}{p(x)}.
$$

Trong đó $c$ là context, $x$ là target, và score dương $f(x,c)$ đo target có phù hợp với context hơn mức xuất hiện nền $p(x)$ hay không. Với $N$ ứng viên, CPC suy ra cận:

$$
I(X;C)\ge \log N-\mathcal{L}_{\mathrm{InfoNCE}}.
$$

Trong đó $I(X;C)$ là mutual information giữa target và context, $N-1$ là số negatives, và $\mathcal{L}_{\mathrm{InfoNCE}}$ là loss kỳ vọng. Tăng $N$ có thể làm cận chặt hơn, nhưng không nên diễn giải InfoNCE như một phép đo mutual information chính xác trong mọi chế độ; mục tiêu thực dụng của nó là học density ratio và một geometry phân biệt hữu ích.

---

## **3. SimCLR: negatives ngay trong minibatch**

SimCLR dùng minibatch gồm $B$ mẫu gốc, tạo hai view mỗi mẫu nên có $2B$ embeddings. Với mỗi anchor, view còn lại của cùng mẫu là positive và $2B-2$ view còn lại là negatives. Loss được tính theo cả hai hướng rồi lấy trung bình:

$$
\mathcal{L}_{\mathrm{SimCLR}}
=
\frac{1}{2B}
\sum_{i=1}^{2B}
\ell_{i,p(i)}.
$$

Trong đó $p(i)$ trả về index của view còn lại từ cùng mẫu gốc với anchor $i$. Mỗi embedding vì thế vừa làm anchor, vừa làm positive hoặc negative cho các anchor khác.

Pipeline tối thiểu:

```python
views_1, views_2 = augment(batch), augment(batch)
h_1, h_2 = encoder(views_1), encoder(views_2)
z_1, z_2 = projector(h_1), projector(h_2)
loss = symmetric_info_nce(z_1, z_2, temperature=0.1)
```

Các ablation chính của SimCLR cho thấy ba yếu tố không thể xem là chi tiết phụ:

- composition của augmentation quyết định pretext task;
- projection head phi tuyến cải thiện representation trước head;
- batch lớn và thời gian train dài cung cấp nhiều negatives hơn và cải thiện kết quả.

Ưu điểm là implementation đơn giản, không có state ngoài optimizer. Nhược điểm là số negatives gắn chặt với batch size; tăng dictionary đồng nghĩa tăng memory, communication và chi phí encoder.

---

## **4. MoCo: queue và momentum dictionary**

MoCo diễn giải contrastive learning như dictionary lookup. Query encoder $f_q$ tạo anchor, key encoder $f_k$ tạo positive key, còn một FIFO queue lưu keys từ các minibatch trước làm negatives. Nhờ đó kích thước dictionary $K$ không còn bị giới hạn bởi batch hiện tại.

Key encoder không được cập nhật trực tiếp bằng gradient từ loss. Thay vào đó:

$$
\theta_k \leftarrow m\theta_k+(1-m)\theta_q.
$$

Trong đó $\theta_q$ và $\theta_k$ lần lượt là tham số query encoder và key encoder, còn $m\in[0,1)$ là momentum coefficient thường gần 1. EMA làm key encoder thay đổi chậm, nên keys cũ trong queue vẫn tương đối nhất quán với keys mới.

Nếu cập nhật $f_k$ đồng thời với $f_q$, các vector trong queue được sinh bởi nhiều encoder khác nhau thay đổi nhanh; dictionary lớn nhưng mất tính nhất quán. Nếu đóng băng $f_k$ hoàn toàn, target trở nên lỗi thời. Momentum update cân bằng hai phía: **dictionary lớn** nhờ queue và **dictionary nhất quán** nhờ encoder chậm.

| Thuộc tính | SimCLR | MoCo |
|---|---|---|
| Nguồn negatives | các view trong minibatch hiện tại | FIFO queue từ nhiều minibatch |
| Kích thước dictionary | gắn với batch size | tách khỏi batch size |
| Target/key encoder | cùng encoder | encoder EMA riêng |
| State ngoài optimizer | không | queue + momentum encoder |
| Điểm mạnh | đơn giản, fully synchronous | nhiều negatives với batch nhỏ hơn |
| Rủi ro | memory/communication của batch lớn | stale keys và tuning momentum |

EMA trong MoCo phục vụ **tính nhất quán của negative dictionary**. Ở DINO/JEPA, EMA target encoder còn đóng vai trò tạo target ổn định không nhận gradient trực tiếp; cơ chế này được phân tích ở **EMA target encoder (mục 5)**.

---

## **5. Giới hạn / Khi nào thất bại**

### False negatives

Random negatives có thể cùng semantic class với anchor. InfoNCE vẫn đẩy chúng ra xa vì chỉ biết identity, tạo mâu thuẫn giữa instance discrimination và semantic grouping. Xác suất gặp false negative tăng khi batch hoặc queue lớn, khi dataset có ít class, hoặc khi nhiều mẫu gần như trùng nghĩa. Debiased Contrastive Learning hiệu chỉnh objective để giảm bias này, nhưng cần thêm giả định về class prior.

### Sampling và compute

Chất lượng phụ thuộc mạnh vào số lượng và độ khó của negatives. SimCLR cần batch lớn; MoCo giảm yêu cầu memory nhưng thêm queue, target encoder và độ trễ representation. Hard-negative mining quá mạnh có thể ưu tiên false negatives, còn negatives quá dễ cho gradient gần bằng 0.

### Shortcut từ augmentation

Nếu hai view chia sẻ artifact không liên quan đến semantics, model có thể nhận diện positive bằng shortcut. Ngược lại, augmentation quá mạnh có thể biến positive pair thành hai nội dung khác nhau. Objective không tự biết invariant nào là đúng.

### Instance discrimination có thể giữ quá nhiều chi tiết

Để phân biệt từng instance, encoder có động cơ giữ texture, background hoặc dấu hiệu nhận dạng riêng mà downstream task không cần. Projection head giúp cô lập một phần áp lực này, nhưng không loại bỏ hoàn toàn phụ thuộc vào data và augmentation.

### Geometry không tự động có semantics

Uniformity trên hypersphere chống collapse nhưng không đảm bảo từng direction có nghĩa hoặc factor được disentangle. Contrastive representation vẫn có thể anisotropic theo subspace, có dimensional collapse cục bộ, hoặc tổ chức neighbourhood không phù hợp downstream task. Cần audit bằng covariance spectrum, effective rank và probes thay vì chỉ nhìn training loss.

---

## **6. Liên hệ với Latent-Anything**

Contrastive learning là baseline quan trọng cho mọi pipeline "predict trong latent" vì nó cho một cơ chế chống collapse rõ ràng và đo được.

- **Layer A — Introspection**: theo dõi positive/negative similarity, alignment, uniformity, effective rank, nearest-neighbour purity và false-negative rate ước lượng. Loss thấp một mình không chứng minh geometry hữu ích.
- **Layer B — Manipulation**: augmentation policy và negative sampler là các phép biến đổi định nghĩa invariance. Chúng cần được lưu trong config để một latent space có thể tái lập và giải thích được.
- **Layer C — Runtime**: SimCLR cần all-gather embeddings giữa accelerator; MoCo cần queue nhất quán, atomic enqueue/dequeue và EMA update đúng thứ tự. Đây là state runtime thật, không chỉ là chi tiết model.
- **`LatentSpace` metadata**: nên ghi normalization, similarity metric, temperature, projection-head boundary và training objective. Cosine geometry trên unit sphere không tương đương Euclidean geometry của pre-projection representation.

Đối với world model, negative sampling còn khó hơn ảnh tĩnh: hai state gần nhau trong cùng trajectory có thể là false negatives dù khác timestep; hai state từ episode khác có thể cùng semantic state. Sampler cần hiểu trajectory, action và temporal neighbourhood. Đây là nơi [`Trajectory`](https://github.com/triet4p/latent-anything/blob/main/docs/ARCHITECTURE.md) ảnh hưởng trực tiếp đến định nghĩa contrastive task.

Contrastive learning giải collapse bằng lực đẩy bên ngoài. Các mục sau chuyển sang target encoder chậm và joint-embedding prediction, nơi model học dự đoán target latent mà không cần so với hàng nghìn negatives.

---

## Liên quan

- [Representation Collapse](02-representation-collapse.md) — giải thích complete collapse và dimensional collapse mà negative term cố ngăn.
- [Stop-gradient và Kiến trúc Bất đối xứng](03-stop-gradient-asymmetric.md) — cơ chế chống collapse không dùng negatives để đối chiếu trực tiếp.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) — augmentation và contrastive task quyết định thông tin nào được giữ hoặc loại bỏ.
- [Đẳng hướng và Bất đẳng hướng](../../03-geometry-structure/research/03-isotropy-anisotropy.md) — uniformity, hypersphere và các failure mode hình học của embedding.
- [Linear Probing](../../05-probing-intervention/research/01-linear-probing.md) — protocol phổ biến để đánh giá feature contrastive sau pretraining.

## Tham khảo

- A. van den Oord, Y. Li, O. Vinyals, *Representation Learning with Contrastive Predictive Coding* (arXiv 2018, arXiv:1807.03748).
- T. Chen, S. Kornblith, M. Norouzi, G. Hinton, *A Simple Framework for Contrastive Learning of Visual Representations* (ICML 2020, arXiv:2002.05709).
- K. He, H. Fan, Y. Wu, S. Xie, R. Girshick, *Momentum Contrast for Unsupervised Visual Representation Learning* (CVPR 2020, arXiv:1911.05722).
- T. Wang, P. Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere* (ICML 2020, arXiv:2005.10242).
- C.-Y. Chuang, J. Robinson, Y.-C. Lin, A. Torralba, S. Jegelka, *Debiased Contrastive Learning* (NeurIPS 2020, arXiv:2007.00224).
