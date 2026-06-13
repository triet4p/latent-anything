# Stop-gradient và Kiến trúc Bất đối xứng

> **TL;DR.** BYOL và SimSiam tránh [collapse](02-representation-collapse.md) **không cần negative, không cần variance/covariance regularization tường minh** — chỉ bằng hai chi tiết kiến trúc: một **predictor** $h$ lệch ở một nhánh, và một **stop-gradient** chặn gradient ở nhánh kia. Loss là $-\cos\big(h(z_1),\,\mathrm{sg}(z_2)\big)$: nhánh target bị coi như *hằng số*. Hiệu ứng: nghiệm collapse trở thành **điểm cân bằng bất ổn định** của động lực học huấn luyện, nên GD không trôi vào đó. Caveat: cơ chế *ngầm* và mong manh — bỏ predictor thì collapse ngay (<1% accuracy), và lý thuyết giải thích vẫn chưa trọn vẹn.

Mục [representation collapse](02-representation-collapse.md) cho thấy huấn luyện invariance một mình sụp về một điểm, và liệt kê bốn họ cơ chế chống lại. VICReg dùng regularization *tường minh*. BYOL/SimSiam đi con đường ngược lại và gây sốc: chúng *vẫn* chỉ tối ưu invariance (kéo hai view lại gần), không hề có lực đẩy nào, vậy mà **không collapse**. Bí mật nằm hoàn toàn ở *cách bố trí gradient*, không ở hàm mất mát. Đây là cơ chế tinh tế nhất của tầng 8 — và là nền móng trực tiếp cho JEPA.

---

## **1. Trực giác / Định nghĩa**

Kiến trúc Siamese chuẩn: hai view $x_1, x_2$ qua *cùng* encoder $f$ ra $z_1, z_2$, ép $z_1\approx z_2$. Đối xứng hoàn toàn này collapse. SimSiam phá đối xứng bằng hai chi tiết:

- **Predictor bất đối xứng** $h$: chỉ một nhánh đi qua thêm một MLP nhỏ $h$, nhánh kia thì không. Hai nhánh không còn cùng hàm.
- **Stop-gradient** $\mathrm{sg}(\cdot)$: nhánh không-predictor bị *đóng băng* khi tính gradient — nó là **mục tiêu cố định** mà nhánh predictor chạy theo, không phải biến được tối ưu đồng thời.

Trực giác: thay vì "kéo hai điểm động về nhau" (dễ cùng nhau trôi về 0), ta biến nó thành "**một nhánh đuổi theo một mục tiêu tạm coi là đứng yên**". Mục tiêu đứng yên đó mang thông tin của input ở bước hiện tại; nhánh kia phải *dự đoán* nó. Một predictor không thể map mọi thứ về hằng số mà vẫn dự đoán đúng các target khác nhau — và stop-gradient ngăn "gian lận" bằng cách kéo luôn target về hằng số.

---

## **2. Cơ chế: SimSiam**

Hàm mất mát là cosine âm, **đối xứng hoá** qua hai view:

$$
\mathcal{L} = \tfrac{1}{2}\,D\big(h(z_1),\,\mathrm{sg}(z_2)\big) + \tfrac{1}{2}\,D\big(h(z_2),\,\mathrm{sg}(z_1)\big),\qquad D(p,t) = -\,\frac{p}{\lVert p\rVert_2}\cdot\frac{t}{\lVert t\rVert_2}.
$$

Trong đó $z_i=f(x_i)$ là embedding, $h$ là predictor MLP, $\mathrm{sg}$ là stop-gradient (forward giữ giá trị, backward cho gradient $=0$), và $D$ là khoảng cách cosine âm. Gradient *chỉ* chảy qua nhánh $h(z_i)$; nhánh $\mathrm{sg}(z_j)$ đóng vai mục tiêu cố định. Cập nhật encoder ở mỗi view dùng view kia làm đích bất biến tạm thời.

### Hai mảnh đều thiết yếu

Thí nghiệm cắt bỏ của Chen & He rất dứt khoát:

- **Bỏ stop-gradient** → collapse ngay; loss tụt xuống $-1$ (cosine cực đại) tức thì vì cả hai nhánh cùng đi về một hằng số.
- **Bỏ predictor $h$** (đặt $h=\text{Id}$) → collapse, top-1 accuracy $<1\%$ trên ImageNet.

Cần *cả hai*: stop-gradient biến bài toán thành đuổi-mục-tiêu-cố-định, predictor cho nhánh đuổi đủ tự do để khớp target mà không cần encoder tự bẹp lại.

### Diễn giải kiểu EM / xen kẽ

Chen & He lập luận stop-gradient khiến SimSiam giống một bài toán tối ưu **xen kẽ hai biến** (như EM hay k-means): một biến là tham số mạng, biến kia là một tập "vector đại diện" ngầm cho mỗi ảnh. Mỗi bước, target $\mathrm{sg}(z)$ đóng băng đóng vai trò "gán cụm" cố định, còn predictor + encoder cập nhật để khớp. Chính sự *xen kẽ* này — không tối ưu đồng thời cả hai nhánh — là thứ tránh nghiệm tầm thường. (Cách giải thích "predictor xấp xỉ kỳ vọng trên augmentation" mà họ nêu thêm về sau bị chỉ ra là chưa chặt.)

### Vì sao collapse trở nên bất ổn định

Tian et al. (2021) phân tích động lực học tuyến tính hoá và cho kết quả sạch hơn: với predictor $h$ và stop-gradient, **nghiệm collapse là điểm cân bằng bất ổn định**. Ma trận predictor học cách *căn chỉnh eigenbasis* với ma trận tương quan của feature; các chiều có phương sai dương được khuếch đại, còn hướng collapse (đưa mọi thứ về 0) có eigenvalue đẩy hệ *ra xa* khỏi nó. Họ dùng đúng insight này để đặt predictor ở **dạng đóng** (DirectPred) từ correlation matrix của feature — bỏ qua việc học $h$ bằng gradient mà vẫn không collapse, xác nhận rằng *vai trò* của predictor là cái mấu chốt, không phải cách học nó.

---

## **3. BYOL vs SimSiam**

Hai phương pháp cùng ý tưởng, khác ở chỗ "mục tiêu cố định" được tạo thế nào:

| | BYOL | SimSiam |
|---|---|---|
| Target encoder | mạng **EMA** (momentum) riêng | **cùng** encoder, chỉ thêm stop-gradient |
| Stop-gradient | có (trên nhánh target) | có (trên nhánh target) |
| Predictor $h$ | có | có |
| Negative pairs | không | không |
| Momentum | cần (ban đầu coi là thiết yếu) | **không** — SimSiam cho thấy momentum không bắt buộc |
| Thông điệp | bootstrap từ một bản sao chậm của chính mình | stop-gradient một mình đã đủ chống collapse |

Đóng góp khái niệm của SimSiam là **tách biến**: nó chứng minh EMA/momentum của BYOL *không* phải nguồn gốc của việc tránh collapse — bỏ momentum, để target = stop-grad của chính mạng, vẫn chạy. Momentum (xem **EMA target encoder (mục 5)**) là một *cải thiện ổn định/chất lượng*, không phải điều kiện sống còn. Lực chống collapse thật sự = predictor + stop-gradient.

---

## **4. Giới hạn / Khi nào thất bại**

**Mong manh với cấu hình.** Tránh collapse ở đây là một "implicit bias" của kiến trúc, không có sàn toán học như VICReg. Đổi batchnorm trong predictor/projector, learning rate, hay warmup sai có thể đưa hệ trở lại collapse — nó hoạt động cho tới khi không.

**Phụ thuộc tuyệt đối vào predictor.** $h=\text{Id}$ là collapse tức thì; cả phương pháp dựa vào một MLP mà vai trò chính xác từng bị giải thích sai. Một thành phần thiết yếu mà cộng đồng còn tranh luận về *vì sao* nó cần — rủi ro cho ai muốn dựng lại từ nguyên lý.

**Lý thuyết chưa trọn.** Có nhiều khung giải thích (EM xen kẽ, dynamics eigenspace, "DirectPred", cả lập luận coi stop-grad đóng vai negative ngầm) nhưng chưa cái nào khép kín hoàn toàn — khác với contrastive (InfoNCE có cận mutual information rõ ràng).

**Không có lực đẩy tường minh để audit.** Vì không có số hạng variance/covariance, ta không có một đại lượng loss để *theo dõi* sức khoẻ chống collapse; phải đo gián tiếp bằng [effective dimension](02-representation-collapse.md) của embedding.

**Vẫn cần augmentation mạnh, phù hợp.** Như mọi joint-embedding, chất lượng phụ thuộc augmentation định nghĩa "cái gì là nuisance"; lệch domain (ngoài ảnh) thì phải thiết kế lại.

---

## **5. Liên hệ với Latent-Anything**

Stop-gradient + predictor là một *mẫu kiến trúc* mà Layer C cần hỗ trợ như công dân hạng nhất, vì JEPA (mục 6–8) xây thẳng trên nó. Điểm tinh tế: bất đối xứng nằm ở **đồ thị tính gradient**, không ở giá trị forward — adapter phải biểu diễn được điều đó.

```python
def simsiam_step(x1, x2, f, h):
    z1, z2 = f(x1), f(x2)                 # shared encoder
    p1, p2 = h(z1), h(z2)                 # predictor on one branch
    # target branch is detached: gradient does NOT flow through z2 / z1 here
    loss = 0.5 * neg_cos(p1, stop_grad(z2)) + 0.5 * neg_cos(p2, stop_grad(z1))
    return loss                           # no negatives, no variance term
```

- **Layer A — Introspection**: vì không có loss-term báo sức khoẻ, Layer A phải *chủ động* giám sát collapse bằng effective dimension / phổ covariance trong lúc train — đây đúng là use case mà diagnostic ở mục trước phục vụ. Có thể trực quan hoá eigenspace của predictor để xác nhận nó đang căn chỉnh với correlation của feature (dấu hiệu "đang chống collapse đúng cách").
- **Layer B — Manipulation**: stop-gradient là một *biến đổi trên đồ thị tính toán* (đóng băng một nhánh) — một primitive Layer B nên lộ ra, vì nó định hình cách tín hiệu học lan truyền khi can thiệp latent.
- **Layer C — Runtime**: phải hỗ trợ đồ thị bất đối xứng (predictor một nhánh, detach nhánh kia) và tuỳ chọn EMA target; đây là khung chạy chung cho BYOL/SimSiam/DINO/JEPA, khác hẳn pipeline đối xứng của contrastive.

Mục này khép "cơ chế chống collapse bằng kiến trúc". Hai con đường còn lại — đẩy negatives ra xa (**contrastive, mục 4**) và làm target encoder chậm bằng **EMA (mục 5)** — bổ sung góc nhìn trước khi tất cả hội tụ ở **JEPA (mục 6)**: predict latent của target từ latent của context, dùng đúng stop-gradient + predictor + EMA học ở đây.

---

## Liên quan

- [Representation Collapse](02-representation-collapse.md) — vấn đề mà stop-gradient giải; effective dimension là cách audit nó ở đây.
- [Masked Autoencoder (MAE)](01-masked-autoencoder-mae.md) — con đường khác: neo bằng pixel reconstruction thay vì kiến trúc bất đối xứng.
- [Policy Gradient trên Imagined Trajectory (Dreamer)](../../07-latent-planning/research/07-policy-gradient-imagined-dreamer.md) — stop-gradient cũng là công cụ trung tâm khi tách actor/critic và chặn gradient qua target value.
- [RSSM — Dreamer](../../06-latent-temporal/research/04-rssm-recurrent-state-space-model.md) — target đóng băng / detach trong huấn luyện latent dynamics.

## Tham khảo

- J.-B. Grill et al., *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning* (BYOL, NeurIPS 2020, arXiv:2006.07733).
- X. Chen, K. He, *Exploring Simple Siamese Representation Learning* (SimSiam, CVPR 2021, arXiv:2011.10566).
- Y. Tian, X. Chen, S. Ganguli, *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs* (DirectPred, ICML 2021, arXiv:2102.06810).
- C. Zhang, K. Zhang, C. Zhang, T. X. Pham, C. D. Yoo, I. S. Kweon, *How Does SimSiam Avoid Collapse Without Negative Samples?* (ICLR 2022, arXiv:2203.16262).
