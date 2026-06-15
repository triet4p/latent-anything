# Probing Classifiers — Survey (Belinkov, 2022)

> **TL;DR.** Probing = train một classifier (probe) dự đoán một thuộc tính từ biểu diễn *đóng băng*; accuracy cao → thuộc tính *decodable* trong biểu diễn. Survey của Belinkov (2022) cảnh báo ba cạm bẫy: (1) **probe complexity confound** — probe mạnh có thể tự học task thay vì *đọc* từ biểu diễn; (2) cần **selectivity / control task** để phân biệt "đọc cấu trúc" với "ghi nhớ"; (3) **decodable ≠ được dùng** — thông tin có trong latent không có nghĩa model *dùng* nó (cần phương pháp nhân quả). Các tiến bộ: control tasks, **MDL probing** (đo độ *dễ* trích, không chỉ accuracy), và **amnesic probing** (xóa thuộc tính rồi đo ảnh hưởng hành vi). Caveat: probing là chẩn đoán tương quan; muốn kết luận nhân quả phải can thiệp.

Tầng 5 dựng các *công cụ* probe ([linear](../../05-probing-intervention/research/01-linear-probing.md), [nonlinear](../../05-probing-intervention/research/02-nonlinear-probing.md), [TCAV](../../05-probing-intervention/research/03-tcav.md)). Mục này là *bản đồ phương pháp luận*: **khi nào dùng probe gì**, và cách tránh kết luận sai — đúng thứ Layer A cần để introspection đáng tin.

---

## 1. Probing là gì và hứa hẹn gì

Ý tưởng cơ bản: nếu một thuộc tính (part-of-speech, sentiment, tọa độ vật lý...) *được mã hóa* trong biểu diễn, thì một classifier đơn giản huấn luyện trên biểu diễn (đóng băng) sẽ dự đoán được nó. Accuracy probe = thước đo "thông tin này có decodable không".

Hứa hẹn: rẻ, đơn giản, cho phép phân tích **theo lớp** (thông tin xuất hiện ở lớp nào), và là cách trực tiếp trả lời "latent encode gì". Đây là nền của introspection có giám sát.

---

## 2. Ba cạm bẫy (shortcomings)

### (a) Probe complexity confound

Một probe *quá mạnh* (MLP sâu) có thể *tự học task* từ tín hiệu yếu, làm accuracy cao ngay cả khi thông tin không thực sự "có sẵn" trong biểu diễn theo cách hữu ích. Câu hỏi nhập nhằng: accuracy cao do *biểu diễn chứa thông tin* hay do *probe đủ mạnh để trích từ noise*? Đây là lý do [linear probe](../../05-probing-intervention/research/01-linear-probing.md) (probe yếu) được ưa cho câu hỏi "thông tin có *tuyến tính* không", còn [nonlinear probe](../../05-probing-intervention/research/02-nonlinear-probing.md) cho "thông tin có *tồn tại* không" (upper bound) — chọn độ phức tạp probe theo câu hỏi.

### (b) Selectivity và control task

Hewitt & Liang (2019): so accuracy probe trên *task thật* với trên một **control task** (cùng cấu trúc nhưng nhãn *ngẫu nhiên*). **Selectivity = acc(task thật) − acc(control)**. Nếu probe đạt accuracy cao cả trên nhãn ngẫu nhiên → nó đang *ghi nhớ*, không đọc cấu trúc; selectivity cao mới chứng tỏ biểu diễn thực sự encode thuộc tính. Đây là kiểm soát then chốt mà mọi probe nên báo cáo.

### (c) Decodable ≠ được dùng (correlation ≠ causation)

Cạm bẫy nặng nhất: probe accuracy cao chỉ nói thông tin *có thể trích*, **không** nói model *dùng* nó để ra quyết định. Thông tin có thể decodable nhưng nằm im, không ảnh hưởng output. Để kết luận model *dùng* một thuộc tính, phải **can thiệp** ([activation patching](../../05-probing-intervention/research/05-activation-patching.md), amnesic probing) chứ không chỉ quan sát — đúng phân biệt [causal vs observational](../../05-probing-intervention/research/04-causal-intervention-vs-observational.md).

---

## 3. Các tiến bộ (advances)

| Phương pháp | Ý tưởng |
|---|---|
| **Control tasks / selectivity** | so với nhãn ngẫu nhiên để loại memorization (mục 2b). |
| **MDL / information-theoretic probing** | đo *độ dài mô tả tối thiểu* (Voita & Titov 2020): thông tin *dễ trích* tới đâu, không chỉ accuracy cuối. Một thuộc tính encode "đẹp" thì probe học nhanh, MDL ngắn. |
| **Amnesic probing** | (Elazar et al. 2021) *xóa* một thuộc tính khỏi biểu diễn (vd INLP — chiếu lên nullspace của probe lặp) rồi đo hành vi model thay đổi ra sao → bằng chứng *nhân quả* model có dùng thuộc tính. |
| **Probing nhiều lớp / theo thời gian** | theo dòng thông tin qua lớp, gần [logit lens](../../05-probing-intervention/research/10-logit-lens-tuned-lens.md). |

MDL biến probing từ "accuracy nhị phân" thành "độ dễ trích" liên tục — robust hơn với probe complexity. Amnesic probing đóng khoảng cách decodable→used bằng can thiệp.

---

## 4. Khi nào dùng gì (cẩm nang)

- Câu hỏi "thông tin có *tuyến tính* không?" → **linear probe** (probe yếu, kiểm với control task).
- Câu hỏi "thông tin có *tồn tại* không (upper bound)?" → **nonlinear probe**, nhưng cẩn thận complexity confound; báo cáo selectivity.
- Câu hỏi "encode *đẹp/dễ trích* tới đâu?" → **MDL probing**.
- Câu hỏi "model có *dùng* thông tin không?" → **can thiệp** (amnesic probing, activation patching) — *không* dùng accuracy đơn thuần.
- Câu hỏi "concept có hướng nhân quả với output?" → **TCAV** / steering.

Quy tắc vàng: *probe accuracy là tương quan; kết luận nhân quả cần can thiệp.*

---

## 5. Giới hạn / Khi nào thất bại

**Bản thân probing là tương quan.** Ngay cả với control task, probe vẫn chỉ nói về *decodability*; usage cần can thiệp.

**Chọn probe là chủ quan.** Độ phức tạp probe, dataset, regularization đều ảnh hưởng kết luận; không có lựa chọn "trung lập".

**Control task không hoàn hảo.** Thiết kế control task tốt khó; nhãn ngẫu nhiên có thể không khớp cấu trúc task thật.

**Phụ thuộc nhãn giám sát.** Probing cần thuộc tính *có nhãn* — không phát hiện được feature *chưa biết* (đó là chỗ [SAE](02-towards-monosemanticity.md) không giám sát bù vào).

**Lớp/biểu diễn cụ thể.** Kết luận gắn với lớp được probe; thông tin có thể di chuyển/biến đổi qua lớp.

---

## 6. Liên hệ với Latent-Anything

Survey này là *kim chỉ nam phương pháp* cho Layer A: nó định nghĩa cách probe *đúng* và tránh kết luận sai. Một `ProbeMethod` của Layer A nên ép buộc các kiểm soát này.

```python
class ProbeMethod(Protocol):
    def fit(self, latent: np.ndarray, labels: np.ndarray, complexity: str) -> float: ...  # acc
    def selectivity(self, latent: np.ndarray, labels: np.ndarray) -> float: ...   # task - control
    def mdl(self, latent: np.ndarray, labels: np.ndarray) -> float: ...           # description length
    def amnesic_effect(self, latent: np.ndarray, labels: np.ndarray) -> float: ...# causal usage
```

- **Layer A — Introspection**: framework nên phơi bày *cả bộ* — accuracy + selectivity + MDL + amnesic — chứ không chỉ accuracy, để báo cáo introspection trung thực ("decodable" vs "được dùng"). Đây là chuẩn chất lượng cho mọi câu trả lời probe.
- **Layer B — Manipulation**: amnesic probing (xóa thuộc tính bằng nullspace projection) *là* một manipulation — họ hàng [subspace projection](../../04-latent-computation/research/04-subspace-projection.md); biết hướng thuộc tính cho phép xóa/giữ có chủ đích.
- **Layer C — Runtime**: probing nhiều lớp × nhiều thuộc tính là nhiều lần fit classifier — Layer C batch và cache activation để chạy hiệu quả.

Survey khép phần "phương pháp probe có giám sát" của tầng. Hai mục cuối chuyển sang công cụ *không giám sát* để *nhìn* latent chiều cao: **UMAP** và **PaCMAP**.

---

## Liên quan

- [Linear Probing](../../05-probing-intervention/research/01-linear-probing.md) — probe yếu cho câu hỏi tuyến tính.
- [Nonlinear Probing](../../05-probing-intervention/research/02-nonlinear-probing.md) — upper bound; cẩn thận complexity confound.
- [Causal Intervention vs Observational](../../05-probing-intervention/research/04-causal-intervention-vs-observational.md) — decodable ≠ used; nền của amnesic probing.
- [Activation Patching](../../05-probing-intervention/research/05-activation-patching.md) — can thiệp để chứng minh usage.
- [Towards Monosemanticity](02-towards-monosemanticity.md) — SAE không giám sát, phát hiện feature *chưa có nhãn*.

## Tham khảo

- Y. Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances* (Computational Linguistics 48(1), 2022, arXiv:2102.12452).
- J. Hewitt, P. Liang, *Designing and Interpreting Probes with Control Tasks* (EMNLP 2019, arXiv:1909.03368).
- E. Voita, I. Titov, *Information-Theoretic Probing with Minimum Description Length* (EMNLP 2020, arXiv:2003.12298).
- Y. Elazar et al., *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals* (TACL 2021, arXiv:2006.00995).
