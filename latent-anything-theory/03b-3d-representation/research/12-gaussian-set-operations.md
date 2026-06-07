# Gaussian set operations — add, remove, split, merge trên tập Gaussian

> **TL;DR.** Gaussian set operations là các thao tác trực tiếp lên **tập primitive** của 3D Gaussian Splatting: thêm Gaussian mới, bỏ Gaussian thừa, tách một Gaussian lớn thành nhiều Gaussian nhỏ hơn, hay gộp nhiều Gaussian gần nhau thành biểu diễn gọn hơn. Ý tưởng cốt lõi có thể viết gọn là $\mathcal{S}_{t+1} = (\mathcal{S}_t \setminus \mathcal{R}_t) \cup \mathcal{A}_t$, trong đó $\mathcal{R}_t$ là tập Gaussian bị remove/prune còn $\mathcal{A}_t$ là tập Gaussian mới sinh ra bởi add, clone hoặc split; `merge` là phép nén ngược chiều nhằm thay nhiều primitive gần-trùng bằng ít primitive hơn. Caveat lớn nhất là các operator này rất mạnh nhưng cũng rất dễ phá vỡ ổn định tối ưu, temporal identity và fidelity nếu không có tiêu chuẩn rõ cho khi nào nên tách, bỏ hay gộp.

Sau [Dynamic 3DGS](11-dynamic-3dgs.md), câu hỏi không còn chỉ là "Gaussian này di chuyển thế nào theo thời gian?" mà còn là "tập Gaussian có còn **giữ nguyên số lượng và tổ chức** qua thời gian hay không?". Trong scene thật, câu trả lời thường là không: vùng mới lộ ra có thể cần Gaussian mới, Gaussian mờ/floaters cần bị bỏ, vùng quá thô cần được chia nhỏ, còn vùng dư thừa lại nên được nén lại.

Đó là chỗ explicit representation của 3DGS lộ ra một ưu thế rất lớn so với implicit field kiểu NeRF. Với một scene được mã hóa thành **một tập phần tử hữu hạn**, có thể suy nghĩ trực tiếp bằng ngôn ngữ của set: thêm phần tử, bỏ phần tử, tách phần tử, gom phần tử. Điều này nghe đơn giản, nhưng chính nó làm Gaussian-centric world model trở nên tự nhiên hơn nhiều khi cần editing, tracking, memory management hay state transition có thay đổi topology.

---

## **1. Trực giác: scene không chỉ là giá trị của Gaussian, mà còn là số lượng Gaussian**

Với 3DGS tĩnh ở mức tối giản, scene có thể viết như:

$$
\mathcal{S} = \{z_i\}_{i=1}^{N},
\qquad
z_i = (\mu_i,\Sigma_i,o_i,\mathbf{a}_i)
$$

trong đó $z_i$ là Gaussian thứ $i$ với mean $\mu_i$, covariance $\Sigma_i$, opacity $o_i$ và appearance $\mathbf{a}_i$. Công thức này nhấn mạnh rằng representation không phải một hàm liên tục ẩn như NeRF, mà là một **tập hữu hạn các primitive** có thể đếm được.

Một khi đã có tập hữu hạn, bài toán học không còn chỉ là cập nhật tham số của từng phần tử, mà còn bao gồm cập nhật **cardinality** và **structure** của cả tập. Có thể viết abstract như:

$$
\mathcal{S}_{t+1} = \Phi(\mathcal{S}_t)
$$

trong đó $\Phi$ không chỉ dịch chuyển hay đổi covariance của từng Gaussian, mà còn có thể sinh thêm, xóa bớt hoặc tái tổ chức primitive. Biểu thức này có nghĩa là evolution của scene state trong Gaussian space là một phép biến đổi trên **set of primitives**, không chỉ trên vector tham số cố định chiều.

Đây là khác biệt có tính bản chất so với implicit radiance field:

- với NeRF, muốn "thêm một phần" vào scene thường phải sửa lại toàn bộ hàm mật độ/màu;
- với Gaussian set, có thể suy nghĩ cục bộ hơn: thêm vài primitive, bỏ vài primitive, hay chia một primitive to thành nhiều primitive nhỏ.

---

## **2. Operator chuẩn nhất trong 3DGS gốc: clone, split và prune**

Paper và mã nguồn chính thức của [3D Gaussian Splatting](06-3d-gaussian-splatting.md) đã cho thấy rất rõ rằng training của 3DGS không chỉ là gradient descent trên một tập Gaussian cố định. Thay vào đó, nó dùng **interleaved optimization and density control**: vừa tối ưu tham số, vừa thay đổi chính tập Gaussian.

Ở dạng khái niệm, vòng cập nhật có thể viết là:

$$
\mathcal{S}_{k+1}
=
\operatorname{Prune}\Big(
\operatorname{Split}(
\operatorname{Clone}(\mathcal{S}_k)
)\Big)
$$

trong đó `Clone` nhân thêm Gaussian ở những vùng cần mật độ cao hơn, `Split` tách Gaussian lớn thành nhiều Gaussian nhỏ hơn, còn `Prune` loại bỏ Gaussian không còn hữu ích. Biểu thức này có nghĩa là chất lượng của 3DGS không đến từ tối ưu liên tục thuần túy, mà đến từ việc **thích nghi số lượng primitive** trong lúc học.

Trong implementation chính thức của GraphDECO:

- `densify_and_clone(...)` sao chép các Gaussian có gradient đủ lớn nhưng còn tương đối nhỏ;
- `densify_and_split(...)` tách các Gaussian có gradient lớn và kích thước đủ to;
- `densify_and_prune(...)` sau đó loại các Gaussian có opacity quá thấp hoặc quá lớn trên màn hình/world scale.

Điểm quan trọng là ngay từ paper gốc, "density control" đã là một phần bản chất của 3DGS chứ không phải mẹo phụ. Scene chất lượng cao đòi hỏi tập Gaussian phải **tự tổ chức lại** trong quá trình học.

---

## **3. Add và split: tại sao phải sinh Gaussian mới?**

Thao tác `add` có hai hình thức phổ biến nhất trong 3DGS:

- **clone**: thêm một bản sao gần như giữ nguyên thuộc tính của Gaussian cũ;
- **split**: thay một Gaussian lớn bằng nhiều Gaussian nhỏ hơn để biểu diễn chi tiết tốt hơn.

Trong mã gốc của 3DGS, `split` được thực hiện bằng cách lấy Gaussian đang có, lấy mẫu các vị trí mới quanh nó theo scale và rotation hiện tại, rồi thay Gaussian to bằng nhiều Gaussian nhỏ hơn. Điều này phản ánh một trực giác rất tự nhiên:

- nếu một vùng còn thiếu chi tiết nhưng đang được một Gaussian quá to che phủ,
- thì cách tốt hơn không phải là ép Gaussian đó tự mang mọi chi tiết,
- mà là **phân rã** nó thành nhiều primitive có độ phân giải cục bộ cao hơn.

Một cách viết trừu tượng cho `split` là:

$$
z \;\longrightarrow\; \{z'_1,\dots,z'_m\}
$$

trong đó một Gaussian gốc $z$ được thay bằng $m$ Gaussian con $\{z'_j\}$. Biểu thức này có nghĩa là local support của một primitive rộng được phân hoạch lại thành nhiều support nhỏ hơn, giúp scene bám bề mặt và chi tiết tốt hơn.

Paper **A New Split Algorithm for 3D Gaussian Splatting** làm rõ hơn chính ý tưởng này: split không nên chỉ là heuristic "bẻ đôi đại khái", mà nên giữ tương thích về đặc tính toán học và appearance để mô hình sau split vẫn gần với Gaussian ban đầu nhưng đều và bám surface hơn.

Về mặt world model, `add` và `split` là những operator rất quan trọng vì chúng cho phép latent state:

- tăng local capacity ở nơi prediction còn kém;
- tạo primitive mới khi vật thể mới xuất hiện hoặc khi disocclusion lộ ra vùng chưa từng thấy;
- tách một object coarse thành cấu trúc tinh hơn khi cần planning hay manipulation ở mức chi tiết.

---

## **4. Remove và prune: tại sao phải bỏ Gaussian đi?**

Nếu `add` và `split` làm representation giàu hơn, thì `remove` và `prune` giữ cho representation không phình vô hạn.

Một cách viết abstract là:

$$
\mathcal{S}' = \mathcal{S} \setminus \mathcal{R},
\qquad
\mathcal{R} = \{z_i \in \mathcal{S} \mid \text{score}(z_i) < \tau\}
$$

trong đó $\mathcal{R}$ là tập Gaussian bị loại, `score` là độ quan trọng nào đó của Gaussian, còn $\tau$ là ngưỡng prune. Công thức này có nghĩa là pruning là một phép chọn lọc phần tử trên set theo một tiêu chí đóng góp, thay vì sửa nhẹ mọi Gaussian đồng đều.

Trong thực hành, lý do prune có thể khác nhau:

- opacity quá thấp nên Gaussian gần như không đóng góp vào render;
- footprint quá lớn hoặc sai chỗ, dễ tạo artifact;
- Gaussian là floater hay phần dư thừa không cần cho fidelity;
- bộ nhớ và tốc độ render buộc phải giảm cardinality của set.

Các công trình như **LightGaussian**, **Compact 3D Gaussian Splatting**, và **PUP 3D-GS** cho thấy prune không chỉ là thao tác dọn rác, mà là một bài toán tối ưu thật sự. Chúng khác nhau ở cách chấm điểm Gaussian:

- LightGaussian thiên về ước lượng mức quan trọng toàn cục và phục hồi chất lượng sau prune;
- Compact 3DGS nhấn mạnh giảm số Gaussian và nén thuộc tính trên cả static lẫn dynamic radiance fields;
- PUP 3D-GS xây một pruning score có cơ sở toán học hơn từ xấp xỉ độ nhạy của lỗi reconstruction.

Điểm rút ra là `remove` trong Gaussian space hoàn toàn có thể là một operator học được hoặc tối ưu được, chứ không chỉ là ngưỡng thủ công.

---

## **5. Merge: operator tự nhiên nhưng ít canonical hơn**

Nếu `split` là thay một Gaussian to bằng nhiều Gaussian nhỏ, thì `merge` là ý tưởng ngược lại:

$$
\{z_1,\dots,z_m\}
\;\longrightarrow\;
\tilde{z}
$$

trong đó nhiều Gaussian gần-trùng được thay bởi một Gaussian đại diện $\tilde{z}$. Biểu thức này có nghĩa là scene được nén cục bộ bằng cách đánh đổi một phần độ chi tiết để giảm số primitive.

Tuy nhiên, có một nuance quan trọng: trong literature 3DGS nền tảng hiện tại, `merge` **không canonical bằng** `split/clone/prune`.

- paper gốc và repo gốc có clone, split, prune khá rõ;
- pruning/compression papers chủ yếu nói về remove, quantize, refine;
- còn một "merge operator chuẩn" với công thức duy nhất thì chưa thật sự trở thành chuẩn nền tảng chung của 3DGS như split.

Vì vậy, nên hiểu `merge` ở đây theo hai nghĩa gần nhau:

1. **inverse conceptually** của split: gom những primitive quá dày đặc hoặc quá giống nhau;
2. **mục tiêu của simplification/compression**: thay nhiều Gaussian bằng biểu diễn gọn hơn mà quality giảm ít nhất.

Trong một world model Gaussian-centric, merge vẫn cực kỳ tự nhiên vì nó cho phép:

- nén memory khi rollout dài;
- gom các primitive quá vi mô thành object-part macro-state;
- tạo coarse state cho planning xa, rồi split lại khi cần decode chi tiết.

Nhưng khác với split, merge cần tiêu chuẩn tốt về:

- Gaussian nào đủ gần để được gộp;
- thuộc tính nào phải bảo toàn sau gộp;
- gộp ở object space, world space hay render space.

---

## **6. So sánh các operator trên tập Gaussian**

| Operator | Tác dụng trực giác | Mục tiêu chính | Có canonical trong 3DGS gốc không? | Rủi ro chính |
|---|---|---|---|---|
| Add | sinh primitive mới | tăng capacity cục bộ | có, dưới dạng densification | bùng nổ cardinality |
| Clone | nhân thêm Gaussian gần như giữ nguyên state | tăng mật độ ở vùng còn thiếu | có | redundancy, overfitting |
| Split | thay Gaussian to bằng nhiều Gaussian nhỏ | bám chi tiết/surface tốt hơn | có | artifact nếu split sai hướng |
| Remove / Prune | bỏ Gaussian ít hữu ích | giảm dư thừa, tăng tốc, dọn floater | có | mất chi tiết quan trọng |
| Merge | gom nhiều Gaussian thành ít Gaussian hơn | nén state, coarse abstraction | chưa thật canonical | mất cấu trúc, blur cục bộ |

Bảng này cho thấy Gaussian set operations không phải các "thủ thuật tách rời", mà là một hệ thao tác điều khiển **độ phân giải và cardinality** của latent representation.

---

## **7. Tại sao explicit Gaussian set amenable với world model hơn implicit NeRF?**

Đây là chỗ roadmap nhấn rất đúng: cùng là radiance-field-like representation, nhưng Gaussian set dễ trở thành state cho world model hơn NeRF nếu bài toán cần can thiệp cấu trúc.

Với Gaussian set, có thể viết transition tổng quát như:

$$
\mathcal{Z}_{t+1}
=
F_{\text{param}}(\mathcal{Z}_t, a_t)
\;\oplus\;
F_{\text{set}}(\mathcal{Z}_t, a_t)
$$

trong đó $F_{\text{param}}$ cập nhật tham số của các Gaussian đang tồn tại, còn $F_{\text{set}}$ quyết định operator set-level nào cần áp dụng như add, prune, split hay merge. Ký hiệu $\oplus$ ở đây chỉ rằng transition đầy đủ gồm **cả cập nhật tham số lẫn cập nhật cấu trúc tập**.

Với implicit NeRF, state thường là trọng số của một MLP hay grid latent. Muốn "bỏ vật thể này", "tách vùng này", "thêm primitive mới" thường không có một operator cục bộ rõ ràng; thay vào đó phải sửa một hàm toàn cục có ảnh hưởng lan rộng.

Ngược lại, Gaussian set hỗ trợ trực tiếp các thao tác kiểu:

- object removal;
- local editing;
- compositional insertion;
- persistent tracking theo ID/group;
- memory-budget-aware simplification.

Các paper như **Gaussian Grouping** minh họa rất rõ ưu thế này: khi scene đã được phân nhóm theo object/instance trên chính các Gaussian, các thao tác remove, recolor, inpaint hay recomposition trở nên tự nhiên vì primitive đã explicit và địa phương hóa.

---

## **8. Giới hạn / Khi nào Gaussian set operations thất bại**

- **Operator quyết định sai lúc.** Split quá sớm tạo redundancy; prune quá sớm làm mất chi tiết chưa kịp học.
- **Identity drift trong scene động.** Một Gaussian bị split rồi prune rồi sinh lại có thể làm notion về "cùng primitive" trở nên lỏng.
- **Merge dễ làm mất topology cục bộ.** Gom các Gaussian tưởng như gần nhau nhưng thực ra thuộc hai surface khác nhau sẽ gây blur hoặc dính hình học.
- **Heuristic dễ dataset-specific.** Nhiều rule dựa vào opacity, gradient hay screen size hoạt động tốt ở scene này nhưng không chắc chuyển được sang scene khác.
- **Chi phí book-keeping tăng.** Một world model thật sự dùng set operations phải theo dõi history, parent-child relation, group ID hoặc uncertainty của primitive.
- **Không tự động giải quyết semantics.** Việc hai Gaussian gần nhau hình học không có nghĩa chúng nên merge ở mức object semantics.

Nói ngắn gọn, explicit operator giúp world model thao tác được trên latent state, nhưng càng cho phép thay đổi cấu trúc mạnh thì càng cần rule tốt để tránh state trở nên khó ổn định và khó diễn giải.

---

## **9. Liên hệ với Latent-Anything**

Đây là một trong những mục 3B chạm rất mạnh đến "Anything" trong Latent-Anything. Nếu latent chỉ là Gaussian parameters cố định, model mới chỉ represent một scene bằng set primitive. Khi có Gaussian set operations, latent bắt đầu có khả năng **tự tổ chức lại representation**.

Điều này mở ra các hướng rất hợp với ba layer:

- Layer A có thể probe primitive importance, uncertainty, split-worthiness hay prune-worthiness của từng Gaussian.
- Layer B có thể can thiệp trực tiếp: ép split một vùng, cấm prune một object, merge một cluster để test abstraction.
- Layer C có thể chạy rollout với budget cố định bằng cách thêm operator prune/merge trong khi vẫn decode bằng [Gaussian Rasterization](09-gaussian-rasterization.md).

Ở góc nhìn world model, một state update lúc này không chỉ là:

$$
z_{t+1} = f(z_t, a_t)
$$

mà gần hơn với:

$$
\mathcal{Z}_{t+1} = \operatorname{EditSet}\big(f(\mathcal{Z}_t, a_t)\big)
$$

trong đó $f$ cập nhật dynamics còn `EditSet` áp các operator như add/remove/split/merge. Công thức này có nghĩa là latent transition không còn bị khóa vào một không gian cố định chiều, mà có thể thay đổi **độ lớn và cấu trúc của state** theo nhu cầu của scene.

Đó là bước đệm rất quan trọng trước khi đi sang các tầng sau về temporal state, planning và world modeling thật sự.

---

## Liên quan

- [3D Gaussian Splatting](06-3d-gaussian-splatting.md) — nguồn gốc trực tiếp của clone, split, prune thông qua density control.
- [Covariance Matrix trong 3DGS](07-covariance-matrix-3dgs.md) — split/merge đều đụng trực tiếp đến cách scale và orientation của Gaussian được tái tham số hóa.
- [Gaussian Rasterization](09-gaussian-rasterization.md) — mọi operator trên set cuối cùng đều được đánh giá qua ảnh render từ rasterizer.
- [Gaussian parameters là latent variable](10-gaussian-parameters-latent-variable.md) — khi Gaussian là latent state, set operations trở thành operator trực tiếp trên latent.
- [Dynamic 3DGS](11-dynamic-3dgs.md) — scene động gần như bắt buộc phải nghĩ đến birth/death/split/merge của primitive qua thời gian.

## Tham khảo

- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (ACM Transactions on Graphics 2023, arXiv:2308.04079).
- GraphDECO, *gaussian-splatting official implementation* (GitHub repository, accessed June 7, 2026).
- Feng, Cao, Chen, Mu, Martin, Hu, *A New Split Algorithm for 3D Gaussian Splatting* (arXiv:2403.09143).
- Kheradmand, Rebain, Sharma, Sun, Tseng, Isack, Kar, Tagliasacchi, Yi, *3D Gaussian Splatting as Markov Chain Monte Carlo* (arXiv:2404.09591).
- Fan, Wang, Wen, Zhu, Xu, Wang, *LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS* (arXiv:2311.17245).
- Hanson, Tu, Singla, Jayawardhana, Zwicker, Goldstein, *PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting* (CVPR 2025, arXiv:2406.10219).
- Lee, Rho, Sun, Ko, Park, *Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields* (arXiv:2408.03822).
- Ye, Wang, Cao, Chen, Yuan, Li, Zhang, Zhu, Chai, *Gaussian Grouping: Segment and Edit Anything in 3D Scenes* (ECCV 2024, arXiv:2312.00732).
