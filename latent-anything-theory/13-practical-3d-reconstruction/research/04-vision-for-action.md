# CVPR 2026: Từ "nhìn thấy gì" đến "làm gì" — Triết lý mới trong thị giác máy

> **TL;DR.** CVPR 2026 đánh dấu một sự dịch chuyển nền tảng: ranh giới giữa *"nhìn thấy cái gì"* (perception) và *"làm cái gì"* (action) đang bị xoá nhoà. Các hệ thống end-to-end học trực tiếp từ pixel đến hành động đang vượt trội so với pipeline có các bước trung gian do con người thiết kế (3D reconstruction, segmentation, object detection làm bước riêng). Luận đề: thị giác sinh học tiến hoá không phải để "vẽ bản đồ 3D trong đầu một cách vô định", mà để sinh tồn — nhìn thấy vật cản thì né, nhìn thấy quả táo thì hái. AI cũng nên như vậy: thị giác chỉ có ý nghĩa khi phục vụ trực tiếp cho một mục tiêu hành động cụ thể trong môi trường thời gian thực.

---

## **1. Hai câu chuyện của CVPR 2026**

CVPR 2026 kể hai câu chuyện song song, tưởng như đối lập nhưng thực chất bổ trợ:

### 1.1 Câu chuyện thứ nhất: 3D foundation model đang thành hình

Một nửa các paper đỉnh cao đang xây dựng **feed-forward 3D reconstruction** như một primitive — thay thế hoàn toàn pipeline hình học cổ điển:

- **[VGGT-Ω](02-feed-forward-3d-recon.md)** — feed-forward 3D từ ảnh/video, scale như LLM, rẻ, nhanh, và *cũng* cải thiện VLA.
- **[D4RT](https://arxiv.org/abs/2605.15195)** — 4D reconstruction động như một "trả lời câu hỏi về điểm bất kỳ trong không-thời gian".
- **[SAM 3D](https://arxiv.org/abs/2511.16624)** — 3D-from-single-image in-the-wild, phá data barrier bằng data engine model-in-the-loop + DPO.
- **[Structured Latents (O-Voxel)](https://arxiv.org/abs/2605.15195)** — native 3D generation với 4B flow-matching, PBR material.
- **[FUSER](https://arxiv.org/abs/2605.15195)** — feed-forward multi-view point-cloud registration, không cần pairwise matching.

Điểm chung: **3D không còn là optimization per-scene nữa, mà là inference một lần** — đúng như cách LLM sinh văn bản.

### 1.2 Câu chuyện thứ hai: nhìn là để làm

Nhưng nửa còn lại của CVPR 2026 đang đặt câu hỏi sâu hơn: **dựng 3D để làm gì?** Và câu trả lời ngày càng rõ: **để phục vụ hành động**.

- **[NitroGen](https://arxiv.org/abs/2605.15195)** — video-action foundation model: học trực tiếp từ 40k giờ gameplay (1000+ games) để sinh hành động từ pixel. Không có bước "dựng 3D → hiểu scene → lập kế hoạch" tách biệt.
- **[SparseWorld-TC](https://arxiv.org/abs/2605.15195)** — occupancy world model có điều kiện quỹ đạo: dự báo tương lai của occupancy grid từ một quỹ đạo hành động. Không cần VAE, không cần BEV.
- **[SocialNav](https://arxiv.org/abs/2605.15195)** — navigation xã hội: VLM brain + action expert + SAFE-GRPO, tối ưu trực tiếp social compliance.
- **[VGGT-Segmentor](https://arxiv.org/abs/2605.15195)** — tri giác ego↔exo cho embodied agent, zero-shot vượt supervised.

Điểm chung: **perception và action được train cùng nhau, không còn là hai module tách biệt**.

### 1.3 Và hồi kết: fidelity ≠ physics

**[PAI-Bench](https://arxiv.org/abs/2605.15195)** đóng vai trò "người cầm cân nảy mực": video generation model trông rất thật nhưng **không tuân thủ vật lý**. MLLM đạt 64.7% trong khi con người 93.2% ở causal reasoning. Bài học: *nhìn giống thật chưa đủ* — cần hiểu và dự đoán được động học.

---

## **2. "Nhìn thấy gì" và "Làm gì" — tại sao ranh giới bị xoá nhoà?**

### 2.1 Góc nhìn sinh học

Thị giác của con người không tiến hoá để "render ra một mô hình 3D hoàn hảo trong đầu". Nó tiến hoá để **sinh tồn**:

- Nhìn thấy vật cản → **né**.
- Nhìn thấy quả táo → **hái**.
- Nhìn thấy khuôn mặt quen → **tiếp cận** (hoặc tránh).

Toàn bộ hệ thống thị giác sinh học là một **perception-action loop** khép kín: mỗi tín hiệu thị giác được xử lý với một *affordance* ngầm — "vật này có thể làm gì với nó". Không có bước "dựng mesh 3D → gán semantic label → suy luận → lập kế hoạch" như pipeline computer vision truyền thống.

### 2.2 Góc nhìn kỹ thuật: tại sao pipeline thất bại

Các pipeline có bước trung gian do con người thiết kế (vd: ảnh → depth → point cloud → segmentation → object detection → planning) có một vấn đề cốt tử: **mỗi bước được tối ưu cho một objective riêng, không phải cho end task**.

- Depth estimation được train để giảm RMSE với ground-truth depth — nhưng *sai 5cm ở gần* và *sai 5cm ở xa* có ý nghĩa hoàn toàn khác cho bài toán grasping.
- Segmentation được train để tối đa mIoU — nhưng *phân đoạn sai một pixel ở rìa vật thể* có thể làm hỏng một pha grasping, trong khi *phân đoạn sai ở nền* có thể vô hại.
- Object detection được train cho AP — nhưng *miss một object* có thể là thảm hoạ cho navigation, trong khi *false positive* có thể chỉ gây chậm.

Khi các module này được train độc lập, **gradient không thể chảy từ end-task về early module**. Hậu quả: early module không biết feature nào *thực sự quan trọng* cho hành động cuối cùng.

### 2.3 End-to-end learning: để gradient làm việc

End-to-end learning giải quyết vấn đề này bằng cách **cho gradient từ action loss chảy ngược về toàn bộ perception stack**:

$$ \theta^* = \arg\min_\theta \mathbb{E}_{(o,a^*)} \left[ \mathcal{L}_{\text{action}}(f_\theta(o), a^*) \right] $$

trong đó $f_\theta$ là một mạng duy nhất từ observation $o$ đến action $a$, không có bước trung gian nào được giám sát riêng. Mạng tự học *cách biểu diễn observation* sao cho có ích nhất cho action.

Đây chính là triết lý của JEPA (LeCun, 2022), NitroGen, và hàng loạt paper CVPR 2026.

---

## **3. Nhưng 3D vẫn quan trọng — chỉ là nó nên ở đâu?**

Luận điểm "end-to-end, không cần 3D trung gian" **không có nghĩa là 3D không quan trọng**. Nó có nghĩa là:

- **3D nên là một biểu diễn ẩn** (latent representation), không phải một đầu ra trung gian được giám sát tường minh.
- **3D geometry nên được học** từ tín hiệu action, không phải từ depth ground-truth.
- **3D structure nên emerge** như một công cụ để dự đoán tương lai và lập kế hoạch — đúng như cách [RSSM trong Dreamer](https://github.com/triet4p/latent-anything/blob/main/docs/THEORY.md) dùng latent state để plan.

Nói cách khác: **3D vẫn là trung tâm, nhưng nó là latent 3D, không phải explicit 3D**. VGGT-Ω là một bước lai: nó sinh explicit 3D (camera, depth, point), nhưng các register token *cũng* là một latent biểu diễn dùng được cho VLA. Sự hội tụ này cho thấy ranh giới giữa explicit và latent 3D cũng đang mờ đi.

---

## **4. Hàm ý cho thiết kế hệ thống AI**

Từ CVPR 2026, có thể rút ra vài nguyên tắc thiết kế:

1. **Train perception với action signal, không chỉ perception signal.** Nếu downstream task là grasping, depth estimation nên được train với grasping success làm tín hiệu — không chỉ depth RMSE.
2. **3D là phương tiện, không phải mục đích.** Dựng mesh 3D tuyệt đẹp là vô nghĩa nếu agent không dùng nó để làm gì. Giá trị của 3D nằm ở khả năng *dự đoán hệ quả của hành động* và *lập kế hoạch*.
3. **Data engine > hand-crafted module.** SAM 3D chứng minh: annotation pipeline model-in-the-loop + preference learning vượt qua thiết kế module thủ công về cả scale lẫn chất lượng.
4. **Foundation model cho perception-action, không chỉ perception.** NitroGen, VGGT-Ω register→VLA, SocialNav đều chỉ về một hướng: cùng một model cho cả nhìn và làm.

---

## **5. Liên hệ với Latent-Anything**

Đây chính là triết lý nền của Latent-Anything:

- **Latent space là cầu nối perception-action**: Latent-Anything thiết kế latent space không phải để "mô tả thế giới", mà để *phục vụ downstream task* — dù là manipulation (Layer B), planning (Tầng 7), hay prediction (Tầng 8).
- **Không có "decoder để ngắm"**: Tinh thần JEPA — predict trong latent, không decode — là hiện thân của triết lý "nhìn để làm, không phải nhìn để vẽ".
- **3DGS parameters là latent**: [Gaussian parameters](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) không được giám sát bởi "3D ground-truth", mà bởi *khả năng dự đoán observation tương lai* hoặc *chất lượng action*.
- **ModelAdapter là perception-action interface**: Mỗi ModelAdapter dịch observation sang latent state — nhưng latent state đó được thiết kế để phục vụ transition model và policy, không phải để tái tạo observation.

CVPR 2026 xác nhận rằng hướng đi của Latent-Anything — **lấy latent làm trung tâm, không phải explicit representation** — là đúng đắn và đang được cả cộng đồng hội tụ về.

---

## Liên quan

- [COLMAP & SfM/MVS cổ điển](01-colmap-sfm-mvs.md) — hiện thân của paradigm "dựng bản đồ 3D vô định", paradigm đang bị thách thức.
- [Dust3R, Must3R & VGGT-Ω](02-feed-forward-3d-recon.md) — cây cầu giữa explicit và latent 3D.
- [Image-to-Point Cloud Feature Back-Projection](03-image-to-pointcloud-backprojection.md) — một cách đưa feature 2D vào 3D để phục vụ action.
- [Gaussian Parameters là Latent Variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) — hiện thân của "3D là latent, không phải explicit".
- [Information Bottleneck Principle](../../02-representation-learning/research/01-information-bottleneck.md) — nền tảng lý thuyết cho việc "chỉ giữ thông tin có ích cho action".

## Tham khảo

- LeCun, *A Path Towards Autonomous Machine Intelligence* (OpenReview, 2022) — JEPA và triết lý latent prediction.
- NitroGen (CVPR 2026) — video-action foundation model, 40k giờ gameplay.
- SparseWorld-TC (CVPR 2026) — trajectory-conditioned occupancy world model.
- PAI-Bench (CVPR 2026) — fidelity ≠ physics.
- VGGT-Ω (CVPR 2026 Oral, arXiv:2605.15195) — register token cải thiện VLA.
- Gibson, *The Ecological Approach to Visual Perception* (1979) — affordance theory, nền tảng triết học cho "nhìn để làm".
