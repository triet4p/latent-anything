# Dust3R, Must3R & VGGT-Ω — Feed-forward 3D Reconstruction

> **TL;DR.** Dust3R (2024) đặt nền móng bằng cách cast bài toán stereo reconstruction thành **pointmap regression** — dự đoán trực tiếp toạ độ 3D của mọi pixel từ một cặp ảnh qua một ViT encoder-decoder, bỏ qua toàn bộ pipeline SfM cổ điển. Must3R mở rộng lên multi-view bằng local alignment window và iterative refinement. VGGT (CVPR 2025 Best Paper) nâng lên tầm Transformer thuần — suy camera, depth, point map, track từ 1→hàng trăm view trong một lần truyền. VGGT-Ω (CVPR 2026 Oral) chứng minh paradigm này scale như LLM: register attention cắt 70% bộ nhớ, self-supervision photometric mở khoá video không nhãn, và các register token *cũng* cải thiện VLA. Đây là sự dịch chuyển nền tảng từ "optimize per-scene" sang "inference một lần" trong 3D reconstruction.

Ba bài báo Dust3R → Must3R → VGGT-Ω tạo thành một cung tiến hoá rõ rệt: từ **pairwise pointmap** → **multi-view alignment** → **feed-forward foundation model** cho 3D. Mỗi bước gỡ một nút thắt của [COLMAP](01-colmap-sfm-mvs.md): Dust3R gỡ matching + geometry, Must3R gỡ multi-view inconsistency, VGGT gỡ scale và speed, VGGT-Ω gỡ data barrier. Cùng nhau, chúng định nghĩa một paradigm mới cho 3D reconstruction: **feed-forward, data-driven, scale-able**.

---

## **1. Trực giác: thay geometry pipeline bằng một mạng duy nhất**

Pipeline cổ điển của [COLMAP](01-colmap-sfm-mvs.md) có thể tóm gọn thành một chuỗi module rời rạc:

```
ảnh → SIFT → matching → geometric verification → incremental SfM → BA → MVS → dense point cloud
```

Mỗi module là một thuật toán hình học thuần tuý, không học được từ dữ liệu. Mỗi module có điểm yếu riêng (SIFT cần texture, matching cần overlap, BA cần initialization tốt), và lỗi từ module trước lan sang module sau.

Feed-forward 3D đặt câu hỏi: **nếu có đủ dữ liệu 3D, liệu một mạng duy nhất có thể học toàn bộ chuỗi này không?** Câu trả lời từ Dust3R đến VGGT-Ω là: **có** — và nó không chỉ nhanh hơn, mà còn robust hơn ở những vùng COLMAP thất bại (low-texture, thin structure, dynamic).

---

## **2. Dust3R (Wang et al., 2024) — bỏ matching, bỏ geometry**

### 2.1 Ý tưởng cốt lõi: pointmap regression

Thay vì tìm correspondence rồi triangulate, Dust3R **dự đoán trực tiếp** toạ độ 3D của mọi pixel trong một hệ toạ độ chuẩn:

$$ \hat{\mathbf{X}}_i = \text{Dust3R}(I_1, I_2)_i $$

trong đó $\hat{\mathbf{X}}_i \in \mathbb{R}^3$ là toạ độ 3D của pixel thứ $i$, biểu diễn trong camera frame của ảnh thứ nhất. Công thức này nói rằng: **mỗi pixel không còn là màu nữa — nó là một điểm 3D**.

Đầu ra cho cả hai ảnh là hai **pointmap** $P_1, P_2 \in \mathbb{R}^{H \times W \times 3}$ trong cùng một hệ toạ độ. Từ đó, camera pose có thể được trích xuất bằng Procrustes alignment giữa hai pointmap.

### 2.2 Kiến trúc

- **Encoder**: ViT (Vision Transformer) shared-weight cho cả hai ảnh.
- **Decoder**: hai head riêng — một cho pointmap, một cho confidence map $\in [0,1]^{H \times W}$. Confidence map cho biết mức độ tin cậy của dự đoán tại mỗi pixel (cực kỳ quan trọng để multi-view alignment sau này).
- **Loss**: regression loss trên pointmap $\|P_{\text{pred}} - P_{\text{gt}}\|$, weighted bởi confidence.

### 2.3 Tại sao nó là breakthrough

- **Không cần correspondence**: không SIFT, không matching, không RANSAC. Mạng tự học mối liên hệ giữa các pixel qua attention.
- **Dense output**: mọi pixel đều có toạ độ 3D, không chỉ sparse keypoint. Texture-less regions cũng có dự đoán (nhờ prior học được).
- **Confidence-aware**: confidence map cho phép downstream module biết chỗ nào đáng tin, chỗ nào không.

### 2.4 Hạn chế

- **Pairwise**: Dust3R xử lý từng cặp ảnh độc lập. Muốn multi-view cần alignment hậu kỳ.
- **Scale không nhất quán giữa các cặp**: mỗi cặp có scale riêng → cần global alignment.

---

## **3. Must3R (Leroy et al., 2024) — từ pairwise lên multi-view**

Must3R giải quyết vấn đề lớn nhất của Dust3R: **multi-view consistency**.

### 3.1 Cơ chế: local alignment window

Must3R giữ nguyên tinh thần Dust3R (ViT encoder + pointmap regression) nhưng thêm hai thành phần:

1. **Local alignment window**: thay vì xử lý từng cặp độc lập, Must3R align các pointmap cục bộ trong một cửa sổ trượt. Với $k$ ảnh liên tiếp, nó dự đoán đồng thời pointmap của tất cả $k$ ảnh trong một hệ toạ độ chung cục bộ.

2. **Iterative refinement**: các window được merge dần qua nhiều vòng, mỗi vòng tinh chỉnh alignment dựa trên confidence của vòng trước. Đây là một dạng **coordinate-ascent** trong không gian pointmap.

### 3.2 So sánh với COLMAP

| | COLMAP | Dust3R | Must3R |
|---|---|---|---|
| Input | N ảnh | 2 ảnh | N ảnh |
| Correspondence | SIFT + matching | Học implicit qua attention | Học implicit + local alignment |
| Pose estimation | Epipolar geometry | Procrustes từ pointmap | Pointmap alignment trong window |
| Dense output | Không (cần MVS riêng) | Có (dense pointmap) | Có (dense pointmap) |
| Tốc độ | Hàng giờ | <1 giây/cặp | Vài giây cho N ảnh |
| Texture-less | Thất bại | Có prior | Có prior |
| Động | Thất bại | Hạn chế | Hạn chế |

---

## **4. VGGT (CVPR 2025 Best Paper) — Transformer thuần cho 3D**

VGGT nâng cấp triệt để: thay vì ViT encoder + head riêng, nó dùng **một Transformer duy nhất** nhận toàn bộ ảnh đầu vào và xuất **camera pose, depth map, point map, track** trong một lần truyền.

### 4.1 Khác biệt kiến trúc

- **Input**: $N$ ảnh được patchify và flatten thành sequence token, cộng với positional embedding.
- **Global self-attention**: mọi token từ mọi ảnh attend lẫn nhau — đây là sức mạnh (thông tin toàn cục) và cũng là điểm yếu ($O(N^2)$ complexity).
- **Multi-head output**: cùng một backbone cho ra camera, depth, point, track qua các head chuyên biệt.

### 4.2 VGGT chứng minh điều gì

VGGT là bằng chứng đầu tiên rằng **feed-forward 3D reconstruction có thể cạnh tranh với optimization-based pipeline ở độ chính xác**, trong khi nhanh hơn hàng trăm lần. CVPR 2025 Best Paper là sự công nhận rằng đây là một paradigm shift, không phải incremental improvement.

---

## **5. VGGT-Ω (CVPR 2026 Oral) — scale như LLM**

Nếu VGGT trả lời "có khả thi không", VGGT-Ω trả lời "có thể scale được không". Đây là phiên bản scale-up với ba đóng góp then chốt:

### 5.1 Register attention — cắt $O(N^2)$ xuống gần tuyến tính

Global self-attention giữa mọi token của mọi frame tốn $O((N\cdot T)^2)$ bộ nhớ (N frame, T token/frame). VGGT-Ω thay bằng **register attention**: một nhóm nhỏ $R$ learnable token ("register") đóng vai trò bottleneck:

```
frames f1..fN        ┌── register attention ──┐
    │                │ frame→register: O(N·T·R)│
    ├──► registers   │ register↔register: O(R²)│
    │                └─────────────────────────┘
    ▼
    scene tokens → single dense head → depth, camera, features (static & dynamic)
```

Trong đó $R \ll N \cdot T$ (vài chục token so với hàng nghìn), tổng chi phí attention giảm từ $O((N\cdot T)^2)$ xuống $O(N\cdot T \cdot R + R^2) \approx O(N\cdot T)$. Kết quả: **~30% bộ nhớ** của VGGT gốc.

### 5.2 Self-supervision photometric — nuốt video không nhãn

VGGT-Ω dùng **photometric consistency** giữa các frame video như một tín hiệu giám sát *không cần 3D ground-truth*:

$$ \mathcal{L}_{\text{photo}} = \|I_t - \text{warp}(I_{t+1}; \hat{D}_t, \hat{T}_{t \to t+1})\|_1 $$

trong đó $\hat{D}_t$ là depth dự đoán, $\hat{T}_{t \to t+1}$ là camera pose tương đối dự đoán, và $\text{warp}$ là phép chiếu + nội suy. Công thức này cho phép huấn luyện trên **video bất kỳ**, không cần 3D GT — đây là cách vượt qua khan hiếm dữ liệu 3D.

Nhờ đó, VGGT-Ω train với **15–20× supervised + tới 100× unsupervised data** so với VGGT gốc.

### 5.3 Kết quả & ý nghĩa

- **Sintel camera +77%**: cải thiện mạnh nhất ở cảnh động — điểm yếu cũ của VGGT.
- **Register → VLA**: các scene token từ register attention cũng cải thiện Vision-Language-Action model — cùng một biểu diễn 3D phục vụ cả perception lẫn hành động.
- **Scaling curve**: accuracy/robustness tăng dự đoán được theo model+data, giống LLM.

---

## **6. Tổng kết: từ COLMAP đến VGGT-Ω**

| Khía cạnh | COLMAP (2016) | Dust3R (2024) | VGGT (2025) | VGGT-Ω (2026) |
|---|---|---|---|---|
| Paradigm | Optimization per-scene | Pairwise regression | Transformer feed-forward | Scaled feed-forward |
| Input | N ảnh | 2 ảnh | N ảnh | N ảnh + video |
| Correspondence | Explicit (SIFT) | Implicit (attention) | Implicit (global attn) | Implicit (register attn) |
| Output | Sparse PC + pose | Dense pointmap | Dense + cam + track | Dense + cam + features |
| Texture-less | ✗ | ✓ | ✓ | ✓ |
| Dynamic | ✗ | Hạn chế | Hạn chế | ✓ (Sintel +77%) |
| Data needed | 0 (zero-shot) | 3D GT | 3D GT | 3D GT + video |
| Speed | Hàng giờ | Giây/cặp | Giây | Giây |
| Memory | — | — | — | ~30% VGGT |
| VLA | — | — | — | ✓ |

---

## **7. Liên hệ với Latent-Anything**

Feed-forward 3D reconstruction trực tiếp liên quan đến hai trụ cột của Latent-Anything:

- **ModelAdapter cho world model**: VGGT-Ω sinh ra camera pose + depth + features từ ảnh — đây chính là một **perception backbone** có thể cắm trực tiếp vào world model. Các scene token (register) là một dạng latent biểu diễn toàn cục của scene, có thể dùng làm input cho transition model.
- **Latent space của 3D**: Dust3R/VGGT chứng minh rằng 3D geometry có thể được mã hoá trong không gian ẩn của Transformer. Pointmap, depth, camera pose đều là các *code* trong một latent space structured. VGGT-Ω tiến thêm một bước khi *cùng* latent space đó serve được VLA.
- **Từ "nhìn" đến "làm"**: Register token của VGGT-Ω nối thẳng sang VLA là hiện thân của luận đề CVPR 2026 — thị giác chỉ có ý nghĩa khi phục vụ hành động.

---

## Liên quan

- [COLMAP & SfM/MVS cổ điển](01-colmap-sfm-mvs.md) — tiền thân, đối trọng về mặt paradism.
- [Image-to-Point Cloud Feature Back-Projection](03-image-to-pointcloud-backprojection.md) — hướng bổ sung: đưa feature 2D vào point cloud.
- [CVPR 2026: Từ "nhìn thấy gì" đến "làm gì"](04-vision-for-action.md) — triết lý mới mà VGGT-Ω là hiện thân.
- [3D Gaussian Splatting](../../03b-3d-representation/research/06-3d-gaussian-splatting.md) — hưởng lợi từ camera pose do feed-forward 3D sinh ra.
- [VGGT-Segmentor (CVPR 2026)](https://arxiv.org/abs/2605.15195) — ứng dụng trực tiếp cho embodied perception.

## Tham khảo

- Wang et al., *Dust3R: Geometric 3D Vision Made Easy* (CVPR 2024, arXiv:2312.14132).
- Leroy et al., *Must3R: Multi-View 3D from a Single Transformer* (ECCV 2024).
- Weinzaepfel et al., *VGGT: Visual Geometry Grounded Transformer* (CVPR 2025 Best Paper, arXiv:2503.11651).
- Weinzaepfel et al., *VGGT-Ω: Scaling Feed-Forward 3D Reconstruction* (CVPR 2026 Oral, arXiv:2605.15195).
