# Gaussian Rasterization — decoder GPU cho tập Gaussian 3D

> **TL;DR.** Gaussian rasterization là bước biến tập Gaussian 3D đã có geometry và appearance thành ảnh 2D bằng pipeline `project -> bound -> bin by tile -> sort by depth -> alpha composite`, thay vì ray marching như [NeRF](02-nerf.md). Công thức cốt lõi ở mức pixel vẫn là $\alpha_i(p)=o_i\exp\!\big(-\tfrac12(p-\mu_i')^\top{\Sigma_i'}^{-1}(p-\mu_i')\big)$ và $C(p)=\sum_i T_i(p)\alpha_i(p)c_i(p)$, nhưng điểm quyết định tốc độ nằm ở việc chỉ rasterize những Gaussian thực sự chạm vào từng tile màn hình. Đổi lại, đây vẫn là một xấp xỉ visibility bằng alpha blending, nên sorting, footprint quá to, và overlap dày đặc đều có thể tạo artifact.

Sau [Covariance Matrix trong 3DGS](07-covariance-matrix-3dgs.md) và [Spherical Harmonics](08-spherical-harmonics.md), mọi thành phần của một Gaussian gần như đã đủ: vị trí, shape, opacity, và màu theo hướng nhìn. Gaussian rasterization là mảnh ghép cuối để biến các primitive đó thành ảnh. Nếu coi tập Gaussian là latent state của một world model, thì rasterizer chính là **decoder gần-deterministic**: từ latent set sang observation mà không cần một mạng lớn ở bước cuối.

---

## **1. Trực giác: không truy vấn trường dọc tia, mà vẽ trực tiếp primitive lên ảnh**

Trong [Volume Rendering & Ray Marching](03-volume-rendering-ray-marching.md), mỗi pixel đòi hỏi lấy nhiều mẫu dọc theo một tia 3D rồi cộng tích phân. Gaussian rasterization đảo góc nhìn đó: scene đã được lưu thành một tập primitive explicit, nên thay vì hỏi "dọc tia này có gì?", ta hỏi "Gaussian nào thực sự ảnh hưởng tới tile/pixel này?".

Một Gaussian sau khi chiếu xuống camera trở thành một ellipse mềm trên image plane. Với Gaussian thứ $i$, ta có:

$$
G_i = (\mu_i, \Sigma_i, o_i, \mathbf{a}_i)
\;\longrightarrow\;
(\mu_i', \Sigma_i', o_i, c_i(\mathbf{d}))
$$

trong đó $\mu_i'$ là tâm ảnh chiếu 2D, $\Sigma_i'$ là covariance 2D của ellipse trên màn hình, $o_i$ là opacity, và $c_i(\mathbf{d})$ là màu nhìn theo hướng camera hiện tại, thường được lấy từ [Spherical Harmonics](08-spherical-harmonics.md). Biểu thức này có nghĩa là Gaussian 3D không được render như một point sprite tròn, mà như một footprint ellipse có hướng, có độ mờ và có màu phụ thuộc góc nhìn.

Vì vậy, Gaussian rasterization có thể được hiểu như một dạng point-based rendering hiện đại:

- vẫn giữ tinh thần splatting của họ Surface / EWA Splatting;
- nhưng primitive là Gaussian 3D dị hướng học được;
- và toàn bộ pipeline được thiết kế để khả vi, nên tối ưu ngược từ ảnh vẫn đi được.

---

## **2. Preprocess: từ Gaussian 3D sang footprint 2D khả rasterize**

Implementation chính thức của `diff-gaussian-rasterization` bắt đầu bằng một bước preprocess cho từng Gaussian:

1. frustum culling;
2. project mean 3D xuống ảnh;
3. đẩy covariance 3D qua phép chiếu để lấy covariance 2D;
4. suy ra bán kính bao ngoài trên màn hình và tile rectangle bị chạm.

Về mặt toán học, covariance trên ảnh vẫn đi theo công thức:

$$
\Sigma_i' = J_i W \Sigma_i W^\top J_i^\top
$$

trong đó $W$ là phép biến đổi world-to-camera và $J_i$ là Jacobian của phép chiếu phối cảnh tại Gaussian thứ $i$. Công thức này nói rằng ellipsoid 3D được tuyến tính hóa cục bộ thành một ellipse 2D tương ứng với góc nhìn hiện tại.

Ở implementation, rasterizer không giữ toàn bộ $\Sigma_i'$ dưới dạng ma trận đầy đủ trong bước sau, mà chuyển nó sang dạng **conic** bằng cách nghịch đảo covariance 2D:

$$
Q_i = {\Sigma_i'}^{-1}
=
\begin{bmatrix}
a_i & b_i\\
b_i & c_i
\end{bmatrix}
$$

trong đó $(a_i,b_i,c_i)$ là ba phần tử độc lập của ma trận đối xứng $2\times 2$. Biểu thức này có nghĩa là Gaussian 2D sau đó có thể được evaluate nhanh bằng một quadratic form tại từng pixel mà không cần nghịch đảo lại.

Code chính thức còn làm thêm hai việc rất thực dụng:

- loại Gaussian có determinant bằng `0` sau phép chiếu, vì footprint khi đó không khả dụng về mặt số;
- cộng một low-pass nhỏ vào đường chéo covariance 2D để mỗi Gaussian ít nhất rộng cỡ một pixel, giúp chống aliasing và tránh footprint quá sắc.

Sau đó, bán kính bao ngoài trên ảnh được lấy từ trị riêng lớn nhất của covariance 2D, rồi dùng để tạo một **bounding rectangle** theo tile. Đây là chỗ quan trọng: Gaussian không được quét qua mọi pixel toàn màn hình, mà chỉ qua những tile mà footprint của nó có thể chạm.

---

## **3. Tile-based binning: biến một scene thành các danh sách việc theo tile**

Nếu sau bước projection ta vẫn để "mỗi pixel tự kiểm tra mọi Gaussian", chi phí sẽ bùng nổ. Rasterizer của 3DGS né điều đó bằng cách tổ chức workload theo tile màn hình.

Ý tưởng là:

- màn hình được chia thành lưới tile cỡ cố định;
- mỗi Gaussian biết rectangle tile nào nó overlap;
- Gaussian được **duplicate theo số tile mà nó chạm vào**;
- mỗi duplicate mang khóa `(tile_id, depth)` để còn sort.

Trong mã chính thức, kernel `duplicateWithKeys` tạo key-value cho mọi overlap Gaussian/tile. Key được pack theo dạng:

$$
\text{key} = (\text{tile\_id}, \text{depth})
$$

trong đó `tile_id` quyết định Gaussian thuộc workload của tile nào, còn `depth` quyết định thứ tự front-to-back trong tile đó. Ý nghĩa của cách pack này là: chỉ cần radix sort một lần trên toàn bộ danh sách duplicate, ta sẽ thu được các Gaussian **đã được nhóm theo tile và đã được sắp theo độ sâu trong từng tile**.

Sau sort, một kernel khác xác định khoảng chỉ số `[start, end)` của từng tile trong danh sách đã sắp. Khi đó, mỗi tile có thể được xử lý độc lập như một "mini render queue" riêng.

Đây là khác biệt rất lớn so với một rasterizer ngây thơ:

| | Rasterize ngây thơ | Gaussian rasterization theo tile |
|---|---|---|
| Pixel cần kiểm tra | gần như mọi Gaussian | chỉ Gaussian nằm trong tile của nó |
| Sắp xếp visibility | khó tổ chức | đã gói sẵn trong key `(tile, depth)` |
| Tận dụng GPU | kém locality | tốt hơn nhờ tile-local batches |
| Chi phí ở vùng rỗng | vẫn dễ lãng phí | gần như không có Gaussian thì tile gần như rỗng |

Nói ngắn gọn, tile binning biến bài toán từ "toàn ảnh đối toàn scene" thành "tile nhỏ đối danh sách Gaussian cục bộ".

---

## **4. Per-tile rasterization: mỗi pixel tích lũy màu theo transmittance**

Sau khi có danh sách Gaussian đã sort cho từng tile, kernel render chính chạy theo kiểu:

- một block xử lý một tile;
- mỗi thread phụ trách một pixel trong tile;
- cả block cùng fetch Gaussian theo batch vào shared memory;
- mỗi pixel đi qua danh sách Gaussian theo thứ tự gần camera.

Với pixel $p$, footprint 2D của Gaussian $i$ được evaluate bằng quadratic form:

$$
e_i(p)
=
\frac{1}{2}(p-\mu_i')^\top Q_i (p-\mu_i')
$$

trong đó $Q_i={\Sigma_i'}^{-1}$ là conic inverse, còn $(p-\mu_i')$ là vector từ tâm Gaussian tới pixel đang xét. Biểu thức này đo pixel nằm sâu bao nhiêu trong ellipse Gaussian: càng xa tâm theo metric của covariance thì giá trị càng lớn.

Từ đó, opacity hiệu dụng của Gaussian tại pixel là:

$$
\alpha_i(p) = o_i \exp\!\big(-e_i(p)\big)
$$

trong đó $o_i$ là opacity học được còn số mũ Gaussian làm alpha rơi dần khỏi tâm. Trong implementation, alpha còn bị chặn trên để tránh bất ổn số và bị bỏ qua nếu quá nhỏ để không tốn compute cho đóng góp không đáng kể.

Màu cuối cùng được cộng theo front-to-back alpha compositing:

$$
C(p)=\sum_i T_i(p)\,\alpha_i(p)\,c_i(p),
\qquad
T_i(p)=\prod_{j<i}(1-\alpha_j(p))
$$

trong đó $c_i(p)$ là màu của Gaussian tại hướng nhìn hiện tại và $T_i(p)$ là transmittance còn sót lại trước khi gặp Gaussian thứ $i$. Công thức này có nghĩa là Gaussian phía trước vừa đóng góp màu, vừa che bớt cơ hội đóng góp của Gaussian phía sau.

Implementation chính thức còn dùng **early termination**:

- nếu transmittance $T$ của pixel đã rất nhỏ, pixel đó gần như opaque;
- thread có thể dừng sớm thay vì tiếp tục đọc toàn bộ Gaussian còn lại trong tile.

Đây là một nguồn tăng tốc quan trọng trong các vùng đã bị che phủ gần hết.

---

## **5. Vì sao rasterization này nhanh hơn ray marching**

So với [NeRF](02-nerf.md) hay [Instant-NGP](05-instant-ngp.md), tốc độ của Gaussian rasterization đến từ ba chỗ cộng hưởng:

1. **Không march trong không gian rỗng.**  
   Gaussian nào không chạm tile thì tile đó không cần quan tâm tới nó.

2. **Không cần query MLP hàng trăm lần mỗi pixel.**  
   Geometry đã explicit; appearance ở 3DGS gốc cũng là closed-form từ SH chứ không phải một mạng nặng.

3. **Workload được tổ chức để hợp GPU.**  
   Tile-local batching, radix sort theo key, và shared-memory fetch giúp locality tốt hơn nhiều so với sampling dọc tia.

Về khái niệm, có thể xem Gaussian rasterization là điểm giữa của hai thế giới:

- giữ compositing / transmittance giống volume rendering;
- nhưng tổ chức compute như rasterization primitive-based của đồ họa thời gian thực.

Đó là lý do paper gốc gọi đây là một **visibility-aware rendering algorithm**: không chỉ vẽ Gaussian, mà còn vẽ theo cách awareness với overlap và depth để vừa nhanh vừa đủ đúng cho novel-view synthesis.

---

## **6. Giới hạn / Khi nào rasterizer thất bại**

- **Sorting chỉ là xấp xỉ visibility.** Alpha blending front-to-back không phải một mô hình quang học đầy đủ cho mọi loại transparency hay multiple scattering.
- **Footprint quá to gây overdraw mạnh.** Một Gaussian phình lớn có thể chạm rất nhiều tile, làm số duplicate tăng nhanh và chi phí sort/rasterize tăng theo.
- **Overlap dày đặc làm tile nặng.** Khi quá nhiều Gaussian cùng chạm một tile, lợi ích locality vẫn còn nhưng chi phí blend trong tile tăng rõ rệt.
- **Sai depth hoặc sai covariance gây artifact ngay trên ảnh.** Thứ tự compositing sai dễ tạo popping, halo hoặc occlusion lỗi.
- **Phụ thuộc mạnh vào preprocess hợp lệ.** Covariance 2D suy biến, radius ước lượng lệch, hay culling sai đều khiến Gaussian biến mất hoặc lan sai vùng.
- **Không tương đương hoàn toàn với render vật lý.** Đây là decoder thực dụng, differentiable và rất nhanh, chứ không phải một renderer physically exact.

Nói cách khác, Gaussian rasterization mạnh vì nó biết "đủ đúng để tối ưu và hiển thị nhanh", chứ không phải vì nó mô phỏng ánh sáng một cách hoàn toàn trung thực.

---

## **7. Liên hệ với Latent-Anything**

Nếu Gaussian set là latent state, thì Gaussian rasterization cho thấy một dạng decoder rất khác so với decoder neural thường thấy:

- đầu vào là một **set structured latent tokens**;
- đầu ra là ảnh 2D;
- bản thân decoder gần như không cần học, chủ yếu là hình học + compositing.

Điều này cực kỳ hợp với Latent-Anything:

- Layer A có thể inspect trực tiếp Gaussian nào chạm tile nào, Gaussian nào đóng góp màu nhiều nhất cho từng vùng ảnh.
- Layer B có thể manipulate latent set rồi đo hiệu ứng render rất rõ ràng, gần như tức thời.
- Layer C có thể dùng rasterizer như observation head deterministic cho world model kiểu LeWM.

Quan trọng hơn, Gaussian rasterization biến latent space từ một vector khó diễn giải thành một pipeline có thể audit từng bước:

1. primitive nào còn sống sau culling,
2. primitive nào chạm vùng ảnh nào,
3. primitive nào thực sự thắng trong alpha compositing.

Đó là đúng tinh thần "latent as a first-class object": không chỉ lưu state, mà còn thấy được state đó đi tới observation như thế nào.

Note này cũng đặt nền trực tiếp cho **Gaussian parameters là latent variable** ở mục tiếp theo: khi encoder xuất ra một Gaussian set và decoder chỉ là rasterizer, latent space của world model trở nên explicit hơn hẳn latent vector thuần túy.

---

## Liên quan

- [3D Gaussian Splatting](06-3d-gaussian-splatting.md) — Gaussian rasterization là nửa "decoder" cụ thể của 3DGS.
- [Covariance Matrix trong 3DGS](07-covariance-matrix-3dgs.md) — rasterizer dùng covariance đã chiếu để tạo ellipse và conic trên image plane.
- [Spherical Harmonics](08-spherical-harmonics.md) — rasterizer cần màu theo hướng nhìn để blend từng Gaussian đúng cách.
- [Volume Rendering & Ray Marching](03-volume-rendering-ray-marching.md) — cùng dùng transmittance/alpha compositing, nhưng Gaussian rasterization tránh sampling dọc tia dày đặc.
- [NeRF](02-nerf.md) — đối chiếu trực tiếp nhất giữa decoder implicit theo ray và decoder explicit theo primitive.

## Tham khảo

- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (ACM Transactions on Graphics 2023, arXiv:2308.04079).
- Graphdeco-Inria, *diff-gaussian-rasterization* official implementation (GitHub repository; `preprocessCUDA`, `duplicateWithKeys`, `identifyTileRanges`, `renderCUDA`).
- Zwicker, Pfister, van Baar, Gross, *Surface Splatting* (SIGGRAPH 2001).
- Zwicker, Pfister, van Baar, Gross, *EWA Splatting* (IEEE TVCG 2002).
