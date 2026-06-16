# Spherical Harmonics — nền toán: basis của hàm trên mặt cầu

> **TL;DR.** Spherical harmonics $Y_l^m$ là **basis trực chuẩn** của không gian các hàm bình phương khả tích trên mặt cầu $S^2$, sinh ra như **hàm riêng của toán tử Laplace–Beltrami** với trị riêng $-l(l+1)$ — chúng là "Fourier của mặt cầu". Hai định lý làm chúng hữu dụng: *addition theorem* $\sum_{m} Y_l^m(\mathbf{x})\overline{Y_l^m(\mathbf{y})} = \tfrac{2l+1}{4\pi}P_l(\mathbf{x}\!\cdot\!\mathbf{y})$ gói cả band $l$ thành một đại lượng **bất biến quay**, và *Funk–Hecke* nói mọi tích chập với kernel phụ thuộc góc trên $S^2$ đều **chéo hoá** trên basis này. Caveat: vì là basis toàn cục, cắt ngắn ở degree $L$ gây ringing kiểu Gibbs và số hệ số tăng bậc hai $(L+1)^2$.

Note này là **nền toán** của spherical harmonics, tách khỏi [note ứng dụng 3DGS ở tầng 3B](../../03b-3d-representation/research/08-spherical-harmonics.md) (vốn tập trung vào view-dependent color encoding). Ở đây ta hỏi câu gốc hơn: *vì sao* SH là basis tự nhiên cho hàm trên mặt cầu, chúng thừa hưởng tính chất gì từ nhóm quay $SO(3)$, và vì sao hai định lý addition + Funk–Hecke là lý do mọi nơi cần "phân tích tín hiệu theo hướng" đều dùng tới chúng.

---

## **1. Trực giác / Định nghĩa**

Fourier series phân tích một hàm tuần hoàn 1D thành tổng các sin/cos — và lý do sâu xa là sin/cos chính là **hàm riêng của toán tử đạo hàm bậc hai** $\tfrac{d^2}{dx^2}$ trên đường tròn. Spherical harmonics là phép loại suy *chính xác* của ý đó lên mặt cầu: chúng là hàm riêng của "Laplacian trên mặt cầu".

Cụ thể, mọi hàm $f \in L^2(S^2)$ (bình phương khả tích trên mặt cầu) khai triển được:

$$ f(\theta,\phi) = \sum_{l=0}^{\infty}\sum_{m=-l}^{l} a_{lm}\, Y_l^m(\theta,\phi) $$

trong đó $(\theta,\phi)$ là hai góc cầu (cực và phương vị), $l \ge 0$ là **degree** (band), $m \in \{-l,\dots,l\}$ là **order**, còn $a_{lm}$ là hệ số khai triển. Ý nghĩa: thay vì lưu một hàm liên tục trên vô hạn hướng, ta lưu một dãy số $a_{lm}$ — đúng tinh thần "đổi sang miền tần số" của Fourier, nhưng tần số ở đây là *tần số góc trên mặt cầu*.

Hai sự thật đếm cần nhớ:

- Với mỗi degree $l$ cố định có đúng $2l+1$ basis (các order $m$).
- Giữ mọi band tới $L$ cho tổng số basis:

$$ \sum_{l=0}^{L}(2l+1) = (L+1)^2 $$

trong đó $L$ là degree cắt ngắn cao nhất. Số chiều của không gian xấp xỉ tăng **bậc hai** theo $L$ — chi tiết này quay lại ám ảnh ở phần giới hạn.

## **2. Cơ chế / Công thức**

### 2.1. SH là hàm riêng của Laplace–Beltrami

Toán tử Laplace–Beltrami $\Delta_{S^2}$ là phiên bản Laplacian "nội tại" trên mặt cầu (phần góc của Laplacian 3D). Spherical harmonics được *định nghĩa* như nghiệm riêng của nó:

$$ \Delta_{S^2}\, Y_l^m = -\,l(l+1)\, Y_l^m $$

trong đó trị riêng $-l(l+1)$ chỉ phụ thuộc degree $l$, **không** phụ thuộc order $m$. Đây là gốc rễ của mọi tính chất: vì $\Delta_{S^2}$ giao hoán với phép quay, các không gian riêng của nó (mỗi band $l$, chiều $2l+1$) là **bất biến dưới $SO(3)$** — quay một SH bậc $l$ luôn cho một tổ hợp tuyến tính của các SH cùng bậc $l$. Trị riêng $-l(l+1)$ chính là "bình phương tần số góc": band cao dao động nhanh hơn theo hướng.

### 2.2. Dạng tường minh và liên hệ Legendre

Trên basis góc chuẩn, SH (phức) viết được qua **associated Legendre polynomials** $P_l^m$:

$$ Y_l^m(\theta,\phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}\; P_l^m(\cos\theta)\, e^{im\phi} $$

trong đó thừa số căn là hằng chuẩn hoá để basis **trực chuẩn**, $P_l^m(\cos\theta)$ nắm sự phụ thuộc vào góc cực $\theta$, và $e^{im\phi}$ là thành phần Fourier theo góc phương vị $\phi$. Tính trực chuẩn phát biểu gọn:

$$ \int_{S^2} Y_l^m(\mathbf{d})\,\overline{Y_{l'}^{m'}(\mathbf{d})}\; d\Omega = \delta_{ll'}\,\delta_{mm'} $$

nghĩa là tích trong $L^2(S^2)$ của hai basis khác nhau bằng 0, của một basis với chính nó bằng 1. Hệ quả: hệ số khai triển lấy bằng phép chiếu $a_{lm} = \int_{S^2} f(\mathbf{d})\,\overline{Y_l^m(\mathbf{d})}\,d\Omega$ — y hệt cách Fourier lấy hệ số bằng tích trong với sóng cơ sở.

### 2.3. Real spherical harmonics

SH phức tiện cho lý thuyết nhưng đồ hoạ/ML dùng **real SH** $Y_{lm}$ để tránh số phức. Chúng là tổ hợp tuyến tính của cặp $\pm m$:

$$
Y_{lm} =
\begin{cases}
\sqrt{2}\,(-1)^m\,\Im\big(Y_l^{|m|}\big) & m < 0\\[4pt]
Y_l^0 & m = 0\\[4pt]
\sqrt{2}\,(-1)^m\,\Re\big(Y_l^{m}\big) & m > 0
\end{cases}
$$

trong đó $\Re,\Im$ là phần thực/ảo. Real SH vẫn trực chuẩn và phủ đúng cùng không gian band — chỉ là chọn một basis thực thay vì phức. Đây chính là dạng được lưu trong các hệ số `features` của 3DGS.

### 2.4. Addition theorem — gói một band thành đại lượng bất biến quay

Định lý cộng (addition theorem) là kết quả trung tâm:

$$ \sum_{m=-l}^{l} Y_l^m(\mathbf{x})\,\overline{Y_l^m(\mathbf{y})} = \frac{2l+1}{4\pi}\, P_l(\mathbf{x}\cdot\mathbf{y}) $$

trong đó $\mathbf{x},\mathbf{y}\in S^2$ là hai hướng đơn vị, $P_l$ là **Legendre polynomial** bậc $l$ (một biến), và $\mathbf{x}\cdot\mathbf{y}=\cos\gamma$ là cosin góc giữa hai hướng. Ý nghĩa rất mạnh: tổng tích chéo trên toàn bộ một band **không phụ thuộc hệ toạ độ**, chỉ phụ thuộc góc tương đối $\gamma$. Đây là phiên bản mặt cầu của đẳng thức Fourier $\sum_m e^{im(x-y)}$, và là lý do toán học khiến SH "đóng kín" dưới phép quay.

### 2.5. Funk–Hecke — vì sao convolution trên mặt cầu chéo hoá

Định lý Funk–Hecke nói: với một kernel **chỉ phụ thuộc góc** $k(\mathbf{x}\cdot\mathbf{y})$, phép tích chập trên mặt cầu biến mỗi SH thành chính nó nhân một vô hướng:

$$ \int_{S^2} k(\mathbf{x}\cdot\mathbf{y})\, Y_l^m(\mathbf{y})\, d\Omega(\mathbf{y}) = \lambda_l\, Y_l^m(\mathbf{x}), \qquad \lambda_l = 2\pi\!\int_{-1}^{1} k(t)\,P_l(t)\,dt $$

trong đó $\lambda_l$ là **trị riêng chỉ phụ thuộc band** $l$ (không phụ thuộc $m$), tính bằng một tích phân 1D của kernel với Legendre $P_l$. Nói cách: SH **chéo hoá** mọi toán tử bất biến quay trên $S^2$, hệt như Fourier chéo hoá mọi tích chập bất biến tịnh tiến trên đường thẳng. Đây là nền toán cho spherical CNN và mọi filter "rotation-equivariant" trên dữ liệu hướng.

## **3. Biến thể / Trường hợp**

| | Fourier ($S^1$) | Spherical harmonics ($S^2$) |
|---|---|---|
| Miền | đường tròn / đoạn tuần hoàn | mặt cầu đơn vị |
| Basis | $e^{inx}$ | $Y_l^m(\theta,\phi)$ |
| Sinh ra từ | hàm riêng $\tfrac{d^2}{dx^2}$ | hàm riêng $\Delta_{S^2}$, trị riêng $-l(l+1)$ |
| Số basis tới bậc $L$ | $2L+1$ (tuyến tính) | $(L+1)^2$ (bậc hai) |
| Bất biến của | tịnh tiến | quay $SO(3)$ |
| "Convolution theorem" | nhân theo từng mode | Funk–Hecke: nhân theo từng band |

| | **Complex SH** $Y_l^m$ | **Real SH** $Y_{lm}$ |
|---|---|---|
| Giá trị | số phức | số thực |
| Dùng cho | lý thuyết, vật lý lượng tử | đồ hoạ, 3DGS, ML |
| Quan hệ | cơ sở gốc | tổ hợp thực của cặp $\pm m$ |

Tổng quát hơn, trên $S^{d-1}$ vẫn có SH với trị riêng $-l(l+d-2)$ và addition theorem dùng **Gegenbauer polynomials** thay cho Legendre — khung lý thuyết không đổi, chỉ đổi đa thức trực giao.

## **4. Giới hạn / Khi nào thất bại**

- **Ringing (Gibbs trên mặt cầu).** Vì mỗi $Y_l^m$ có giá đỡ **toàn cục** (phủ cả mặt cầu), xấp xỉ một hàm có biên sắc bằng hữu hạn band gây dao động giả quanh chỗ gián đoạn — đúng họ hàng của hiện tượng Gibbs trong Fourier truncation. SH dở với tín hiệu sắc (specular hẹp, cạnh cứng).
- **Bùng nổ bậc hai.** Tăng $L$ để bắt chi tiết góc làm số hệ số tăng $(L+1)^2$: bộ nhớ và chi phí evaluate đều phình. Vì thế thực tế hiếm khi vượt $L=3\!-\!4$ cho appearance.
- **Không cục bộ hoá được.** Muốn "chỉnh màu ở một hướng nhỏ" mà không động tới hướng khác là khó, vì basis trải khắp cầu — ngược hẳn với wavelet/spherical-Gaussian vốn có giá đỡ cục bộ.
- **Nhạy convention chuẩn hoá.** Có nhiều quy ước thừa số ($4\pi$, Condon–Shortley $(-1)^m$, Schmidt semi-normalized…); trộn hai quy ước làm hệ số sai mà không báo lỗi.

## **5. Liên hệ với Latent-Anything**

Note này là **tầng móng toán** đỡ cho nhiều thứ phía trên trong framework:

- **Đỡ trực tiếp cho appearance latent của 3DGS.** [Note SH ở tầng 3B](../../03b-3d-representation/research/08-spherical-harmonics.md) dùng đúng khai triển $\mathbf{c}(\mathbf{d})=\sum_{lm}\mathbf{c}_{lm}Y_l^m(\mathbf{d})$ để gắn hàm màu theo hướng cho mỗi Gaussian; các sự thật "16 basis ở `sh_degree=3`", "band 0 là DC term", "ringing với specular" đều là hệ quả trực tiếp của $(L+1)^2$, trực chuẩn, và giá đỡ toàn cục ở đây.
- **Bất biến/đẳng biến quay là tài sản cho 3D latent.** Addition theorem và Funk–Hecke cho ta cách xây *đặc trưng bất biến quay* (năng lượng mỗi band $\sum_m |a_{lm}|^2$ không đổi khi quay scene) và *toán tử đẳng biến quay*. Điều này bổ sung góc nhìn cho [Lie groups](05-lie-groups-and-lie-algebra.md): Lie group lo *tham số hoá* phép quay của Gaussian/pose, còn SH lo *phân tích tín hiệu* sống trên hướng — hai mảnh ghép của cùng câu chuyện $SO(3)$.
- **Một dạng basis expansion để introspection.** Như [Positional Encoding](../../03b-3d-representation/research/04-positional-encoding.md) khai triển toạ độ không gian, SH khai triển hướng. Khi latent của một world model chứa hệ số SH, Layer A có thể *probe theo band*: band thấp = màu trung bình, band cao = mức độ phụ thuộc góc nhìn — một trục diễn giải sạch nhờ tính trực chuẩn.

Tóm lại, SH cho Latent-Anything một công cụ chuẩn để **biểu diễn, nén, và phân rã mọi đại lượng phụ thuộc hướng** trong latent — và quan trọng hơn, một đảm bảo hình học (bất biến quay) mà các basis ad-hoc không có.

---

## Liên quan

- [Spherical Harmonics trong 3DGS (tầng 3B)](../../03b-3d-representation/research/08-spherical-harmonics.md) — ứng dụng trực tiếp: view-dependent color encoding; note hiện tại là phần toán nền của nó.
- [Lie groups và Lie algebra](05-lie-groups-and-lie-algebra.md) — cùng xoay quanh $SO(3)$: Lie lo tham số hoá phép quay, SH lo phân tích hàm trên mặt cầu.
- [Positional Encoding](../../03b-3d-representation/research/04-positional-encoding.md) — cùng là basis expansion, nhưng trên toạ độ không gian thay vì hướng.
- [Hình học Riemannian](../../03-geometry-structure/research/04-riemannian-geometry.md) — Laplace–Beltrami là toán tử Riemannian sinh ra SH.

## Tham khảo

- Claus Müller, *Spherical Harmonics* (Lecture Notes in Mathematics 17, Springer 1966).
- Kendall Atkinson, Weimin Han, *Spherical Harmonics and Approximations on the Unit Sphere: An Introduction* (Springer Lecture Notes in Mathematics 2044, 2012).
- Feng Dai, Yuan Xu, *Approximation Theory and Harmonic Analysis on Spheres and Balls* (Springer 2013) — addition theorem, Funk–Hecke.
- Ravi Ramamoorthi, Pat Hanrahan, *An Efficient Representation for Irradiance Environment Maps* (SIGGRAPH 2001).
- Taco Cohen, Mario Geiger, Jonas Köhler, Max Welling, *Spherical CNNs* (ICLR 2018, arXiv:1801.10130) — Funk–Hecke và đẳng biến $SO(3)$.
- Peter-Pike Sloan, *Stupid Spherical Harmonics (SH) Tricks* (GDC 2008) — real SH thực hành.
