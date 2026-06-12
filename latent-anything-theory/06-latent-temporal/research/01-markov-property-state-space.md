# Markov Property và State Space

> **TL;DR.** Một state đúng phải tóm tắt đủ quá khứ để phân phối của tương lai, sau khi đã biết action hiện tại, không còn phụ thuộc trực tiếp vào toàn bộ history: $p(s_{t+1}\mid s_{0:t},a_{0:t})=p(s_{t+1}\mid s_t,a_t)$. State-space model tách dynamics $p(s_{t+1}\mid s_t,a_t)$ khỏi observation model $p(o_t\mid s_t)$, nhờ đó có thể suy luận và rollout trên state ẩn thay vì pixel. Caveat chính là Markov property phụ thuộc vào cách chọn state: một observation đơn lẻ thường không Markov, còn latent học được chỉ gần Markov nếu nó giữ đủ thông tin dự báo và không alias các tình huống có tương lai khác nhau.

Từ Tầng 1 đến Tầng 5, latent chủ yếu được xem như biểu diễn của một observation hoặc một tập dữ liệu. Tầng 6 thêm trục thời gian: latent tại bước $t$ phải đóng vai trò **state** của một hệ động lực, nghĩa là nó không chỉ mô tả hiện tại mà còn phải mang đủ thông tin để dự báo điều gì có thể xảy ra tiếp theo.

Đây là bước chuyển quan trọng. Một embedding tốt cho reconstruction, clustering, hay probing chưa chắc là state tốt cho dynamics. Nếu hai history khác nhau bị encoder nén thành cùng một latent nhưng dẫn tới hai tương lai khác nhau, transition model không thể dự báo đúng cả hai. Markov property là tiêu chuẩn hình thức để phát hiện đúng vấn đề đó.

---

## **1. State không đồng nghĩa với observation**

Giả sử camera chỉ quan sát vị trí của một quả bóng tại từng frame. Hai frame có thể giống hệt nhau về vị trí, nhưng trong một history quả bóng đang đi sang trái, còn trong history kia nó đang đi sang phải. Observation hiện tại không chứa vận tốc, nên cùng một ảnh có thể dẫn tới hai frame kế tiếp khác nhau.

Một state vật lý hợp lý hơn là:

$$
s_t = (x_t, v_t),
$$

trong đó $x_t$ là vị trí và $v_t$ là vận tốc tại thời điểm $t$. Khi state chứa cả hai đại lượng, quy luật chuyển động có thể dự báo bước kế tiếp mà không cần đọc lại toàn bộ chuỗi vị trí trước đó.

Ví dụ này cho thấy ba khái niệm khác nhau:

| Khái niệm | Vai trò | Ví dụ quả bóng |
|---|---|---|
| **Observation $o_t$** | dữ liệu cảm biến nhìn thấy tại bước $t$ | ảnh hoặc vị trí $x_t$ |
| **History $h_t$** | toàn bộ thông tin đã quan sát và action đã thực hiện | $(o_0,a_0,\ldots,o_t)$ |
| **State $s_t$** | bản tóm tắt đủ của history cho dự báo và điều khiển | vị trí + vận tốc |

State không nhất thiết phải là biến vật lý có thể đo trực tiếp. Trong một state-space model, nó có thể là biến ẩn được suy ra từ observation history. Điều cốt lõi không phải state "trông có nghĩa" đến đâu, mà là nó có giữ đủ thông tin liên quan tới tương lai hay không.

---

## **2. Markov property**

### 2.1 Quá trình không có action

Một quá trình bậc nhất có Markov property nếu:

$$
p(s_{t+1}\mid s_0,s_1,\ldots,s_t)
=
p(s_{t+1}\mid s_t).
$$

Ở đây $s_{0:t}=(s_0,\ldots,s_t)$ là toàn bộ state history. Phương trình nói rằng sau khi đã biết state hiện tại $s_t$, các state xa hơn trong quá khứ không bổ sung thông tin để dự báo $s_{t+1}$.

Điều này **không** nói rằng quá khứ không ảnh hưởng tới tương lai. Ảnh hưởng của quá khứ đã được nén vào $s_t$. Nếu phép nén làm mất một biến còn ảnh hưởng tới tương lai, biến được gọi là "state" đó chưa thực sự Markov.

### 2.2 Quá trình có action

Trong control và reinforcement learning, transition còn phụ thuộc vào action:

$$
p(s_{t+1}\mid s_{0:t},a_{0:t})
=
p(s_{t+1}\mid s_t,a_t).
$$

Trong đó $a_{0:t}=(a_0,\ldots,a_t)$ là action history. Phương trình có nghĩa là phân phối của state kế tiếp chỉ cần state hiện tại và action vừa áp dụng; các state và action cũ không cần xuất hiện trực tiếp trong transition kernel.

Nếu reward cũng chỉ phụ thuộc vào hiện tại, có thể viết:

$$
p(r_t\mid s_{0:t},a_{0:t})
=
p(r_t\mid s_t,a_t),
$$

trong đó $r_t$ là reward tại bước $t$. Đây là điều kiện để bài toán điều khiển có thể dùng policy và value function trên state hiện tại thay vì trên toàn bộ history.

### 2.3 Markov property phụ thuộc vào biểu diễn

Cùng một hệ có thể Markov dưới một cách chọn state nhưng không Markov dưới cách chọn khác. Với hệ cơ học bậc hai, vị trí $x_t$ thường không đủ, còn cặp $(x_t,v_t)$ có thể đủ. Với delay điều khiển hai bước, state còn phải chứa action đang nằm trong hàng đợi.

Một quá trình Markov bậc $k$:

$$
p(x_{t+1}\mid x_{0:t})
=
p(x_{t+1}\mid x_{t-k+1:t})
$$

có thể chuyển thành Markov bậc nhất bằng state mở rộng:

$$
s_t=(x_{t-k+1},\ldots,x_t).
$$

Trong đó $x_{t-k+1:t}$ là cửa sổ $k$ bước gần nhất. Ý nghĩa là "bậc Markov" không phải thuộc tính bất biến của dữ liệu; nó phụ thuộc vào lượng history được đóng gói vào state.

---

## **3. State-space model**

State-space model tách một hệ tuần tự thành hai cơ chế:

1. **Transition model** mô tả state tiến hóa theo thời gian.
2. **Observation model** mô tả state sinh ra dữ liệu cảm biến như thế nào.

Với action, mô hình tổng quát là:

$$
s_{t+1} \sim p_\theta(s_{t+1}\mid s_t,a_t),
\qquad
o_t \sim p_\theta(o_t\mid s_t).
$$

Trong đó $\theta$ là tham số mô hình, $p_\theta(s_{t+1}\mid s_t,a_t)$ là dynamics, còn $p_\theta(o_t\mid s_t)$ là observation hoặc emission model. Hai phương trình tách "thế giới thay đổi ra sao" khỏi "cảm biến nhìn thế giới ra sao".

Phân phối chung của cả trajectory factorize thành:

$$
p_\theta(s_{0:T},o_{0:T}\mid a_{0:T-1})
=
p(s_0)\,p_\theta(o_0\mid s_0)
\prod_{t=0}^{T-1}
p_\theta(s_{t+1}\mid s_t,a_t)\,
p_\theta(o_{t+1}\mid s_{t+1}).
$$

Trong đó $s_{0:T}$ và $o_{0:T}$ lần lượt là state trajectory và observation trajectory; $p(s_0)$ là prior của state đầu; mỗi thừa số trong tích chỉ nối hai state kề nhau và observation tương ứng. Factorization này là hệ quả trực tiếp của Markov property và conditional independence của observation khi đã biết state.

Trường hợp tất định là một trường hợp riêng:

$$
s_{t+1}=f_\theta(s_t,a_t),
\qquad
\hat{o}_t=g_\theta(s_t).
$$

Trong đó $f_\theta$ là transition function và $g_\theta$ là decoder hoặc renderer. Khi môi trường có uncertainty, stochasticity, hay thông tin bị che khuất, dùng cả phân phối thường phù hợp hơn một dự báo điểm duy nhất.

### State-space tuyến tính cổ điển

Kalman đưa bài toán lọc tuyến tính về dạng state-space:

$$
s_{t+1}=A_ts_t+B_ta_t+w_t,
\qquad
o_t=C_ts_t+v_t.
$$

Trong đó $A_t$ là state-transition matrix, $B_t$ ánh xạ action vào state, $C_t$ ánh xạ state sang observation, còn $w_t$ và $v_t$ là process noise và observation noise. Dạng tuyến tính-Gaussian cho phép cập nhật posterior của state theo đệ quy; các latent state-space model hiện đại giữ nguyên cấu trúc transition/emission nhưng thay các ánh xạ tuyến tính bằng neural network.

---

## **4. Partial observability và belief state**

Markov property thường được giả định cho **state thật của môi trường**, không phải cho observation. Khi observation chỉ cho thấy một phần state, hệ trở thành partially observable:

$$
p(s_{t+1}\mid s_t,a_t)
\quad\text{là Markov, nhưng}\quad
p(o_{t+1}\mid o_t,a_t)
\quad\text{không nhất thiết Markov}.
$$

Ở đây state ẩn có dynamics bậc nhất, nhưng observation hiện tại có thể thiếu vận tốc, object bị occlude, biến nội tại của robot, hoặc context dài hạn. Vì thế một policy chỉ nhận $o_t$ có thể gặp state aliasing: cùng observation nhưng action tối ưu khác nhau.

Một cách chuẩn để khôi phục tính Markov là duy trì **belief state**:

$$
b_t(s)=p(s_t=s\mid o_{0:t},a_{0:t-1}).
$$

Trong đó $b_t$ là phân phối hậu nghiệm trên state thật sau khi quan sát history. Belief state tóm tắt uncertainty còn lại về thế giới, thay vì buộc agent chọn một state estimate duy nhất.

Belief được cập nhật đệ quy:

$$
b_{t+1}(s')
\propto
p(o_{t+1}\mid s')
\int p(s'\mid s,a_t)b_t(s)\,ds.
$$

Trong đó $s$ là state hiện tại, $s'$ là state kế tiếp, tích phân thực hiện bước prediction qua dynamics, còn $p(o_{t+1}\mid s')$ thực hiện bước correction bằng observation mới. Kết quả $b_{t+1}$ phụ thuộc vào belief trước, action hiện tại và observation mới, nên không cần lưu trực tiếp toàn bộ history.

Belief state chính xác thường là phân phối vô hạn chiều hoặc rất đắt để biểu diễn. Trong deep learning, một recurrent hidden state hay stochastic latent state thường đóng vai trò **xấp xỉ belief**:

$$
z_t = e_\phi(z_{t-1},a_{t-1},o_t).
$$

Trong đó encoder/filter $e_\phi$ cập nhật latent $z_t$ từ latent trước, action trước và observation mới. Mục tiêu là làm $z_t$ trở thành bản tóm tắt hữu hạn chiều đủ tốt cho prediction, reward và control.

---

## **5. Khi nào một latent state có thể xem là Markov?**

Một latent $z_t$ không trở thành Markov chỉ vì transition model được viết dưới dạng $p(z_{t+1}\mid z_t,a_t)$. Kiến trúc có thể giả định Markov, nhưng dữ liệu và encoder chưa chắc thỏa giả định đó.

Điều kiện trực giác là **predictive sufficiency**: sau khi biết $z_t$ và future actions, history cũ không được giúp dự báo các đại lượng liên quan tới tương lai tốt hơn đáng kể. Có thể viết mục tiêu lý tưởng:

$$
p(o_{t+1:T},r_{t:T}\mid h_t,a_{t:T-1})
\approx
p(o_{t+1:T},r_{t:T}\mid z_t,a_{t:T-1}),
$$

trong đó $h_t$ là observation-action history và $z_t=e_\phi(h_t)$ là latent state. Dấu xấp xỉ nhấn mạnh rằng latent học được hữu hạn chiều thường chỉ gần đủ, không phải sufficient statistic chính xác.

Có ít nhất ba mức "đủ" khác nhau:

| Mức state | Phải giữ thông tin gì? | Rủi ro |
|---|---|---|
| **Reconstruction-sufficient** | đủ để tái tạo observation hiện tại | có thể bỏ mất vận tốc hoặc biến ẩn chỉ ảnh hưởng tương lai |
| **Prediction-sufficient** | đủ để dự báo observation tương lai | có thể giữ texture không liên quan tới task |
| **Control-sufficient** | đủ để dự báo reward, termination và hệ quả của action | có thể không tái tạo đầy đủ pixel nhưng vẫn plan đúng |

Vì vậy reconstruction loss đơn lẻ không bảo đảm Markov state. PlaNet và các latent dynamics model sau đó học transition trực tiếp trong latent, dùng multi-step prediction hoặc variational objectives để ép representation giữ thông tin cần cho rollout. DVBF cũng cho thấy việc backpropagate qua transition giúp latent embedding tuân theo state-space assumptions tốt hơn so với học representation tĩnh rồi mới fit dynamics.

### Các dấu hiệu thực nghiệm

Không có một test hữu hạn đơn giản chứng minh tuyệt đối Markov property cho latent học được, nhưng có thể audit bằng các phép kiểm tra sau:

1. **History ablation:** so predictor dùng $(z_t,a_t)$ với predictor được thêm history cũ. Nếu history cải thiện mạnh next-state hoặc reward prediction, $z_t$ còn thiếu state information.
2. **Multi-step rollout:** one-step error nhỏ nhưng error tăng nhanh qua nhiều bước thường báo hiệu state thiếu biến chậm, uncertainty, hoặc transition không ổn định.
3. **Alias search:** tìm các cặp history có $z_t$ gần nhau nhưng phân phối $o_{t+1}$ hay reward khác xa nhau. Đây là bằng chứng trực tiếp của state aliasing.
4. **Action-conditional test:** kiểm tra cùng latent dưới nhiều action. Một state có thể dự báo tốt trên behavior policy nhưng thất bại với action chưa được dữ liệu khám phá.
5. **Probe phần history còn sót:** train probe dự báo future target từ history residual sau khi đã điều kiện trên $z_t$. Tín hiệu còn lại cho thấy latent chưa phải bản tóm tắt đủ.

---

## **6. Biến thể của state representation**

| Representation | Cách mang history | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| **Observation hiện tại** | không mang history | rẻ, đơn giản | chỉ đúng khi observation đã fully observed và Markov |
| **Frame stack / cửa sổ hữu hạn** | ghép $k$ observation gần nhất | dễ implement, thêm vận tốc ngầm | không xử lý tốt dependency dài hoặc độ trễ thay đổi |
| **Deterministic recurrent state** | nén history vào hidden vector | cập nhật nhanh, compact | dễ quên uncertainty và trộn nhiều khả năng vào một vector |
| **Stochastic latent state** | biểu diễn posterior trên state ẩn | giữ multimodality và uncertainty tốt hơn | inference và training phức tạp hơn |
| **Belief state chính xác** | posterior đầy đủ trên state thật | sufficient statistic chuẩn cho POMDP | thường bất khả thi ở hệ lớn và liên tục |

Không có representation nào luôn tốt nhất. Frame stack có thể đủ cho Atari đơn giản; robot có contact, occlusion và sensor delay thường cần recurrent hoặc stochastic state. Structured representation như tập Gaussian 3D còn phải xử lý permutation, cardinality và correspondence, nhưng đổi lại giữ geometry rõ hơn vector dày đặc.

---

## **7. Giới hạn / Khi nào giả định Markov thất bại**

**State bị thiếu biến ẩn.** Camera không thấy vận tốc, friction, intention của agent khác, hay object nằm sau occlusion. Transition nhìn từ state thiếu sẽ có vẻ stochastic hoặc phụ thuộc history.

**Độ trễ và memory của hệ chưa được đưa vào state.** Actuator delay, hysteresis, vật liệu biến dạng, hay queue trong hệ thống phần mềm làm tương lai phụ thuộc các input cũ. Cần augment state bằng delay buffer hoặc biến nội tại.

**Môi trường không stationary.** Nếu dynamics thay đổi theo thời gian do wear, domain shift, hay đối thủ đang học, cùng $(s_t,a_t)$ có thể sinh transition khác nhau ở các giai đoạn khác nhau. Time index hoặc context về regime phải trở thành một phần state.

**Latent aliasing.** Encoder tối ưu reconstruction hoặc compression có thể gộp hai history nhìn giống nhau nhưng khác dynamics. Transition model sau đó buộc phải average hai tương lai, thường tạo prediction mờ hoặc uncertainty giả.

**Một bước đúng không bảo đảm rollout đúng.** Teacher forcing cho transition luôn nhận latent từ encoder, trong khi rollout nhận latent do chính model dự báo. Distribution shift này có thể làm lỗi tích lũy dù one-step likelihood tốt.

**Finite-dimensional state có thể không đủ.** Với một số hệ partially observed phức tạp, sufficient statistic chính xác là cả một belief distribution. Ép nó vào vector nhỏ tạo trade-off không tránh khỏi giữa compression và predictive sufficiency.

**Markov không đồng nghĩa causal correctness.** Một latent có thể dự báo tốt trên dữ liệu quan sát nhưng học correlation phụ thuộc behavior policy. Nếu chưa thấy đủ intervention theo action, transition learned được có thể thất bại khi planning ra khỏi data distribution.

---

## **8. Liên hệ với Latent-Anything**

Markov property đặt contract đầu tiên cho `Trajectory` và các world-model adapter:

- `Trajectory[t]` phải đại diện cho state tại bước $t$, không chỉ là embedding của frame $t$.
- `ModelAdapter.encode_state(...)` có thể cần observation-action history hoặc recurrent cache; interface không nên mặc định `state = encode(observation)`.
- `LatentSpace` cần metadata mô tả state là deterministic vector, stochastic distribution, structured set, hay belief approximation.
- Transition method phải khai báo action conditioning, Markov order, observation model và uncertainty semantics.
- Layer A cần diagnostic cho state aliasing, residual history information và multi-step rollout drift.
- Layer B chỉ nên rollout hoặc plan trên latent sau khi state sufficiency đã được kiểm tra trên action distribution liên quan.

Với [Gaussian parameters là latent variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md), state có geometry tường minh nhưng chưa tự động Markov: Gaussian set hiện tại vẫn có thể thiếu velocity, occluded geometry hoặc correspondence qua thời gian. Dynamic adapter có thể phải bổ sung motion features, recurrent memory hay posterior uncertainty vào mỗi state.

Khái niệm này cũng nối trực tiếp tới các mục còn lại của Tầng 6:

- **Latent transition model** hiện thực hóa $p(z_{t+1}\mid z_t,a_t)$.
- **Stochastic transition** biểu diễn uncertainty thay vì average nhiều tương lai.
- **RSSM** kết hợp recurrent deterministic memory với stochastic state.
- **Kalman filter và variants** thực hiện recursive belief update trong các state-space model có cấu trúc.
- **Latent trajectory** là chuỗi state Markov hoặc gần Markov để smoothing, compare và rollout.

---

## Liên quan

- [Gaussian Parameters là Latent Variable](../../03b-3d-representation/research/10-gaussian-parameters-latent-variable.md) — ví dụ structured latent có thể trở thành state của world model, nhưng cần thêm dynamics và memory để đạt tính Markov.
- [Information Bottleneck](../../02-representation-learning/research/01-information-bottleneck.md) — state learning cũng là bài toán nén history: bỏ nuisance nhưng giữ thông tin cần cho prediction và control.
- [Causal Intervention vs Observational Study](../../05-probing-intervention/research/04-causal-intervention-vs-observational.md) — transition dùng cho planning phải đúng dưới can thiệp bằng action, không chỉ fit correlation trên trajectory quan sát.
- [Optimal Transport trong Latent Space](../../04-latent-computation/research/07-optimal-transport-in-latent.md) — có thể so state-occupancy distribution giữa trajectory, nhưng OT tĩnh không kiểm tra được Markov property hay thứ tự thời gian.

## Tham khảo

- R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems* (Transactions of the ASME--Journal of Basic Engineering 1960, 82:35--45, DOI: 10.1115/1.3662552).
- K. J. Åström, *Optimal Control of Markov Processes with Incomplete State Information* (Journal of Mathematical Analysis and Applications 1965, 10(1):174--205, DOI: 10.1016/0022-247X(65)90154-X).
- M. Karl, M. Soelch, J. Bayer, P. van der Smagt, *Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data* (ICLR 2017, arXiv:1605.06432).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (ICML 2019, arXiv:1811.04551).
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
