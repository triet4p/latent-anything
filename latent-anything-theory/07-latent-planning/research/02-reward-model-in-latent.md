# Reward Model trong Latent

> **TL;DR.** Reward model là một head $\hat r_\theta(z_t)\to\mathbb{R}$ đọc trực tiếp từ latent state, học có giám sát từ các cặp $(z,r)$ thật, rồi được gọi *bên trong* imagined rollout để planner ước lượng return mà không cần environment thật hay decoder. Công thức then chốt là $\hat J=\sum_t\gamma^t\hat r_\theta(\hat z_t)+\dots$ — toàn bộ planning đứng trên reward head này. Caveat lớn nhất: planner chủ động tìm action tối đa $\hat r_\theta$, nên reward head sai-mà-lạc-quan ở vùng out-of-distribution sẽ bị khai thác (reward hacking / overoptimization).

[Mục 1](01-model-based-vs-model-free-rl.md) đã nêu imagined return là số hạng trung tâm của planning trong latent. Số hạng đó có ba thành phần học được: reward $\hat r_\theta$, value $\hat v_\xi$ và continuation $\hat c_\theta$. Mục này đi vào thành phần đầu tiên — **reward model trong latent** — vì nếu planner đánh giá sai reward, mọi tầng planner phía trên (MPC, CEM, Dreamer) đều tối ưu sai mục tiêu.

---

## **1. Trực giác / Định nghĩa**

Trong một world model latent, agent không quan sát reward thật trong lúc tưởng tượng — nó chỉ có chuỗi latent state $\hat z_0,\hat z_1,\dots$ do [transition model](../../06-latent-temporal/research/02-latent-transition-model.md) sinh ra. Để chấm điểm một trajectory tưởng tượng, ta cần một hàm trả về reward *từ latent state*:

$$
\hat r_\theta:\;\mathcal{Z}\to\mathbb{R},\qquad \hat r_t=\hat r_\theta(\hat z_t).
$$

Trong đó $\mathcal{Z}$ là latent space và $\hat r_t$ là predicted reward tại bước $t$. Đây thường là một MLP nhỏ ("reward head") gắn lên latent state, song song với reward chỉ là một trong nhiều head (value, continuation, decoder) cùng đọc một latent.

Điểm mấu chốt là **decoder không nằm trong vòng lặp**: reward được đọc thẳng từ $z$, nên planner đánh giá hàng nghìn imagined step mà không bao giờ phải sinh lại observation. Đây chính là lý do reward model trong latent rẻ và là tiền đề cho planning hiệu quả.

---

## **2. Học reward head: giám sát từ transition thật**

Reward head được train *supervised*, tách khỏi vòng RL, trên các transition thật trong replay buffer:

$$
\mathcal{L}_r(\theta)=\mathbb{E}_{(o_{\le t},r_t)\sim\mathcal{D}}\big[\,\ell\big(\hat r_\theta(z_t),\,r_t\big)\,\big],\qquad z_t=\operatorname{Enc}_\phi(o_{\le t}).
$$

Trong đó $r_t$ là reward thật environment trả về, $z_t$ là latent encode từ observation thật, và $\ell$ là loss hồi quy. Vì target là reward đã quan sát, đây là supervised regression thuần — ổn định hơn nhiều so với học policy. Trong RSSM/Dreamer, reward head dùng *posterior* state (đã được observation hiệu chỉnh) để học, nhưng lúc planning lại được gọi trên *prior* imagined state; chênh lệch posterior–prior này là một nguồn sai số cần lưu ý.

### Reward của state hay của (state, action)?

Hai dạng tham số hóa thường gặp:

$$
\hat r_\theta(z_t)\qquad\text{so với}\qquad \hat r_\theta(z_t,a_t).
$$

Trong đó dạng đầu giả định reward xác định bởi state đến (Dreamer dùng $\hat r(z_t)$, gắn reward vào latent state sau transition), còn dạng sau cho phép reward phụ thuộc trực tiếp action (phổ biến trong control có action cost). Lựa chọn phải khớp với MDP gốc: nếu reward thật là $r(s_t,a_t)$ thì head chỉ-state buộc latent phải mã hóa đủ thông tin action gần nhất.

---

## **3. Mã hóa reward: regression vs distributional (symlog two-hot)**

Loss đơn giản nhất là MSE: $\ell=(\hat r-r)^2$. Nó hoạt động khi reward có scale đồng nhất, nhưng kém ổn định khi reward magnitude biến thiên lớn giữa các task/môi trường (sparse $+1$ ở Atari vs continuous dày đặc ở control).

DreamerV3 thay regression bằng **distributional reward head** với hai kỹ thuật:

1. **symlog transform** nén biên độ:

    $$
    \operatorname{symlog}(x)=\operatorname{sign}(x)\,\ln\!\big(1+|x|\big),\qquad \operatorname{symexp}(x)=\operatorname{sign}(x)\,\big(e^{|x|}-1\big).
    $$

    Trong đó $\operatorname{symexp}$ là nghịch đảo để giải mã về reward gốc. Phép nén này làm reward lớn không lấn át gradient mà vẫn giữ dấu và xử lý được giá trị âm — khác $\log$ thường.

2. **two-hot encoding**: reward (đã symlog) được biểu diễn thành phân phối trên một lưới $B$ bin cố định (DreamerV3 dùng 255 bin). Giá trị nằm giữa hai bin $b_k\le y< b_{k+1}$ được mã hóa thành vector "two-hot" đặt khối lượng lên đúng hai bin lân cận:

    $$
    w_k=\frac{b_{k+1}-y}{b_{k+1}-b_k},\qquad w_{k+1}=1-w_k,
    $$

    và head tối ưu cross-entropy với target two-hot này. Trong đó $w_k,w_{k+1}$ là trọng số sao cho kỳ vọng của phân phối two-hot bằng đúng $y$. Dự đoán reward = kỳ vọng phân phối, rồi $\operatorname{symexp}$ trả về scale gốc.

Lợi ích: cùng một cấu hình hyperparameter hoạt động trên dải reward magnitude rất rộng mà không cần normalize thủ công — yếu tố giúp DreamerV3 chạy "out-of-the-box" trên nhiều domain. Critic ([value function](https://github.com/triet4p/latent-anything/blob/main/docs/THEORY.md), mục tiếp theo) dùng đúng cơ chế symlog two-hot.

| | MSE regression | symlog two-hot (distributional) |
|---|---|---|
| Output | scalar | phân phối trên bin |
| Reward scale | nhạy, cần normalize | bất biến scale nhờ symlog |
| Sparse/spiky reward | dễ bị nuốt | giữ được nhờ cross-entropy |
| Độ phức tạp | tối thiểu | thêm bin + decode kỳ vọng |

---

## **4. Reward model trong vòng planning**

Khi đã có head, imagined return mà mọi planner tối ưu là:

$$
\hat J(a_{0:H-1}\mid z_0)=\sum_{t=0}^{H-1}\Big(\textstyle\prod_{j<t}\hat c_{j}\Big)\gamma^t\,\hat r_\theta(\hat z_t)+\Big(\textstyle\prod_{j<H}\hat c_j\Big)\gamma^H\,\hat v_\xi(\hat z_H),\qquad \hat z_{t+1}=\hat f_\theta(\hat z_t,a_t).
$$

Trong đó $\hat c_j$ là continuation probability ngắt đóng góp sau terminal, và value head bootstrap phần sau horizon. Toàn bộ tổng này chỉ gọi $\hat f_\theta,\hat r_\theta,\hat c_\theta,\hat v_\xi$ — không environment, không decoder. Reward head do đó là "thước đo" duy nhất cho phần reward trực tiếp; nếu nó lệch, không planner nào sửa được vì chúng tin vào chính nó.

Với stochastic transition, lưu ý $\hat r_\theta(\mathbb{E}[z])\ne\mathbb{E}[\hat r_\theta(z)]$ khi reward head phi tuyến — planning trên mean state có thể lệch khỏi expected reward, nên particle rollout đánh giá $\frac1P\sum_p\hat r_\theta(\hat z^{(p)})$ chính xác hơn (xem [Rollout §10](../../06-latent-temporal/research/07-rollout-latent-imagination.md)).

---

## **5. Giới hạn / Khi nào thất bại**

**Reward hacking / overoptimization.** Đây là failure mode nghiêm trọng nhất. Reward head chỉ là *proxy* cho reward thật; planner không lấy mẫu ngẫu nhiên mà chủ động $\arg\max_a\hat J$. Tối ưu một proxy cố định thường cải thiện theo proxy nhưng chỉ cải thiện theo reward thật trong một giai đoạn đầu, sau đó performance thật quay đầu giảm — đường cong overoptimization kinh điển. Càng search mạnh, càng dễ tìm ra "lỗ hổng" nơi head cho điểm cao sai.

**Out-of-distribution queries.** Reward head học trên replay states, nhưng planning kéo state vào vùng chưa thấy. Ngoài support của data, head có thể gán reward cao cho state tệ. Mitigation: uncertainty penalty / ensemble reward, ràng buộc state trong support ([density estimation](../../04-latent-computation/research/06-density-estimation.md)), pessimistic aggregation.

**Reward shortcut trong latent.** Encoder có thể nén bỏ chính factor cần để dự reward đúng ở task mới, chỉ giữ tín hiệu đủ khớp reward training. Latent "đủ tốt cho reward cũ" không đảm bảo transfer.

**Sparse reward.** Reward hiếm (một $+1$ ở cuối episode) làm head bias về 0; two-hot cross-entropy đỡ hơn MSE nhưng không xóa vấn đề tín hiệu thưa.

**Posterior–prior mismatch.** Head học trên posterior state nhưng dùng trên prior imagined state; calibration có thể trôi theo horizon.

---

## **6. Liên hệ với Latent-Anything**

Reward head là một thành phần của `ModelAdapter` và là điểm Layer A cần audit kỹ:

```python
class ModelAdapter(Protocol):
    def reward(self, z: np.ndarray) -> np.ndarray: ...        # r̂(z): không cần decode
    def continuation(self, z: np.ndarray) -> np.ndarray: ...  # ĉ(z): xác suất episode tiếp tục
```

- **Layer A — Introspection**: đo reward calibration theo horizon (predicted vs realized reward trên holdout), phát hiện vùng head over-confident, vẽ reward landscape $\hat r_\theta(z)$ để thấy planner đang bị kéo về đâu. Đây là tuyến phòng thủ chính chống reward hacking.
- **Layer B — Manipulation**: khi edit/branch trajectory, reward head cho phép chấm điểm lại candidate tức thì trong latent — nền cho các planner ở mục sau (MPC/CEM/MPPI).
- **Layer C — Runtime**: reward head phải batch được cùng candidate/particle/time; runtime nên cho phép gắn uncertainty penalty hoặc ensemble vào reward để giảm exploitation theo budget.

Mục tiếp theo, **Value function trong latent**, bổ sung số hạng bootstrap $\hat v_\xi(\hat z_H)$ — thứ cho phép rút ngắn horizon và giảm phụ thuộc reward dài hạn của reward head.

---

## Liên quan

- [Model-based vs Model-free RL](01-model-based-vs-model-free-rl.md) — đặt khung imagined return mà reward head phục vụ.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — nơi reward head được gọi, kèm continuation và return.
- [Latent Transition Model](../../06-latent-temporal/research/02-latent-transition-model.md) — sinh state mà reward head đọc.
- [RSSM — Recurrent State Space Model](../../06-latent-temporal/research/04-rssm-recurrent-state-space-model.md) — posterior/prior state và các head reward/value/continuation.
- [Density Estimation](../../04-latent-computation/research/06-density-estimation.md) — phát hiện state rời support, mitigation cho overoptimization.

## Tham khảo

- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, *Mastering Diverse Domains through World Models* (DreamerV3, 2023, arXiv:2301.04104).
- D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, *Dream to Control: Learning Behaviors by Latent Imagination* (ICLR 2020, arXiv:1912.01603).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (PlaNet, ICML 2019, arXiv:1811.04551).
- D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, D. Mané, *Concrete Problems in AI Safety* (2016, arXiv:1606.06565).
- L. Gao, J. Schulman, J. Hilton, *Scaling Laws for Reward Model Overoptimization* (ICML 2023, arXiv:2210.10760).
- J. Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero, Nature 2020, arXiv:1911.08265).
