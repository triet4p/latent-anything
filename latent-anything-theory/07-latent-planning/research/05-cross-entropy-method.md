# Cross-Entropy Method (CEM)

> **TL;DR.** CEM là optimizer population-based giải bước $\arg\max_{a_{0:H-1}}\hat J$ bên trong MPC: giữ một phân phối Gaussian trên action sequence, lặp lại "sample → chấm điểm → giữ elite → refit phân phối về elite" cho tới khi nó co cụm quanh nghiệm tốt. Refit chính là cực tiểu KL/cross-entropy giữa phân phối tham số và empirical của elite, tức MLE trên elite. Caveat: tốn nhiều lần gọi model mỗi bước, và phương sai dễ collapse sớm → kẹt local optimum nếu elite fraction/scheduling sai.

[Mục 4](04-model-predictive-control.md) để ngỏ một câu hỏi: bước $\arg\max\hat J$ trong MPC giải bằng gì? Random shooting (sample đều rồi chọn tốt nhất) là baseline nhưng phí mẫu — phần lớn candidate ngẫu nhiên là rác. CEM thay "sample đều" bằng một phân phối *thích nghi*: nó học chỗ nào trong không gian action đáng sample hơn, qua từng iteration. PlaNet và PETS đều dùng CEM làm planner trong latent space.

---

## **1. Trực giác / Định nghĩa**

CEM ra đời (Rubinstein, 1997) như một thủ tục **adaptive importance sampling** để ước lượng xác suất *rare event*, dùng cross-entropy (KL divergence) làm thước đo gần nhau giữa hai phân phối. Mối nối với tối ưu: tìm nghiệm tốt nhất tương đương "lấy mẫu được một sự kiện hiếm" — sample rơi đúng vùng return cao. CEM dần *dịch chuyển* phân phối sample để sự kiện hiếm đó trở nên thường xuyên.

Trực giác trong planning: bắt đầu bằng một đám mây action sequence rộng (chưa biết gì), chấm điểm tất cả, giữ lại nhúm tốt nhất ("elite"), rồi *vẽ lại* đám mây bám quanh nhúm elite đó. Lặp vài lần, đám mây co về vùng action tốt. Đây là tìm kiếm ngẫu nhiên *có học*.

---

## **2. Thuật toán**

Phân phối thường là Gaussian đường chéo trên toàn chuỗi action $\mathbf{a}=a_{0:H-1}\in\mathbb{R}^{H\cdot d_a}$:

$$
q_\eta(\mathbf{a})=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2)).
$$

Trong đó $\mu$ là chuỗi action trung bình và $\sigma^2$ là phương sai mỗi chiều. Mỗi iteration $i$:

1. **Sample** $N$ candidate $\mathbf{a}^{(n)}\sim q_{\eta_i}$.
2. **Đánh giá** bằng rollout model: $J^{(n)}=\hat J(\mathbf{a}^{(n)}\mid z_t)=\sum_k\gamma^k\hat r(\hat z_k)+\gamma^H\hat v(\hat z_H)$.
3. **Chọn elite**: lấy $K=\lceil\rho N\rceil$ candidate điểm cao nhất ($\rho$ là elite fraction, ví dụ 0.1).
4. **Refit** phân phối về elite bằng MLE:

    $$
    \mu_{i+1}=\frac{1}{K}\sum_{e\in\mathcal{E}}\mathbf{a}^{(e)},\qquad \sigma^2_{i+1}=\frac{1}{K}\sum_{e\in\mathcal{E}}\big(\mathbf{a}^{(e)}-\mu_{i+1}\big)^2.
    $$

    Trong đó $\mathcal{E}$ là tập elite. Đây đúng là mean/variance mẫu của elite — phân phối mới đặt khối lượng lên vùng elite vừa tìm được.
5. Lặp tới khi hội tụ (hoặc đủ số iteration), trả $\mu$ cuối làm nghiệm; MPC thực thi $a_t=\operatorname{first}(\mu)$.

Vì phân phối ngày càng co quanh nghiệm tốt, các iteration sau cần ít sample hơn để giữ chất lượng — CEM tập trung ngân sách vào nơi đáng giá thay vì rải đều.

---

## **3. Vì sao gọi là "cross-entropy"**

Bước refit (4) không tùy tiện: nó là nghiệm của bài toán cực tiểu **cross-entropy / KL** giữa phân phối mục tiêu (tập trung trên vùng tốt) và họ tham số $q_\eta$. Với một ngưỡng $\gamma_i$ và phân phối lý tưởng tỉ lệ với $\mathbb{1}[J(\mathbf{a})\ge\gamma_i]\,q(\mathbf{a})$, cập nhật cross-entropy

$$
\eta_{i+1}=\arg\max_\eta\;\frac{1}{K}\sum_{e\in\mathcal{E}}\log q_\eta(\mathbf{a}^{(e)})
$$

chính là **maximum likelihood trên elite**. Trong đó vế phải là log-likelihood của elite dưới $q_\eta$; với Gaussian, MLE cho ra đúng công thức mean/variance ở §2. Tên gọi đến từ việc tối thiểu cross-entropy $H(p,q_\eta)$ tương đương tối đa log-likelihood — đó là lý do "Cross-Entropy Method".

---

## **4. Biến thể**

| | Random shooting | CEM | iCEM / colored CEM |
|---|---|---|---|
| Phân phối sample | đều, cố định | Gaussian, refit theo elite | + colored noise, momentum, clipping |
| Sample-efficiency | thấp | trung bình | cao (real-time) |
| Action tương quan thời gian | không mô hình | đường chéo (bỏ qua tương quan) | colored noise tạo smooth action |

Các kỹ thuật thực dụng: **warm-start** (dịch $\mu$ của bước trước sang một bước để khởi tạo bước sau — receding horizon), **momentum** trên $\mu,\sigma$ để mượt giữa các bước, **giữ elite cũ** (CEM-with-memory), **floor cho $\sigma$** chống collapse, và **colored noise** (iCEM, Pinneri 2021) tạo action sequence trơn theo thời gian, tăng mạnh sample-efficiency cho real-time planning. PETS kết hợp CEM với **particles** qua probabilistic ensemble để đánh giá candidate dưới uncertainty.

---

## **5. Giới hạn / Khi nào thất bại**

**Compute mỗi bước.** Mỗi bước MPC chạy $I$ iteration × $N$ sample × $H$ rollout (× $P$ particles) lần gọi model. Trong latent thì rẻ, nhưng vẫn là ràng buộc real-time nặng cho control tần số cao.

**Premature collapse.** Nếu $\sigma^2$ co quá nhanh (elite fraction quá nhỏ, không có floor/momentum), phân phối kẹt ở local optimum đầu tiên gặp và mất khả năng khám phá. Đây là failure mode thực dụng phổ biến nhất.

**Diagonal Gaussian bỏ qua tương quan.** Action ở các bước thời gian thường nên tương quan (mượt); Gaussian đường chéo độc lập từng chiều sinh chuỗi giật cục — lý do iCEM dùng colored noise.

**Non-convex/multimodal.** CEM bám một mode; nếu nhiều mode tốt cách xa, mean của elite có thể rơi vào "thung lũng" giữa các mode — nghiệm vô nghĩa.

**Model & value exploitation.** CEM tối ưu $\hat J$ mạnh hơn random shooting, nên *cũng* khai thác model/value sai mạnh hơn (optimizer's curse). Tối ưu tốt một proxy xấu không phải điều tốt — xem [reward overoptimization](02-reward-model-in-latent.md).

**Nhạy hyperparameter.** $N$, $I$, $\rho$, floor $\sigma$, momentum đều ảnh hưởng lớn và phụ thuộc task.

---

## **6. Liên hệ với Latent-Anything**

CEM là một **optimizer plugin** thay thế được cho bước $\arg\max$ trong MPC — đúng tinh thần interface ở [mục 4](04-model-predictive-control.md):

```python
def cem_plan(z_t, model, objective, H, n=256, elite_frac=0.1, iters=6, mu=None):
    mu = np.zeros((H, A_DIM)) if mu is None else mu
    sigma = np.ones((H, A_DIM))
    for _ in range(iters):
        seqs = mu + sigma * randn(n, H, A_DIM)            # sample from belief
        Z = model.rollout(z_t, seqs, H)                   # latent rollout (batched)
        J = objective.evaluate(Z, seqs)                   # Σ γ^k r̂ + γ^H v̂
        elite = seqs[J.argsort()[-int(elite_frac * n):]]  # keep top fraction
        mu, sigma = elite.mean(0), elite.std(0)           # refit (MLE)
    return mu                                             # MPC executes mu[0]
```

- **Layer A — Introspection**: theo dõi đường co của $\sigma$ (phát hiện premature collapse), phân bố return của elite qua iteration, và gap giữa imagined return của $\mu$ với realized return.
- **Layer B — Manipulation**: CEM là cách tinh hơn để "tối ưu action sequence" so với random shooting trong cùng API biến đổi trajectory.
- **Layer C — Runtime**: lộ ra ngân sách $(N,I,\rho)$, warm-start state giữa các bước, batching candidate/particle; cho phép đổi optimizer (shooting ↔ CEM ↔ MPPI) mà không đụng dynamics adapter.

Mục tiếp theo, **MPPI**, thay bước "chọn cứng elite" của CEM bằng trung bình *mềm* có trọng số mũ theo return — thường tốt hơn cho continuous control vì dùng được thông tin của *mọi* candidate, không chỉ nhúm elite.

---

## Liên quan

- [Model Predictive Control (MPC)](04-model-predictive-control.md) — vòng receding-horizon mà CEM giải bước tối ưu bên trong.
- [Reward Model trong Latent](02-reward-model-in-latent.md) — objective CEM tối đa; cảnh báo overoptimization.
- [Value Function trong Latent](03-value-function-in-latent.md) — terminal value trong $\hat J$ mà CEM dùng.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — random shooting và CEM như planning bằng imagined rollout.
- [Density Estimation](../../04-latent-computation/research/06-density-estimation.md) — giữ candidate trong support, chống exploitation.

## Tham khảo

- R. Y. Rubinstein, *Optimization of Computer Simulation Models with Rare Events* (European J. of Operational Research 1997).
- P.-T. de Boer, D. P. Kroese, S. Mannor, R. Y. Rubinstein, *A Tutorial on the Cross-Entropy Method* (Annals of Operations Research 2005).
- D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, *Learning Latent Dynamics for Planning from Pixels* (PlaNet, ICML 2019, arXiv:1811.04551).
- K. Chua, R. Calandra, R. McAllister, S. Levine, *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* (PETS, NeurIPS 2018, arXiv:1805.12114).
- C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek, G. Martius, *Sample-efficient Cross-Entropy Method for Real-time Planning* (iCEM, CoRL 2020, arXiv:2008.06389).
