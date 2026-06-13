# MPPI — Model Predictive Path Integral

> **TL;DR.** MPPI là optimizer sampling-based cho MPC, thay bước "chọn cứng elite" của [CEM](05-cross-entropy-method.md) bằng **trung bình mềm có trọng số mũ**: mỗi candidate control sequence nhận trọng số $w^{(n)}\propto\exp(\text{return}^{(n)}/\lambda)$ (softmax theo nhiệt độ $\lambda$), rồi nominal control cập nhật bằng weighted average của nhiễu. Nó đến từ control information-theoretic / free energy, dùng *mọi* candidate thay vì chỉ nhúm elite, nên thường mượt và sample-hiệu-quả hơn CEM cho continuous control — đây là planner mặc định của TD-MPC. Caveat: nhạy nhiệt độ $\lambda$, covariance thường cố định, và weighted average vẫn có thể rơi vào "thung lũng" giữa nhiều mode.

[CEM](05-cross-entropy-method.md) bỏ đi phần lớn candidate (chỉ giữ top-$K$) — phí thông tin, và bước "0/1" theo elite làm cập nhật giật. MPPI giữ lại *tất cả* candidate, chỉ cho điểm cao trọng số lớn hơn theo một hàm mũ trơn. Trực giác đơn giản; nền lý thuyết (path integral / free energy) thì sâu, và nó là optimizer trong vòng MPC của TD-MPC trên latent dynamics.

---

## **1. Trực giác / Định nghĩa**

Cả CEM lẫn MPPI đều: sample nhiều chuỗi action quanh một nominal $\mu$, rollout model, chấm điểm. Khác biệt nằm ở **cách dùng điểm số**:

- **CEM**: sắp xếp, giữ top-$K$ (elite), vứt phần còn lại, refit mean/var về elite. Trọng số là 0 hoặc 1.
- **MPPI**: không vứt gì. Mỗi candidate nhận trọng số *liên tục* tỉ lệ $\exp(\text{return}/\lambda)$ — candidate tốt hơn được ưu tiên mượt, candidate tệ vẫn đóng góp một chút. Nominal cập nhật bằng trung bình có trọng số.

Ẩn dụ: CEM là "bầu cử winner-take-elite"; MPPI là "bỏ phiếu có trọng số", nơi mọi candidate có tiếng nói tỉ lệ với chất lượng của nó. Tham số $\lambda$ (nhiệt độ) điều khiển độ "gắt": $\lambda$ nhỏ → gần như chỉ nghe candidate tốt nhất; $\lambda$ lớn → trung bình đều.

---

## **2. Cơ chế / Công thức**

Tham số hóa control bằng nhiễu quanh nominal sequence $\mu=(\mu_0,\dots,\mu_{H-1})$:

$$
\mathbf{a}^{(n)}=\mu+\varepsilon^{(n)},\qquad \varepsilon^{(n)}\sim\mathcal{N}(0,\Sigma).
$$

Trong đó $\varepsilon^{(n)}$ là perturbation của candidate $n$. Rollout model cho mỗi candidate, tính return $R^{(n)}=\hat J(\mathbf{a}^{(n)}\mid z_t)$ (hoặc cost $S^{(n)}=-R^{(n)}$). **Trọng số mềm** bằng softmax theo nhiệt độ:

$$
w^{(n)}=\frac{\exp\!\big(R^{(n)}/\lambda\big)}{\sum_{m}\exp\!\big(R^{(m)}/\lambda\big)}.
$$

Trong đó $\lambda>0$ là nhiệt độ (inverse temperature theo cost), và $\sum_n w^{(n)}=1$. Candidate return cao nhận trọng số lớn theo hàm mũ. **Cập nhật nominal** bằng weighted average của perturbation:

$$
\mu\;\leftarrow\;\mu+\sum_{n} w^{(n)}\,\varepsilon^{(n)}.
$$

Trong đó nominal dịch theo trung bình có trọng số của các nhiễu — về phía vùng return cao. MPC thực thi $a_t=\operatorname{first}(\mu)$, rồi shift $\mu$ một bước (warm-start) cho lần sau.

### Vai trò của nhiệt độ $\lambda$

$$
\lambda\to 0:\ w\to\text{one-hot tại }\arg\max R\quad(\text{chọn cứng candidate tốt nhất}),\qquad \lambda\to\infty:\ w\to\text{đều}\quad(\text{trung bình mọi candidate}).
$$

Trong đó $\lambda$ là núm exploration–exploitation: nhỏ thì gắt và tham lam (dễ nhiễu/khai thác), lớn thì khám phá nhưng chậm. CEM (hard elite) là một điểm trung gian rời rạc; MPPI cho một phổ liên tục.

---

## **3. Nền lý thuyết: free energy và importance sampling**

MPPI không phải heuristic — nó suy ra từ **stochastic optimal control** và information theory. Free energy của bài toán:

$$
\mathcal{F}(S,P,x_0,\lambda)=-\lambda\,\log\,\mathbb{E}_{P}\!\Big[\exp\!\big(-\tfrac{1}{\lambda}S(V)\big)\Big].
$$

Trong đó $S(V)$ là cost-to-go của trajectory sinh bởi control noise $V$, $P$ là phân phối base (uncontrolled), và kỳ vọng lấy trên nhiễu. Tối ưu hóa được lift thành bài toán **KL-regularized** trên không gian phân phối: tìm phân phối control tối ưu $q^\star\propto\exp(-S/\lambda)\,p$, rồi xấp xỉ nó bằng importance sampling. Cập nhật weighted-average ở §2 chính là ước lượng Monte Carlo của kỳ vọng dưới $q^\star$, với trọng số $w^{(n)}$ là **importance weights**. Nói cách khác, MPPI tối thiểu KL giữa phân phối control điều khiển được và phân phối tối ưu — cùng họ với variational inference MPC. Đây là lý do trọng số có dạng mũ: nó không tùy chọn mà rơi ra từ KL/free energy.

---

## **4. MPPI vs CEM**

| | CEM | MPPI |
|---|---|---|
| Dùng candidate nào | chỉ top-$K$ elite | tất cả, trọng số mũ |
| Trọng số | 0/1 (cứng) | $\exp(R/\lambda)$ (mềm) |
| Cập nhật | mean **và** covariance từ elite | mean bằng weighted average (Σ thường cố định) |
| Núm chính | elite fraction $\rho$ | nhiệt độ $\lambda$ |
| Nền lý thuyết | cross-entropy / rare-event | free energy / KL optimal control |
| Hợp với | tối ưu tổng quát, noisy | continuous control, action mượt |

Điểm mạnh MPPI: dùng *mọi* thông tin candidate (không vứt), cập nhật mượt hơn, ít nhạy với việc "đúng $K$ elite". TD-MPC chọn MPPI làm planner bên trong vòng MPC latent vì lý do này. Điểm yếu so CEM: covariance cố định nên không tự thích nghi scale tìm kiếm như CEM (trừ khi thêm covariance adaptation).

---

## **5. Biến thể**

- **Smooth MPPI**: nhiễu trắng tạo action giật; lift bằng colored/correlated noise hoặc lọc input để control trơn (quan trọng cho robot thật).
- **Covariance adaptation**: cập nhật cả $\Sigma$ (mượn ý CEM) để thích nghi scale.
- **Tsallis / generalized VI-MPC**: MPPI và CEM là các trường hợp đặc biệt của variational inference SOC với các divergence khác nhau (Tsallis $r$): $r\to\infty$ cho ra CEM, exponential weighting cho ra MPPI.
- **Automatic temperature tuning**: chọn $\lambda$ theo effective sample size để tránh tuning thủ công.
- **GPU parallel**: toàn bộ candidate rollout song song — MPPI thiết kế cho parallel computation, hợp với latent rollout batched.

---

## **6. Giới hạn / Khi nào thất bại**

**Nhạy nhiệt độ $\lambda$.** $\lambda$ quá nhỏ → cập nhật bám một candidate nhiễu (variance cao, dễ khai thác model sai); quá lớn → trung bình ì, hội tụ chậm. Hiệu năng phụ thuộc mạnh vào $\lambda$ và scale của return.

**Covariance cố định.** Không tự co/giãn vùng tìm kiếm; nếu $\Sigma$ ban đầu sai scale, MPPI kém hiệu quả (CEM thích nghi tốt hơn ở khía cạnh này).

**Weighted average qua nhiều mode.** Giống CEM mean, trung bình có trọng số của hai mode tốt cách xa có thể rơi vào "thung lũng" giữa chúng — control vô nghĩa. $\lambda$ nhỏ giảm thiểu (tập trung một mode) nhưng tăng variance.

**Model & value exploitation.** Là optimizer mạnh nên cũng khai thác mạnh vùng model/value lạc quan sai (optimizer's curse), y như [reward overoptimization](02-reward-model-in-latent.md).

**Phụ thuộc nominal/warm-start.** Sample quanh $\mu$ hiện tại; nếu nominal xa nghiệm và $\Sigma$ hẹp, candidate không phủ tới vùng tốt.

**Sample count trong chiều cao.** Action sequence dài/chiều cao cần nhiều sample; ràng buộc real-time vẫn áp.

---

## **7. Liên hệ với Latent-Anything**

MPPI là một **optimizer plugin** khác cho cùng interface MPC ở [mục 4](04-model-predictive-control.md) — thay được CEM mà không đụng dynamics adapter:

```python
def mppi_plan(z_t, model, objective, H, n=512, lam=1.0, sigma=0.5, mu=None):
    mu = np.zeros((H, A_DIM)) if mu is None else mu
    eps = sigma * randn(n, H, A_DIM)                 # perturbations
    seqs = mu + eps
    R = objective.evaluate(model.rollout(z_t, seqs, H), seqs)   # returns
    w = softmax(R / lam)                             # soft, information-theoretic weights
    mu = mu + np.tensordot(w, eps, axes=1)           # weighted-average update
    return mu                                        # MPC executes mu[0]; then shift mu
```

- **Layer A — Introspection**: theo dõi *effective sample size* $1/\sum_n (w^{(n)})^2$ (chẩn đoán $\lambda$: ESS quá nhỏ = quá gắt, quá lớn = quá đều), và gap imagined vs realized return.
- **Layer B — Manipulation**: MPPI là cách thứ ba (sau shooting, CEM) để tối ưu action sequence trong cùng API biến đổi trajectory.
- **Layer C — Runtime**: lộ ra $\lambda,\Sigma,N$, batching candidate song song (MPPI rất hợp GPU), warm-start nominal giữa các bước.

Hết MPPI, ta đã đi qua các planner *decision-time* (sample-based): shooting → CEM → MPPI, đều phải tối ưu lại mỗi bước. Mục tiếp theo, **Dreamer (policy gradient trên imagined trajectory)**, chuyển hướng: *amortize* planning thành một actor học được bằng cách backprop gradient qua dynamics khả vi, để khỏi search online.

---

## Liên quan

- [Cross-Entropy Method (CEM)](05-cross-entropy-method.md) — optimizer hard-elite mà MPPI làm mềm bằng exponential weighting.
- [Model Predictive Control (MPC)](04-model-predictive-control.md) — vòng receding-horizon mà MPPI giải bước tối ưu.
- [Reward Model trong Latent](02-reward-model-in-latent.md) — return trong trọng số softmax; cảnh báo overoptimization.
- [Value Function trong Latent](03-value-function-in-latent.md) — terminal value trong return của candidate.
- [Rollout và Latent Imagination](../../06-latent-temporal/research/07-rollout-latent-imagination.md) — candidate rollout song song trong latent.

## Tham khảo

- G. Williams, A. Aldrich, E. A. Theodorou, *Model Predictive Path Integral Control: From Theory to Parallel Computation* (J. Guidance, Control, and Dynamics 2017).
- G. Williams, P. Drews, B. Goldfain, J. M. Rehg, E. A. Theodorou, *Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving* (IEEE Transactions on Robotics 2018).
- M. Okada, T. Taniguchi, *Variational Inference MPC for Bayesian Model-based Reinforcement Learning* (CoRL 2019, arXiv:1907.04202).
- Z. Wang, O. So, J. Gibson, B. Vlahov, M. Gandhi, G. Williams, E. Theodorou, *Variational Inference MPC using Tsallis Divergence* (RSS 2021, arXiv:2104.00241).
- N. Hansen, X. Wang, H. Su, *Temporal Difference Learning for Model Predictive Control* (TD-MPC, ICML 2022, arXiv:2203.04955).
- N. Hansen, H. Su, X. Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control* (ICLR 2024, arXiv:2310.16828).
