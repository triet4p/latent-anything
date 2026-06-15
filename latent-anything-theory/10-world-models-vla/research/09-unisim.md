# UniSim (Yang et al., 2023)

> **TL;DR.** UniSim ("Learning Interactive Real-World Simulators") học một **universal simulator** của tương tác thế giới thực qua generative modeling: một giao diện thống nhất **action-in → video-out** (điều kiện trên lịch sử + hành động, sinh frame kế bằng **video diffusion**, rollout tự hồi quy cho mô phỏng dài). Chìa khóa scale đa-domain: gom dữ liệu *cực kỳ đa dạng* (vật thể, cảnh, hoạt động người, navigation, manipulation, panorama, render) vào *một* model bằng cách ép mọi hành động — motor control, mô tả ngôn ngữ ("lau bàn"), chuyển động camera — về **một không gian action chung**. Caveat: pixel-space diffusion rất đắt, không phải latent compact để plan; chất lượng phụ thuộc độ phủ dữ liệu.

[GAIA-1](../../09-discrete-latent/research/08-gaia-1.md) và [Genie](../../09-discrete-latent/research/09-genie.md) là world model mạnh nhưng *theo domain* (lái xe, game). UniSim (Yang et al., 2023) hỏi câu lớn nhất nhóm: có thể học *một* simulator cho **mọi** loại tương tác thực không? Câu trả lời định hình đúng vấn đề trung tâm của Latent-Anything — *handle multi-domain latent* — nên đây là mục khép tầng world model về phía "universal".

---

## 1. Trực giác: một giao diện cho mọi domain

Vấn đề khi gộp nhiều domain: dữ liệu robot, video người, navigation, render game... có *observation* và *action* định dạng khác nhau hoàn toàn. UniSim thống nhất bằng một quan sát đơn giản: gần như mọi tương tác đều có thể đóng khung là **"cho cảnh hiện tại + một hành động, cảnh thay đổi thế nào"** — tức **action-in-video-out**. Nếu mọi domain nói cùng một giao diện đó, một model duy nhất học được từ tất cả.

Mấu chốt thứ hai: **action không cần cùng modality**. Motor control của robot, câu lệnh "lau bàn", và chuyển động camera trích từ video — tất cả được map vào *một không gian conditioning chung*. Nhờ vậy dữ liệu không-nhãn-action (video người, panorama) vẫn đóng góp được qua action trích từ camera motion.

---

## 2. Cơ chế: video diffusion + rollout tự hồi quy

UniSim mô hình hóa simulator như một **observation prediction model** điều kiện trên lịch sử hữu hạn và hành động:

$$
p_\theta\big(o_{t+1:t+k} \mid o_{t-h:t},\, a_t\big),
$$

trong đó $o$ là frame quan sát, $a_t$ là hành động (đa modality, map vào conditioning chung), $h$ là độ dài lịch sử, và phân phối được tham số hóa bằng một **video diffusion model**. Sinh một đoạn video bằng denoising (như [diffusion decoder của GAIA-1](../../09-discrete-latent/research/08-gaia-1.md)); rollout *tự hồi quy* — nối đoạn vừa sinh vào lịch sử rồi sinh tiếp — cho mô phỏng dài, nhất quán.

Khác biệt thiết kế quan trọng với phần còn lại của tầng: UniSim sinh **pixel** (high-fidelity video), không phải latent compact. Nó là *simulator* để render tương tác chân thực, không phải latent gọn để plan nhanh (đối cực [LeWM](08-lewm.md)). Đổi lại độ chân thực và phổ quát, nó tốn kém hơn nhiều.

### Dữ liệu đa dạng → khả năng phổ quát

UniSim train trên hỗn hợp rộng: vật thể, cảnh tĩnh, hoạt động người, motion trong navigation và manipulation, panorama scan, và cả simulation/rendering. Mỗi nguồn đóng góp một mảnh: render cho động lực vật lý sạch, video người cho đa dạng ngữ nghĩa, robot cho action có nhãn. Giao diện chung cho phép *transfer* giữa chúng.

---

## 3. Vì sao quan trọng cho framework

UniSim là minh chứng cho luận điểm *multi-domain* mà Latent-Anything dựa vào: **một giao diện thống nhất cho phép một hệ thống nuốt nhiều nguồn latent/action khác nhau**. Ứng dụng nó chứng minh:

| Ứng dụng | Ý nghĩa |
|---|---|
| Train high-level VLM policy | simulator sinh kinh nghiệm cho policy ngôn ngữ |
| Train low-level control | rollout làm môi trường ảo cho điều khiển |
| Zero-shot sim-to-real | bắc cầu sim→real, giảm gap |
| Sinh dữ liệu (video captioning) | dùng làm máy sinh dữ liệu tổng quát |

So với cả tầng: Dreamer/MuZero/LeWM tối ưu *latent compact để plan*; GAIA/Genie là *token/diffusion theo domain*; UniSim là *generative simulator pixel-space phổ quát*. Ba thiết kế này phủ không gian world model, và framework cần adapter cho cả ba — UniSim là ca "đa domain, action đa modality" khó nhất.

---

## 4. Giới hạn / Khi nào thất bại

**Pixel diffusion rất đắt.** Sinh video chất lượng cao bằng diffusion tốn tính toán; rollout dài chậm — không hợp planning realtime (ngược hẳn LeWM/TD-MPC2).

**Không latent compact để reasoning.** UniSim render observation, không phơi bày một latent gọn để probe/plan — kém tiện cho introspection so với JEPA/value-equivalence.

**Phụ thuộc độ phủ dữ liệu.** "Universal" chỉ tới mức dữ liệu phủ; domain/tương tác ngoài hỗn hợp train vẫn ngoài tầm.

**Action chung là xấp xỉ.** Map mọi modality về một không gian conditioning là tiện nhưng lossy — action trích từ camera motion không chính xác bằng motor control thật.

**Compounding error.** Rollout tự hồi quy pixel-space tích lũy lỗi và artifact qua thời gian như mọi world model.

---

## 5. Liên hệ với Latent-Anything

UniSim là *case study* cho tham vọng multi-domain của framework: nó cho thấy một **giao diện chuẩn** (action-in-video-out, action space chung) là cách gộp nhiều nguồn — đúng tinh thần plugin-first của Latent-Anything.

```python
class UniSimAdapter(Protocol):
    def encode_action(self, action: Any, modality: str) -> np.ndarray: ...   # motor|language|camera -> chung
    def simulate(self, history: np.ndarray, action: np.ndarray, steps: int) -> np.ndarray: ...  # -> video
    history_len: int
```

- **Layer A — Introspection**: UniSim là ca khó cho introspection (pixel-space, không latent gọn) — nhưng chính vì vậy nó thúc framework định nghĩa *interface* introspection trên *hành vi sinh* (rollout) thay vì trên latent nội bộ: probe "model làm gì khi nhận action X".
- **Layer B — Manipulation**: không gian action chung *là* một bề mặt manipulation thống nhất — chèn action đa modality (lệnh ngôn ngữ, motor) để lái mô phỏng, đúng kiểu steering đa nguồn.
- **Layer C — Runtime**: rollout diffusion pixel-space là workload nặng nhất tầng; Layer C phải xử lý đánh đổi fidelity vs cost, và lập lịch render lười — đối cực với LeWM nhẹ.

UniSim khép Tầng bổ sung World Models & VLA: cùng [LeWM](08-lewm.md) (anchor compact JEPA), [Dreamer](01-dreamerv1.md)/[MuZero](05-muzero.md) (planning), và [OpenVLA](06-openvla.md)/[π0](07-pi0.md) (VLA), nó hoàn tất bức tranh các thiết kế latent quy mô lớn mà framework phải bao. Các tầng bổ sung còn lại quay về *công cụ* (interpretability) và *toán nền*.

---

## Liên quan

- [GAIA-1](../../09-discrete-latent/research/08-gaia-1.md) — cùng dùng video diffusion decode; GAIA theo domain lái xe, UniSim phổ quát.
- [Genie](../../09-discrete-latent/research/09-genie.md) — cùng action-controllable world generation; Genie học latent action, UniSim dùng action space đa modality chung.
- [Tokenized World Model](../../09-discrete-latent/research/07-tokenized-world-model.md) — đối chiếu token vs pixel-diffusion world model.
- [LeWM](08-lewm.md) — đối cực: latent compact gọn để plan vs simulator pixel phổ quát.
- [π0](07-pi0.md) — flow/diffusion sinh; cùng họ generative cho hành động/quan sát.

## Tham khảo

- M. Yang, Y. Du, K. Ghasemipour, J. Tompson, D. Schuurmans, P. Abbeel, *Learning Interactive Real-World Simulators* (UniSim, ICLR 2024, arXiv:2310.06114).
- J. Ho et al., *Video Diffusion Models* (NeurIPS 2022, arXiv:2204.03458) — nền video diffusion.
- A. Hu et al., *GAIA-1: A Generative World Model for Autonomous Driving* (2023, arXiv:2309.17080).
- J. Bruce et al., *Genie: Generative Interactive Environments* (ICML 2024, arXiv:2402.15391).
