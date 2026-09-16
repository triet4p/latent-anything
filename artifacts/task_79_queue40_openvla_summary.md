# Sprint 79 queue position 40 — OpenVLA prerequisite closure

`THY-X01-OPENVLA` remains D0 in M14 lane L19. This task closes the local
planning/provenance work only; it does not cross the remote CUDA boundary or
promote evidence.

## Authoritative upstream findings

- Fine-tuned task checkpoint: [`openvla/openvla-7b-finetuned-libero-spatial@962318cec55ac10993ff0f5f43eda9a270b4c873`](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial/commit/962318cec55ac10993ff0f5f43eda9a270b4c873).
  Its Hub metadata is public, non-gated, and disabled=false; the model is
  `OpenVLAForActionPrediction`, BF16, with 15,082,474,368 bytes in its
  safetensors index.
- Base checkpoint: [`openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f`](https://huggingface.co/openvla/openvla-7b/commit/47a0ec7fc4ec123775a391911046cf33cf9ed83f).
- The model cards declare MIT. The [official OpenVLA README](https://github.com/openvla/openvla#pretrained-vlas)
  separately warns that pretrained Llama-2-derived weights inherit the Llama
  Community License. Both notices are retained; this is not a commercial-use
  approval.
- Data: [`openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`](https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551),
  subset `libero_spatial_no_noops`, with 52,970 transitions and 432
  trajectories in the pinned checkpoint statistics. Dataset metadata declares
  MIT and public, non-gated access.
- Environment: [`Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01`](https://github.com/Lifelong-Robot-Learning/LIBERO/commit/8f1084e3132a39270c3a13ebe37270a43ece2a01),
  `libero_spatial`, 10 tasks × 50 trials, 10 initial stabilization steps,
  256-pixel observations, and up to 220 control steps per episode.

## Contract recorded

The new [`artifacts/m14/l19-openvla.config.json`](m14/l19-openvla.config.json)
records the real Transformers loading path, exact prompt and image contract,
seven 256-bin action tokens, `libero_spatial` unnormalization, gripper
postprocessing/sign inversion, finite `(7,)` environment actions, paired
intervention/control requirement, success gate (>=0.80 against the published
LIBERO-Spatial reference), runtime versions, A100 80GB reference profile,
network policy, and cleanup. The blocked receipt is
[`artifacts/m14/l19-openvla.json`](m14/l19-openvla.json).

## Local readiness and blocker

Only non-secret markers were checked. `lerobot` and `transformers` are
installed, but the OpenVLA checkpoint is not cached, network opt-in is disabled,
no HF token is configured, and `torch.cuda.is_available()` is false. The
Windows checkout has no production OpenVLA adapter/capture implementation.
Existing ACT, Diffusion Policy, and SmolVLA adapters cannot be relabeled as
OpenVLA evidence. No source/test integration was added, and D0 is retained.

The next action is an authorized Linux NVIDIA CUDA run in a disposable
checkout/cache using Python 3.10.13, PyTorch 2.2.0, Transformers 4.40.1,
tokenizers 0.19.1, flash-attn 2.5.5, and the upstream A100 80GB reference
profile. Provision the pinned model/data, implement and review the real adapter,
then run the 500-trial paired LIBERO-Spatial intervention evaluation and retain
a signed artifact. No remote command was run here. Sprint 79 line 595 and lines
596+ were not touched.

Queue regeneration provenance is explicit: the authoritative builder ran after
the source/doc/test commit at
`60c9c1900bf9a7a03ae81fdd65ef0f2a90ab3f91`; the final delivery commit that
contains the regenerated queue is `e007a48bd429ea80a7908f52aaf1d11126175ee2`.
