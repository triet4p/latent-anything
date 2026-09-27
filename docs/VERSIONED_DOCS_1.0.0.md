# v1.0.0 Versioned Documentation and Input-Revisions Record

**Package version:** `1.0.0`  
**Protected release tag:** `v1.0.0`  
**Release commit:** `a449ca33b4b83ce109c29db7cf471919820b1d56`  
**Annotated tag object:** `3bdb7ff2c5e09def32e39b9b4628750729104b22`

This is a post-release companion record for the published `1.0.0` package. It binds the bounded signed-off diagnostic claims to their exact benchmark, model, and dataset revisions and records the blocked SmolVLA lane. It was added after the release commit; it is not represented as content of tag `v1.0.0`. The protected tag is unchanged.

## Immutable release and evidence revisions

- Release commit [`a449ca33b4b83ce109c29db7cf471919820b1d56`](https://github.com/triet4p/latent-anything/commit/a449ca33b4b83ce109c29db7cf471919820b1d56) is the exact source revision for package version `1.0.0`; the annotated tag object above points to it. Exact-SHA CI run [36250365395](https://github.com/triet4p/latent-anything/actions/runs/36250365395) and audited release run [36251907256](https://github.com/triet4p/latent-anything/actions/runs/36251907256) are recorded in the [release notes](RELEASE_NOTES_1.0.0.md).
- The accepted Sprint 80 fresh-root replay used source commit [`52415535ba1889c49e67be65429985a7ed6f5c22`](https://github.com/triet4p/latent-anything/commit/52415535ba1889c49e67be65429985a7ed6f5c22), run `20260925-065327-115093-14640`. Its three bounded cases and limits are documented in the [Sprint 80 depth-evidence report](SPRINT_80_DEPTH_EVIDENCE.md).
- The historical blocked SmolVLA v1 attempt used source commit [`24850717ca3f55afbd8d19dc28aeca587744f996`](https://github.com/triet4p/latent-anything/commit/24850717ca3f55afbd8d19dc28aeca587744f996). Its failure is preserved in the [Task 80.27 record](../artifacts/task_80.27_smolvla_secondary_lane_summary.md); the prospective v2 manifest remains unexecuted.

The machine-readable [release input revision archive](../artifacts/benchmark_revision_archive_1.0.0.json) lists the repo-relative manifest paths, exact manifest-file SHA-256 values, locked commitments, model and dataset IDs/revisions, selected-data digests, license sources, and evidence run/source identifiers. Each manifest URL is pinned to the release commit rather than a moving branch.

## Supported claim boundary

| Signed-off case | Pinned inputs | Boundary |
| --- | --- | --- |
| Encoder v3 model-weight lesion | Locally trained four-dimensional linear autoencoder; `sklearn.datasets.load_digits`, `scikit-learn==1.9.0`, held-out split digest `b3ed6f2d420b7dc8de35bb8adc0088b49f76a3b05d31057a0a48e37430966148`. | One prospective controlled lesion and one fixed brightness-bin task; not a natural production defect, general encoder/dataset result, or GPU claim. |
| Transformer target-evidence-v2 | `openai-community/gpt2@e7da7f221d5bf496a48136c0cd264e630fe9fcc8`; `Salesforce/wikitext@f776294184f13b8ff2337b3841cf9269a6216d1e`, `wikitext-2-raw-v1` validation selection. | Section-header attribute separability and localization at `transformer.h.0` under the frozen protocol; not causal feature-use evidence or a general model/dataset claim. |
| Transformer stability supplement | The same pinned GPT-2 and WikiText revisions, with the separately predeclared grouped-split protocol. | Separate stability threshold evidence; it does not retroactively change the immutable target-evidence-v2 artifact's `threshold: null`. |

## Explicitly blocked SmolVLA lane

The `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de` policy and `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` dataset revisions are recorded only to preserve provenance. The v1 proof captured `(64, 768)` while asserting `(64, 1024)`; the frozen estimator also rejects 64 samples for 768 dimensions. Execution stopped before diagnosis, no validator-clean diagnostic artifact exists, and the prospective 768-to-32 v2 protocol has not been run. SmolVLA remains **BLOCKED** and non-gating; these identifiers do not promote a VLA, GPU, or CUDA claim.

## License and redistribution boundary

The archive contains identifiers, revisions, hashes, license declarations, source paths, and evidence references only. It does **not** copy model weights, dataset rows, raw text, simulator assets, or cached model/dataset files. Hashes identify bytes; they are not substitutes for the bytes and do not grant redistribution rights.

- The GPT-2 model card at its pinned revision declares MIT. The WikiText card at its pinned revision declares CC BY-SA 3.0 and GFDL. No model or WikiText bytes are included.
- The SmolVLA checkpoint and LeRobot LIBERO dataset cards at their pinned revisions declare Apache-2.0. The license labels do not change the lane's blocked status; no checkpoint or dataset bytes are included, and this record is not a legal review.
- The encoder model is a locally trained benchmark artifact referenced by digest, not an upstream pretrained checkpoint. Project evidence records the scikit-learn digits fixture as BSD-3-Clause; the benchmark manifest pins the package version and selected-data digest. The archive contains no checkpoint or digits data.

## Documentation publication status

This versioned record is maintained in the repository and is content-hash-verifiable
through the machine-readable archive. It is a post-release documentation
companion, not an edit to the release tag or its five package assets.
The corrected immutable source snapshot is the annotated
[docs-v1.0.0-r1 tag](https://github.com/triet4p/latent-anything/tree/docs-v1.0.0-r1).
It supersedes the unchanged [docs-v1.0.0 tag](https://github.com/triet4p/latent-anything/tree/docs-v1.0.0),
whose archive recorded the SmolVLA v1 manifest's Windows CRLF checkout bytes
instead of the pinned release commit's canonical LF bytes. The corrected archive
records the exact 5,068-byte manifest from release commit
`a449ca33b4b83ce109c29db7cf471919820b1d56` (SHA-256
`285f540624b389ca274425ef264ca4bbc91be7bad38521879cfc9c27dd2d5ddf`); the
prior checkout-form digest was `4961f5f8d764314d12d92a5f9712cdf3c7b0e27d0283b51d137f579b4298a73f`
for 5,212 bytes.

### Immutable r1 snapshot: hash authority and editorial limits

The public `docs-v1.0.0-r1` tag (commit
[`3cbc925de464a090bc1d65a2fc8858a11a0dcbf3`](https://github.com/triet4p/latent-anything/commit/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3))
and its metadata-only revision archive remain unchanged. The archive's
immutable record of manifest paths, canonical Git-blob SHA-256 values, and byte
lengths identifies the source bytes at the pinned release commit; those hashes
do not replace the source files. That authority does not make every copied
documentation page in the immutable tag current. Several r1 copies still
describe `1.0.0` as an unpublished candidate even though the release is
published. Treat the listed lines as historical wording, not as current
publication status:

| Immutable r1 copy and stale lines | Corrected current-source document |
| --- | --- |
| [`API_REFERENCE.md:12–14`](https://github.com/triet4p/latent-anything/blob/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/API_REFERENCE.md#L12) frames the API as an unreleased candidate. | [API reference](https://github.com/triet4p/latent-anything/blob/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/API_REFERENCE.md#L12), corrected in `379aaec`. |
| [`API_COMPATIBILITY.md:3–6, 58–60`](https://github.com/triet4p/latent-anything/blob/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/API_COMPATIBILITY.md#L3) calls the API an unpublished candidate. | [Compatibility ledger](https://github.com/triet4p/latent-anything/blob/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/API_COMPATIBILITY.md#L3), corrected in `379aaec`. |
| [`EVIDENCE_GAP_PLAN.md:36, 45–46`](https://github.com/triet4p/latent-anything/blob/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/EVIDENCE_GAP_PLAN.md#L36) calls this the candidate boundary and says the package/API snapshot is unpublished. | [Evidence-gap plan](https://github.com/triet4p/latent-anything/blob/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/EVIDENCE_GAP_PLAN.md#L36), corrected in `379aaec`. |
| [`PLUGIN_AUTHOR_GUIDE.md:8–15, 62–64`](https://github.com/triet4p/latent-anything/blob/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/PLUGIN_AUTHOR_GUIDE.md#L8) makes the published 1.x contract conditional on release and calls the distribution unavailable. | [Plugin author guide](https://github.com/triet4p/latent-anything/blob/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/PLUGIN_AUTHOR_GUIDE.md#L8), corrected in `379aaec`. |
| [`PLUGIN_TEMPLATE.md:6–8`](https://github.com/triet4p/latent-anything/blob/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/PLUGIN_TEMPLATE.md#L6) calls `1.0.0` an unpublished candidate that cannot be installed. | [Plugin template](https://github.com/triet4p/latent-anything/blob/379aaec2a7936edb2b5ba08a125c4002d1f58a39/docs/PLUGIN_TEMPLATE.md#L6), corrected in `379aaec`. |

Use the linked current-source documents for API, evidence, and plugin
publication status. This clarification does not move or recreate either docs
tag, change the protected `v1.0.0` release, or alter any release asset.

The live [GitHub Pages site](https://triet4p.github.io/latent-anything/) is the
separate Latent-Anything Theory site; `mkdocs.yml` sets
`docs_dir: latent-anything-theory`. The combined deployment workflow publishes
that theory site at `/` together with the task-oriented `1.0.0` API guide at
`/api/` (built from `docs/public/` via `mkdocs-api.yml`) from one merged tree
on `theory-v*`/`guide-v*` tags or manual `workflow_dispatch`. Until that
combined deployment lands and is HTTP-verified, the intended
`/api/` URL remains pending and this record does not claim the guide is live
on Pages.