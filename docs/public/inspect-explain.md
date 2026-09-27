# Choose a method to inspect or explain a representation

Start from the question, not the plot. A projection can show geometry; a probe tests whether a declared label is readable; a clusterer groups samples; a density model ranks unusual samples; attribution methods test sensitivity to a chosen target; and temporal methods analyze ordered states. These are different claims, not interchangeable explanations.

This page describes selected APIs from the published `latent-anything==1.0.0` release. It is an instructional route, not a catalog of every stable export. The only executed example here is the small CPU digits workflow below. The UMAP, nonlinear-probe, GMM, SAE, TCAV, Integrated Gradients, and temporal sections are conditional method-selection guidance—not claimed execution or evidence for an arbitrary model.

## Choose by question

| Question | Starting method | Labels needed to fit? | What the result can answer |
| --- | --- | --- | --- |
| What structure is visible in a lower-dimensional view? | PCA or UMAP | No | How a fitted projection arranges these vectors under its objective |
| Can this representation predict a named property? | Linear probe; compare with a bounded MLP probe | Yes, one label per sample | Whether the property is linearly or nonlinearly accessible on the evaluated split |
| Do samples form groups, or does a sample look unusual relative to a reference distribution? | KMeans for groups; Gaussian-mixture density for density/OOD scores | No labels to fit; ID/OOD labels only for external evaluation | Assignment geometry, or relative density-based unusualness in one representation |
| Are learned sparse features active, stable, and useful for a declared task? | SAE feature evaluation | No for basic metrics; aligned labels/model inputs for optional cross-checks | Reconstruction, activation, stability, and bounded cross-check measures |
| Does a human-defined concept direction affect a selected target? | TCAV | Positive concept and reference examples; a scalar target | Target-gradient sensitivity along a learned concept direction |
| Which selected activation dimensions contribute to a selected transformer logit? | Activation-space Integrated Gradients | No class labels; requires a model, input, and explicit scalar target | Attribution along a baseline-to-activation path for one chosen target |
| Where does an ordered latent sequence smooth or change phase? | Trajectory smoothing, change-point detection, or DTW | No for analysis; boundary labels for evaluation | Geometry-aware smoothing, velocity-change candidates, or sequence alignment cost |

If you first need to define the sample unit, axes, coordinate identity, or trajectory shape, see [latent primitives and adapters](latent-primitives.md). Keep the model, checkpoint revision, layer, pooling/token selection, and preprocessing attached to the representation: matching feature dimensions alone does not make two representations comparable.

## Prerequisites and common input contract

Use Python `>=3.12,<3.15` and the published base package from PyPI: `python -m pip install "latent-anything==1.0.0"`. The base release includes NumPy, scikit-learn, UMAP, and PyTorch; the example uses only CPU and the local scikit-learn digits dataset. Do not download a checkpoint to run it.

Most methods below consume a numeric matrix `X` with shape `(n_samples, n_features)`: rows are the analysis units and columns are the features of one selected representation. Supply a one-dimensional label array of shape `(n_samples,)` only when the method is supervised. Token, image, time, and layer axes are not automatically understood as sample or feature axes. Decide whether to pool, select a token/layer, or analyze each group separately, and record that choice. For geometry-bound methods, use points from the same representation identity and the geometry expected by that method.

## Project to inspect structure: PCA or UMAP

**Use PCA** for a deterministic linear summary of variance, or **UMAP** when a nonlinear neighborhood layout is the question. Neither needs labels to fit. Fit on a declared reference/training matrix, then use that fitted object to transform the samples you intend to compare. Labels may be overlaid afterward for display or external checks; do not feed them into an unsupervised fit and then describe the resulting separation as discovery.

Both methods take a two-dimensional NumPy matrix and return a two-dimensional NumPy array with the requested component count. PCA centers the data and returns linear principal coordinates; `explained_variance_ratio_` summarizes variance captured by each component. UMAP approximates local-neighborhood structure and has stochastic choices, so set `random_state` when reproducibility matters. A small UMAP `min_dist` or visually distinct islands are not evidence of well-separated real populations.

Read PCA axes as high-variance directions, not necessarily task-relevant directions. Read UMAP primarily as a local-neighborhood visualization: its global distances, cluster sizes, and apparent gaps can change with hyperparameters, random seed, and sample composition. Compare multiple seeds/parameter settings and check whether a suspected neighborhood is also present in the original representation space. Neither plot establishes classification performance, a causal mechanism, or that the model uses the displayed feature.

Released implementations: [PCA](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/methods/pca.py) and [UMAP](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/methods/umap.py).

## Probe a labeled property: linear first, then bounded nonlinear

A probe answers: *can a simple readout recover this label from this representation on held-out examples?* `LinearProbe.fit(features, labels)` expects a 2D feature matrix and aligned 1D labels with at least two classes. It makes a stratified train/validation/test split; feature standardization, when enabled, is fitted on the training split only. Configure the split and seed deliberately. Its default row-wise split is not group-aware: if rows share a prompt, subject, sequence, or source document, keep related rows together in an independently designed evaluation instead of treating the default score as leakage-safe for that design.

`LinearProbeResult.accuracy` is held-out test accuracy. `val_accuracy` is `0.0` when validation is disabled (`val_size=0`), so treat it as a measured validation score only when validation was configured; `classes`, `predictions`, `probabilities`, `coefficients`, `intercept`, split masks, config, and provenance help interpret the result. Coefficient size is affected by feature scaling, regularization, class coding, and correlated dimensions; it is not a causal feature ranking. Compare against majority-class and shuffled-label controls on the same split, and repeat across predeclared seeds when stability matters. The released `cross_seed_evaluation` summarizes mean accuracy, confidence interval, and per-seed results.

Use an `MLPProbe` only after the linear baseline, with bounded hidden sizes, regularization, early stopping, and a validation split. `compare_probes` returns a typed `ProbeComparison` with linear/nonlinear accuracy, their gap, confidence-interval fields, and a qualitative classification. Its helper compares one seed; its interval fields are zero-width and do not measure repeat-run uncertainty. The associated shuffled-label memorization check is important: a nonlinear gain that also appears on shuffled labels can reflect probe capacity rather than useful structure. `MLPProbeResult` retains held-out predictions and architecture details; `MLPProbe.predict()` is not available for new data in the released contract.

A positive held-out probe result supports accessibility of that label under this sampling, split, representation, and probe capacity. It does not show that the base model relies on the label, that the representation improves the end task, or that the relationship is causal. Use a separately controlled intervention for a causal-use claim. The [bounded diagnostic guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md) is about its two named frozen diagnoses, not a generic endorsement of every probe run.
Released contracts: [linear probe](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/probes.py) and [nonlinear MLP probe](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/mlp_probe.py).


## Discover groups or score distribution shift: KMeans and GMM

### KMeans for unlabeled groups

`KMeans.fit_predict(X)` expects flat `(n_samples, n_features)` points and a chosen `n_clusters >= 2`. The default configuration standardizes features before fitting; the seed and number of initializations are part of the run definition. Supported geometries are Euclidean and unit-norm. If you have known labels, compare the resulting assignments afterward with permutation-invariant external metrics such as adjusted Rand index; labels are not used to fit KMeans.

The typed `KMeansResult` contains `assignments`, `centers`, `cluster_sizes`, `inertia`, `silhouette_score`, optional per-sample silhouettes, and a nearest-versus-second-nearest distance `confidence` margin. That margin is a confidence proxy, not a calibrated probability. Cluster IDs are arbitrary and can permute between fits. Check cluster stability across seeds and compare to an appropriate baseline before treating structure as robust.

A silhouette score or visually compact cluster does not establish a natural class, semantic meaning, causal role, or downstream performance. KMeans also requires a plausible `k` and is sensitive to scaling, outliers, and non-spherical structure.

### GMM density for relative OOD scoring

Use `GaussianMixtureDensity` when the question is whether held-out points look unusual relative to a declared in-distribution representation. Fit the GMM on training ID examples only; calibrate on a separate held-out ID set; then score separate ID and OOD evaluation sets. Keep the source representation identity, geometry, split provenance, and preprocessing fixed across those stages. The estimator rejects a mismatched declared identity and supports flat Euclidean or unit-norm representations.

`DensityResult.log_density` is fitted log density, `responsibilities` are mixture-component probabilities, and `calibrated_ood_score` is an empirical tail rank when calibration has been performed (higher means more unusual). Without a calibration set the same field contains the raw negative log-density score, not a calibrated probability. `DensityEvaluationReport` keeps both evaluation results, `DensityMetrics` (AUROC, AUPRC, Brier score, and score means), and split provenance. Do not fit or tune against the OOD evaluation set; report multiple seeds and compare with a simple Mahalanobis baseline when appropriate.

A density tail is relative to the fitted representation distribution, not a semantic error detector. High density does not guarantee correctness; low density does not prove an input is invalid. Performance depends on the chosen ID population, fit/calibration separation, dimensionality, and shift type. AUROC/AUPRC on one declared ID/OOD split are not guarantees under future deployment shifts.
Released contracts: [KMeans](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/clustering.py) and [Gaussian-mixture density/OOD](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/density.py).


## Evaluate sparse-autoencoder features

Use `SAEFeatureEvaluation` to assess a sparse decomposition of a fixed activation matrix, not as a substitute for inspecting the source model or examples. `fit(train_activations, val_data=..., source_representation_identity=...)` fits on training activations and evaluates on held-out activations. Each row must be one sample from the same layer/model/checkpoint coordinate system. If `val_data` is omitted, the evaluator makes a seeded validation split and requires at least the configured minimum number of validation rows. Preserve identity and split provenance when comparing results.

`SAEEvaluationResult` reports train and validation reconstruction MSE, mean L0/L1 activity, dead-feature counts/fraction, per-feature activation frequencies and norms, validation activations, and decoder weights. A low reconstruction error alone does not establish interpretable features. Feature indices are fit-specific: use the cross-seed stability result, which matches decoder directions, rather than comparing column numbers across independently trained SAEs. `rank()` / the feature atlas can show top- and bottom-activating examples; those examples are hypotheses to inspect, not automatic semantic labels.

Optional feature cross-checks have extra prerequisites. Labels supplied to the probe check must align with the SAE validation activations; that check includes shuffled labels. Concept sensitivity and steering agreement require the compatible model, token inputs, attention mask, scalar target, and layer. A feature is not thereby established as a cause: agreement with a separately controlled intervention is stronger evidence than activation frequency, example ranking, or probe accuracy alone. Stability is not guaranteed merely because the same SAE width was used.
Released contracts: [SAE feature evaluation](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/sae_evaluation.py) and the [SAE transform method](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/methods/sae.py).


## Explain a concept or a selected target: TCAV and Integrated Gradients

These are conditional, model-backed paths. The released implementations target a decoder-only transformer-style interface with named `transformer.h.{layer}` modules; they are not general methods for arbitrary model objects, and they do not include a model checkpoint or tokenizer. Use the pinned model setup and specialist evidence where applicable rather than treating this prose as a verified run.

### TCAV: sensitivity along a concept direction

TCAV needs positive concept activations and reference activations of shape `(n_examples, n_features)` from the same representation layer, coordinate system, model revision, and preprocessing. `ConceptDataset` records concept name and provenance and requires at least two examples from each group. It also needs evaluation token IDs, a matching attention mask, and a scalar `TransformerLogitTarget` (one token logit at a selected sequence position and batch item). Choose `mean_diff` or `linear_separator` direction fitting and predeclare the target/comparison family.

A `TCAVScore.sensitivity` is the fraction of evaluated examples with positive directional derivative of the chosen target along the learned direction; it is not a probability that the concept is present. `TCAVResult` includes seed aggregation and uncertainty, random-concept baseline scores, empirical and Bonferroni-corrected p-values, significance, and optional intervention agreement. Read concept-direction stability and held-out concept/reference separability too. Control the concept examples, reference examples, target choice, random concepts, seeds, and number of comparisons. A significant sensitivity is a target-specific association, not proof that the model causally uses the human concept.

### Integrated Gradients: one activation path, one scalar logit

`IntegratedGradients` attributes one selected residual-block activation vector at `target_layer`, `activation_position`, and `activation_batch_index` to one `TransformerLogitTarget`. Its input IDs and attention mask are 2D arrays with matching shapes. It integrates along a straight path from a declared zero, batch-mean, or explicit activation baseline using the chosen integration rule and number of steps. This is activation-space attribution; it does not attribute input tokens, explain every output, or support an arbitrary module naming scheme.

`IntegratedGradientsResult.attributions` has one value per selected hidden dimension. Read `target_input`, `target_baseline`, `attribution_sum`, `completeness_error`, `n_steps`, baseline, target, layer, and provenance together: completeness compares the sum with the selected target change along that path. Repeat with step counts, baselines, target choices, and a randomized control using the sensitivity evaluator. A small completeness error checks numerical accounting for that path; it does not certify the baseline is meaningful, the attribution is unique, or the identified dimensions are causal.

For the release-specific inputs and result types, consult [TCAV](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/tcav.py), [Integrated Gradients](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/integrated_gradients.py), and the [API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md). The [Sprint 80 evidence report](https://github.com/triet4p/latent-anything/blob/main/docs/SPRINT_80_DEPTH_EVIDENCE.md) distinguishes accepted evidence, a separate stability supplement, blocked work, and historical negatives; method availability does not erase those boundaries.

## Analyze a time-ordered latent sequence

Use a `Trajectory` for an ordered matrix of points shaped `(n_points, dim)` and a matching `LatentSpace` for geometry-aware distances. Keep time order and one consistent representation identity. `Trajectory` stores sequence order, not timestamps or a time axis inferred from arbitrary leading dimensions; keep a separate index-to-time mapping if the sampling interval is irregular.

- `smooth_trajectory` applies a centered moving average with an odd positive window and uniform or triangular weights. It returns a `SmoothedTrajectory` containing the new trajectory, provenance, and mean/maximum distortion from the original under the selected geometry. Check that the window does not erase real transitions and report distortion.
- `detect_change_points` looks for robust local changes in adjacent latent velocity. Its `ChangePointResult` contains boundary indices, half-open segments, scores, confidence, velocity, threshold, geometry, and provenance. These boundaries are candidate changes in the selected latent path—not automatically semantic events or timestamps.
- `evaluate_boundaries` compares predicted indices against pre-annotated boundaries with an explicit tolerance and reports precision, recall, F1, and matching counts. Use known synthetic transitions or independently annotated sequences as controls; map indices to clock time using your own recorded sampling metadata.
- `compute_dtw` aligns query and reference sequences in one `LatentSpace`, returning a typed distance, alignment path, point costs, normalization, and provenance. Choose a window, cost limit, and normalization before comparing sequences; check the `max_cells` bound for memory. DTW accommodates unequal sequence lengths, but an alignment is not evidence that two trajectories have the same cause or outcome.

For these APIs see the released [temporal methods](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/temporal.py), [DTW](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/dtw.py), and [trajectory contract](https://github.com/triet4p/latent-anything/blob/v1.0.0/src/latent_anything/trajectory.py).

## Runnable CPU example: projection, probe, and clusters on known data

This uses the bundled scikit-learn digits dataset, selecting only digits 0 and 1. PCA and KMeans do not consume labels; the probe does, and KMeans labels are compared only afterward. The probe owns its stratified held-out split. KMeans is fitted and scored on the same rows here, so its adjusted Rand index is descriptive agreement—not a held-out prediction score. The PCA fit also uses the full subset and is only a visualization projection, not part of the probe's evaluation.

Run after installing `latent-anything==1.0.0`:

```python
import numpy as np
from sklearn.datasets import load_digits

from latent_anything.clustering import KMeans, KMeansConfig, compare_with_labels
from latent_anything.methods import PCA
from latent_anything.probes import LinearProbe, LinearProbeConfig


digits = load_digits()
keep = np.isin(digits.target, (0, 1))
features = digits.data[keep].astype(np.float64)
labels = digits.target[keep].astype(np.int64)
assert set(np.unique(labels)) == {0, 1}

projection = PCA(n_components=2)
projection.fit(features)
coordinates = projection.transform(features)
explained = float(projection.explained_variance_ratio_.sum())
assert coordinates.shape == (features.shape[0], 2)
assert explained > 0.30

probe = LinearProbe(
    LinearProbeConfig(test_size=0.25, val_size=0.0, random_state=82)
)
probe_result = probe.fit(
    features,
    labels,
    provenance={"dataset": "sklearn load_digits, classes 0 and 1"},
)
assert probe_result.accuracy >= 0.95
assert probe_result.predictions.shape == (int(probe_result.test_indices.sum()),)

cluster_result = KMeans(
    KMeansConfig(n_clusters=2, n_init=10, random_state=82)
).fit_predict(features)
cluster_agreement = compare_with_labels(cluster_result.assignments, labels)
ari = cluster_agreement["adjusted_rand_index"]
assert ari > 0.50

print(f"digits 0/1: {features.shape[0]} examples, {features.shape[1]} input features")
print(f"PCA: coordinates={coordinates.shape}, explained_variance={explained:.3f}")
print(f"LinearProbe: held-out accuracy={probe_result.accuracy:.3f}")
print(f"KMeans: sizes={cluster_result.cluster_sizes.tolist()}, ARI={ari:.3f}")
```

The thresholds are meaningful checks on this well-separated two-digit subset; the printed measurements are observations about this data and configuration only. They are not evidence for other models, datasets, probe labels, or deployed performance.

## Evidence boundaries and next steps

The accepted `1.0` diagnostic evidence is limited to two frozen ordinary-DL cases—the prospective linear-autoencoder lesion on a held-out digits split and the pinned GPT-2/WikiText target-evidence-v2 case—plus the separately frozen transformer stability supplement. The supplement does not retroactively alter the immutable target-evidence-v2 artifact. Those case-specific controls and intervention results do not mean that PCA/UMAP separation or probe accuracy alone explains a model, proves causal use, or validates task performance. Arbitrary-model and arbitrary-dataset diagnosis, GPU/CUDA claims, and the blocked non-gating SmolVLA lane are outside that evidence scope. `DiagnosticWorkflow` is not a stable public import.

For release signatures and exact result schemas, use the [frozen API reference](https://github.com/triet4p/latent-anything/blob/v1.0.0/docs/API_REFERENCE.md) and [1.0.0 API snapshot](https://github.com/triet4p/latent-anything/blob/v1.0.0/artifacts/api_freeze_snapshot_1.0.0.json). For the named diagnosis runs and their controls, see the [AI engineer guide](https://github.com/triet4p/latent-anything/blob/main/docs/AI_ENGINEER_GUIDE.md) and [Sprint 80 evidence report](https://github.com/triet4p/latent-anything/blob/main/docs/SPRINT_80_DEPTH_EVIDENCE.md). The site's `/api/` route is not publicly deployed or verified by a local build; see [guide home](index.md) for the publication boundary and other task routes.
