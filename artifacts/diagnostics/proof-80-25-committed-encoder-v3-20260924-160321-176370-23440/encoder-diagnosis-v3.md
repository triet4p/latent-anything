# Diagnostic Report report-80-23-encoder-model-lesion-v3

## Identifiers
render schema: diagnostic-report-render-v1
artifact schema: diagnostic-artifact-v1
report schema: diagnostic-report-schema-v1
run id: ade7b3144207bb88
artifact digest: fedfe97792d8f08fb5a343e2d259109000de4a79255efcb34442098d257e1f93
report digest: 8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194
manifest: sprint80-core-encoder-autoencoder-collapse-model-lesion-v3
manifest sha256: f823efc1b016ca9ad62c0f207e17d138947040a39f5d1c1f42ab6d0d65f5954f
taxonomy sha256: 77728d92e97e237dd9e07cefe862ab94d684dc8a99319eb3bd124e35352a57a5
report schema sha256: 8dd0b2a248284e068e693991f76147b3424f334acf55ac7a800fea5cd1b0aa4b
workflow identity: 51355c2a22cc8ebef694b6e789d16ac6d7608f7513262758fe708cbc28d811ee
request id: proof-80-23-v3-model-lesion
validator: passed

## Symptom
description: A prospective model-level lesion zeroed encoder weight row 0 and its bias in a copy of the train-only linear autoencoder before heldout encoding; the frozen per-feature variance ratio was 0.
metrics: bottleneck-feature-variance-ratio
status: supported
family: collapse_rank_loss
bounded conclusion: supported
reason: metrics flag a defect predeclared thresholds with required controls passed
hypothesis: The localized dim0 coordinate is the model's predeclared linear image-mean direction: the train-only encoder emits centered global brightness on this coordinate and the fixed downstream readout thresholds it using the training median.
hypothesis status: supported

## Location
axes: sample, feature, slice
representation: linear_pca_autoencoder:bottleneck:latent_dim=4
location: feature brightness-axis-dim0 (confidence 1.0, status supported)

## Evidence
metric bottleneck-effective-rank: estimate 2.9576990034956805, interval [2.875274672590536, 2.9839066845293694], status observed
metric bottleneck-singular-spread: estimate 0.0, interval [0.0, 0.0], status observed
metric bottleneck-feature-variance-ratio: estimate 0.0, interval [0.0, 0.0], status observed
control control-benign-low-variance: passed
control control-healthy-counterexample: passed
control control-null-shuffle: passed
family evidence collapse_rank_loss: collapse_healthy_counterexample=observed, collapse_negative_control=observed, collapse_provenance=observed, collapse_rank_profile=observed, collapse_variance_or_singular_values=observed
explanation h-brightness-axis-dim0-v3: probe outcome supported
evidence stage-record-capture: 2a78208b19117f3322c488aaa96a66808254dd3d0a2b82ea051979982004206b
evidence stage-record-detect: 9f1738367b9eee150f5b3065fa8b937c6df29ef6c2c3cd2657c22d5b1b9d0412
evidence stage-record-localize: ebb664af871d7d0a57d4294b8b59e59eff5c4f9eedc27912c78ca4f5398487f0
evidence stage-record-explain: 4d70cd7b1f4eea577d1648425490d4e92829f8df32740f8211060c04ead4b5ef
evidence stage-record-intervene: 22506f6412b76ffc3c135cde2e2438edeb11b4949d80c5b44f5484ad6fc3beb2
evidence stage-record-compare: 1270382c4dc081abfb042de065dd16862b7c5f3f72348cb38d64daf029afd41c
evidence stage-record-report: 031daeeccf5e63c374860b2ca536007f3a1668105013b6418bae3f27d358784c
evidence capture-capture-linear-autoencoder-bottleneck-v3-record: 95fb45cc3d9dee15335ab45acf02d337b1959bdbe21ef14c3932f061053e0bda
evidence stage-record-intervene-restore-lesioned-encoder-dim0-v3: 63daf9c06ac88458a0b4ed4e93a265c98e9f3d15d571af1ccec9d6d870dc3c14
evidence comparison-cmp-healthy-vs-model-lesion-v3-record: 512371d5725ae31e86aed529bf795548e719dc70327254e4cb843ea82f2ce0a9
evidence explanation-h-brightness-axis-dim0-v3-record: 0d46d2c72ac10ad4e4e8939fe1275b2d6084187352ace50ff316b2439a88f46a
evidence diagnostic-report: 8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194

## Causal
intervention patch on brightness-axis-dim0: intervention-restore-lesioned-encoder-dim0-v3-record, conclusion supported
record stage-record-intervene-restore-lesioned-encoder-dim0-v3: 63daf9c06ac88458a0b4ed4e93a265c98e9f3d15d571af1ccec9d6d870dc3c14
claim intervention-restore-lesioned-encoder-dim0-v3: supported (claim allowed yes)

## Comparison
comparison cmp-healthy-vs-model-lesion-v3: observed (classification both), baseline linear-autoencoder-heldout-healthy-v3 (baseline-92829e57aaaebf52), candidate linear-autoencoder-heldout-lesion-v3 (lesion-81a29cc8d92b3816)
metrics: bottleneck-feature-variance-ratio, heldout-brightness-bin-accuracy
representation bottleneck-feature-variance-ratio: signed delta -0.8494925859561244, absolute delta 0.8494925859561244, tolerance 0.05, changed yes
task heldout-brightness-bin-accuracy: signed delta -0.4638888888888889, absolute delta 0.4638888888888889, tolerance 0.05, changed yes
record comparison-cmp-healthy-vs-model-lesion-v3-record: 512371d5725ae31e86aed529bf795548e719dc70327254e4cb843ea82f2ce0a9

## Limitations
limitation lim-controlled-model-scope-v3: This is a prospective controlled linear-autoencoder weight lesion on one frozen digits split, not evidence that the historical ConvVAE or a production encoder has the same defect. (affects detect, localization, intervene, compare; blocking no)
limitation lim-task-scope-v3: The downstream task is a predeclared train-median brightness bin, not a general-purpose digit recognition or cross-dataset utility measure. (affects explain, compare; blocking no)

## Next action
action: retain the frozen v3 proof as prospective evidence and keep historical v1/v2 results unchanged
rationale: the artifact binds the model-level lesion, causal controls, task-utility comparison, and validated report
requires evidence symptom-model-weight-lesion-v3
requires evidence stat-bottleneck-feature-variance-ratio
requires evidence cmp-healthy-vs-model-lesion-v3
