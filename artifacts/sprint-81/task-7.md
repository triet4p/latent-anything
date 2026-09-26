# Task Summary: Sprint 81 Task 7 — Post-Publication Consumer Smoke

**Sprint:** 81  
**Task:** 7 — Verify post-publication install, links, plugin discovery, and one lightweight end-to-end example (`[~]`)  
**Plan status:** Remains `[~]`; Main owns evidence review and any status change.

## Outcome

A fresh CPython 3.13.3 environment installed `latent-anything==1.0.0` from public PyPI with `uv`'s cache disabled. The installed module came from that environment's `site-packages`; an independently fetched PyPI wheel matched the published SHA-256. The immutable GitHub/PyPI release and corrected `docs-v1.0.0-r1` links resolved, and the separately installed external hello-world plugin was listed without import, explicitly loaded, constructed through `ObjectSpec`, called, and reported with provenance. The release-tagged lightweight CPU encoder diagnostic completed all seven workflow stages, passed its acceptance gate and independent report validator, and emitted a deterministic rendered report.

The public PyPI long description and immutable `docs-v1.0.0-r1` plugin author/template guides still contain pre-release wording that says `1.0.0` is unpublished or pending. Those records cannot be changed without a new publication/tag, which this task does not authorize. The current working-tree plugin guides were corrected to describe the published `1.x` contract; the immutable public copies remain an open documentation finding.

## Changed Files

- [`docs/PLUGIN_AUTHOR_GUIDE.md`](../../docs/PLUGIN_AUTHOR_GUIDE.md) — corrected the current guide's version-1 contract and published support-policy wording.
- [`docs/PLUGIN_TEMPLATE.md`](../../docs/PLUGIN_TEMPLATE.md) — corrected the current template's framework dependency/publication wording.
- `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.html`, `graphify-out/.graphify_analysis.json`, `graphify-out/.graphify_labels.json`, `graphify-out/.graphify_labels.json.sig`, and `graphify-out/manifest.json` — scoped graph refresh outputs.
- `graphify-out/cache/ast/v0.9.68-s4/` — three structural-extraction cache records for the two plugin guides and this artifact.
- [`artifacts/sprint-81/task-7.md`](task-7.md) — this verification record.

The sprint plan remains owned by Main and its Task 7 checkbox was not changed. No package source, PyPI release asset, GitHub Release, or tag was changed.

## Verification

### Public package install and release links

Commands, run outside the repository source tree:

```powershell
uv venv C:/Users/admin/AppData/Local/Temp/omp-sprint81-task7-pypi-20260927 --python 3.13
uv pip install --python C:/Users/admin/AppData/Local/Temp/omp-sprint81-task7-pypi-20260927/Scripts/python.exe --index-url https://pypi.org/simple --no-cache latent-anything==1.0.0
C:/Users/admin/AppData/Local/Temp/omp-sprint81-task7-pypi-20260927/Scripts/python.exe -c "import importlib.metadata, latent_anything, sys; d=importlib.metadata.distribution('latent-anything'); print('version='+d.version); print('module='+str(latent_anything.__file__)); print('prefix='+sys.prefix); print('python='+sys.version.split()[0])"
```

`uv` resolved and installed 40 packages from `https://pypi.org/simple`; no editable install, local source wheel, or uv cache was used. The import check reported `version=1.0.0`, Python `3.13.3`, and `latent_anything.__file__` under `.../omp-sprint81-task7-pypi-20260927/Lib/site-packages/latent_anything/`, with `sys.prefix` set to the isolated environment. The published wheel was independently fetched and hashed against PyPI JSON: `latent_anything-1.0.0-py3-none-any.whl`, 635,199 bytes, SHA-256 `3f7081d4cb8994c6a719d85d51a3c0ccff76171673c5a1dc33737d7770b408ea`. The published sdist was 886,293 bytes, SHA-256 `36b6b3950fd791cf9ffa5eeac79050c5052d5b9902d1c98a9a2b7a6d38e8e3a9`. These match the task 5 release record. Direct GETs of both PyPI package files and their GitHub Release counterparts returned HTTP 200 and matching size/hash; GitHub `SHA256SUMS` returned HTTP 200 and contained those same two digests.

The following public URLs returned HTTP 200 in direct link checks (raw text was also fetched for the tagged docs):

- [PyPI 1.0.0 JSON](https://pypi.org/pypi/latent-anything/1.0.0/json) and [PyPI 1.0.0 project page](https://pypi.org/project/latent-anything/1.0.0/).
- [GitHub v1.0.0 Release](https://github.com/triet4p/latent-anything/releases/tag/v1.0.0); public tag ref resolves annotated tag object `3bdb7ff2c5e09def32e39b9b4628750729104b22` to exact release commit `a449ca33b4b83ce109c29db7cf471919820b1d56`.
- [Corrected docs-v1.0.0-r1 tag](https://github.com/triet4p/latent-anything/tree/docs-v1.0.0-r1); public tag ref resolves annotated tag object `653b34f7608913a49531026a266e93deb6818f8a` to commit `3cbc925de464a090bc1d65a2fc8858a11a0dcbf3`.
- Commit-pinned [versioned docs companion](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/VERSIONED_DOCS_1.0.0.md), 6,611 bytes / SHA-256 `f92766f5fda3998559d58247bde9a75bd86b07479648f1c8bea9c809d0df8e28`.
- Commit-pinned [metadata-only revision archive](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/artifacts/benchmark_revision_archive_1.0.0.json), 13,507 bytes / SHA-256 `269b9148cedb0d459c901944ac0c7d7bbd188f90724162a1319108b855b100ff`.
- Commit-pinned public [plugin author guide](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/PLUGIN_AUTHOR_GUIDE.md) and [plugin template](https://raw.githubusercontent.com/triet4p/latent-anything/3cbc925de464a090bc1d65a2fc8858a11a0dcbf3/docs/PLUGIN_TEMPLATE.md) both returned HTTP 200; author guide 4,907 bytes / SHA-256 `140d47ba84752545f2549d9283f2ee15f0efda529f4a99450a47325a6050f48b`, template 1,593 bytes / SHA-256 `921aa439e36f40a3b7bc9e89e1500b9326da1e00d4695386d8c290446e3f0b9f`.

### External plugin discovery and behavior

The external fixture distribution was installed separately into a target directory, while the core framework remained the public PyPI install above:

```powershell
uv pip install --python C:/Users/admin/AppData/Local/Temp/omp-sprint81-task7-pypi-20260927/Scripts/python.exe --index-url https://pypi.org/simple --no-cache --no-deps --target C:/Users/admin/AppData/Local/Temp/omp-sprint81-task7-pypi-20260927/plugin-site tests/fixtures/sprint73_hello_plugin
```

A clean child process (`PYTHONPATH=""`, `PYTHONNOUSERSITE="1"`) added only `plugin-site` to `sys.path`, then used the installed framework's `list_entry_points`, `load_entry_points`, `Registry`, `ObjectSpec`, and `build_from_config`. It asserted the framework import path was in the isolated venv, listing did not import the plugin module, explicit loading did import it, there was one supported entry point and no issues, provenance identified an external package, and the built adapter returned the expected value. The temporary assertion script was `C:/Users/admin/AppData/Local/Temp/omp-sprint81-task7-pypi-20260927/plugin_smoke.py` and was removed after the smoke.

Exact runtime command:

```powershell
powershell -NoProfile -Command '$env:PYTHONPATH=""; $env:PYTHONNOUSERSITE="1"; & "C:\Users\admin\AppData\Local\Temp\omp-sprint81-task7-pypi-20260927\Scripts\python.exe" "C:\Users\admin\AppData\Local\Temp\omp-sprint81-task7-pypi-20260927\plugin_smoke.py"'
```

Observed assertion output:

```json
{"after_listing":false,"after_loading":true,"before_listing":false,"call_result":"hi:world","framework_path":"C:\\Users\\admin\\AppData\\Local\\Temp\\omp-sprint81-task7-pypi-20260927\\Lib\\site-packages\\latent_anything\\__init__.py","framework_version":"1.0.0","issues":[],"listed":[["hello-world","latent_anything.adapter","latent-anything-hello-plugin","0.1.0"]],"loaded":["hello-world"],"provenance":{"distribution":"latent-anything-hello-plugin","entry_point_group":"latent_anything.adapter","entry_point_value":"latent_anything_hello:HelloAdapter","plugin_api_version":"1","source":"external","version":"0.1.0"}}
```

### Installed-package lightweight diagnostic

The smoke used the release-tagged encoder-v3 example from commit `a449ca33b4b83ce109c29db7cf471919820b1d56`, fetched outside the source tree (8,442 bytes / SHA-256 `63f3f9c8806cfc4a51a6232d0138981e97526fc8aada59496a5a6d207bf3094f`). Its two proof scripts were verified byte-for-byte against that release commit: `scripts/sprint80_task80_23_v3_proof.py` (51,208 bytes / SHA-256 `87891b0c75d4ddcfe890792e421ac4108801d08aa49a9d3a234da764481cb53f`) and `scripts/sprint80_task80_23_proof.py` (50,540 bytes / SHA-256 `4b574d04a1a58c9508798197da3b6cda4909ba203b5f74f97b383e89a57f2922`). Required frozen repository inputs were staged in the external scratch directory; they were not imported as framework code. The environment used the release guide's `scikit-learn==1.9.0` pin.

Exact final invocation (from the external scratch directory):

```powershell
powershell -NoProfile -Command '$env:PYTHONPATH=""; $env:PYTHONNOUSERSITE="1"; & "C:\Users\admin\AppData\Local\Temp\omp-sprint81-task7-pypi-20260927\Scripts\python.exe" "C:\Users\admin\AppData\Local\Temp\omp-sprint81-task7-pypi-20260927\scripts\ai_engineer_example_encoder.py" --output artifacts/diagnostics/task7-public-pypi-encoder-v3-release-20260927'
```

Run `a6db6481e3483c1a` completed with acceptance passed. Capture, detect, localize, explain, intervene, compare, and report all ran; the independent diagnostic-report validator passed and report rendering was deterministic. The controlled lesion was localized to `brightness-axis-dim0`; the lesion report recorded feature-variance ratio `0.0`, singular spread `0.0`, and effective rank `2.9576990034956805`. Healthy-counterexample, benign-low-variance, and null-shuffle controls passed. The healthy-vs-lesion held-out brightness-bin accuracy was `1.0` vs `0.5361111111111111` (delta `-0.4638888888888889`); the report's restore-lesioned-dim0 intervention was supported. Report limitation: this is a prospective controlled linear-autoencoder lesion on one frozen digits split, not evidence about historical ConvVAE or production encoders; the brightness-bin task is not general digit recognition or cross-dataset utility.

Output path: `C:/Users/admin/AppData/Local/Temp/omp-sprint81-task7-pypi-20260927/artifacts/diagnostics/task7-public-pypi-encoder-v3-release-20260927/`. Run artifact SHA-256 `a4ed225a29c9e346a005759d57c7a40bb3c22e9904223b015a947abacda8f7d6`; report SHA-256 `8fb1528f39d355a311d2a8db39e9cf7db6455e1c7392915e21f377a05d13f194`; rendered report SHA-256 `9aa1c46f3be3dc77ee5300a057f8eb0dca207f6453d44c96a8ef46d9c639dbf9`. No GPU or pretrained-model download was used.

### Scope and findings

- Public PyPI JSON reports version `1.0.0` and the correct wheel/sdist hashes, but its immutable description still says the `1.0.0` candidate has not been tagged/released, PyPI's first upload is pending, and installation uses only `0.9.0`. It also calls the candidate unpublished in the Quick Start.
- Public `docs-v1.0.0-r1` plugin guide/template links resolve, but those immutable files still say 1.0.0 is unpublished and make the stable plugin contract/dependency range conditional on publication. The current working-tree `docs/PLUGIN_AUTHOR_GUIDE.md` and `docs/PLUGIN_TEMPLATE.md` now use the published version wording; no immutable docs tag was moved or recreated.
- No PyPI upload, GitHub Release asset, package tag, or docs tag was changed. Correcting the published PyPI long description or the immutable r1 files would require a separately authorized publication/tag.
- No project-wide test suite, build, formatter, or linter was run. Verification was limited to the specified clean public install, public link/hash checks, plugin consumer scenario, and CPU diagnostic example.

### Graph update

- **Scoped extraction and merge:** Graphify's incremental scan found 320 changed files overall. To avoid absorbing unrelated or sibling changes, only the two updated plugin guides and this Task 7 artifact were re-extracted. Structural Markdown extraction produced 23 nodes / 23 edges; inline semantic extraction produced 11 concept nodes / 11 `references` edges. Starting from 16,723 nodes / 39,325 edge keys / 37 hyperedges, `build_merge` produced 16,742 nodes / 39,347 edges / 37 hyperedges. It preserved all 16,708 nodes, 39,313 edge keys, and 37 hyperedges outside the three selected sources. The manifest was stamped only for those three selected documents. The graph's inherited `built_at_commit` remains `a449ca33b4b83ce109c29db7cf471919820b1d56`.
- **Clustering and integrity:** four `graphify.exe cluster-only .` passes completed. The first run found 1,055 communities against 1,070 saved labels and renamed 213 by hub; its second pass was clean. After canonicalizing Task 7 semantic node IDs, the next pass found 1,061 communities against 1,055 labels and renamed 29 by hub; the fourth pass was clean. Final clustering found 1,061 communities and regenerated the aggregate HTML view with 1,061 community nodes / 1,855 cross-community edges. `graphify.exe diagnose multigraph --json` reported 16,742 nodes / 39,347 edges, zero non-object edges, missing or dangling endpoints, external references, self-loops, duplicate edges, collapsed endpoint pairs, or post-build errors. It reported one unverified node, the `π0 (Pi0)` entry from `latent-anything-theory/10-world-models-vla/research/07-pi0.md`; that same node existed in the pre-refresh snapshot and is outside this task's scope.
- **Final artifact-only refresh:** after recording the graph results here, a final source-scoped extraction replaced only the Task 7 artifact's 10 Markdown structural nodes / 11 edges and four semantic concepts / four edges. It preserved all 16,728 nodes, 39,332 edge keys, and 37 hyperedges outside that artifact; totals remain 16,742 nodes / 39,347 edges / 37 hyperedges. The four semantic IDs use the canonical `artifacts_sprint_81_task_7_...` slug. The final Graphify scan queued 318 other files, but none of the three task-owned documents remained queued; no unrelated files were re-extracted.
- **Graph backup/version notes:** Graphify 0.9.68 emitted its known warning that the installed skill copy is 0.9.32. The first cluster pass warned that 1,070 saved community labels did not match 1,055 communities and renamed 213 labels by hub; the second pass completed without that warning. No LLM label pass was run. Before refreshing, the current pre-task graph was preserved outside the repository at `C:/Users/admin/AppData/Local/Temp/omp-sprint81-task7-pypi-20260927/graphify-before-task7/graph.json` (SHA-256 `bb27960771f41cb7426cdfd207f8c99eeeab9813d97568ed55e3fac0fe0de3ce`). `GRAPHIFY_NO_BACKUP=1` prevented Graphify from overwriting the existing Task 6 dated safety snapshot; its `graph.json` hash remained `c25f27bd97ff122b9f38d49eecea35984b36832c5a2ea2ac8f39b880c092d4f0`.
