# Visualization — Browser / Manual Visual QA Checklist

The automated tests assert the *structure* of the renderer inputs and the
figures (schema, trace types, hover payloads, downsampling). They cannot
verify that the interactive figure *behaves* well in a browser. Use this
checklist before merging any change to `src/latent_anything/visualization/`
or to the interactive walkthrough artifacts.

Requirement: each interactive chart must be accompanied by the quantitative
metrics it displays (per [docs/PLAN.md](PLAN.md): *"a visually clean
projection is not accepted as an explanation by itself"*). The charts in
`scripts/interactive_viz_walkthrough.py` follow this rule.

## Environment

1. Install the optional extra: `uv sync --extra viz`.
2. Start a notebook: `uv run jupyter notebook` (or `jupyter lab`).
3. Open a fresh notebook and run:

   ```python
   from latent_anything.visualization import render
   from scripts.interactive_viz_walkthrough import build_digits_views
   views = build_digits_views()
   explorer = render(views["kmeans"])
   explorer.show()
   ```

## 1. Interaction basics (2D)

- [ ] The figure renders in the notebook cell without errors.
- [ ] Hovering a point shows its label, category, and per-point metadata
      (confidence / probability / OOD score) in a tooltip.
- [ ] The legend lists each category; clicking a legend entry toggles that
      category's visibility.
- [ ] Box/lasso selection tools in the mode bar select a subset of points.
- [ ] Zoom / pan / autoscale work and the hover overlay stays aligned.

## 2. Trajectory overlays

- [ ] The trajectory overlay is drawn as a connected line with markers on top
      of the background scatter.
- [ ] Hovering a trajectory point shows its step index and per-step metadata.
- [ ] The overlay does not affect selection of background points.

## 3. 3D explorer

- [ ] `projection_from_trajectory(..., coordinates3d)` produces an interactive
      3D scene with orbit / zoom controls.
- [ ] Hover text and per-point metadata render correctly in 3D.

## 4. Notebook widget path

- [ ] `explorer.show()` in a notebook displays the ipywidgets container
      (Plotly figure + inspection panel) — not just raw HTML.
- [ ] The inspection panel shows the view summary (point count + metrics).
- [ ] Hovering a point updates the inspection panel with that point's
      metadata.
- [ ] When the same code runs as a plain script (`python script.py`),
      `explorer.show()` degrades to a static HTML string without errors.

## 5. Static export

- [ ] `explorer.save("out.html")` writes a self-contained HTML file that
      opens in a browser without internet access (plotly.js is inlined).
- [ ] `explorer.save("out.png")` writes a valid PNG (kaleido backend).
- [ ] The PNG renders the title and the metrics annotation legibly at
      `1200x800` and above.

## 6. Responsiveness / downsampling

The declared targets are `DEFAULT_POINT_LIMIT_2D = 50_000` and
`DEFAULT_POINT_LIMIT_3D = 20_000` (see
`src/latent_anything/visualization/data.py`).

- [ ] A view with more points than the limit renders smoothly; the metrics
      annotation reports `downsampled (dropped N)`.
- [ ] Downsampling is deterministic: rendering the same view twice with the
      same seed keeps the same points (verified by
      `tests/test_visualization_data.py`).
- [ ] Trajectory overlays are never downsampled — they remain fully visible
      even when the background scatter is thinned.

## 7. Layout / typography

- [ ] Titles and axis labels are legible at default sizes.
- [ ] The metrics annotation (top-left) does not overlap the legend or data.
- [ ] Category colors are distinguishable (qualitative palette cycles; pairs
      like blue/orange/teal are not adjacent-confusable).
