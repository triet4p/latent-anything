"""Built-in plugin registrations — registry-first built-in adapters and methods.

This module is the **single stable import location** where all built-in
adapters and methods are registered into ``GLOBAL_REGISTRY``. It exists
as a separate module so that ``latent_anything.registry`` remains
infrastructure-only — no knowledge of concrete adapter or method classes.

By importing this module (or importing ``latent_anything`` which re-exports
it), all built-in entries become available in ``GLOBAL_REGISTRY`` without
callers having to import each class directly.

Internal plugin extraction contract
-----------------------------------
1. **Registration-only.** This module does **not** re-export classes.
   Built-in classes remain importable from their canonical locations
   (``latent_anything.adapters``, ``latent_anything.methods``).

2. **Deterministic order.** Registrations are ordered by kind (adapters
   first, then Layer A, then Layer B) and alphabetically within each
   group. This matches the iteration order of ``GLOBAL_REGISTRY.list()``.

3. **No circular imports.** This module imports from the concrete
   sub-packages (adapters, methods), which in turn import from
   infrastructure (latent_space, trajectory, protocols). Neither adapters
   nor methods import ``registry`` or ``_plugin_builtins``, so the
   dependency graph stays acyclic.

4. **One-to-one with built-in classes.** Every adapter and method that
   ships with the ``latent_anything`` package is registered here. When a
   new built-in class is added, a corresponding ``GLOBAL_REGISTRY.register``
   call must be added here.

5. **Entry-point readiness.** When the project introduces external plugin
   discovery via Python ``importlib.metadata`` entry points (future), the
   external-plugin loader will populate a **separate** ``Registry``
   instance. Built-in registrations remain here regardless.
"""

from __future__ import annotations

from latent_anything.adapters.conv_vae import ConvVAE
from latent_anything.adapters.gaussian_renderer import GaussianRendererAdapter
from latent_anything.adapters.hidden_state import HiddenStateAdapter
from latent_anything.adapters.random_projection import RandomProjection
from latent_anything.adapters.vae import VAE
from latent_anything.clustering import KMeans
from latent_anything.density import GaussianMixtureDensity
from latent_anything.geodesic import DensityGeodesic
from latent_anything.integrated_gradients import IntegratedGradients
from latent_anything.methods.activation_patch import ActivationPatch
from latent_anything.methods.lerp import Lerp
from latent_anything.methods.pca import PCA
from latent_anything.methods.sae import SAE
from latent_anything.methods.steering import SteeringVector
from latent_anything.methods.umap import UMAP
from latent_anything.mlp_probe import MLPProbe
from latent_anything.probes import LinearProbe
from latent_anything.projection import SubspaceProjection
from latent_anything.registry import GLOBAL_REGISTRY, KIND_ADAPTER, KIND_ANALYSIS, KIND_INTERVENTION
from latent_anything.sae_evaluation import SAEFeatureEvaluation
from latent_anything.tcav import TCAV

# ── Register adapters ───────────────────────────────────────────────

GLOBAL_REGISTRY.register(
    KIND_ADAPTER,
    "conv_vae",
    ConvVAE,
    description="Convolutional VAE for image batches — explicit learned latent (mode i)",
    protocol="ModelAdapter, DecodableAdapter, FlatBatchDecodableAdapter",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ADAPTER,
    "gaussian_renderer",
    GaussianRendererAdapter,
    description="2D Gaussian splat renderer — deterministic decode (mode iii)",
    protocol="ModelAdapter, DecodableAdapter",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ADAPTER,
    "hidden_state",
    HiddenStateAdapter,
    description="Hidden-state activations — no-explicit-latent (mode ii)",
    protocol="ModelAdapter",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ADAPTER,
    "random_projection",
    RandomProjection,
    description="Random fixed-weight projection (mode i-like, stateless)",
    protocol="ModelAdapter, DecodableAdapter, FlatBatchDecodableAdapter",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ADAPTER,
    "vae",
    VAE,
    description="Variational Autoencoder — explicit learned latent (mode i)",
    protocol="ModelAdapter, DecodableAdapter, FlatBatchDecodableAdapter",
    source="built-in",
)

# ── Register Layer A methods ────────────────────────────────────────

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "pca",
    PCA,
    description="Principal Component Analysis — linear dimensionality reduction",
    protocol="Method",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "sae",
    SAE,
    description="Sparse Autoencoder — neural/trained with L1 sparsity",
    protocol="Method",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "umap",
    UMAP,
    description="Uniform Manifold Approximation and Projection — nonlinear dim-reduction",
    protocol="Method",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "kmeans",
    KMeans,
    description="K-means clustering — latent structure discovery with geometry checks and diagnostics",
    protocol="KMeans",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "gaussian_mixture_density",
    GaussianMixtureDensity,
    description="Representation-bound Gaussian-mixture density and calibrated OOD scoring",
    protocol="GaussianMixtureDensity",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "linear_probe",
    LinearProbe,
    description="Label-aware linear classification probe — logistic regression with leakage-guarded split",
    protocol="LinearProbe",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "mlp_probe",
    MLPProbe,
    description="Bounded nonlinear MLP probe — bounded-capacity MLP with early stopping",
    protocol="MLPProbe",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "tcav",
    TCAV,
    description="TCAV analysis — concept-sensitivity via directional derivatives",
    protocol="TCAV",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "integrated_gradients",
    IntegratedGradients,
    description="Activation-space Integrated Gradients for scalar transformer logits",
    protocol="IntegratedGradients",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_ANALYSIS,
    "sae_evaluation",
    SAEFeatureEvaluation,
    description="Sparse-autoencoder feature evaluation — reconstruction, sparsity, stability, cross-check, atlas",
    protocol="SAEFeatureEvaluation",
    source="built-in",
)

# ── Register Layer B methods ────────────────────────────────────────

GLOBAL_REGISTRY.register(
    KIND_INTERVENTION,
    "activation_patch",
    ActivationPatch,
    description="Activation patching — B-Method #3 (model-mediated data→data)",
    protocol="BMethod",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_INTERVENTION,
    "density_geodesic",
    DensityGeodesic,
    description="Density-penalized geodesic path interpolation — non-Euclidean path optimization",
    protocol="DensityGeodesic",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_INTERVENTION,
    "lerp",
    Lerp,
    description="Linear/geodesic interpolation — B-Method #1 (stateless latent→latent)",
    protocol="BMethod",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_INTERVENTION,
    "steering",
    SteeringVector,
    description="Steering vector — B-Method #2 (stateful latent→latent via contrast)",
    protocol="BMethod",
    source="built-in",
)

GLOBAL_REGISTRY.register(
    KIND_INTERVENTION,
    "subspace_projection",
    SubspaceProjection,
    description="Orthonormal subspace projection — project onto / remove a fitted subspace",
    protocol="SubspaceProjection",
    source="built-in",
)
