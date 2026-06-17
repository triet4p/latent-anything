"""Introspection methods for latent space analysis."""

from latent_anything.methods.pca import PCA as PCA
from latent_anything.methods.protocols import Method as Method
from latent_anything.methods.sae import SAE as SAE
from latent_anything.methods.umap import UMAP as UMAP

__all__ = ["Method", "PCA", "SAE", "UMAP"]
