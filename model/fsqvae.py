import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.module import Encoder, Decoder


class FSQ(nn.Module):
    """Finite Scalar Quantizer (FSQ) implementation.

    Attributes:
        levels: List of integers specifying the number of levels per dimension
        eps: Small epsilon value for numerical stability
        codebook_size: Total number of possible codes in the codebook
    """

    def __init__(self, levels: List[int] = [5, 5, 5, 8, 8], eps: float = 1e-3):
        super().__init__()

        # Validate input
        if not isinstance(levels, (list, tuple)):
            raise TypeError("levels must be a list or tuple of integers")
        if any(not isinstance(l, int) for l in levels):
            raise ValueError("All elements in levels must be integers")

        self.levels = levels
        self.eps = eps

        # Register buffers for device management
        self.register_buffer("levels_tensor", torch.tensor(self.levels))

        # Precompute basis for code indexing
        basis = [1]
        for i in range(len(self.levels) - 1):
            basis.append(basis[-1] * self.levels[i])
        self.register_buffer("basis_tensor", torch.tensor(basis))

        self.codebook_size = np.prod(self.levels)

    @property
    def num_dimensions(self) -> int:
        """Return the number of quantization dimensions."""
        return len(self.levels)

    @property
    def codebook_len(self) -> int:
        """Return the total codebook size."""
        return self.codebook_size

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound input tensor to valid quantization range.

        Args:
            z: Input tensor of shape (..., num_dimensions)

        Returns:
            Bounded tensor in the range [-1, 1]
        """
        levels = self.levels_tensor
        device = z.device

        # Calculate bounding parameters
        half_l = (levels - 1) * (1 - self.eps) / 2
        offset = torch.where(
            levels % 2 == 1,
            torch.tensor(0.0, device=device),
            torch.tensor(0.5, device=device),
        )
        shift = torch.tan(offset / half_l)

        return torch.tanh(z + shift) * half_l - offset

    def round_ste(self, z: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator rounding.

        Preserves gradients through the rounding operation.
        """
        zhat = torch.round(z)
        return z + (zhat - z).detach()

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize continuous latent vectors.

        Args:
            z: Input tensor of shape (..., num_dimensions)

        Returns:
            Quantized tensor with same shape as input
        """
        quantized = self.round_ste(self.bound(z))
        half_width = torch.div(self.levels_tensor, 2, rounding_mode="floor").to(
            z.device
        )
        return quantized / half_width

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """Convert quantized codes to codebook indices.

        Args:
            zhat: Quantized tensor of shape (..., num_dimensions)

        Returns:
            Indices tensor of shape (...)
        """
        scaled = self._scale_and_shift(zhat)
        return (scaled * self.basis_tensor).sum(dim=-1).to(torch.int32)

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert codebook indices back to quantized codes.

        Args:
            indices: Index tensor of shape (...)

        Returns:
            Quantized codes tensor of shape (..., num_dimensions)
        """
        indices = indices.unsqueeze(-1)
        codes = torch.zeros(
            indices.shape[:-1] + (len(self.levels),), device=indices.device
        )

        # Vectorized implementation for better performance
        basis = self.basis_tensor
        for i, level in enumerate(self.levels):
            codes[..., i] = (indices // basis[i]) % level

        return self._scale_and_shift_inverse(codes)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full quantization forward pass.

        Args:
            z: Input tensor of shape (..., num_dimensions)

        Returns:
            tuple: (quantized tensor, code indices)
        """
        quantized = self.quantize(z)
        indices = self.codes_to_indexes(quantized)
        return quantized, indices

    # Helper methods with type hints
    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = torch.div(self.levels_tensor, 2, rounding_mode="floor").to(
            zhat_normalized.device
        )
        return zhat_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = torch.div(self.levels_tensor, 2, rounding_mode="floor").to(
            zhat.device
        )
        return (zhat - half_width) / half_width


class FSQVAE(nn.Module):
    """FSQ Variational Autoencoder with Encoder/Decoder architecture.

    Attributes:
        levels: List of integers specifying FSQ quantization levels
    """

    def __init__(self, levels: List[int] = [5, 5, 5, 8, 8]):
        super().__init__()

        self.encoder = Encoder(embedding_num=len(levels))
        self.decoder = Decoder(embedding_num=len(levels))
        self.fsq = FSQ(levels=levels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full VAE forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            tuple: (reconstructed tensor, code indices)
        """
        z = self.encoder(x)
        quantized, indices = self.fsq(z)
        x_recon = self.decoder(quantized)
        return x_recon, indices

    def calculate_recon_loss(
        self, recon_x: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Calculate reconstruction loss (MSE)."""
        return F.mse_loss(recon_x, x)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to discrete indices."""
        z = self.encoder(images)
        _, indices = self.fsq(z)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices back to images."""
        quantized = self.fsq.indexes_to_codes(indices)
        return self.decoder(quantized)


if __name__ == "__main__":
    # Example test case with type hints
    def model_test():
        """Basic functionality test for FSQVAE"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running test on {device}")

        # Test parameters
        batch_size = 4
        channels = 3
        height = width = 256

        # Initialize model and test data
        model = FSQVAE(levels=[5, 5, 5, 8, 8]).to(device)
        x = torch.randn(batch_size, channels, height, width).to(device)

        # Forward pass
        recon_x, indices = model(x)

        # Output shapes
        print("\nTest Results:")
        print(f"Input shape: {x.shape}")
        print(f"Reconstruction shape: {recon_x.shape}")
        print(f"Indices shape: {indices.shape}")

        # Codebook verification
        test_code = torch.tensor([0.4, 0.6, -0.8, 0.2, 0.5], device=device)
        quantized = model.fsq.quantize(test_code)
        idx = model.fsq.codes_to_indexes(quantized)
        reconstructed = model.fsq.indexes_to_codes(idx)

        print("\nCodebook Test:")
        print(f"Original code: {test_code.cpu().numpy()}")
        print(f"Quantized code: {quantized.cpu().numpy()}")
        print(f"Reconstructed code: {reconstructed.cpu().numpy()}")
        print(f"Index: {idx.item()}")

    model_test()
