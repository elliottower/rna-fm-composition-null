"""Model adapters for activation-space audit.

Each adapter provides a common interface: load, tokenize, and extract
per-layer hidden states from a sequence model. The hidden state at each
layer is the tensor that feeds into the next layer — the common object
across transformer, SSM, and hybrid architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch


ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register_adapter(key: str):
    def decorator(cls):
        ADAPTER_REGISTRY[key] = cls
        return cls
    return decorator


class ModelAdapter(ABC):
    name: ClassVar[str]
    d_model: ClassVar[int]
    n_layers: ClassVar[int]
    token_resolution: ClassVar[str]
    architecture: ClassVar[str]
    bidirectional: ClassVar[bool]

    @abstractmethod
    def load(self, pretrained: bool = True) -> None:
        ...

    @abstractmethod
    def tokenize(self, seq: str) -> torch.Tensor:
        ...

    @abstractmethod
    def get_all_layer_embeddings(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Return list of (seq_len, d_model) tensors, one per layer.

        This is the per-layer hidden state — the common object across
        architectures for fair comparison of effective dimensionality.
        """
        ...

    def get_final_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.get_all_layer_embeddings(tokens)[-1]

    def get_layer_types(self) -> list[str] | None:
        """For hybrid models (Evo), return per-layer type labels.

        Returns None for pure architectures (all layers same type).
        For Evo/StripedHyena: ['hyena', 'hyena', 'attention', 'hyena', ...].
        """
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "token_resolution": self.token_resolution,
            "architecture": self.architecture,
            "bidirectional": self.bidirectional,
            "layer_types": self.get_layer_types(),
        }
