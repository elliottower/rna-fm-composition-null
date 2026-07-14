from __future__ import annotations

from pathlib import Path

import torch

from . import ModelAdapter, register_adapter


@register_adapter("rnafm")
class RNAFMAdapter(ModelAdapter):
    name = "RNA-FM"
    d_model = 640
    n_layers = 12
    token_resolution = "nucleotide"
    architecture = "BERT (full attention)"
    bidirectional = True

    NUC_TO_ID = {"A": 5, "C": 6, "G": 7, "U": 8}

    def __init__(self, weight_path: str = "pretrained/pytorch_model.bin"):
        self.weight_path = Path(weight_path)
        self.model = None

    def load(self, pretrained: bool = True):
        from transformers import BertConfig, BertModel

        if pretrained and not self.weight_path.exists():
            raise FileNotFoundError(f"RNA-FM weights not found at {self.weight_path}")

        max_pos = 1024
        if pretrained:
            state = torch.load(self.weight_path, map_location="cpu", weights_only=False)
            max_pos = state["model.embeddings.position_embeddings.weight"].shape[0]

        config = BertConfig(
            vocab_size=28, hidden_size=640, num_hidden_layers=12,
            num_attention_heads=20, intermediate_size=5120,
            max_position_embeddings=max_pos,
        )
        self.model = BertModel(config)

        if pretrained:
            clean_state = {}
            for k, v in state.items():
                if not k.startswith("model."):
                    continue
                nk = k[len("model."):]
                nk = nk.replace("attention.layer_norm", "attention.output.LayerNorm")
                if "encoder.layer." in nk and ".layer_norm." in nk and "attention" not in nk:
                    nk = nk.replace("layer_norm", "output.LayerNorm")
                elif nk.startswith("embeddings.layer_norm"):
                    nk = nk.replace("embeddings.layer_norm", "embeddings.LayerNorm")
                clean_state[nk] = v

            self.model.load_state_dict(clean_state, strict=False)

        self.model.eval()

    def tokenize(self, seq: str) -> torch.Tensor:
        tokens = [2] + [self.NUC_TO_ID.get(c, 1) for c in seq] + [3]
        return torch.tensor([tokens])

    @torch.no_grad()
    def get_all_layer_embeddings(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        out = self.model(tokens, output_hidden_states=True)
        return [hs[0, 1:-1, :] for hs in out.hidden_states]
