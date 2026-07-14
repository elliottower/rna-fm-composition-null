from __future__ import annotations

import torch

from . import ModelAdapter, register_adapter


@register_adapter("nt")
class NTAdapter(ModelAdapter):
    name = "Nucleotide-Transformer-v2"
    d_model = 512
    n_layers = 12
    token_resolution = "6mer"
    architecture = "ESM (attention + GLU)"
    bidirectional = True

    MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or self.MODEL_ID
        self.model = None
        self.tokenizer = None

    def load(self, pretrained: bool = True):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_id, trust_remote_code=True,
            attn_implementation="eager",
        )
        if not pretrained:
            self.model.apply(self._reinit_weights)
        self.model.eval()

    @staticmethod
    def _reinit_weights(module):
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            return
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.dim() >= 2:
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                torch.nn.init.zeros_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def tokenize(self, seq: str) -> torch.Tensor:
        seq_dna = seq.replace("U", "T")
        encoded = self.tokenizer(seq_dna, return_tensors="pt")
        return encoded["input_ids"]

    @torch.no_grad()
    def get_all_layer_embeddings(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        out = self.model(tokens, output_hidden_states=True)
        return [hs[0, 1:-1, :] for hs in out.hidden_states]
