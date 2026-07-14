from __future__ import annotations

import torch

from . import ModelAdapter, register_adapter


@register_adapter("dnabert2")
class DNABERT2Adapter(ModelAdapter):
    name = "DNABERT-2"
    d_model = 768
    n_layers = 12
    token_resolution = "bpe"
    architecture = "BERT + ALiBi"
    bidirectional = True

    def __init__(self, model_id: str = "zhihan1996/DNABERT-2-117M"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self, pretrained: bool = True):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        if pretrained:
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True,
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True,
            )
            self.model.apply(self._reinit_weights)
        self.model.eval()

    @staticmethod
    def _reinit_weights(module):
        if isinstance(module, (torch.nn.LayerNorm,)):
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
        enc = self.tokenizer(seq_dna, return_tensors="pt")
        return enc["input_ids"]

    @torch.no_grad()
    def get_all_layer_embeddings(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        # MosaicBERT ignores output_hidden_states and returns a tuple.
        # Use the last hidden state from out[0] as the only valid layer.
        out = self.model(tokens)
        last_hidden = out[0] if isinstance(out, tuple) else out
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.unsqueeze(0)
        return [last_hidden[0, 1:-1, :]]
