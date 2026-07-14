from __future__ import annotations

import torch

from . import ModelAdapter, register_adapter


@register_adapter("splicebert")
class SpliceBERTAdapter(ModelAdapter):
    name = "SpliceBERT"
    d_model = 512
    n_layers = 6
    token_resolution = "nucleotide"
    architecture = "BERT"
    bidirectional = True

    def __init__(self, model_id: str = "multimolecule/splicebert"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self, pretrained: bool = True):
        from multimolecule import RnaTokenizer, SpliceBertModel

        self.tokenizer = RnaTokenizer.from_pretrained(self.model_id)
        if pretrained:
            self.model = SpliceBertModel.from_pretrained(
                self.model_id, attn_implementation="eager",
            )
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_id)
            config._attn_implementation = "eager"
            self.model = SpliceBertModel(config)
        self.model.eval()

    def tokenize(self, seq: str) -> torch.Tensor:
        enc = self.tokenizer(seq, return_tensors="pt")
        return enc["input_ids"]

    @torch.no_grad()
    def get_all_layer_embeddings(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        out = self.model(tokens, output_hidden_states=True)
        return [hs[0, 1:-1, :] for hs in out.hidden_states]
