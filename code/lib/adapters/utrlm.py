from __future__ import annotations

import torch

from . import ModelAdapter, register_adapter


@register_adapter("utrlm")
class UTRLMAdapter(ModelAdapter):
    name = "UTR-LM"
    d_model = 128
    n_layers = 6
    token_resolution = "nucleotide"
    architecture = "Transformer encoder"
    bidirectional = True

    def __init__(self, model_id: str = "multimolecule/utrlm-te_el"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self, pretrained: bool = True):
        from multimolecule import RnaTokenizer, UtrLmModel

        self.tokenizer = RnaTokenizer.from_pretrained(self.model_id)
        if pretrained:
            self.model = UtrLmModel.from_pretrained(
                self.model_id, attn_implementation="eager",
            )
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_id)
            config._attn_implementation = "eager"
            self.model = UtrLmModel(config)
        self.model.eval()

    def tokenize(self, seq: str) -> torch.Tensor:
        enc = self.tokenizer(seq, return_tensors="pt")
        return enc["input_ids"]

    @torch.no_grad()
    def get_all_layer_embeddings(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        out = self.model(tokens, output_hidden_states=True)
        return [hs[0, 1:-1, :] for hs in out.hidden_states]
