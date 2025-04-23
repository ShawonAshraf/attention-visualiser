import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding
from .base import BaseAttentionVisualiser
import numpy as np


class AttentionVisualiserPytorch(BaseAttentionVisualiser):
    def __init__(
        self, model: AutoModel, tokenizer: AutoTokenizer, config: dict = None
    ) -> None:
        super().__init__(model, tokenizer, config)

    def compute_attentions(self, encoded_input: BatchEncoding) -> tuple:
        with torch.no_grad():
            output = self.model(**encoded_input, output_attentions=True)
        attentions = output.attentions
        return attentions

    def get_attention_vector_mean(
        self, attention: torch.Tensor, axis: int = 0
    ) -> np.ndarray:
        return torch.mean(attention, dim=axis).detach().cpu().numpy()
