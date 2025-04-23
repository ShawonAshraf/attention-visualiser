from .base import BaseAttentionVisualiser

import jax.numpy as jnp
from transformers import FlaxAutoModel, AutoTokenizer
import numpy as np
from transformers import BatchEncoding


class AttentionVisualiserFlax(BaseAttentionVisualiser):
    def __init__(
        self, model: FlaxAutoModel, tokenizer: AutoTokenizer, config: dict = None
    ) -> None:
        super().__init__(model, tokenizer, config)

    def compute_attentions(self, encoded_input: BatchEncoding) -> tuple:
        output = self.model(**encoded_input, output_attentions=True)
        attentions = output.attentions
        return attentions

    def get_attention_vector_mean(
        self, attention: jnp.ndarray, axis: int = 0
    ) -> np.ndarray:
        vector_np = np.array(attention)
        return np.mean(vector_np, axis=axis)
