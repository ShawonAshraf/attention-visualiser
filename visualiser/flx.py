from .base import BaseAttentionVisualiser

import jax.numpy as jnp
from transformers import FlaxAutoModel, AutoTokenizer
import numpy as np
from transformers import BatchEncoding


class AttentionVisualiserFlax(BaseAttentionVisualiser):
    """Attention visualizer for Flax-based transformer models.

    This class implements the abstract methods from BaseAttentionVisualiser
    specifically for models implemented in Flax/JAX. It handles the extraction
    and processing of attention weights from Flax transformer models.

    Attributes:
        model: A Flax-based transformer model from Hugging Face
        tokenizer: A tokenizer matching the model
        config: Dictionary containing visualization configuration parameters
    """

    def __init__(
        self, model: FlaxAutoModel, tokenizer: AutoTokenizer, config: dict = None
    ) -> None:
        """Initialize the Flax-specific attention visualizer.

        Args:
            model: A Flax-based transformer model from Hugging Face
            tokenizer: A tokenizer matching the model
            config: Optional dictionary with visualization parameters
        """
        super().__init__(model, tokenizer, config)

    def compute_attentions(self, encoded_input: BatchEncoding) -> tuple:
        """Compute attention weights for the given input using a Flax model.

        Runs the Flax model with output_attentions flag set to True and
        extracts the attention weights from the model output.

        Args:
            encoded_input: The encoded input from the tokenizer

        Returns:
            A tuple containing attention weights from all layers of the model
        """
        if encoded_input == self.current_input:
            # return from cache
            return self.cache

        # else recompute
        output = self.model(**encoded_input, output_attentions=True)
        attentions = output.attentions

        # update cache and current input
        self.current_input = encoded_input
        self.cache = attentions

        return attentions

    def get_attention_vector_mean(
        self, attention: jnp.ndarray, axis: int = 0
    ) -> np.ndarray:
        """Calculate mean of JAX attention vectors along specified axis.

        Converts JAX ndarray to NumPy array and computes the mean.

        Args:
            attention: JAX ndarray containing attention weights
            axis: Axis along which to compute the mean (default: 0)

        Returns:
            NumPy array of mean attention values
        """
        vector_np = np.array(attention)
        return np.mean(vector_np, axis=axis)
