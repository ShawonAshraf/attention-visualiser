import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, FlaxAutoModel
from transformers import BatchEncoding
from loguru import logger
from einops import rearrange
import jax.numpy as jnp
from abc import ABC, abstractmethod
import numpy as np


class BaseAttentionVisualiser(ABC):
    def __init__(
        self,
        model: AutoModel | FlaxAutoModel,
        tokenizer: AutoTokenizer,
        config: dict = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        logger.info(f"Model config: {self.model.config}")

        if not config:
            self.config = {
                "figsize": (15, 15),
                "cmap": "viridis",
                "annot": True,
                "xlabel": "",
                "ylabel": "",
            }
            logger.info(f"Setting default visualiser config: {self.config}")
        else:
            logger.info(f"Visualiser config: {config}")
            self.config = config

    def id_to_tokens(self, encoded_input: BatchEncoding) -> list[str]:
        tokens = self.tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
        return tokens

    @abstractmethod
    def compute_attentions(self, encoded_input: BatchEncoding) -> tuple:
        pass

    @abstractmethod
    def get_attention_vector_mean(
        attention: torch.Tensor | jnp.ndarray, axis: int = 0
    ) -> np.ndarray:
        pass

    def visualise_attn_layer(self, idx: int, encoded_input: BatchEncoding) -> None:
        tokens = self.id_to_tokens(encoded_input)

        attentions = self.compute_attentions(encoded_input)
        n_attns = len(attentions)

        # idx must no exceed attn_heads
        assert idx < n_attns, (
            f"index must be less than the number of attention outputs in the model, which is: {n_attns}"
        )

        # setting idx = -1 will get the last attention layer activations but
        # the plot title will also show -1
        if idx < 0:
            idx = n_attns + idx

        # get rid of the additional dimension since single input
        attention = rearrange(attentions[idx], "1 a b c -> a b c")
        # take mean over dim 0
        attention = self.get_attention_vector_mean(attention)

        plt.figure(figsize=self.config.get("figsize"))
        sns.heatmap(
            attention,
            cmap=self.config.get("cmap"),
            annot=self.config.get("annot"),
            xticklabels=tokens,
            yticklabels=tokens,
        )

        plt.title(f"Attention Weights for Layer idx: {idx}")
        plt.xlabel(self.config.get("xlabel"))
        plt.ylabel(self.config.get("ylabel"))
        plt.show()
