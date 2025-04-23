import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding
from loguru import logger


class AttentionVisualiser:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, config: dict = None) -> None:
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
            
    def get_num_attn_layers(self) -> int:
        return 0

    def id_to_tokens(self, encoded_input: BatchEncoding) -> list[str]:
        tokens = self.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
        return tokens


    @torch.no_grad()
    def compute_attentions(self, encoded_input: BatchEncoding) -> torch.Tensor:
        output = self.model(**encoded_input, output_attentions=True)
        attentions = output.attentions
        return attentions

    def visualise_attn_layer(self, idx: int, encoded_input: BatchEncoding) -> None:
        # total number of attention heads in the model
        attn_heads = self.model.config.num_attention_heads
        
        # idx must no exceed attn_heads
        assert idx < attn_heads, \
            f"index must be less than the number of attention heads in the model, which is: {attn_heads}"
            
        # setting idx = -1 will get the last attention layer activations but
        # the plot title will also show -1
        if idx == -1:
            idx = attn_heads - 1
        
        tokens = self.id_to_tokens(encoded_input)
        attentions = self.compute_attentions(encoded_input)

        # get rid of the additional dimension since single input
        attention_weights = attentions[idx].squeeze()
        # take mean over dim 0 
        attention_weights = attention_weights.mean(dim=0)

        plt.figure(figsize=self.config.get("figsize"))
        sns.heatmap(
            attention_weights.cpu().numpy(), 
            cmap=self.config.get("cmap"), 
            annot=self.config.get("annot"),
            xticklabels=tokens, 
            yticklabels=tokens
        )
        
        plt.title(f"Attention Weights for Layer idx: {idx + 1}")
        plt.xlabel(self.config.get("xlabel"))
        plt.ylabel(self.config.get("ylabel"))
        plt.show()
