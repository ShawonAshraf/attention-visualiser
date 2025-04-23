import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding


class AttentionVisualiser:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, config: dict = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        
        if not config:
            self.config = {
                "figsize": (15, 15),
                "cmap": "viridis",
                "annot": True,
                "xlabel": "",
                "ylable": "",
            }
        else:
            self.config = config
            
    def get_num_attn_layers(self) -> int:
        return 0

    def id_to_tokens(self, encoded_input: BatchEncoding) -> list[str]:
        tokens = self.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
        return tokens


    @torch.no_grad()
    def compute_attentions(self, encoded_input: BatchEncoding) -> torch.Tensor:
        output = self.model(**encoded_input, output_attentions=True)
        return output.attentions

    def visualise_attn_layer(self, idx: int, encoded_input: BatchEncoding) -> None:
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
        
        plt.title(f"Attention Weights for Layer {idx}")
        plt.xlabel(self.config.get("xlabel"))
        plt.ylabel(self.config.get("ylabel"))
        plt.show()
