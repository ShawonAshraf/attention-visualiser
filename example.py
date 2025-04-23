from visualiser import AttentionVisualiserPytorch
from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    model_name = "openai-community/openai-gpt"

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "Look on my Works, ye Mighty, and despair!"
    encoded_inputs = tokenizer.encode_plus(text, truncation=True, return_tensors="pt")

    visualiser = AttentionVisualiserPytorch(model, tokenizer)
    visualiser.visualise_attn_layer(0, encoded_inputs)
