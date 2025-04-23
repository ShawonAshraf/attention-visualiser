# attention-visualiser

a module to visualise attention layer activations from transformer based models from huggingface

## installation

```bash
pip install git+https://github.com/ShawonAshraf/attention-visualiser
```

## usage

### pytorch
```python
from visualiser import AttentionVisualiserPytorch
from transformers import AutoModel, AutoTokenizer

# visualising activations from gpt
model_name = "openai-community/openai-gpt"

model = AutoModel.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Look on my Works, ye Mighty, and despair!"
encoded_inputs = tokenizer.encode_plus(text, truncation=True, return_tensors="pt")

visualiser = AttentionVisualiserPytorch(model, tokenizer)

# visualise from the first attn layer
visualiser.visualise_attn_layer(0, encoded_inputs)

```

### flax

```python
from visualiser import AttentionVisualiserFlax
from transformers import AutoModel, AutoTokenizer, FlaxAutoModel


model_name = "bert-base-cased"
model = FlaxAutoModel.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Look on my Works, ye Mighty, and despair!"
encoded_inputs = tokenizer.encode_plus(text, truncation=True, return_tensors="jax")

visualiser = AttentionVisualiserFlax(model, tokenizer)

# visualise from the first attn layer
visualiser.visualise_attn_layer(0, encoded_inputs)
```

An example colab notebook can be found [here](https://colab.research.google.com/drive/1N5uuRPcM90CPtEPnTaeWcA9PNKzzZaK-?usp=sharing).


## local dev

```bash
# env setup

uv sync
source .venv/bin/activate

# tests
uv run pytest
```

alternatively you can use the devcontainer.
