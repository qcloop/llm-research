from text_generator import TextGenerator
from models.dummy import DummyGPTModel
from activations.gelu import GELU
from tokenizer.encoder import Encoder
import torch
from torch import nn
import matplotlib.pyplot as plt

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}


def _sample_dim():
    torch.manual_seed(112)
    batch_example = torch.randn(2, 5)  # 2 inputs with 5 dimensions
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    print("dim=-1, keepdim=true")
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print(f"Mean: {mean}")
    print(f"Var: {var}")
    print("#################")

    print("dim=0, keepdim=true")
    mean = out.mean(dim=0, keepdim=True)
    var = out.var(dim=0, keepdim=True)

    print(f"Mean: {mean}")
    print(f"Var: {var}")
    print("#################")

    print("dim=1, keepdim=false")
    mean = out.mean(dim=1, keepdim=True)
    var = out.var(dim=1, keepdim=True)

    print(f"Mean: {mean}")
    print(f"Var: {var}")
    print("#################")

    print("dim=0, keepdim=false")
    mean = out.mean(dim=0, keepdim=False)
    var = out.var(dim=0, keepdim=False)
    print(f"Mean: {mean}")
    print(f"Var: {var}")
    print("#################")

    print("dim=0, keepdim=true")
    mean = out.mean(dim=-2, keepdim=True)
    var = out.var(dim=-2, keepdim=True)
    print(f"Mean: {mean}")
    print(f"Var: {var}")
    print("#################")


def __model_sample():
    text = ["Every effort moves you", "Every day holds a"]
    batch = Encoder.convert(text)
    print(batch)

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)

    print("#################")
    _sample_dim()
    print("#################")


def __gelu_sample():
    gelu = GELU()
    relu = nn.ReLU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def __text_generation(input: str, max_new_tokens: int):
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }
    generator = TextGenerator(config=GPT_CONFIG_124M)

    return generator.generate(input=input, max_new_tokens=max_new_tokens)


def main():
    result = __text_generation("Here we go Panthers", 10)
    print(result)


if __name__ == "__main__":
    main()
