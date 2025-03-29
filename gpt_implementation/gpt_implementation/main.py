from gpt.model import DummyGPTModel
from tokenizer.encoder import Encoder
import torch
from torch import nn

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


def main():
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


if __name__ == "__main__":
    main()
