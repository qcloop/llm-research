from gpt.model import DummyGPTModel
from tokenizer.encoder import Encoder
import torch

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}


def main():
    text = ["Every effort moves you", "Every day holds a"]
    batch = Encoder.convert(text)
    print(batch)

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)


if __name__ == "__main__":
    main()
