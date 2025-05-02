from text_generator import TextGenerator
from models.dummy import DummyGPTModel
from activations.gelu import GELU
from tokenizer.encoder import Encoder
import torch
from torch import nn
import matplotlib.pyplot as plt
import tiktoken

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


def __tensor_fun():
    x = torch.zeros(2, 1, 2, 1, 2)
    print(x)
    print(x.shape)
    # Expected output: torch.Size([2, 1, 2, 1, 2])
    print("^^^^^^^^^^^^^")

    y = torch.squeeze(x)
    print(y)
    print(y.shape)
    # Expected output: torch.Size([2, 2, 2])
    print("^^^^^^^^^^^^^")
    z = torch.squeeze(x, 3)
    print(z)
    print(z.shape)
    # Expected output: torch.Size([2, 2, 1, 2]) 


def ___probas_target():
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
    
    inputs = torch.tensor([[16833, 3626, 6100],  
                       [40,    1107, 588]])  
    
    targets = torch.tensor([[3626, 6100, 345  ],  
                        [1107,  588, 11311]]) 
    
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    
    with torch.no_grad():
        logits = generator.model(inputs)
        
    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size
    
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    
    text_idx = 1
    print(f"Targets batch 1: {token_ids_to_text(targets[text_idx], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[text_idx].flatten(), tokenizer)}")
    
    #text_idx = 1
    print(f"targets[text_idx]: {targets[text_idx]}")
    target_probas_1 = probas[text_idx, [0,1,2], [ 1107,   588, 11311]]
    print("Text 1:", target_probas_1)
    
    
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def main():
  #result = __text_generation("Here we go Panthers", 10)
  #print(result)
  #my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  #print(my_array[-20:])
  #print(my_array[:-20])
  
  ___probas_target()
 


if __name__ == "__main__":
    main()
