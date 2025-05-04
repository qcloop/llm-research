from models.gpt_v2 import GPTV2
from open_ai.pretrained_weights import (
    download_gpt2_files,
    load_params_from_file,
    load_weights_into_gpt,
)
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

    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])

    targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    with torch.no_grad():
        logits = generator.model(inputs)

    probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary
    print(probas.shape)  # Shape: (batch_size, num_tokens, vocab_size

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)

    text_idx = 1
    print(f"Targets batch 1: {token_ids_to_text(targets[text_idx], tokenizer)}")
    print(
        f"Outputs batch 1: {token_ids_to_text(token_ids[text_idx].flatten(), tokenizer)}"
    )

    text_idx = 0
    print(f"targets[text_idx]: {targets[text_idx]}")
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    print(f"targets[text_idx]: {targets[text_idx]}")
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print("Log probas:", log_probas)
    avg_log_probas = torch.mean(log_probas)
    print("Log probas:", avg_log_probas * -1)

    # using cross entropy, should produce same result
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten(0)
    print("logits_flats:", logits_flat.shape)
    print("targets_flat:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("loss:", loss)
    perpelexity = torch.exp(avg_log_probas * -1)
    print("perpelexity:", perpelexity)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


pre_trained_model_dir = (
    "/Users/uzokirov/Development/Python/llm-research/pre-trained_weights/124M"
)


def download_pretrained_weights():
    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    download_gpt2_files(model_size, pre_trained_model_dir)


def load_model_params():
    settings, params = load_params_from_file(pre_trained_model_dir, "hparams.json")
    print("Settings: ", settings)
    print("Params dictionary keys: ", params.keys())
    print(params["wte"])
    print("Token embedding weight tensor dimensions: ", params["wte"].shape)

def use_pretrained_weights():
    # Define model configurations in a dictionary for compactness
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTV2(NEW_CONFIG)
    gpt.eval()

    settings, params = load_params_from_file(pre_trained_model_dir, "hparams.json")
    load_weights_into_gpt(gpt, params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    gpt.to(device)
    torch.manual_seed(123)
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=gpt,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=35,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def main():
    # result = __text_generation("Here we go Panthers", 10)
    # print(result)
    # my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # print(my_array[-20:])
    # print(my_array[:-20])

    # ___probas_target()
    # download_pretrained_weights()
    # load_model_params()
    use_pretrained_weights()


if __name__ == "__main__":
    main()
