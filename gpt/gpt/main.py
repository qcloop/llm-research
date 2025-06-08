import matplotlib.pyplot as plt
import tiktoken
import torch
from activations.gelu import GELU
from finetuning.classifier import (calc_accuracy_loader, calc_loss_loader,
                                   classifier_model_init, classify_review,
                                   make_model_trainable,
                                   train_classifier_model)
from finetuning.spam_data_set import (create_data_loaders,
                                      create_test_data_set,
                                      create_train_data_set,
                                      create_validation_data_set)
from instruction.data_set import (custom_collate_draft_1,
                                  custom_collate_draft_2, custom_collate_fn,
                                  format_input, load_raw_data_set)
from models.dummy import DummyGPTModel
from models.gpt_v2 import GPTV2
from multiclassifier.classifier import (mc_calc_accuracy_loader,
                                        mc_calc_loss_loader,
                                        mc_classifier_model_init,
                                        mc_classify_expense,
                                        mc_make_model_trainable,
                                        mc_train_classifier_model)
from multiclassifier.expenses_data_set import (mc_create_data_loaders,
                                               mc_create_test_data_set,
                                               mc_create_train_data_set,
                                               mc_create_validation_data_set)
from open_ai.pretrained_weights import (download_gpt2_files,
                                        load_params_from_file,
                                        load_weights_into_gpt)
from text_generator import TextGenerator
from tokenizer.encoder import Encoder
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
        context_size=GPT_CONFIG_124M["context_length"],
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


def classify_spam_no_spam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_dict = torch.load(
        "/Users/uzokirov/Development/Python/llm-research/model_files/review_classifier.pth",
        map_location=device,
        weights_only=True,
    )
    model = classifier_model_init()
    model.load_state_dict(model_state_dict)

    text = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash award or a $2000 award:\n\n\n"
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    print(classify_review(text, model, tokenizer, device, max_length=120))

    text_2 = "Hey Norm, just wanted to check if we're still on for dinner tonight? Let me know!"
    print(classify_review(text_2, model, tokenizer, device, max_length=120))


def classify_spam_no_spam_trainer():
    # prepare_training_data_set()
    train_data_set = create_train_data_set()
    test_data_set = create_test_data_set()
    validation_data_set = create_validation_data_set()
    train_loader, validation_loader, test_loader = create_data_loaders(
        train_data_set, validation_data_set, test_data_set
    )

    print("Train loader:")
    for input_batch, target_batch in train_loader:
        pass

    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(validation_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    model = classifier_model_init()
    trainable_model = make_model_trainable(model)

    inputs = train_data_set.tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)  # shape: (batch_size, num_tokens)

    with torch.no_grad():
        outputs = trainable_model(inputs)

    print("Outputs:\n", outputs)
    print(
        "Outputs dimensions:", outputs.shape
    )  # shape: (batch_size, num_tokens, num_class
    # result = classify_spam(model, text)
    # print(f"{result}")

    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    print("Class label:", label.item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainable_model.to(device)

    torch.manual_seed(
        123
    )  # For reproducibility due to the shuffling in the training data loader

    train_accuracy = calc_accuracy_loader(
        train_loader, trainable_model, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
        validation_loader, trainable_model, device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, trainable_model, device, num_batches=10
    )

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    with (
        torch.no_grad()
    ):  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(
            train_loader, trainable_model, device, num_batches=5
        )
        val_loss = calc_loss_loader(
            validation_loader, trainable_model, device, num_batches=5
        )
        test_loss = calc_loss_loader(
            test_loader, trainable_model, device, num_batches=5
        )

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    train_classifier_model(trainable_model, train_loader, validation_loader, device)

    torch.save(
        model.state_dict(),
        "/Users/uzokirov/Development/Python/llm-research/model_files/review_classifier.pth",
    )


def instructions_tuning():
    data = load_raw_data_set()
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))  

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    ) 

    print(custom_collate_draft_1(batch))

    print("/n/n/n Source Target Collated Combo\n\n\n")
    print(custom_collate_draft_2(batch))


    print("/n/n/n Source Target Collated Combo With Sentinel value -100\n\n\n")
    print(custom_collate_fn(batch))

    logits_1 = torch.tensor(
    [[-1.0, 1.0, 50256],  # 1st training example
     [-0.5, 50256, 50256],
     [-0.5, 50246, 50246]
     ]  # 2nd training example
)
   # targets_1 = torch.tensor([0, 1])
    targets_2 = torch.tensor([0, 1, -100])
    tokenizer = tiktoken.get_encoding("gpt2")

    entry = {
        "instruction": "Edit the following sentence for grammar.",
        "input": "He go to the park every day.",
        "output": "He goes to the park every day."
    }
    encoded = tokenizer.encode(format_input(entry))

    decoded = tokenizer.decode([50256])


    #loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    loss_2 = torch.nn.functional.cross_entropy(logits_1, targets_2)
    #print(loss_1)
    print("\n\n\n")
    print(loss_2)

    print("\n\n\n")
    print(f"decoded {decoded}")
    print(f"encoded {encoded}")
    print("\n\n\n")
    

def main():
    # result = __text_generation("Here we go Panthers", 10)
    # print(result)
    # my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # print(my_array[-20:])
    # print(my_array[:-20])

    # ___probas_target()
    # download_pretrained_weights()
    # load_model_params()
    # use_pretrained_weights()
    # classify_spam_no_spam()
    # instructions_tuning()
    
    classify_expenses_trainer()
    classify_expenses()


def classify_expenses_trainer():
    train_data_set = mc_create_train_data_set()
    test_data_set = mc_create_test_data_set()
    validation_data_set = mc_create_validation_data_set()
    train_loader, validation_loader, test_loader = mc_create_data_loaders(
        train_data_set, validation_data_set, test_data_set
    )

    print("Train loader:")
    for input_batch, target_batch in train_loader:
        pass

    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    print(f"{len(train_loader)} training batches")
    print(f"{len(validation_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    model = mc_classifier_model_init()
    trainable_model = mc_make_model_trainable(model)

    inputs = train_data_set.tokenizer.encode("Online subscriptions")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)  # shape: (batch_size, num_tokens)

    with torch.no_grad():
        outputs = trainable_model(inputs)

    print("Outputs:\n", outputs)
    print(
        "Outputs dimensions:", outputs.shape
    )  
    

    probas = outputs[:, -1, :] #torch.softmax(outputs[:, -1, :], dim=-1)
    print("Probas: ", probas)
    label = torch.argmax(probas)
    print("Expenses Line Number:", label.item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainable_model.to(device)

    torch.manual_seed(
        123
    )  # For reproducibility due to the shuffling in the training data loader

    train_accuracy = mc_calc_accuracy_loader(
        train_loader, trainable_model, device, num_batches=3
    )
    val_accuracy = mc_calc_accuracy_loader(
        validation_loader, trainable_model, device, num_batches=3
    )
    test_accuracy = mc_calc_accuracy_loader(
        test_loader, trainable_model, device, num_batches=3
    )

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    with (
        torch.no_grad()
    ):  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = mc_calc_loss_loader(
            train_loader, trainable_model, device, num_batches=5
        )
        val_loss = mc_calc_loss_loader(
            validation_loader, trainable_model, device, num_batches=5
        )
        test_loss = mc_calc_loss_loader(
            test_loader, trainable_model, device, num_batches=5
        )

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    mc_train_classifier_model(trainable_model, train_loader, validation_loader, device)

    torch.save(
        model.state_dict(),
        "/Users/uzokirov/Development/Python/llm-research/model_files/expenses_classifier.pth",
    )


def classify_expenses():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_dict = torch.load(
        "/Users/uzokirov/Development/Python/llm-research/model_files/expenses_classifier.pth",
        map_location=device,
        weights_only=True,
    )
    model = mc_classifier_model_init()
    model.load_state_dict(model_state_dict)

    text = (
        "Equipment repairs"
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    print(mc_classify_expense(text, model, tokenizer, device, max_length=4))

    text_2 = "Business travel expenses"
    print(mc_classify_expense(text_2, model, tokenizer, device, max_length=4))



if __name__ == "__main__":
    main()
