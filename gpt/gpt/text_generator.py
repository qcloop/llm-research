import tiktoken
import torch
from models.gpt_v2 import GPTV2


class TextGenerator:
    def __init__(self, config):
        torch.manual_seed(123)
        self.config = config
        self.model = GPTV2(config)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model.eval()  # disable dropout

    def generate(self, input: str, max_new_tokens: int):
        encoded = self.tokenizer.encode(input)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
        print("\nInput text:", input)
        print("Encoded input text:", encoded)
        print("encoded_tensor.shape:", encoded_tensor.shape)
        result = self.__generate_text_simple(
            model=self.model,
            idx=encoded_tensor,
            max_new_tokens=max_new_tokens,
            context_size=self.config["context_length"],
        )
        print(result)
        print(result.squeeze(0).tolist())
        return self.tokenizer.decode(result.squeeze(0).tolist())

    def __generate_text_simple(self, model, idx, max_new_tokens, context_size):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            print(f"****context_size {context_size}")
            print(f"****idx {idx}")
            idx_cond = idx[:, -context_size:]
            print(
                f"****idx_cond {idx_cond}, {self.tokenizer.decode(idx_cond.squeeze(0).tolist())}"
            )

            # Get the predictions
            with torch.no_grad():
                logits = model(idx_cond)

            # Focus only on the last time step
            # (batch, n_token, vocab_size) becomes (batch, vocab_size)
            print(f"logits {logits}")
            logits = logits[:, -1, :]
            print(f"logits after {logits}")
            # Get the idx of the vocab entry with the highest logits value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx
