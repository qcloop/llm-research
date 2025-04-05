import torch
import tiktoken
from models.gpt_v2 import GPTV2


class TextGenerator:
    def __init__(self, config):
        torch.manual_seed(123)
        self.config = config
        self.model = GPTV2(config)
        self.model.eval()  # disable dropout

    def generate(self, input: str, max_new_tokens: int):
        tokenizer = tiktoken.get_encoding("gpt2")
        encoded = tokenizer.encode(input)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        result = self.__generate_text_simple(
            model=self.model,
            idx=encoded_tensor,
            max_new_tokens=10,
            context_size=self.config["context_length"],
        )
        return tokenizer.decode(result.squeeze(0).tolist())

    def __generate_text_simple(self, model, idx, max_new_tokens, context_size):
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
