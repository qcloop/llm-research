import tiktoken
import torch


class Encoder:
    tokenizer = tiktoken.get_encoding("gpt2")

    @staticmethod
    def convert(text: list[str]):
        batch = []
        for item in text:
            batch.append(torch.tensor(Encoder.tokenizer.encode(item)))
        return torch.stack(batch, dim=0)
