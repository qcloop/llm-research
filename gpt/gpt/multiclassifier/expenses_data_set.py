import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class ExpensesDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Description"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Line"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length



train_data_set_path = (
    "/Users/uzokirov/Development/Python/llm-research/expenses_data_set/train.csv"
)
validation_data_set_path = (
    "/Users/uzokirov/Development/Python/llm-research/expenses_data_set/validation.csv"
)
test_data_set_path = (
    "/Users/uzokirov/Development/Python/llm-research/expenses_data_set/test.csv"
)


def mc_create_test_data_set():
    return mc_create_data_set(test_data_set_path, max_length=4)


def mc_create_train_data_set():
    return mc_create_data_set(train_data_set_path, max_length=None)


def mc_create_validation_data_set():
    return mc_create_data_set(validation_data_set_path, 4)


def mc_create_data_set(path: str, max_length: int):
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    dataset = ExpensesDataset(csv_file=path, max_length=max_length, tokenizer=tokenizer)
    print(f"dataset {dataset.max_length}")
    return dataset


def mc_create_data_loaders(
    train_dataset: ExpensesDataset, val_dataset: ExpensesDataset, test_dataset: ExpensesDataset
):
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


