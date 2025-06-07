import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

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
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
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
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)


def prepare_training_data_set():
    balanced_df = create_data_sets()
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # Test size is implied to be 0.2 as the remainder

    train_df.to_csv(
        "/Users/uzokirov/Development/Python/llm-research/spam_data_set/train.csv",
        index=None,
    )
    validation_df.to_csv(
        "/Users/uzokirov/Development/Python/llm-research/spam_data_set/validation.csv",
        index=None,
    )
    test_df.to_csv(
        "/Users/uzokirov/Development/Python/llm-research/spam_data_set/test.csv",
        index=None,
    )


train_data_set_path = (
    "/Users/uzokirov/Development/Python/llm-research/spam_data_set/train.csv"
)
validation_data_set_path = (
    "/Users/uzokirov/Development/Python/llm-research/spam_data_set/validation.csv"
)
test_data_set_path = (
    "/Users/uzokirov/Development/Python/llm-research/spam_data_set/test.csv"
)


def create_test_data_set():
    return create_data_set(test_data_set_path, max_length=120)


def create_train_data_set():
    return create_data_set(train_data_set_path, max_length=None)


def create_validation_data_set():
    return create_data_set(validation_data_set_path, 120)


def create_data_set(path: str, max_length: int):
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    dataset = SpamDataset(csv_file=path, max_length=max_length, tokenizer=tokenizer)
    print(dataset.max_length)
    return dataset


def create_data_loaders(
    train_dataset: SpamDataset, val_dataset: SpamDataset, test_dataset: SpamDataset
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


def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def create_data_sets():
    data_file_path = "/Users/uzokirov/Development/Python/llm-research/spam_data_set/SMSSpamCollection.tsv"
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print(df)
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    print(balanced_df)
    return balanced_df


def create_balanced_dataset(df):
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df
