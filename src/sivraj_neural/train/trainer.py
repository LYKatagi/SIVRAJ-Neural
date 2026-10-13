from pathlib import Path

from sivraj_neural.token import Tokenizer, Vocabulary
from sivraj_neural.train import IntentDataset


class Trainer:
    def __init__(
        self,
        dataset_path: str | Path = "data/intents.json",
        vocabulary_path: str | Path = "data/vocab.json",
    ):
        self.dataset = IntentDataset(dataset_path)
        self.tokenizer = Tokenizer()
        self.vocabulary = Vocabulary(vocabulary_path)

        self.tags = sorted(self.dataset.get_tags())
        self.tag_to_id = {tag: index for index, tag in enumerate(self.tags)}


def prepare_data(self) -> list[dict]:
    prepared_data = []

    for text, tag in self.dataset.get_examples():
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.vocabulary.encode(tokens)

        prepared_data.append(
            {
                "text": text,
                "tokens": tokens,
                "input_ids": token_ids,
                "tag": tag,
                "label": self.tag_to_id[tag],
            }
        )

    return prepared_data


def summary(self) -> None:
    print("=== SIVRAJ-Neural Trainer ===")
    print(f"Exemplos: {len(self.dataset)}")
    print(f"Intenções: {len(self.tags)}")
    print(f"Vocabulário: {len(self.vocabulary.token_to_id)}")
    print("\nClasses:")

    for tag, label in self.tag_to_id.items():
        print(f"  {label}: {tag}")
