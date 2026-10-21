from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sivraj_neural.model.model import IntentClassifier
from sivraj_neural.token.tokenizer import Tokenizer
from sivraj_neural.token.vectorizer import Vectorizer
from sivraj_neural.token.vocabulary import Vocabulary
from sivraj_neural.train.dataset import IntentDataset


class Trainer:
    def __init__(
        self,
        dataset_path: str | Path = "data/intents.json",
        vocabulary_path: str | Path = "data/vocab.json",
        model_path: str | Path = "data/model.pth",
        hidden_size: int = 32,
        learning_rate: float = 0.001,
        batch_size: int = 8,
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size deve ser positivo.")

        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo.")

        if batch_size <= 0:
            raise ValueError("batch_size deve ser positivo.")

        self.dataset = IntentDataset(dataset_path)
        self.vocabulary = Vocabulary(vocabulary_path)
        self.tokenizer = Tokenizer()
        self.vectorizer: Vectorizer | None = None

        self.tags = sorted(self.dataset.get_tags())
        self.tag_to_id = {tag: index for index, tag in enumerate(self.tags)}

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_path = Path(model_path)

        self.model: IntentClassifier | None = None

    def prepare_data(self) -> DataLoader:
        """Constrói o vocabulário e prepara os dados para treinamento."""
        examples = self.dataset.get_examples()

        if not examples:
            raise ValueError("O dataset não contém exemplos.")

        # O vocabulário precisa corresponder aos dados de treinamento.
        token_sequences = [
            self.tokenizer.tokenize(text) for text, _ in examples
        ]

        self.vocabulary.build(token_sequences)

        self.vectorizer = Vectorizer(
            vocabulary=self.vocabulary,
            tokenizer=self.tokenizer,
        )

        vectors = torch.stack(
            [self.vectorizer.transform(text) for text, _ in examples]
        )

        labels = torch.tensor(
            [self.tag_to_id[tag] for _, tag in examples],
            dtype=torch.long,
        )

        dataset = TensorDataset(vectors, labels)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def train(self, epochs: int = 100) -> list[float]:
        """Treina o classificador e retorna a perda de cada época."""
        if epochs <= 0:
            raise ValueError("epochs deve ser positivo.")

        loader = self.prepare_data()

        self.model = IntentClassifier(
            input_size=len(self.vocabulary),
            hidden_size=self.hidden_size,
            num_classes=len(self.tags),
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        losses = []

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_examples = 0

            for inputs, labels in loader:
                optimizer.zero_grad()

                logits = self.model(inputs)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

            average_loss = total_loss / total_examples
            losses.append(average_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Época {epoch + 1:3}/{epochs} | Loss: {average_loss:.4f}"
                )

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": len(self.vocabulary),
                "hidden_size": self.hidden_size,
                "num_classes": len(self.tags),
                "tags": self.tags,
                "tag_to_id": self.tag_to_id,
            },
            self.model_path,
        )

        print(f"\nModelo salvo em: {self.model_path}")

        return losses
