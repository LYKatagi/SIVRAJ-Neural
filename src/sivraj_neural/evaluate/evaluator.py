import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sivraj_neural.model.model import IntentClassifier
from sivraj_neural.token.tokenizer import Tokenizer
from sivraj_neural.token.vectorizer import Vectorizer
from sivraj_neural.token.vocabulary import Vocabulary
from sivraj_neural.train.dataset import IntentDataset


class Evaluator:
    def __init__(
        self,
        dataset_path: str | Path = "data/intents.json",
        validation_split: float = 0.2,
        hidden_size: int = 32,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        seed: int = 42,
    ):
        if not 0 < validation_split < 1:
            raise ValueError("validation_split deve estar entre 0 e 1.")

        if hidden_size <= 0:
            raise ValueError("hidden_size deve ser positivo.")

        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo.")

        if batch_size <= 0:
            raise ValueError("batch_size deve ser positivo.")

        self.dataset = IntentDataset(dataset_path)
        self.validation_split = validation_split
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed

        self.tokenizer = Tokenizer()
        self.vocabulary = Vocabulary()
        self.vectorizer: Vectorizer | None = None

        self.tags = sorted(self.dataset.get_tags())
        self.tag_to_id = {tag: index for index, tag in enumerate(self.tags)}

    def split_data(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Separa os exemplos de cada intenção em treino e validação."""
        grouped = defaultdict(list)

        for example, tag in self.dataset.get_examples():
            grouped[tag].append((example, tag))

        rng = random.Random(self.seed)
        training_data = []
        validation_data = []

        for tag, examples in sorted(grouped.items()):
            if len(examples) < 2:
                raise ValueError(
                    f"A intenção '{tag}' precisa de pelo menos "
                    "2 exemplos para treinamento e validação."
                )

            examples = examples.copy()
            rng.shuffle(examples)

            validation_count = round(len(examples) * self.validation_split)
            validation_count = max(1, validation_count)
            validation_count = min(
                validation_count,
                len(examples) - 1,
            )

            validation_data.extend(examples[:validation_count])
            training_data.extend(examples[validation_count:])

        rng.shuffle(training_data)
        rng.shuffle(validation_data)

        return training_data, validation_data

    def evaluate(self, epochs: int = 100) -> dict:
        """Treina um modelo e avalia seu desempenho em exemplos separados."""
        if epochs <= 0:
            raise ValueError("epochs deve ser positivo.")

        training_data, validation_data = self.split_data()

        # Constrói o vocabulário exclusivamente com os dados de treino.
        token_sequences = [
            self.tokenizer.tokenize(text) for text, _ in training_data
        ]

        self.vocabulary.build(token_sequences)

        self.vectorizer = Vectorizer(
            vocabulary=self.vocabulary,
            tokenizer=self.tokenizer,
        )

        training_vectors = torch.stack(
            [self.vectorizer.transform(text) for text, _ in training_data]
        )

        training_labels = torch.tensor(
            [self.tag_to_id[tag] for _, tag in training_data],
            dtype=torch.long,
        )

        training_loader = DataLoader(
            TensorDataset(training_vectors, training_labels),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model = IntentClassifier(
            input_size=len(self.vocabulary),
            hidden_size=self.hidden_size,
            num_classes=len(self.tags),
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
        )

        model.train()

        for _ in range(epochs):
            for inputs, labels in training_loader:
                optimizer.zero_grad()

                logits = model(inputs)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

        # Avalia apenas os exemplos que não participaram do treino.
        model.eval()

        validation_vectors = torch.stack(
            [self.vectorizer.transform(text) for text, _ in validation_data]
        )

        validation_labels = torch.tensor(
            [self.tag_to_id[tag] for _, tag in validation_data]
        )

        with torch.no_grad():
            logits = model(validation_vectors)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

        confusion = torch.zeros(
            len(self.tags),
            len(self.tags),
            dtype=torch.int64,
        )

        for expected, predicted in zip(
            validation_labels,
            predictions,
            strict=False,
        ):
            confusion[expected, predicted] += 1

        correct = int((predictions == validation_labels).sum().item())
        total = len(validation_data)

        per_intent = {}

        for index, tag in enumerate(self.tags):
            true_positive = int(confusion[index, index].item())

            actual_count = int(confusion[index, :].sum().item())
            predicted_count = int(confusion[:, index].sum().item())

            precision = (
                true_positive / predicted_count if predicted_count else 0.0
            )

            recall = true_positive / actual_count if actual_count else 0.0

            per_intent[tag] = {
                "precision": precision,
                "recall": recall,
                "support": actual_count,
            }

        accuracy = correct / total if total else 0.0

        macro_precision = sum(
            metrics["precision"] for metrics in per_intent.values()
        ) / len(self.tags)

        macro_recall = sum(
            metrics["recall"] for metrics in per_intent.values()
        ) / len(self.tags)

        results = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "total_examples": total,
            "correct_predictions": correct,
            "tags": self.tags,
            "per_intent": per_intent,
            "confusion_matrix": confusion.tolist(),
        }

        print("\n=== SIVRAJ-Neural: Avaliação ===")
        print(f"Exemplos de validação: {total}")
        print(f"Previsões corretas: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Macro precision: {macro_precision:.2%}")
        print(f"Macro recall: {macro_recall:.2%}")

        print("\n=== Métricas por intenção ===")

        for tag, metrics in per_intent.items():
            print(
                f"{tag}: "
                f"precision={metrics['precision']:.2%}, "
                f"recall={metrics['recall']:.2%}, "
                f"support={metrics['support']}"
            )

        print("\n=== Matriz de confusão ===")
        print("Linhas = intenção real; colunas = previsão.")
        print("Ordem das intenções:", self.tags)

        for row in confusion.tolist():
            print(row)

        return results
