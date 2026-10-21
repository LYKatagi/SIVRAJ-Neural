from pathlib import Path

import torch

from sivraj_neural.model.model import IntentClassifier
from sivraj_neural.token.tokenizer import Tokenizer
from sivraj_neural.token.vectorizer import Vectorizer
from sivraj_neural.token.vocabulary import Vocabulary


class Inference:
    def __init__(
        self,
        model_path: str | Path = "data/model.pth",
        vocabulary_path: str | Path = "data/vocab.json",
        confidence_threshold: float = 0.70,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold deve estar entre 0 e 1.")

        self.confidence_threshold = confidence_threshold
        self.model_path = Path(model_path)

        self.vocabulary = Vocabulary(vocabulary_path)
        self.vocabulary.load()

        self.tokenizer = Tokenizer()
        self.vectorizer = Vectorizer(
            vocabulary=self.vocabulary,
            tokenizer=self.tokenizer,
        )

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True,
        )

        input_size = checkpoint["input_size"]
        hidden_size = checkpoint["hidden_size"]
        num_classes = checkpoint["num_classes"]

        if input_size != len(self.vocabulary):
            raise ValueError(
                "O tamanho do vocabulário não corresponde "
                "ao tamanho de entrada do modelo."
            )

        self.tags = checkpoint["tags"]

        if len(self.tags) != num_classes:
            raise ValueError(
                "O número de intenções não corresponde "
                "ao número de classes do modelo."
            )

        self.model = IntentClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError("O texto deve ser uma string.")

        if not text.strip():
            raise ValueError("O texto não pode estar vazio.")

        tokens = self.tokenizer.tokenize(text)

        known_tokens = [
            token
            for token in tokens
            if token in self.vocabulary
            and token
            not in {
                self.vocabulary.PAD_TOKEN,
                self.vocabulary.UNK_TOKEN,
            }
        ]

        if not known_tokens:
            return {
                "text": text,
                "intent": "unknown",
                "confidence": 0.0,
                "reason": "no_known_tokens",
            }

        vector = self.vectorizer.transform(text)
        logits = self.model(vector.unsqueeze(0))

        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_id = int(torch.argmax(probabilities).item())

        confidence = float(probabilities[predicted_id].item())

        if confidence < self.confidence_threshold:
            return {
                "text": text,
                "intent": "unknown",
                "confidence": confidence,
                "reason": "low_confidence",
            }

        return {
            "text": text,
            "intent": self.tags[predicted_id],
            "confidence": confidence,
            "reason": None,
        }
