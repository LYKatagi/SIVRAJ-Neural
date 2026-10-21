import torch

from sivraj_neural.token.tokenizer import Tokenizer
from sivraj_neural.token.vocabulary import Vocabulary


class Vectorizer:
    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer: Tokenizer | None = None,
    ):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer or Tokenizer()

    def transform(self, text: str) -> torch.Tensor:
        """Converte uma frase em um vetor bag-of-words."""
        if not isinstance(text, str):
            raise TypeError("O texto deve ser uma string.")

        tokens = self.tokenizer.tokenize(text)
        token_ids = self.vocabulary.encode(tokens)

        vector = torch.zeros(
            len(self.vocabulary),
            dtype=torch.float32,
        )

        for token_id in token_ids:
            if token_id > 1:
                vector[token_id] = 1.0

        return vector

    def transform_batch(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        """Converte várias frases em uma matriz de vetores."""
        if not texts:
            raise ValueError("A lista de textos não pode estar vazia.")

        return torch.stack([self.transform(text) for text in texts])
