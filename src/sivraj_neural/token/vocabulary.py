import json
from collections.abc import Iterable
from pathlib import Path


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, path: str | Path = "data/vocab.json"):
        self.path = Path(path)
        self.token_to_id: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.id_to_token: dict[int, str] = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def add(self, token: str) -> int:
        """Adiciona um token se ele ainda não existir."""
        if not isinstance(token, str):
            raise TypeError("O token deve ser uma string.")

        if not token:
            raise ValueError("O token não pode ser vazio.")

        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        return self.token_to_id[token]

    def build(self, token_sequences: Iterable[Iterable[str]]) -> None:
        """Constrói um vocabulário determinístico a partir dos tokens."""
        tokens = set()

        for sequence in token_sequences:
            for token in sequence:
                if not isinstance(token, str):
                    raise TypeError("Todos os tokens devem ser strings.")

                if not token:
                    raise ValueError("Os tokens não podem ser vazios.")

                if token not in {self.PAD_TOKEN, self.UNK_TOKEN}:
                    tokens.add(token)

        self.token_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        self.id_to_token = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }

        for token in sorted(tokens):
            self.add(token)

        self.save()

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Converte tokens em IDs; tokens desconhecidos viram <UNK>."""
        unknown_id = self.token_to_id[self.UNK_TOKEN]

        return [self.token_to_id.get(token, unknown_id) for token in tokens]

    def decode(self, token_ids: Iterable[int]) -> list[str]:
        """Converte IDs em tokens."""
        return [
            self.id_to_token.get(token_id, self.UNK_TOKEN)
            for token_id in token_ids
        ]

    def save(self) -> None:
        """Salva o vocabulário em JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w", encoding="utf-8") as file:
            json.dump(
                self.token_to_id,
                file,
                ensure_ascii=False,
                indent=4,
                sort_keys=False,
            )

    def load(self) -> None:
        """Carrega e valida um vocabulário existente."""
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("O vocabulário deve ser um objeto JSON.")

        if data.get(self.PAD_TOKEN) != 0:
            raise ValueError("O token <PAD> deve ter ID 0.")

        if data.get(self.UNK_TOKEN) != 1:
            raise ValueError("O token <UNK> deve ter ID 1.")

        if any(not isinstance(token, str) for token in data):
            raise ValueError("Todos os tokens devem ser strings.")

        ids = list(data.values())

        if any(type(token_id) is not int or token_id < 0 for token_id in ids):
            raise ValueError("Todos os IDs devem ser inteiros não negativos.")

        if len(ids) != len(set(ids)):
            raise ValueError("O vocabulário contém IDs duplicados.")

        if set(ids) != set(range(len(ids))):
            raise ValueError(
                "Os IDs devem ser consecutivos, começando em zero."
            )

        self.token_to_id = data
        self.id_to_token = {
            token_id: token for token, token_id in data.items()
        }
