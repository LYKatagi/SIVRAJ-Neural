import json
from pathlib import Path


class Vocabulary:
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    def __init__(self, path: str | Path = "vocab.json"):
        self.path = Path(path)
        self.token_to_id = dict(self.SPECIAL_TOKENS)

        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as file:
            vocabulary = json.load(file)

        if not isinstance(vocabulary, dict):
            raise ValueError("Vocabulary must be a JSON object.")

        if not all(
            isinstance(token, str)
            and isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id >= 0
            for token, token_id in vocabulary.items()
        ):
            raise ValueError("Invalid vocabulary entries.")

        ids = list(vocabulary.values())

        if len(ids) != len(set(ids)):
            raise ValueError("Vocabulary IDs must be unique.")

        for token, token_id in self.SPECIAL_TOKENS.items():
            if vocabulary.get(token) != token_id:
                raise ValueError(f"Missing or invalid special token: {token}")

        self.token_to_id = vocabulary

    def save(self) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open("w", encoding="utf-8") as file:
            json.dump(
                self.token_to_id,
                file,
                ensure_ascii=False,
                indent=4,
            )

    def add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]

        token_id = max(self.token_to_id.values(), default=-1) + 1

        self.token_to_id[token] = token_id
        self.save()

        return token_id

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.add(token) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

        return [id_to_token.get(token_id, "<UNK>") for token_id in ids]
