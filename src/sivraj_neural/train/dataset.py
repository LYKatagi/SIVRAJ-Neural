import json
from pathlib import Path


class IntentDataset:
    def __init__(self, path: str | Path = "data/intents.json"):
        self.path = Path(path)
        self.intents = self.load()

    def load(self) -> list[dict]:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Dataset must be a JSON object.")

        intents = data.get("intents")

        if not isinstance(intents, list) or not intents:
            raise ValueError(
                "Dataset must contain a non-empty 'intents' list."
            )

        seen_tags = set()

        for intent in intents:
            if not isinstance(intent, dict):
                raise ValueError("Each intent must be an object.")

            tag = intent.get("tag")
            examples = intent.get("examples")

            if not isinstance(tag, str) or not tag.strip():
                raise ValueError("Each intent must have a non-empty tag.")

            if tag in seen_tags:
                raise ValueError(f"Duplicate intent tag: {tag}")

            seen_tags.add(tag)

            if not isinstance(examples, list) or not examples:
                raise ValueError(f"Intent '{tag}' must have examples.")

            if not all(
                isinstance(example, str) and example.strip()
                for example in examples
            ):
                raise ValueError(
                    f"Intent '{tag}' contains an invalid example."
                )

        return intents

    def get_tags(self) -> list[str]:
        return [intent["tag"] for intent in self.intents]

    def get_examples(self) -> list[tuple[str, str]]:
        examples = []

        for intent in self.intents:
            tag = intent["tag"]

            for example in intent["examples"]:
                examples.append((example, tag))

        return examples

    def __len__(self) -> int:
        return sum(len(intent["examples"]) for intent in self.intents)
