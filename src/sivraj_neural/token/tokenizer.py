import re
import unicodedata


class Tokenizer:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = unicodedata.normalize("NFC", text)
        text = text.strip()

        if self.lowercase:
            text = text.lower()

        return text

    def tokenize(self, text: str) -> list[str]:
        text = self.normalize(text)

        return re.findall(
            r"\w+|[^\w\s]",
            text,
            flags=re.UNICODE,
        )

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)
