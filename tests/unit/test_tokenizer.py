import pytest

from sivraj_neural.token import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


def test_basic_tokenization(tokenizer):
    assert tokenizer.tokenize("Abra o VS Code!") == [
        "abra",
        "o",
        "vs",
        "code",
        "!",
    ]


def test_punctuation(tokenizer):
    assert tokenizer.tokenize("Olá, SIVRAJ!") == [
        "olá",
        ",",
        "sivraj",
        "!",
    ]


def test_accents(tokenizer):
    assert tokenizer.tokenize("Ação rápida") == [
        "ação",
        "rápida",
    ]


def test_extra_spaces(tokenizer):
    assert tokenizer.tokenize("  olá   mundo  ") == [
        "olá",
        "mundo",
    ]


def test_empty_string(tokenizer):
    assert tokenizer.tokenize("") == []


def test_non_string_input(tokenizer):
    with pytest.raises(TypeError):
        tokenizer.tokenize(123)


def test_case_preservation():
    tokenizer = Tokenizer(lowercase=False)

    assert tokenizer.tokenize("Olá SIVRAJ") == [
        "Olá",
        "SIVRAJ",
    ]
