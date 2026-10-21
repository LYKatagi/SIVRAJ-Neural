from sivraj_neural.token.tokenizer import Tokenizer
from sivraj_neural.token.vectorizer import Vectorizer
from sivraj_neural.token.vocabulary import Vocabulary


def test_vectorizer_output_size(tmp_path):
    vocabulary = Vocabulary(tmp_path / "vocab.json")
    vocabulary.build([["abrir", "navegador"]])

    vectorizer = Vectorizer(vocabulary, Tokenizer())
    vector = vectorizer.transform("abrir navegador")

    assert vector.shape == (len(vocabulary),)
    assert vector.dtype.name if False else vector.dtype.is_floating_point


def test_vectorizer_unknown_words(tmp_path):
    vocabulary = Vocabulary(tmp_path / "vocab.json")
    vocabulary.build([["abrir", "navegador"]])

    vectorizer = Vectorizer(vocabulary)
    vector = vectorizer.transform("palavra_desconhecida")

    assert vector.sum().item() == 0


def test_vectorizer_batch(tmp_path):
    vocabulary = Vocabulary(tmp_path / "vocab.json")
    vocabulary.build([["abrir", "navegador"]])

    vectorizer = Vectorizer(vocabulary)
    batch = vectorizer.transform_batch(["abrir navegador", "abrir"])

    assert batch.shape == (2, len(vocabulary))
