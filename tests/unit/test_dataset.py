import json

import pytest

from sivraj_neural.train import IntentDataset


def test_load_dataset(tmp_path):
    path = tmp_path / "intents.json"

    data = {
        "intents": [
            {
                "tag": "greeting",
                "examples": ["oi", "olá"],
            },
            {
                "tag": "open_app",
                "examples": ["abra o vscode"],
            },
        ]
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )

    dataset = IntentDataset(path)

    assert len(dataset) == 3
    assert dataset.get_tags() == ["greeting", "open_app"]


def test_get_examples(tmp_path):
    path = tmp_path / "intents.json"

    data = {
        "intents": [
            {
                "tag": "greeting",
                "examples": ["oi", "olá"],
            }
        ]
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )

    dataset = IntentDataset(path)

    assert dataset.get_examples() == [
        ("oi", "greeting"),
        ("olá", "greeting"),
    ]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        IntentDataset("arquivo_inexistente.json")


def test_empty_intents(tmp_path):
    path = tmp_path / "intents.json"
    path.write_text('{"intents": []}', encoding="utf-8")

    with pytest.raises(ValueError):
        IntentDataset(path)


def test_duplicate_tags(tmp_path):
    path = tmp_path / "intents.json"

    data = {
        "intents": [
            {"tag": "greeting", "examples": ["oi"]},
            {"tag": "greeting", "examples": ["olá"]},
        ]
    }

    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        IntentDataset(path)
