import torch

from sivraj_neural.infer import Inference
from sivraj_neural.model.model import IntentClassifier
from sivraj_neural.token.vocabulary import Vocabulary


def test_inference_loads_trained_model(tmp_path):
    vocabulary_path = tmp_path / "vocab.json"
    model_path = tmp_path / "model.pth"

    vocabulary = Vocabulary(vocabulary_path)
    vocabulary.build(
        [
            ["ola"],
            ["abrir", "navegador"],
        ]
    )

    tags = ["greeting", "open_app"]

    model = IntentClassifier(
        input_size=len(vocabulary),
        hidden_size=8,
        num_classes=len(tags),
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(vocabulary),
            "hidden_size": 8,
            "num_classes": len(tags),
            "tags": tags,
            "tag_to_id": {tag: index for index, tag in enumerate(tags)},
        },
        model_path,
    )

    inference = Inference(
        model_path=model_path,
        vocabulary_path=vocabulary_path,
    )

    result = inference.predict("ola")

    assert result["intent"] in tags
    assert 0.0 <= result["confidence"] <= 1.0


def test_inference_rejects_empty_text(tmp_path):
    vocabulary_path = tmp_path / "vocab.json"
    model_path = tmp_path / "model.pth"

    vocabulary = Vocabulary(vocabulary_path)
    vocabulary.build([["ola"], ["adeus"]])

    model = IntentClassifier(
        input_size=len(vocabulary),
        hidden_size=8,
        num_classes=2,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(vocabulary),
            "hidden_size": 8,
            "num_classes": 2,
            "tags": ["greeting", "farewell"],
            "tag_to_id": {
                "greeting": 0,
                "farewell": 1,
            },
        },
        model_path,
    )

    inference = Inference(
        model_path=model_path,
        vocabulary_path=vocabulary_path,
    )

    try:
        inference.predict("   ")
    except ValueError:
        pass
    else:
        raise AssertionError("Era esperado um ValueError.")
