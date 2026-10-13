import pytest
import torch

from sivraj_neural.model import IntentClassifier

def test_output_shape():
    model = IntentClassifier(
    input_size=20,
    hidden_size=16,
    num_classes=6,
    )


    inputs = torch.zeros(4, 20)
    outputs = model(inputs)

    assert outputs.shape == (4, 6)


def test_single_example():
    model = IntentClassifier(
    input_size=10,
    hidden_size=8,
    num_classes=3,
    )


    inputs = torch.zeros(1, 10)
    outputs = model(inputs)

    assert outputs.shape == (1, 3)


def test_invalid_input_size():
    with pytest.raises(ValueError):
        IntentClassifier(
        input_size=0,
        hidden_size=8,
        num_classes=3,
        )

def test_invalid_num_classes():
    with pytest.raises(ValueError):
        IntentClassifier(
        input_size=10,
        hidden_size=8,
        num_classes=1,
        )
