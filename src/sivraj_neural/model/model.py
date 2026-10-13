import torch
from torch import nn


class IntentClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
    ):
        super().__init__()

        if input_size < 1:
            raise ValueError("input_size must be positive.")

        if hidden_size < 1:
            raise ValueError("hidden_size must be positive.")

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
