
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


model = NeuralNetwork()

print(model)