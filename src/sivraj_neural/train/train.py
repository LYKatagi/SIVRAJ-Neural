import torch
from torch import nn

from sivraj_neural.model import NeuralNetwork

def train():
    # Dados de treinamento
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ])


    model = NeuralNetwork()

    loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )


    # Treinamento
    for epoch in range(5000):
        predictions = model(X)

        loss = loss_function(
            predictions,
            y,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(
                f"Época: {epoch + 1}, "
                f"Erro: {loss.item():.4f}"
            )


    # Salvar os parâmetros treinados
    torch.save(
        model.state_dict(),
        "model.pth",
    )

    print("Treinamento concluído!")