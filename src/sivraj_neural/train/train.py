import argparse

from sivraj_neural.train.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina o classificador de intenções do SIVRAJ-Neural."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de épocas de treinamento.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taxa de aprendizado.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Quantidade de exemplos por lote.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Quantidade de neurônios na camada oculta.",
    )

    args = parser.parse_args()

    trainer = Trainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
    )

    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
