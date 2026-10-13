from .trainer import Trainer


def main() -> None:
    trainer = Trainer()
    trainer.summary()

    data = trainer.prepare_data()

    print("\n=== Exemplos preparados ===")

    for item in data[:5]:
        print(f"Texto: {item['text']}")
        print(f"Tokens: {item['tokens']}")
        print(f"IDs: {item['input_ids']}")
        print(f"Intenção: {item['tag']}")
        print(f"Label: {item['label']}")
        print()

    if __name__ == "__main__":
        main()
