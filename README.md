# SIVRAJ-Neural

A lightweight, independent neural network project built with PyTorch for command interpretation and basic conversational interactions.

SIVRAJ-Neural is designed to become the intelligence component of the SIVRAJ assistant, with a focus on recognizing user intentions, classifying commands, and supporting simple conversations.

## Features

**Planned features:**

* [ ] Intent classification using PyTorch.
* [ ] Text preprocessing and tokenization.
* [ ] Custom vocabulary management.
* [ ] Training and evaluation pipelines.
* [ ] Model checkpoint saving and loading.
* [ ] Basic conversational intent recognition.
* [ ] Command argument extraction.
* [ ] Confidence-based prediction rejection.
* [ ] Integration with the SIVRAJ assistant.

## Requirements

* Python 3.11 or newer.
* PyTorch.
* Git (recommended).

A CPU is sufficient for the initial experiments.

## Installation

Clone the repository:

```bash
git clone https://github.com/LYKatagi/SIVRAJ-Neural.git
cd SIVRAJ-Neural
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch using the official installation selector:

https://pytorch.org/get-started/locally/

Install the project in editable mode with its development dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Project Structure

```text
SIVRAJ-Neural/
├── src/
│   └── sivraj_neural/
│       ├── __init__.py
│       ├── dataset.py
│       ├── inference.py
│       ├── model.py
│       ├── tokenizer.py
│       └── trainer.py
├── data/
│   └── intents.json
├── models/
├── tests/
├── train.py
├── pyproject.toml
├── .gitignore
├── .gitattributes
└── README.md
```

Some files and directories may not exist yet. They will be added as development progresses.

## Development

Run the test suite:

```bash
python -m pytest
```

Check code quality:

```bash
ruff check .
```

Format the code:

```bash
ruff format .
```

## How It Works

The initial architecture is planned around an intent-classification pipeline:

1. Receive a text input.
2. Normalize and tokenize the text.
3. Convert tokens into numerical representations.
4. Run the input through a trained neural network.
5. Predict the most likely intent.
6. Validate the prediction and its confidence.
7. Pass the result to the appropriate application logic.

The initial model will focus on classifying intents such as:

* `greeting`
* `gratitude`
* `open_app`
* `get_time`
* `unknown`

The model will be developed and evaluated independently of the main SIVRAJ application.

## Safety

Neural network predictions are not guaranteed to be correct.

Predictions must be validated before they are used to trigger system operations. Application execution should use an explicit allowlist of supported actions and validated arguments.

The model must not be trusted to execute arbitrary operating system commands.

## License

This project is distributed under the MIT License.

See [LICENSE](LICENSE) for details.
