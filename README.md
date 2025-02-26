# Build an LLM for Shakespeare style text generation

This repository contains a decoder-only transformer model implemented using PyTorch. The project includes scripts for training the model on text data and performing inference to generate text.

## Features

- Multi-head attention mechanism
- Feed-forward neural network
- Layer normalization
- Dropout for regularization

## Requirements

- Python 3.8+
- PyTorch
- tiktoken

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sdeepthikumar/shakespearean-llm.git
   cd shakespearean-llm
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, ensure you have your training data in `data/input.txt` and run:
```bash
python scripts/train.py
```
This will train the model and save checkpoints in the `logs/` directory.

### Inference

To perform inference using the trained model, run:
```bash
python scripts/inference.py
```
This script will load the model and generate text based on a sample input.

## Project Structure

- `models/`: Contains the model architecture.
- `scripts/`: Contains scripts for training and inference.
- `notebooks/`: Contains Jupyter notebooks executed in colab.
- `data/`: Directory for input data files.
- `logs/`: Directory where training logs and model checkpoints are saved.


