import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
from datetime import datetime
from models.transformer import DecoderOnlyTransformer, Config
import tiktoken
import torch.nn as nn

class TextDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        self.num_samples = len(text) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return (
            self.text[start:start+self.seq_len],
            self.text[start+1:start+self.seq_len+1]
        )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_memory(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024 ** 2)
    return total_size

def train(model, dataset, config, epochs, lr=1e-4, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Model Size: {count_parameters(model) / 1e6:.2f}M parameters")
    print(f"Estimated Model Memory Usage: {estimate_memory(model):.2f} MB")

    model.train()
    step = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            step += 1
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, config.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Step {step}: Loss = {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f" Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

        if avg_loss < 0.099999:
            print("Early stopping as loss is below target")
            break

        torch.save(model.state_dict(), f"{log_dir}/model_epoch_{epoch+1}.pt")
        print("Checkpoint saved..")

        with open(f"{log_dir}/training_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()} - Step {step} - Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}\n")

if __name__ == '__main__':
    config = Config()
    model = DecoderOnlyTransformer(config)

    with open('data/input.txt', 'r') as fp:
        text = fp.read()

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor(enc.encode(text), dtype=torch.long)

    dataset = TextDataset(idx, seq_len=512)
    train(model, dataset, config, epochs=100)

    torch.save(model.state_dict(), "final_model.pt")
    print("Model training complete. Upload to GitHub and Hugging Face Spaces.") 