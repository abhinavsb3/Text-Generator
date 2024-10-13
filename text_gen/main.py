import torch
from config import data_path, vocab_size, device
from data.dataset import load_data
from models.bigram_language_model import BigramLanguageModel
from utils import encode, decode, get_batch
from trainings.train import train_model
from torch.optim import AdamW

data_path = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


if __name__ == "__main__":
    # Load the data and mappings
    text, chars, vocab_size, stoi, itos = load_data(data_path)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))  # Train-validation split
    train_data = data[:n]
    val_data = data[n:]

    # Initialize the model and optimizer
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Train the model
    train_model(model, optimizer, train_data, val_data)

    # Generate some text using the trained model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with an empty context
    generated_text = model.generate(context, max_new_tokens=2000)
    print(decode(generated_text[0].tolist(), itos))  # Decode and print the generated text