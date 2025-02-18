import torch
from text_gen.models.bigram_language_model import BigramLanguageModel
from text_gen.data.dataset import load_data  # Import function that provides decode & itos
from text_gen.utils import decode

# Load the tokenizer mappings
_, _, vocab_size, stoi, itos = load_data('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')  # Assuming load_data returns these

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BigramLanguageModel(vocab_size)  # Initialize the model
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))  # Load trained model
model.to(device)
model.eval()  # Set to evaluation mode

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(context, max_new_tokens=2000)

# Decode and print the generated text
print(decode(generated_text[0].tolist(), itos))
