import requests 

def load_data(file_path):
    if file_path.startswith('http'):
        response = requests.get(file_path)
        text = response.text
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    return text, chars, vocab_size, stoi, itos
