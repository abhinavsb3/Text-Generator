# train.py

import torch
from text_gen.utils import get_batch
from text_gen.models.bigram_language_model import BigramLanguageModel
from text_gen.config import max_iters, eval_interval, learning_rate, eval_iters, device
from text_gen.utils import encode, decode

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, None, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train_model(model, optimizer, train_data, val_data):
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if torch.cuda.is_available() and iter % 100 == 0:
                print("training loop print succesfull")
                print(torch.cuda.memory_summary()) #chat gpt suggested line

        xb, yb = get_batch('train', None, train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
