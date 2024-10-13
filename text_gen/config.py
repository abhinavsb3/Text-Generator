import torch
from utils import 

batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0
vocab_size =len(chars) #64 is test value this is the vocab size real value =len(chars) and chars should import from the utils


torch.manual_seed(1337)

