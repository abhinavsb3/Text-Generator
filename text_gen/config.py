import torch

batch_size = 16 #128#increasing model complexity by increasing parameters
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("CUDA is available")
else:
    print("CUDA is not available")

eval_iters = 200
n_embd = 64#128#
n_head = 4#8#
n_layer = 4#8
dropout = 0
# vocab_size =len(chars) 


torch.manual_seed(1337)

