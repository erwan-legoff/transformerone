import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
context_length = 8 # what is the maximum context length for predictions?
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

print(torch.cuda.is_available())  # Doit retourner True si CUDA est actif
print(torch.cuda.device_count())  # Nombre de GPU disponibles
print(torch.cuda.get_device_name(0))  # Nom du premier GPU détecté
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('shakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
vocabulary = sorted(list(set(text)))
vocabulary_size = len(vocabulary)
# create a mapping from characters to integers
string_to_int = { char:int for int,char in enumerate(vocabulary) }
int_to_string = { int:char for int,char in enumerate(vocabulary) }
tokenize = lambda s: [string_to_int[c] for c in s] # encoder: take a string, output a list of integers
detokenize = lambda l: ''.join([int_to_string[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
tokenized_data = torch.tensor(tokenize(text), dtype=torch.long)
training_data_size = int(0.9*len(tokenized_data)) # first 90% will be train, rest val
training_data = tokenized_data[:training_data_size]
evaluation_data = tokenized_data[training_data_size:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = training_data if split == 'train' else evaluation_data
    random_offsets = torch.randint(len(data) - context_length, (batch_size,))
    starting_token = torch.stack([data[offset:offset+context_length] for offset in random_offsets])
    solution_token = torch.stack([data[offset+1:offset+context_length+1] for offset in random_offsets])
    starting_token, solution_token = starting_token.to(device), solution_token.to(device)
    return starting_token, solution_token

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, starting_tokens, solution_tokens=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(starting_tokens) # (B,T,C)

        if solution_tokens is None:
            loss = None
        else:
            batch_size, time_steps, channel_size = logits.shape
            logits = logits.view(batch_size*time_steps, channel_size)
            solution_tokens = solution_tokens.view(batch_size*time_steps)
            loss = F.cross_entropy(logits, solution_tokens)

        return logits, loss

    def generate(self, starting_tokens, max_new_token_number):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_token_number):
            # get the predictions
            logits, loss = self(starting_tokens)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            starting_tokens = torch.cat((starting_tokens, next_token), dim=1) # (B, T+1)
        return starting_tokens

model = BigramLanguageModel()
initialized_model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    batched_starting_token, batched_solution_token = get_batch('train')

    # evaluate the loss
    logits, loss = model(batched_starting_token, batched_solution_token)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
starting_context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(detokenize(initialized_model.generate(starting_context, max_new_token_number=500)[0].tolist()))