import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 256 # how many independent sequences will we process in parallel?
context_length = 30 # what is the maximum context length for predictions?
max_iters = 30000
eval_interval = 1000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
embedding_dimension_count = 64
head_count = 4

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
tokenize = lambda string: [string_to_int[character] for character in string] # encoder: take a string, output a list of integers
detokenize = lambda integers: ''.join([int_to_string[integer] for integer in integers]) # decoder: take a list of integers, output a string

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






class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dimension_count, head_size, bias=False)
        self.query = nn.Linear(embedding_dimension_count, head_size, bias=False)
        self.value = nn.Linear(embedding_dimension_count, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, current_token_context):
        batch_size,time_step_count,channel_count = current_token_context.shape
        key = self.key(current_token_context)
        query = self.query(current_token_context)
        attention_scores = query @ key.transpose(-2,-1) * channel_count**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        causal_attention_scores = attention_scores.masked_fill(self.tril[:time_step_count, :time_step_count] == 0, float('-inf')) # (B,T,T)
        probabilistic_causal_attention = F.softmax(causal_attention_scores, dim=1) # (B,T,T)

        value = self.value(current_token_context) # (B,T,C)
        weighted_aggregation = probabilistic_causal_attention @ value # (B,T,T) @ (B,T,C) => (B,T,C)
        return weighted_aggregation

class MultiHeadAttention(nn.Module):

    def __init__(self, head_count, head_size):
        super().__init__()
        # We create multiple head_attention
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(head_count)])
        
    def forward(self, input_tokens):
        # we concatenate the result of multiple heads
        return torch.cat([head(input_tokens) for head in self.heads], dim=-1)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dimension_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension_count, embedding_dimension_count),
            nn.ReLU(),
        )
    
    def forward(self, input_tokens):
        return self.network(input_tokens)
    
class AttentionThinkingBlock(nn.Module):
    
    def __init__(self, embedding_dimension_count, head_count):
        super().__init__()
        head_size = embedding_dimension_count // head_count
        # Make the tokens talk to each other
        self.attention_network = MultiHeadAttention(head_count, head_size)
        # Make tokens thinks with this new information
        self.feed_forward_network = FeedForwardNetwork(embedding_dimension_count)

    def forward(self, input_tokens):
        attended_tokens = input_tokens + self.attention_network(input_tokens) # We keep a skip connection to improve the retropagation retention
        thought_attended_tokens = attended_tokens + self.feed_forward_network(attended_tokens)
        return thought_attended_tokens

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # it's the identity of the token
        self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimension_count)
        # the position is also embedded
        self.position_embedding_table = nn.Embedding(context_length, embedding_dimension_count)
       
        # Tokens will communicate with each other and think about it multiple times
        self.attention_thinking_block = nn.Sequential(
            AttentionThinkingBlock(embedding_dimension_count, head_count),
            AttentionThinkingBlock(embedding_dimension_count, head_count),
            AttentionThinkingBlock(embedding_dimension_count, head_count)
        )

        # Will convert embeddings to logits
        self.language_modeling_head = nn.Linear(embedding_dimension_count, vocabulary_size)

    def forward(self, input_tokens, solution_tokens=None):
        batch_size, time_steps = input_tokens.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(input_tokens) # (B,T,C)
        
        position_embeddings = self.position_embedding_table(torch.arange(time_steps, device=device))# (T,C)
        
        spatial_meaning_embedding = token_embeddings + position_embeddings
        spatial_meaning_embedding = self.attention_thinking_block(spatial_meaning_embedding)

        logits = self.language_modeling_head(spatial_meaning_embedding) # (B,T,Cvocab_size)

        if solution_tokens is None:
            loss = None
        else:
            batch_size, time_steps, channel_size = logits.shape
            logits = logits.view(batch_size*time_steps, channel_size)
            solution_tokens = solution_tokens.view(batch_size*time_steps)
            loss = F.cross_entropy(logits, solution_tokens)

        return logits, loss

    def generate(self, input_tokens, max_new_token_number):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_token_number):
            context_tokens = input_tokens[:, -context_length:]
            # get the predictions
            logits, loss = self(context_tokens)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            input_tokens = torch.cat((input_tokens, next_token), dim=1) # (B, T+1)
        return input_tokens

model = BigramLanguageModel()
initialized_model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
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