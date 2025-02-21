import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 
context_length = 256 
maximum_training_steps = 50000
evaluation_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iteration_count = 20
# Embedding depth: higher dimensionality captures more nuanced relationships
embedding_dimension_count = 384
head_count = 4
layer_count = 6
dropout = 0.33 # It will randomly silence some neuron (the fraction)
max_new_token_number = 10000

if(torch.cuda.is_available()):
    print(f"{torch.cuda.device_count()} GPU DETECTED: {torch.cuda.get_device_name(0)}")
# ------------

torch.manual_seed(1337) # For reproducibility

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('shakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()


vocabulary = sorted(list(set(text))) # The possible tokens for our model, sorted by ascii value
vocabulary_size = len(vocabulary)
# we create a dictionnary from string to int
string_to_int = { char:int for int,char in enumerate(vocabulary) } 
int_to_string = { int:char for int,char in enumerate(vocabulary) } # we create a dictionnary from int to string

# Each character of the string will be converted to tokens encoded in int
tokenize = lambda string_to_tokenize: [string_to_int[character] for character in string_to_tokenize] 
# each token will be converted into char, and it will be concatenated to form a string
detokenize = lambda tokens_to_stringify: ''.join([int_to_string[integer] for integer in tokens_to_stringify]) 

# we tokenize our dataset
tokenized_data = torch.tensor(tokenize(text), dtype=torch.long)
# we train on 90% of the dataset
training_data_size = int(0.9*len(tokenized_data)) 
training_data = tokenized_data[:training_data_size]
# we eval to the remaining 10%
evaluation_data = tokenized_data[training_data_size:]

# Take 2 random batches in the dataset (input_tokens and solution_tokens)
def get_batch(data_partition_name):
    # According to the type partition name, we choose between training or eval
    data = training_data if data_partition_name == 'train' else evaluation_data
    # the max of the strating_offset for the batch, we substract 1 to avoir y overflow
    max_offset = len(data) - context_length-1
    # we take random offsets from the given dataset (without overflowing the size)
    random_start_offsets = torch.randint(max_offset, (batch_size,))
    # for each random start_offset we take a slice of the context length
    input_tokens = torch.stack([data[offset:offset+context_length] for offset in random_start_offsets])
    # Same but we offset by 1 to have the solution (next token)
    solution_tokens = torch.stack([data[offset+1:offset+1+context_length] for offset in random_start_offsets])
    #We choose the right  device to put the data (gpu/cpu)
    input_tokens, solution_tokens = input_tokens.to(device), solution_tokens.to(device)
    return input_tokens, solution_tokens

# We eval the mean loss for the training data and eval data
# The eval_iteration_count define how many points we take for the data
@torch.no_grad()
def calculate_mean_losses():
    mean_losses = {}
    model.eval()
    for data_partition_name in ['train', 'val']:
        calculate_mean_loss(mean_losses, data_partition_name)
    model.train()
    return mean_losses

# Calculate the mean for one dataset
def calculate_mean_loss(mean_losses, data_partition_name):
    losses = torch.zeros(eval_iteration_count)
    for eval_iteration_number in range(eval_iteration_count):
        inputs, solutions = get_batch(data_partition_name)
        logits, loss = model(inputs, solutions)
        losses[eval_iteration_number] = loss.item()
    mean_losses[data_partition_name] = losses.mean()





# This where attention takes place
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # We store the keys of each token, it's like a profile or a cv
        # it will be compared to other token queries
        self.keys = nn.Linear(embedding_dimension_count, head_size, bias=False)

        # Queries are what the token wants to know about other tokens
        self.queries = nn.Linear(embedding_dimension_count, head_size, bias=False)

        # Values will be the actual info shared according to the matching between queries and keys
        self.values = nn.Linear(embedding_dimension_count, head_size, bias=False)

        # This is the context_window_mask, we block info from the futur tokens
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        
        #  To help share the knowledge between neurons, we randomly silence some neuron
        self.dropouts = nn.Dropout(dropout)
    
    def forward(self, current_token_contexts):
        batch_size,token_count,channel_count = current_token_contexts.shape
        # compute keys
        keys = self.keys(current_token_contexts)

        # compute queries
        queries = self.queries(current_token_contexts)

        # compute attention scores, how much do each care about each other
        # The formule is : dot_product between keys and values  (we transpose to allow dot product)
        # We devide by the square root of embedding dimmensions
        attention_scores = queries @ keys.transpose(-2,-1) * keys.shape[-1]**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        
        # we mask the attention that are before each token
        causal_attention_scores = attention_scores.masked_fill(self.tril[:token_count, :token_count] == 0, float('-inf')) # (B,T,T)
        
        # Now between 0 and 1
        probabilistic_causal_attention = F.softmax(causal_attention_scores, dim=1) # (B,T,T)
        
        # We randomly silence some neurons
        probabilistic_causal_attention = self.dropouts(probabilistic_causal_attention)

        # Compute values
        values = self.values(current_token_contexts) # (B,T,C)

        # We share values pondered by the attention
        shared_informations = probabilistic_causal_attention @ values # (B,T,T) @ (B,T,C) => (B,T,C)
        return shared_informations


# We cumulate several attention heads
class MultiHeadAttention(nn.Module):

    def __init__(self, head_count, head_size):
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(head_count)])
        self.projection = nn.Linear(embedding_dimension_count, embedding_dimension_count)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tokens):
        # we concatenate the result of multiple heads
        self_attended_tokens = torch.cat([head(input_tokens) for head in self.heads], dim=-1)
        projection = self.projection(self_attended_tokens) # We remix all head computation
        projection = self.dropout(projection)
        return projection

# A network to "think"
class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dimension_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension_count, 4*embedding_dimension_count),
            nn.ReLU(),
            nn.Linear(4*embedding_dimension_count, embedding_dimension_count),
            nn.Dropout(dropout),
        )
    
    def forward(self, input_tokens):
        return self.network(input_tokens)
    


class AttentionThinkingBlock(nn.Module):
    
    def __init__(self, embedding_dimension_count, head_count):
        super().__init__()
        head_size = embedding_dimension_count // head_count
        # We apply attention
        self.attention_network = MultiHeadAttention(head_count, head_size)
        # We think about these informations
        self.feed_forward_network = FeedForwardNetwork(embedding_dimension_count)

        self.attention_layer_normalization = nn.LayerNorm(embedding_dimension_count)
        self.feed_forward_layer_normalization = nn.LayerNorm(embedding_dimension_count)

    def forward(self, input_tokens):
        ## We normalize values before attention
        normalized_input_tokens = self.attention_layer_normalization(input_tokens)
        # We apply attention and keep the input to avoid forgetting with back propagation
        attended_tokens = input_tokens + self.attention_network(normalized_input_tokens) 
        ## We normalize values before "thinking"
        normalized_attended_tokens = self.feed_forward_layer_normalization(attended_tokens)
        # We think about the attention new info
        thought_attended_tokens = attended_tokens + self.feed_forward_network(normalized_attended_tokens)
        return thought_attended_tokens

# super simple bigram model
class GptOne(nn.Module):

    def __init__(self):
        super().__init__()
        # The meaning of the token in vector space
        self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimension_count)
        # The info of position in the vector space
        self.position_embedding_table = nn.Embedding(context_length, embedding_dimension_count)
       
        # Attention + thinking
        self.attention_thinking_blocks = nn.Sequential(
            *[AttentionThinkingBlock(embedding_dimension_count, head_count) for _ in range(layer_count)] 
        )
        # normalization
        self.final_layer_normalization = nn.LayerNorm(embedding_dimension_count)

        
        self.language_modeling_head = nn.Linear(embedding_dimension_count, vocabulary_size)

    def forward(self, input_tokens, solution_tokens=None):
        batch_size, time_steps = input_tokens.shape
        
        token_embeddings = self.token_embedding_table(input_tokens) # (B,T,C)
        
        position_embeddings = self.position_embedding_table(torch.arange(time_steps, device=device))# (T,C)
        
        spatial_meaning_embedding = token_embeddings + position_embeddings
        spatial_meaning_embedding = self.attention_thinking_blocks(spatial_meaning_embedding)
        normalized_thought_embedding = self.final_layer_normalization(spatial_meaning_embedding)
        logits = self.language_modeling_head(normalized_thought_embedding) # (B,T,Cvocab_size)

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
            
            logits, loss = self(context_tokens)
            
            logits = logits[:, -1, :] # becomes (B, C)
            
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            input_tokens = torch.cat((input_tokens, next_token), dim=1) # (B, T+1)
        return input_tokens

model = GptOne()
initialized_model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(maximum_training_steps):

    
    if step % evaluation_interval == 0 or step == maximum_training_steps - 1:
        losses = calculate_mean_losses()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    
    random_input_tokens, solution_tokenS = get_batch('train')

    
    logits, loss = model(random_input_tokens, solution_tokenS)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


starting_context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(detokenize(initialized_model.generate(starting_context, max_new_token_number=max_new_token_number)[0].tolist()))