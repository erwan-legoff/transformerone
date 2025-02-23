from datetime import datetime, timedelta
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 
context_length = 256
maximum_training_steps = 10000
evaluation_interval = 1000
eval_iteration_count = 60
learning_rate = 4e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Embedding depth: higher dimensionality captures more nuanced relationships
embedding_dimension_count = 512 
head_count = 8
layer_count = 10
dropout = 0.40 # It will randomly silence some neuron (the fraction)
max_new_token_number = 10000
max_new_token_number_preview = 125
model_file_name = "gpt_wiki_bigram_one"
generate_interval = 500
checkpoint_interval = 2000
time_estimation_interval = 50
short_eval_interval = 100
short_eval_iters = 5

if(torch.cuda.is_available()):
    print(f"{torch.cuda.device_count()} GPU DETECTED: {torch.cuda.get_device_name(0)}")
# ------------


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('wiki.train.tokens', 'r', encoding='utf-8') as f:
    training_text = f.read()

with open('wiki.test.tokens', 'r', encoding='utf-8') as f:
    eval_text = f.read()


chars = sorted(list(set(training_text))) # The possible tokens for our model, sorted by ascii value
# We want to have the possible characters in the training data and their count
def count_bigram_occurences(training_text):
    bigram_occurences = {}
    for c in range(len(training_text)-1):
        bigram = training_text[c]+training_text[c+1]
        if bigram not in bigram_occurences:
            bigram_occurences[bigram] = 1
        else:
            bigram_occurences[bigram] +=1
    return bigram_occurences

def count_char_occurences(training_text):
    char_occurences = {}
    for c in training_text:
        if c not in char_occurences:
            char_occurences[c] = 1
        else:
            char_occurences[c] +=1
    return char_occurences

bigram_occurences = count_bigram_occurences(training_text)
char_occurences = count_char_occurences(training_text)

# On enregistre dans un fichier les occurences
def save_list(occurences, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        if isinstance(occurences, dict):
            for c in occurences:
                f.write(f"{c}: {occurences[c]}\n")
        elif isinstance(occurences, list):
            for item in occurences:
                f.write(f"{item}\n")

save_list(bigram_occurences, 'bigram_occurences.txt')
save_list(char_occurences, 'char_occurences.txt')

# Now we want to keep the most frequent tokens
max_bigram_vocabulary_size = 319
top_bigrams_dict = dict(sorted(bigram_occurences.items(), key=lambda item: item[1], reverse=True)[:max_bigram_vocabulary_size])
max_char_vocabulary_size = 117
top_chars_dict = dict(sorted(char_occurences.items(), key=lambda item: item[1], reverse=True)[:max_char_vocabulary_size])

save_list(top_bigrams_dict, 'top_bigrams.txt')
save_list(top_chars_dict, 'top_chars.txt')

full_vocabulary = list(top_bigrams_dict.keys()) + list(top_chars_dict.keys())

# save in file
save_list(full_vocabulary, 'full_vocabulary.txt')


vocabulary_size = len(full_vocabulary)
# we create a dictionnary from string to int
string_to_int = { string:int for int,string in enumerate(full_vocabulary) } 
int_to_string = { int:string for int,string in enumerate(full_vocabulary) } # we create a dictionnary from int to string

# We will start by searching a matching bigram, if not we will search for a char
def tokenize(text):
    unknown_token = 0 
    int_tokens = []
    c = 0
    while c < len(text):
        # Si possible, essayer de tokeniser un bigramme
        if c < len(text) - 1 and (text[c] + text[c+1]) in string_to_int:
            int_tokens.append(string_to_int[text[c] + text[c+1]])
            c += 2  # On saute deux caractères
        else:
            # Sinon, tokeniser un caractère seul
            token = string_to_int.get(text[c], unknown_token)
            int_tokens.append(token)
            c += 1
    return int_tokens

# each token will be converted into char, and it will be concatenated to form a string
detokenize = lambda int_tokens: ''.join([int_to_string[integer] for integer in int_tokens]) 

# we tokenize our datasets
tokenized_training_data = torch.tensor(tokenize(training_text), dtype=torch.long)
tokenized_evaluation_data = torch.tensor(tokenize(eval_text), dtype=torch.long)


training_data = tokenized_training_data

evaluation_data = tokenized_evaluation_data

def verify_tokenization(tokenized_data, original_text):
    detokenized_text = detokenize(tokenized_data.tolist())
    error_count = sum(1 for a, b in zip(detokenized_text, original_text) if a != b)
    error_percentage = (error_count / len(original_text)) * 100

    missing_chars = {}
    for a, b in zip(detokenized_text, original_text):
        if a != b:
            if b not in missing_chars:
                missing_chars[b] = 1
            else:
                missing_chars[b] += 1

    sorted_missing_chars = sorted(missing_chars.items(), key=lambda item: item[1], reverse=True)

    # Count and display missing words that are in the vocabulary
    missing_words_in_vocab = {}
    for word in full_vocabulary:
        if word in missing_chars:
            missing_words_in_vocab[word] = missing_chars[word]

    print("Missing Words in Vocabulary:")
    for word, count in missing_words_in_vocab.items():
        print(f"{word}: {count}")

    return error_percentage, sorted_missing_chars

# Verification step
error_percentage, sorted_missing_chars = verify_tokenization(tokenized_training_data, training_text)
print(f"Error Percentage: {error_percentage:.2f}%")
print("Most Missing Characters (Top 20):")
for char, count in sorted_missing_chars[:20]:
    print(f"{char}: {count}")

# Write the first 1000 re-detokenized tokens to a text file
def write_first_1000_tokens_to_file(tokenized_data, file_name):
    detokenized_text = detokenize(tokenized_data.tolist()[:1000])
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(detokenized_text)

write_first_1000_tokens_to_file(tokenized_training_data, 'first_1000_tokens.txt')

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

@torch.no_grad()
def calculate_short_mean_losses():
    mean_losses = {}
    model.eval()
    for data_partition_name in ['train', 'val']:
        calculate_short_mean_loss(mean_losses, data_partition_name)
    model.train()
    return mean_losses

def calculate_short_mean_loss(mean_losses, data_partition_name):
    losses = torch.zeros(short_eval_iters)
    for eval_iteration_number in range(short_eval_iters):
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
        probabilistic_causal_attention = F.softmax(causal_attention_scores, dim=-1) # (B,T,T)
        
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
        batch_size, token_count = input_tokens.shape
        
        token_embeddings = self.token_embedding_table(input_tokens) # (B,T,C)
        
        position_embeddings = self.position_embedding_table(torch.arange(token_count, device=device))# (T,C)
        
        spatial_meaning_embedding = token_embeddings + position_embeddings
        spatial_meaning_embedding = self.attention_thinking_blocks(spatial_meaning_embedding)
        normalized_thought_embedding = self.final_layer_normalization(spatial_meaning_embedding)
        logits = self.language_modeling_head(normalized_thought_embedding) # (B,T,Cvocab_size)

        if solution_tokens is None:
            loss = None
        else:
            batch_size, token_count, channel_size = logits.shape
            logits = logits.view(batch_size*token_count, channel_size)
            solution_tokens = solution_tokens.view(batch_size*token_count)
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

def save_checkpoint(model, loss, checkpoint_dir="checkpoints", base_name=model_file_name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    existing = [f for f in os.listdir(checkpoint_dir) if f.startswith(base_name) and f.endswith(".pt")]
    max_index = 0
    for fname in existing:
        try:
            idx = int(fname.split('_')[-2])
            max_index = max(max_index, idx)
        except Exception:
            continue
    new_index = max_index + 1
    loss_int = int(loss * 10000)
    checkpoint_name = f"{base_name}_{new_index}_loss{loss_int}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print("Checkpoint sauvegardé :", checkpoint_path)

def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint chargé depuis :", checkpoint_path)

# load_checkpoint(initialized_model, './checkpoints/gptone_2.pt')
def detokenizeTokens(generated_tokens):
    return detokenize(generated_tokens[0].tolist())



def generate_text(max_new_token_number):
    starting_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(starting_context, max_new_token_number=max_new_token_number)
    generated_text = detokenize(generated_tokens[0].tolist())
    return generated_text

def perform_long_evaluation(step, best_val_loss, no_improvement_count, max_no_improvement):
    print(f"Evaluating losses at step {step}...")
    losses = calculate_mean_losses()
    print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print(f"Current min_loss: {best_val_loss:.4f}")
    
    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= max_no_improvement:
            print(f"Validation loss did not improve for {max_no_improvement} consecutive evaluations. Stopping training.")
            save_checkpoint(initialized_model, losses['val'])
            return True, best_val_loss, no_improvement_count
    return False, best_val_loss, no_improvement_count

starting_context = torch.tensor([tokenize("En 1998, la coupe du monde a été gagnée par")]).to(device)
def generate_and_print_text(max_new_token_number, tokens_per_print=1, starting_context=starting_context):
    print(detokenizeTokens(starting_context), end='', flush=True)
    generated_tokens = starting_context
    for _ in range(max_new_token_number // tokens_per_print):
        
        generated_tokens = model.generate(generated_tokens, tokens_per_print)
        generated_text = detokenizeTokens(generated_tokens)[-1]
        print(generated_text, end='', flush=True)
# create a PyTorch optimizer
def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    starting_timer = time.time()
    print('THE MODEL HAS STARTED TRAINING')
    print(datetime.now())
    
    global best_val_loss
    best_val_loss = float('inf')
    best_short_eval_loss = float('inf')
    no_improvement_count = 0
    max_no_improvement = 6
    short_no_improvement_count = 0
    max_short_no_improvement = 10
    
    for step in range(maximum_training_steps):
        if step % evaluation_interval == 0 or step == maximum_training_steps - 1:
            stop_training, best_val_loss, no_improvement_count = perform_long_evaluation(step, best_val_loss, no_improvement_count, max_no_improvement)
            if stop_training:
                break
        
        if step % short_eval_interval == 0:
            print(f"Performing short evaluation at step {step}...")
            short_losses = calculate_short_mean_losses()
            print(f"step {step}: short train loss {short_losses['train']:.4f}, short val loss {short_losses['val']:.4f}")
            print(f"Current min_short_loss: {best_short_eval_loss:.4f}")
            
            if short_losses['val'] < best_short_eval_loss:
                best_short_eval_loss = short_losses['val']
                short_no_improvement_count = 0
            else:
                short_no_improvement_count += 1
                if short_no_improvement_count >= max_short_no_improvement:
                    stop_training, best_val_loss, no_improvement_count = perform_long_evaluation(step, best_val_loss, no_improvement_count, max_no_improvement)
                    if stop_training:
                        break
        
        if step % checkpoint_interval == 0 or step == maximum_training_steps - 1:
            print(f"Saving checkpoint at step {step}...")
            save_checkpoint(initialized_model, best_val_loss)
        
        if step % generate_interval == 0 or step == maximum_training_steps - 1:
            print(f"Generating text at step {step}...")
            generate_and_print_text(max_new_token_number_preview, tokens_per_print=1, starting_context="Elon Musk est ")
        
        if step % time_estimation_interval == 0 or step == maximum_training_steps - 1:
            print(f"Estimating remaining time at step {step}...")
            current_time = time.time()
            current_training_duration = current_time - starting_timer
            minutes_by_step = current_training_duration / (step + 1) / 60
            remaining_steps = maximum_training_steps - step
            remaining_minutes = remaining_steps * minutes_by_step
            predicted_end_time = datetime.now() + timedelta(minutes=remaining_minutes)
            
            print("="*50)
            print(f"Step: {step}/{maximum_training_steps}")
            print(f"Elapsed Time: {current_training_duration/60:.2f} minutes")
            print(f"Remaining Time: {remaining_minutes:.2f} minutes")
            print(f"Predicted End Time: {predicted_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*50)

        random_input_tokens, solution_tokenS = get_batch('train')
    
        logits, loss = model(random_input_tokens, solution_tokenS)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print('Training has finished :)')
    print(datetime.now())

#load_checkpoint(initialized_model, './checkpoints/gpt_wiki_bigram_one_12_loss17012')
train()





generate_and_print_text(max_new_token_number, tokens_per_print=1)