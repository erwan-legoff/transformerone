from datetime import datetime, timedelta
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from FileUtils import *  # On suppose que ce module reste inchangé
import unicodedata

# --- Fonctions de normalisation Unicode ---
def to_decomposed_unicode(text: str) -> str:
    return unicodedata.normalize('NFD', text)

def to_unified_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

# --- Fonctions de tokenisation/détokénisation ---
def tokenize(text, string_to_int):
    text = to_decomposed_unicode(text)
    unknown_token = 0 
    int_tokens = []
    c = 0
    while c < len(text):
        if c < len(text) - 1 and (text[c] + text[c+1]) in string_to_int:
            int_tokens.append(string_to_int[text[c] + text[c+1]])
            c += 2
        else:
            token = string_to_int.get(text[c], unknown_token)
            int_tokens.append(token)
            c += 1
    return int_tokens

def detokenize(int_tokens, int_to_string):
    return to_unified_unicode(''.join([int_to_string[i] for i in int_tokens]))

# --- Chargement des données ---
def load_data(training_file, evaluation_file):
    with open(training_file, 'r', encoding='utf-8') as f:
        training_text = f.read()
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        eval_text = f.read()
    return training_text, eval_text

# --- Comptage des occurences ---
def count_bigram_occurences(training_text):
    bigram_occurences = {}
    for c in range(len(training_text)-1):
        bigram = training_text[c] + training_text[c+1]
        bigram_occurences[bigram] = bigram_occurences.get(bigram, 0) + 1
    return bigram_occurences

def count_n_gram_occurences(training_text, gram_size):
    bigram_occurences = {}
    for c in range(len(training_text)-gram_size):
        n_gram = training_text[c:c+gram_size]

        bigram_occurences[n_gram] = bigram_occurences.get(n_gram, 0) + 1
    return bigram_occurences

def count_char_occurences(training_text):
    char_occurences = {}
    for c in training_text:
        char_occurences[c] = char_occurences.get(c, 0) + 1
    return char_occurences

def get_top_n_grams(n_gram_occurences, max_size):
    """
    Trie un dictionnaire d'occurrences de n-grams et garde les `max_size` plus fréquents.
    
    :param n_gram_occurences: Dictionnaire contenant les n-grams et leurs occurrences.
    :param max_size: Nombre maximal d'éléments à conserver.
    :return: Dictionnaire trié avec les n-grams les plus fréquents.
    """
    return dict(sorted(n_gram_occurences.items(), key=lambda item: item[1], reverse=True)[:max_size])


import os

def save_sorted_n_gram_occurences(n_gram_occurences, file_name, directory="vocabulary"):
    """
    Trie les occurrences de n-grams par fréquence décroissante et les enregistre dans un fichier.

    :param n_gram_occurences: Dictionnaire contenant les n-grams et leurs occurrences.
    :param file_name: Nom du fichier de sortie.
    :param directory: Dossier où stocker les fichiers.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    sorted_n_grams = sorted(n_gram_occurences.items(), key=lambda item: item[1], reverse=True)

    # Sauvegarde des occurrences triées
    with open(os.path.join(directory, file_name), 'w', encoding='utf-8') as f:
        for n_gram, count in sorted_n_grams:
            f.write(f"{n_gram}\t{count}\n")
    
    return sorted_n_grams  # Renvoie la liste triée pour usage ultérieur


# --- Création des vocabulaires ---
def create_vocabularies(training_text, 
                        max_bigrams=538, max_chars=117, 
                        max_trigrams=1000, max_quadgrams=1500, 
                        max_pentagrams=2000, max_sextegrams=2500, 
                        directory="vocabulary"):

    # 1. **Calcul des occurrences**
    sextegram_occurences = count_n_gram_occurences(training_text, gram_size=6)
    pentagram_occurences = count_n_gram_occurences(training_text, gram_size=5)
    quadgram_occurences = count_n_gram_occurences(training_text, gram_size=4)
    trigram_occurences = count_n_gram_occurences(training_text, gram_size=3)
    bigram_occurences = count_n_gram_occurences(training_text, gram_size=2)
    char_occurences = count_char_occurences(training_text)

    # 2. **Trier et enregistrer les occurrences dans "vocabulary/"**
    sorted_sextegrams = save_sorted_n_gram_occurences(sextegram_occurences, 'sextegram_occurences.txt', directory)
    sorted_pentagrams = save_sorted_n_gram_occurences(pentagram_occurences, 'pentagram_occurences.txt', directory)
    sorted_quadgrams = save_sorted_n_gram_occurences(quadgram_occurences, 'quadgram_occurences.txt', directory)
    sorted_trigrams = save_sorted_n_gram_occurences(trigram_occurences, 'trigram_occurences.txt', directory)
    sorted_bigrams = save_sorted_n_gram_occurences(bigram_occurences, 'bigram_occurences.txt', directory)
    sorted_chars = save_sorted_n_gram_occurences(char_occurences, 'char_occurences.txt', directory)

    # 3. **Sélection des N plus fréquents**
    top_sextegrams = dict(sorted_sextegrams[:max_sextegrams])
    top_pentagrams = dict(sorted_pentagrams[:max_pentagrams])
    top_quadgrams = dict(sorted_quadgrams[:max_quadgrams])
    top_trigrams = dict(sorted_trigrams[:max_trigrams])
    top_bigrams = dict(sorted_bigrams[:max_bigrams])
    top_chars = dict(sorted_chars[:max_chars])

    # 4. **Sauvegarde des N-grams tronqués**
    save_dict_to_file(top_sextegrams, os.path.join(directory, 'top_sextegrams.txt'))
    save_dict_to_file(top_pentagrams, os.path.join(directory, 'top_pentagrams.txt'))
    save_dict_to_file(top_quadgrams, os.path.join(directory, 'top_quadgrams.txt'))
    save_dict_to_file(top_trigrams, os.path.join(directory, 'top_trigrams.txt'))
    save_dict_to_file(top_bigrams, os.path.join(directory, 'top_bigrams.txt'))
    save_dict_to_file(top_chars, os.path.join(directory, 'top_chars.txt'))

    # 5. **Création du vocabulaire combiné**
    full_vocabulary = (
        list(top_sextegrams.keys()) + 
        list(top_pentagrams.keys()) + 
        list(top_quadgrams.keys()) + 
        list(top_trigrams.keys()) + 
        list(top_bigrams.keys()) + 
        list(top_chars.keys())
    )

    # 6. **Sauvegarde du vocabulaire complet**
    save_list_to_file(full_vocabulary, os.path.join(directory, 'full_vocabulary.txt'))

    # 7. **Création des mappings pour la tokenisation**
    vocabulary_size = len(full_vocabulary)
    string_to_int = {string: idx for idx, string in enumerate(full_vocabulary)}
    int_to_string = {idx: string for idx, string in enumerate(full_vocabulary)}

    return vocabulary_size, string_to_int


# --- Préparation des tenseurs de données ---
def prepare_tokenized_data(training_text, eval_text, tokenize_func, string_to_int, max_train_tokens=None):
    tokenized_training_data = torch.tensor(tokenize_func(training_text, string_to_int), dtype=torch.long)
    tokenized_evaluation_data = torch.tensor(tokenize_func(eval_text, string_to_int), dtype=torch.long)
    
    # Si un nombre maximum de tokens est défini, on tronque le dataset d'entraînement
    if max_train_tokens is not None:
        tokenized_training_data = tokenized_training_data[:max_train_tokens]
    
    return tokenized_training_data, tokenized_evaluation_data

# --- Extraction de sous-batch ---
def get_batch(data_partition_name, training_data, evaluation_data, context_length, batch_size, device):
    data = training_data if data_partition_name == 'train' else evaluation_data
    max_offset = len(data) - context_length - 1
    random_start_offsets = torch.randint(max_offset, (batch_size,))
    input_tokens = torch.stack([data[offset:offset+context_length] for offset in random_start_offsets])
    solution_tokens = torch.stack([data[offset+1:offset+1+context_length] for offset in random_start_offsets])
    return input_tokens.to(device), solution_tokens.to(device)

# --- Fonctions d'évaluation des pertes ---
@torch.no_grad()
def calculate_mean_losses(model, training_data, evaluation_data, context_length, batch_size, eval_iteration_count, device, get_batch_func):
    mean_losses = {}
    model.eval()
    for data_partition_name in ['train', 'val']:
        losses = torch.zeros(eval_iteration_count)
        for eval_iteration_number in range(eval_iteration_count):
            inputs, solutions = get_batch_func(data_partition_name, training_data, evaluation_data, context_length, batch_size, device)
            _, loss = model(inputs, solutions)
            losses[eval_iteration_number] = loss.item()
        mean_losses[data_partition_name] = losses.mean()
    model.train()
    return mean_losses

@torch.no_grad()
def calculate_short_mean_losses(model, training_data, evaluation_data, context_length, batch_size, short_eval_iters, device, get_batch_func):
    mean_losses = {}
    model.eval()
    for data_partition_name in ['train', 'val']:
        losses = torch.zeros(short_eval_iters)
        for eval_iteration_number in range(short_eval_iters):
            inputs, solutions = get_batch_func(data_partition_name, training_data, evaluation_data, context_length, batch_size, device)
            _, loss = model(inputs, solutions)
            losses[eval_iteration_number] = loss.item()
        mean_losses[data_partition_name] = losses.mean()
    model.train()
    return mean_losses

# --- Classes du modèle ---
class AttentionHead(nn.Module):
    def __init__(self, head_size, embedding_dimension_count, context_length, dropout):
        super().__init__()
        self.keys = nn.Linear(embedding_dimension_count, head_size, bias=False)
        self.queries = nn.Linear(embedding_dimension_count, head_size, bias=False)
        self.values = nn.Linear(embedding_dimension_count, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropouts = nn.Dropout(dropout)
    
    def forward(self, current_token_contexts):
        batch_size, token_count, channel_count = current_token_contexts.shape
        keys = self.keys(current_token_contexts)
        queries = self.queries(current_token_contexts)
        attention_scores = queries @ keys.transpose(-2, -1) * keys.shape[-1] ** -0.5
        causal_attention_scores = attention_scores.masked_fill(self.tril[:token_count, :token_count] == 0, float('-inf'))
        probabilistic_causal_attention = F.softmax(causal_attention_scores, dim=-1)
        probabilistic_causal_attention = self.dropouts(probabilistic_causal_attention)
        values = self.values(current_token_contexts)
        shared_informations = probabilistic_causal_attention @ values
        return shared_informations

class MultiHeadAttention(nn.Module):
    def __init__(self, head_count, head_size, embedding_dimension_count, context_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, embedding_dimension_count, context_length, dropout) for _ in range(head_count)])
        self.projection = nn.Linear(embedding_dimension_count, embedding_dimension_count)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tokens):
        self_attended_tokens = torch.cat([head(input_tokens) for head in self.heads], dim=-1)
        projection = self.projection(self_attended_tokens)
        projection = self.dropout(projection)
        return projection

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dimension_count, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension_count, 4 * embedding_dimension_count),
            nn.ReLU(),
            nn.Linear(4 * embedding_dimension_count, embedding_dimension_count),
            nn.Dropout(dropout),
        )
    
    def forward(self, input_tokens):
        return self.network(input_tokens)

class AttentionThinkingBlock(nn.Module):
    def __init__(self, embedding_dimension_count, head_count, context_length, dropout):
        super().__init__()
        head_size = embedding_dimension_count // head_count
        self.attention_network = MultiHeadAttention(head_count, head_size, embedding_dimension_count, context_length, dropout)
        self.feed_forward_network = FeedForwardNetwork(embedding_dimension_count, dropout)
        self.attention_layer_normalization = nn.LayerNorm(embedding_dimension_count)
        self.feed_forward_layer_normalization = nn.LayerNorm(embedding_dimension_count)

    def forward(self, input_tokens):
        normalized_input_tokens = self.attention_layer_normalization(input_tokens)
        attended_tokens = input_tokens + self.attention_network(normalized_input_tokens)
        normalized_attended_tokens = self.feed_forward_layer_normalization(attended_tokens)
        thought_attended_tokens = attended_tokens + self.feed_forward_network(normalized_attended_tokens)
        return thought_attended_tokens

class GptOne(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension_count, context_length, dropout, head_count, layer_count, device):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimension_count)
        self.position_embedding_table = nn.Embedding(context_length, embedding_dimension_count)
        self.attention_thinking_blocks = nn.Sequential(
            *[AttentionThinkingBlock(embedding_dimension_count, head_count, context_length, dropout) for _ in range(layer_count)]
        )
        self.final_layer_normalization = nn.LayerNorm(embedding_dimension_count)
        self.language_modeling_head = nn.Linear(embedding_dimension_count, vocabulary_size)

    def forward(self, input_tokens, solution_tokens=None):
        batch_size, token_count = input_tokens.shape
        token_embeddings = self.token_embedding_table(input_tokens)
        position_embeddings = self.position_embedding_table(torch.arange(token_count, device=self.device))
        spatial_meaning_embedding = token_embeddings + position_embeddings
        spatial_meaning_embedding = self.attention_thinking_blocks(spatial_meaning_embedding)
        normalized_thought_embedding = self.final_layer_normalization(spatial_meaning_embedding)
        logits = self.language_modeling_head(normalized_thought_embedding)
        if solution_tokens is None:
            loss = None
        else:
            batch_size, token_count, channel_size = logits.shape
            logits = logits.view(batch_size * token_count, channel_size)
            solution_tokens = solution_tokens.view(batch_size * token_count)
            loss = F.cross_entropy(logits, solution_tokens)
        return logits, loss

    def generate(self, input_tokens, max_new_token_number, context_length):
        for _ in range(max_new_token_number):
            context_tokens = input_tokens[:, -context_length:]
            logits, _ = self(context_tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, next_token), dim=1)
        return input_tokens

# --- Sauvegarde et chargement des checkpoints ---
def save_checkpoint(model, loss, hyperparams, checkpoint_dir="checkpoints", base_name="gpt_wiki_bigram_two"):
    """
    Sauvegarde le checkpoint avec dans le nom les hyperparamètres
    qui définissent intrinsèquement le modèle.
    
    hyperparams: dictionnaire contenant par exemple :
        {
            'head_count': 12,
            'layer_count': 2,
            'embedding_dimension_count': 576,
            'context_length': 364,
            'dropout': 0.10
        }
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Construire une chaîne représentant les hyperparamètres
    hp_str = f"heads{hyperparams['head_count']}_layers{hyperparams['layer_count']}_emb{hyperparams['embedding_dimension_count']}_ctx{hyperparams['context_length']}_drop{hyperparams['dropout']}"
    
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
    checkpoint_name = f"{base_name}_{hp_str}_{new_index}_loss{loss_int}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print("Checkpoint sauvegardé :", checkpoint_path)

def load_checkpoint(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint chargé depuis :", checkpoint_path)

# --- Fonctions de génération de texte ---
def generate_text(model, detokenize_func, int_to_string, max_new_token_number, context_length, device):
    starting_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(starting_context, max_new_token_number, context_length)
    generated_text = detokenize_func(generated_tokens[0].tolist(), int_to_string)
    return generated_text

def generate_and_print_text(model, context_length, detokenize_func, int_to_string, max_new_token_number, tokens_per_print, starting_context):
    # Affiche le contexte initial
    print(detokenize_func(starting_context[0].tolist(), int_to_string), end='', flush=True)
    generated_tokens = starting_context
    for _ in range(max_new_token_number // tokens_per_print):
        # Génère tokens_per_print tokens de plus
        generated_tokens = model.generate(generated_tokens, tokens_per_print, context_length)
        # Extraction des tokens générés lors de cette itération
        new_token_indices = generated_tokens[0].tolist()[-tokens_per_print:]
        new_text = ''.join([int_to_string[t] for t in new_token_indices])
        print(new_text, end='', flush=True)


def generate_and_print_text(model, context_length, detokenize_func, int_to_string,
                            max_new_token_number, tokens_per_print, starting_context, return_text=False):
    """
    Génère et affiche le texte par incréments de tokens.
    Si return_text est True, renvoie le texte généré en plus de l'affichage.
    """
    # Initialisation
    text_generated = detokenize_func(starting_context[0].tolist(), int_to_string)
    print(text_generated, end='', flush=True)
    generated_tokens = starting_context
    
    steps = max_new_token_number // tokens_per_print
    for _ in range(steps):
        generated_tokens = model.generate(generated_tokens, tokens_per_print, context_length)
        full_text = detokenize_func(generated_tokens[0].tolist(), int_to_string)
        # Extraction sur la base des tokens générés et non des caractères
        new_token_indices = generated_tokens[0].tolist()[-tokens_per_print:]
        new_text = ''.join([int_to_string[t] for t in new_token_indices])
        print(new_text, end='', flush=True)
        text_generated += new_text
        
    if return_text:
        return text_generated

def generate_print_and_save_text(model, context_length, detokenize_func, int_to_string,
                                 max_new_token_number, tokens_per_print, starting_context, file_name):
    """
    Appelle generate_and_print_text pour générer et afficher le texte,
    puis sauvegarde l'intégralité dans un fichier.
    """
    final_text = generate_and_print_text(model, context_length, detokenize_func, int_to_string,
                                           max_new_token_number, tokens_per_print, starting_context,
                                           return_text=True)
    time.sleep(10)
    save_str_to_file(final_text, file_name)
    print(f"\nTexte intégral sauvegardé dans '{file_name}'.")


def inspect_characters(text):
    for idx, c in enumerate(text):
        code_point = ord(c)
        name = unicodedata.name(c, "UNKNOWN")
        print(f"{idx:3d} | {repr(c)} | U+{code_point:04X} | {name}")
# Write the first 1000 re-detokenized tokens to a text file
def write_first_1000_tokens_to_file(tokenized_data, file_name, detokenize_func, int_to_string):
    first_1000 = tokenized_data.tolist()[:1000]
    detokenized_text = detokenize_func(first_1000, int_to_string)
    save_str_to_file(detokenized_text, file_name)
# --- Boucle d'entraînement ---
def perform_long_evaluation(step, best_val_loss, no_improvement_count, max_no_improvement,
                            model, training_data, evaluation_data, context_length, batch_size,
                            eval_iteration_count, device, get_batch_func, hyperparams):
    print(f"Evaluating losses at step {step}...")
    losses = calculate_mean_losses(model, training_data, evaluation_data, context_length, batch_size, eval_iteration_count, device, get_batch_func)
    print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print(f"Current min_loss: {best_val_loss:.4f}")
    if losses['val'] < best_val_loss:
        best_val_loss = losses['val']
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= max_no_improvement:
            print(f"Validation loss did not improve for {max_no_improvement} consecutive evaluations. Stopping training.")
            save_checkpoint(model, losses['val'], hyperparams)
            return True, best_val_loss, no_improvement_count
    return False, best_val_loss, no_improvement_count

def train(model, training_data, evaluation_data, context_length, batch_size, maximum_training_steps,
          evaluation_interval, short_eval_interval, checkpoint_interval, generate_interval,
          time_estimation_interval, eval_iteration_count, short_eval_iters, learning_rate, device,
          max_new_token_number_preview, generate_and_print_text_func, get_batch_func,
          calculate_mean_losses_func, calculate_short_mean_losses_func, save_checkpoint_func,
          tokenize_func, string_to_int, detokenize_func, int_to_string, hyperparams):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print('THE MODEL HAS STARTED TRAINING')
    
    best_val_loss = float('inf')
    best_short_eval_loss = float('inf')
    no_improvement_count = 0
    max_no_improvement = 500
    short_no_improvement_count = 0
    max_short_no_improvement = 500
    starting_timer = time.time()
    
    for step in range(maximum_training_steps):
        if step % evaluation_interval == 0 or step == maximum_training_steps - 1:
            stop_training, best_val_loss, no_improvement_count = perform_long_evaluation(
                step, best_val_loss, no_improvement_count, max_no_improvement,
                model, training_data, evaluation_data, context_length, batch_size,
                eval_iteration_count, device, get_batch_func, hyperparams)
            if stop_training:
                break
        
        if step % short_eval_interval == 0:
            print(f"Performing short evaluation at step {step}...")
            short_losses = calculate_short_mean_losses_func(
                model, training_data, evaluation_data, context_length, batch_size, short_eval_iters, device, get_batch_func)
            print(f"step {step}: short train loss {short_losses['train']:.4f}, short val loss {short_losses['val']:.4f}")
            print(f"Current min_short_loss: {best_short_eval_loss:.4f}")
            if short_losses['val'] < best_short_eval_loss:
                best_short_eval_loss = short_losses['val']
                short_no_improvement_count = 0
            else:
                short_no_improvement_count += 1
                if short_no_improvement_count >= max_short_no_improvement:
                    stop_training, best_val_loss, no_improvement_count = perform_long_evaluation(
                        step, best_val_loss, no_improvement_count, max_no_improvement,
                        model, training_data, evaluation_data, context_length, batch_size,
                        eval_iteration_count, device, get_batch_func, hyperparams)
                    if stop_training:
                        break
        
        if step % checkpoint_interval == 0 or step == maximum_training_steps - 1:
            print(f"Saving checkpoint at step {step}...")
            save_checkpoint_func(model, best_val_loss, hyperparams)
        
        if step % generate_interval == 0 or step == maximum_training_steps - 1:
            print(f"Generating text at step {step}...")
            starting_context = torch.tensor(tokenize_func("John Lennon est ", string_to_int), dtype=torch.long, device=device).unsqueeze(0)
            generate_and_print_text_func(model, context_length, detokenize_func, int_to_string, max_new_token_number_preview, 1, starting_context)
        
        if step % time_estimation_interval == 0 or step == maximum_training_steps - 1:
            print(f"Estimating remaining time at step {step}...")
            current_time = time.time()
            current_training_duration = current_time - starting_timer
            minutes_by_step = current_training_duration / (step + 1) / 60
            remaining_steps = maximum_training_steps - step
            remaining_minutes = remaining_steps * minutes_by_step
            predicted_end_time = datetime.now() + timedelta(minutes=remaining_minutes)
            print("=" * 50)
            print(f"Step: {step}/{maximum_training_steps}")
            print(f"Elapsed Time: {current_training_duration / 60:.2f} minutes")
            print(f"Remaining Time: {remaining_minutes:.2f} minutes")
            print(f"Predicted End Time: {predicted_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)
        
        random_input_tokens, solution_tokens = get_batch_func('train', training_data, evaluation_data, context_length, batch_size, device)
        logits, loss = model(random_input_tokens, solution_tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print('Training has finished :)')
    print(datetime.now())

# --- Programme principal ---
if __name__ == '__main__':
    # Définition des hyperparamètres
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64 
    context_length = 250
    maximum_training_steps = 100000
    learning_rate = 2e-3
    head_count = 6
    layer_count = 2
    dropout = 0.10
    embedding_dimension_count = 360 
    evaluation_interval = 800
    eval_iteration_count = 30
    short_eval_interval = 400
    short_eval_iters = 5
    max_new_token_number = 100
    max_new_token_number_preview = 100
    model_file_name = "gpt_wiki_sectegram_one_mini"
    generate_interval = 800
    checkpoint_interval = 5000
    time_estimation_interval = 200

    # Chargement des données
    training_text, eval_text = load_data('../wiki.train.tokens', '../wiki.test.tokens')

    # Création des vocabulaires et mappings
    vocabulary_size, string_to_int, int_to_string = create_vocabularies(training_text)

    # Préparation des tenseurs de données
    tokenized_training_data, tokenized_evaluation_data = prepare_tokenized_data(training_text, eval_text, tokenize, string_to_int)
    print("training set size :")
    print(len(tokenized_training_data))
    print("tokens by iteration :")
    print(len(tokenized_training_data) / (maximum_training_steps * batch_size))
    # Sauvegarde d'extraits
    write_first_1000_tokens_to_file(tokenized_training_data, 'first_1000_tokens.txt', detokenize, int_to_string)
    write_first_2000_chars_to_file(training_text, 'first_2000_chars.txt')

    # Création du modèle
    model = GptOne(vocabulary_size, embedding_dimension_count, context_length, dropout, head_count, layer_count, device)
    model = model.to(device)
    
    # Définir les hyperparamètres pour la sauvegarde
    hyperparams = {
        'head_count': head_count,
        'layer_count': layer_count,
        'embedding_dimension_count': embedding_dimension_count,
        'context_length': context_length,
        'dropout': dropout
    }

    # Entraînement
    train(model,
          tokenized_training_data,
          tokenized_evaluation_data,
          context_length,
          batch_size,
          maximum_training_steps,
          evaluation_interval,
          short_eval_interval,
          checkpoint_interval,
          generate_interval,
          time_estimation_interval,
          eval_iteration_count,
          short_eval_iters,
          learning_rate,
          device,
          max_new_token_number_preview,
          generate_and_print_text,
          get_batch,
          calculate_mean_losses,
          calculate_short_mean_losses,
          save_checkpoint,
          tokenize,
          string_to_int,
          detokenize,
          int_to_string,
          hyperparams)

    # Génération finale et sauvegarde
    starting_context = torch.tensor(tokenize("En 1998, la coupe du monde a été gagnée par", string_to_int), dtype=torch.long, device=device).unsqueeze(0)
    generate_print_and_save_text(model, context_length, detokenize, int_to_string, max_new_token_number, 1, starting_context, 'generated_text.txt')
