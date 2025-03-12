from datetime import datetime
import json
import os
import unicodedata
from FileUtils import *
import torch
from Stats import count_char_occurences, count_n_gram_occurences_optimized_no_ponctuation

def to_decomposed_unicode(text: str) -> str:
    return unicodedata.normalize('NFD', text)

def to_unified_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

def tokenize(text, string_to_int, max_gram_chars= 8):
    text = to_decomposed_unicode(text)
    unknown_token = 0 
    int_tokens = []
    c = 0
    while c < len(text):
        remaining_chars = len(text) - c
        current_char_size = min(max_gram_chars, remaining_chars) 
        not_found = True
        while not_found and current_char_size > 0:
            current_n_gram = text[c:c+current_char_size]
            if(current_n_gram in string_to_int):
                int_tokens.append(string_to_int[current_n_gram]) 
                not_found = False
            else:
                current_char_size-=1
            
        if not_found:
            token = string_to_int.get(text[c], unknown_token)
            int_tokens.append(token)
            c += 1
        else:
            c += current_char_size
    return int_tokens

def detokenize(int_tokens, int_to_string):
    return to_unified_unicode(''.join([int_to_string[i] for i in int_tokens]))

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    string_to_int = tokenizer_data["string_to_int"]
    int_to_string = tokenizer_data["int_to_string"]
    
    print(f"Tokenizer loaded: {tokenizer_path}")
    return string_to_int, int_to_string

def save_tokenizer(string_to_int, int_to_string, tokenization_iteration, max_char_skip, directory="tokenizers"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Construire une chaîne représentant les paramètres du tokenizer
    tp_str = f"iter{tokenization_iteration}_skip{max_char_skip}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh")
    tokenizer_name = f"tokenizer_{tp_str}_{timestamp}.json"
    tokenizer_path = os.path.join(directory, tokenizer_name)
    
    tokenizer_data = {
        "string_to_int": string_to_int,
        "int_to_string": int_to_string
    }
    
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)
    
    print(f"Tokenizer saved: {tokenizer_path}")
    return tokenizer_path

def create_vocabularies_V2(training_text, 
                        max_bigrams=538, 
                        max_trigrams=1000, max_quadgrams=1500, 
                        max_pentagrams=2173, max_sextegrams=7000,
                        max_septegrams = 7000, max_octograms=7000, 
                        directory="vocabulary_v2",
                        tokenization_iteration = 1000,
                        max_char_skip = 100):
    char_occurences = count_char_occurences(training_text)
    sorted_chars = sorted(char_occurences.items(), key=lambda item: item[1], reverse=True)
    top_chars = dict(sorted_chars)
    current_vocabulary = list(top_chars.keys())
    # Juste après avoir construit top_chars et current_vocabulary:
    all_chars_in_text = set(training_text)  # l'ensemble de tous les caractères distincts
    dict_chars = set(current_vocabulary)    # l'ensemble des chars que vous avez retenus

    missing_chars = all_chars_in_text - dict_chars
    if len(missing_chars) > 0:
        print("Caractères manquants (non couverts par le vocabulaire) :", missing_chars)
    else:
        print("Tous les caractères du texte sont couverts par le vocabulaire initial.")
        current_string_to_int = {string: idx for idx, string in enumerate(current_vocabulary)}
        current_int_to_string = {idx: string for idx, string in enumerate(current_vocabulary)}
    
    tokenized_compressed_text = tokenize(training_text,current_string_to_int,max_gram_chars=1)
    # 1. **Count char occurrences**
    for i in range(tokenization_iteration):
        print("bigram_occurences_start")
        bigram_occurences = count_n_gram_occurences_optimized_no_ponctuation(tokenized_compressed_text, gram_size=2, int_to_string=current_int_to_string,max_char_skip=max_char_skip,)
        print("bigram_occurences_end")

        sorted_bigrams = sorted(bigram_occurences.items(), key=lambda item: item[1], reverse=True)

        

        # 3. **Select the top N most frequent**
        if not sorted_bigrams:
            print("Aucun bigram trouvé, fin du processus.")
            break  # Stopper si plus de bigrams à fusionner
        
        current_top_bigram_ints = sorted_bigrams[0][0]
        current_top_bigram_strings = current_int_to_string.get(current_top_bigram_ints[0], "") + \
                                     current_int_to_string.get(current_top_bigram_ints[1], "")


        current_vocabulary.append(current_top_bigram_strings)

        
        new_token_id = len(current_string_to_int)
        current_string_to_int[current_top_bigram_strings] = new_token_id
        current_int_to_string[new_token_id] = current_top_bigram_strings
        print('merge_in_place_start')
        print(i)
        tokenized_compressed_text  = merge_in_place(tokenized_compressed_text, current_top_bigram_ints,new_token_id)
        print('merge_in_place_end')



    # 5. **Create the combined vocabulary**
    full_vocabulary = current_vocabulary
        
    # 6. **Save the full vocabulary**
    save_list_to_file(full_vocabulary, os.path.join(directory, 'full_vocabulary.txt'))

    # 7. **Create mappings for tokenization**
    vocabulary_size = len(full_vocabulary)
    string_to_int = {string: idx for idx, string in enumerate(full_vocabulary)}
    int_to_string = {idx: string for idx, string in enumerate(full_vocabulary)}
    tokenizer_path = save_tokenizer(string_to_int, int_to_string, tokenization_iteration, max_char_skip)
    return vocabulary_size, string_to_int, int_to_string, tokenizer_path

def merge_in_place(token_sequence, bigram, new_token):
    """
    Fusionne le bigram 'bigram' par 'new_token' dans 'token_sequence'.
    Renvoie une NOUVELLE liste de tokens après la fusion.
    
    :param token_sequence: liste d’entiers (IDs de tokens).
    :param bigram: tuple (token_id_1, token_id_2) à fusionner.
    :param new_token: entier représentant l'ID du nouveau token.
    :return: nouvelle liste de tokens où les occurrences de bigram sont remplacées par new_token.
    """
    
    i,j = 0,0
    token_count = len(token_sequence)
    merged_sequence = [None] * token_count
    b0, b1 = bigram
    while i < token_count:
        # Si on est sur l'avant-dernier token, on peut regarder la paire (i, i+1)
        if i < token_count - 1 and token_sequence[i] == b0 and token_sequence[i+1] == b1:
            # On remplace la paire par le nouveau token
            merged_sequence[j]=new_token
            j += 1

            i += 2
        else:
            # Sinon on recopie le token courant tel quel
            merged_sequence[j]=token_sequence[i]
            j += 1
            i += 1

    return merged_sequence[:j]

def prepare_tokenized_data(training_text, eval_text, tokenize_func, string_to_int, max_train_tokens=None):
    tokenized_training_data = torch.tensor(tokenize_func(training_text, string_to_int), dtype=torch.long)
    tokenized_evaluation_data = torch.tensor(tokenize_func(eval_text, string_to_int), dtype=torch.long)
    
    # If a maximum number of tokens is defined, truncate the training dataset
    if max_train_tokens is not None:
        tokenized_training_data = tokenized_training_data[:max_train_tokens]
    
    return tokenized_training_data, tokenized_evaluation_data