def count_bigram_occurences(training_text):
    return count_n_gram_occurences(training_text, 2)

def count_n_gram_occurences(training_data, gram_size):
    bigram_occurences = {}
    for i in range(len(training_data) - gram_size + 1):
        sub = training_data[i : i + gram_size]
        
        # Si training_data est une liste, sub = [x, y, ...] => non hashable
        # Donc on le convertit en tuple
        if isinstance(training_data, list):
            sub = tuple(sub)
        
        bigram_occurences[sub] = bigram_occurences.get(sub, 0) + 1
    return bigram_occurences
import os
import random
import unicodedata

from GP4.FileUtils import save_str_to_file
def count_n_gram_occurences_optimized(training_data, gram_size, max_char_skip = 10):
    bigram_occurences = {}
    c = 0
    while c < len(training_data) - gram_size + 1:
        sub = training_data[c : c + gram_size]
        
        # Si training_data est une liste, sub = [x, y, ...] => non hashable
        # Donc on le convertit en tuple
        if isinstance(training_data, list):
            sub = tuple(sub)
        
        bigram_occurences[sub] = bigram_occurences.get(sub, 0) + 1
        current_max_c = (len(training_data) - gram_size)
        remaining_chars = current_max_c - c
        current_max_skip = min(max_char_skip,remaining_chars)
        # We want to skip randomly to echantillonized our data
        next_offset = random.randint(1,1+current_max_skip)
        c += next_offset
    return bigram_occurences
import re
import re
import random

def count_n_gram_occurences_optimized_no_ponctuation(training_data, gram_size, int_to_string, max_char_skip=10):
    n_gram_occurrences = {}
    c = 0
    # Compilation de la regex pour éviter de la recompiler à chaque itération
    pattern = re.compile(r"[.,;:!?'\"()«»—\-]")
    data_len = len(training_data)
    limit = data_len - gram_size + 1

    while c < limit:
        sub = training_data[c : c + gram_size]
        # Si training_data est une liste, convertissons le sous-ensemble en tuple (hashable)
        if isinstance(training_data, list):
            sub = tuple(sub)

        # Vérifie si au moins un caractère de 'sub' correspond à une ponctuation.
        # Utilisation d'une expression génératrice et pattern.search pour un test court-circuité.
        if not any(pattern.search(int_to_string.get(item, "")) for item in sub):
            n_gram_occurrences[sub] = n_gram_occurrences.get(sub, 0) + 1

        # Calcul du nombre de caractères restants pour définir le saut aléatoire
        remaining = data_len - (c + gram_size)
        current_max_skip = min(max_char_skip, remaining)
        next_offset = random.randint(1, 1 + current_max_skip)
        c += next_offset

    return n_gram_occurrences


def count_char_occurences(training_text):
    char_occurences = {}
    for c in training_text:
        char_occurences[c] = char_occurences.get(c, 0) + 1
    return char_occurences

def get_top_n_grams(n_gram_occurences, max_size):
    """
    Sorts a dictionary of n-gram occurrences and keeps the `max_size` most frequent ones.
    
    :param n_gram_occurences: Dictionary containing n-grams and their occurrences.
    :param max_size: Maximum number of elements to keep.
    :return: Sorted dictionary with the most frequent n-grams.
    """
    return dict(sorted(n_gram_occurences.items(), key=lambda item: item[1], reverse=True)[:max_size])

def save_sorted_n_gram_occurences(n_gram_occurences, file_name, directory="vocabulary"):
    """
    Sorts n-gram occurrences by descending frequency and saves them to a file.

    :param n_gram_occurences: Dictionary containing n-grams and their occurrences.
    :param file_name: Output file name.
    :param directory: Directory to store the files.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    sorted_n_grams = sorted(n_gram_occurences.items(), key=lambda item: item[1], reverse=True)

    # Save sorted occurrences
    with open(os.path.join(directory, file_name), 'w', encoding='utf-8') as f:
        for n_gram, count in sorted_n_grams:
            f.write(f"{n_gram}\t{count}\n")
    
    return sorted_n_grams  # Return the sorted list for later use

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