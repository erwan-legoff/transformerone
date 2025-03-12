from Stats import *
from FileUtils import *
import os
def create_vocabularies(training_text, 
                        max_bigrams=538, max_chars=117, 
                        max_trigrams=1000, max_quadgrams=1500, 
                        max_pentagrams=2173, max_sextegrams=7000,
                        max_septegrams = 7000, max_octograms=7000, 
                        directory="vocabulary"):

    # 1. **Count occurrences**
    octogram_occurences = count_n_gram_occurences(training_text, gram_size=8)
    septegram_occurences = count_n_gram_occurences(training_text, gram_size=7)
    sextegram_occurences = count_n_gram_occurences(training_text, gram_size=6)
    pentagram_occurences = count_n_gram_occurences(training_text, gram_size=5)
    quadgram_occurences = count_n_gram_occurences(training_text, gram_size=4)
    trigram_occurences = count_n_gram_occurences(training_text, gram_size=3)
    bigram_occurences = count_n_gram_occurences(training_text, gram_size=2)
    char_occurences = count_char_occurences(training_text)

    # 2. **Sort and save occurrences in "vocabulary/"**
    sorted_octograms = save_sorted_n_gram_occurences(octogram_occurences, 'octogram_occurences.txt', directory)
    sorted_septegrams = save_sorted_n_gram_occurences(septegram_occurences, 'septegram_occurences.txt', directory)
    sorted_sextegrams = save_sorted_n_gram_occurences(sextegram_occurences, 'sextegram_occurences.txt', directory)
    sorted_pentagrams = save_sorted_n_gram_occurences(pentagram_occurences, 'pentagram_occurences.txt', directory)
    sorted_quadgrams = save_sorted_n_gram_occurences(quadgram_occurences, 'quadgram_occurences.txt', directory)
    sorted_trigrams = save_sorted_n_gram_occurences(trigram_occurences, 'trigram_occurences.txt', directory)
    sorted_bigrams = save_sorted_n_gram_occurences(bigram_occurences, 'bigram_occurences.txt', directory)
    sorted_chars = save_sorted_n_gram_occurences(char_occurences, 'char_occurences.txt', directory)

    # 3. **Select the top N most frequent**
    top_octograms = dict(sorted_octograms[:max_octograms])
    top_septegrams = dict(sorted_septegrams[:max_septegrams])
    top_sextegrams = dict(sorted_sextegrams[:max_sextegrams])
    top_pentagrams = dict(sorted_pentagrams[:max_pentagrams])
    top_quadgrams = dict(sorted_quadgrams[:max_quadgrams])
    top_trigrams = dict(sorted_trigrams[:max_trigrams])
    top_bigrams = dict(sorted_bigrams[:max_bigrams])
    top_chars = dict(sorted_chars[:max_chars])

    # 4. **Save the truncated N-grams**
    save_dict_to_file(top_octograms, os.path.join(directory, 'top_octograms.txt'))
    save_dict_to_file(top_septegrams, os.path.join(directory, 'top_septegrams.txt'))
    save_dict_to_file(top_sextegrams, os.path.join(directory, 'top_sextegrams.txt'))
    save_dict_to_file(top_pentagrams, os.path.join(directory, 'top_pentagrams.txt'))
    save_dict_to_file(top_quadgrams, os.path.join(directory, 'top_quadgrams.txt'))
    save_dict_to_file(top_trigrams, os.path.join(directory, 'top_trigrams.txt'))
    save_dict_to_file(top_bigrams, os.path.join(directory, 'top_bigrams.txt'))
    save_dict_to_file(top_chars, os.path.join(directory, 'top_chars.txt'))

    # 5. **Create the combined vocabulary**
    full_vocabulary = (
        list(top_octograms.keys()) +
        list(top_septegrams.keys()) + 
        list(top_sextegrams.keys()) + 
        list(top_pentagrams.keys()) + 
        list(top_quadgrams.keys()) + 
        list(top_trigrams.keys()) + 
        list(top_bigrams.keys()) + 
        list(top_chars.keys())
    )

    # 6. **Save the full vocabulary**
    save_list_to_file(full_vocabulary, os.path.join(directory, 'full_vocabulary.txt'))

    # 7. **Create mappings for tokenization**
    vocabulary_size = len(full_vocabulary)
    string_to_int = {string: idx for idx, string in enumerate(full_vocabulary)}
    int_to_string = {idx: string for idx, string in enumerate(full_vocabulary)}

    return vocabulary_size, string_to_int, int_to_string
