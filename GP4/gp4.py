import time
import torch

from FileUtils import load_data, write_first_2000_chars_to_file
from Stats import write_first_1000_tokens_to_file
from models.ModelHandler import generate_and_print_text, generate_print_and_save_text, load_checkpoint, save_checkpoint
from models.Trainer import calculate_mean_losses, calculate_short_mean_losses, get_batch, train
from tokenizer_one import create_vocabularies_V2, create_vocabularies_GPT4_like, detokenize, load_tokenizer, prepare_tokenized_data, tokenize
from models.GptOne.GptOne import GptOne


# --- Programme principal ---
if __name__ == '__main__':
    # Définition des hyperparamètres
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    final_testing_sentence = "2+2="
    tokenization_iterations = 1000
    batch_size = 64 
    context_length = 250
    maximum_training_steps = 12000
    learning_rate = 2e-3
    head_count = 6
    layer_count = 3
    dropout = 0.10
    embedding_dimension_count = 360 
    evaluation_interval = 800
    eval_iteration_count = 30
    short_eval_interval = 400
    short_eval_iters = 5
    max_new_token_number = 20000
    max_new_token_number_preview = 100
    model_file_name = "gpt_wiki_bpe_one"
    generate_interval = 1600
    checkpoint_interval = 10000
    time_estimation_interval = 200
    should_train = True
    should_load = False
    model_to_load = "checkpoints/gpt_wiki_bigram_two_heads6_layers3_emb360_ctx250_drop0.1_12_loss27604.pt"
    use_tokenizer = False
    tokenizer_to_load ="tokenizers/tokenizer_iter1000_skip100_2025-03-14_22h.json"
    # Chargement des données
    training_text, eval_text = load_data('./wiki.train.tokens', './wiki.test.tokens')
    
    if use_tokenizer:
        string_to_int, int_to_string = load_tokenizer(tokenizer_to_load)
        vocabulary_size = len(string_to_int)
    else:
        vocabulary_size, string_to_int, int_to_string, tokenizer_path = create_vocabularies_GPT4_like(training_text, tokenization_iterations=tokenization_iterations,max_char_skip=100)
    
    print(f"Taille string_to_int chargée : {len(string_to_int)}")
    print(f"Taille int_to_string chargée : {len(int_to_string)}")

    import json
    with open("tokenizers/tokenizer_iter1000_skip100_2025-03-14_22h.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    missing_keys = [v for v in string_to_int.values() if v not in int_to_string]
    # print(f"Tokens manquants dans int_to_string: {missing_keys}")

    print(f"Nombre total de tokens dans in to string: {len(int_to_string)}")
    
    print("vocab size")
    print(vocabulary_size)

    # Préparation des tenseurs de données
    tokenized_training_data, tokenized_evaluation_data = prepare_tokenized_data(training_text, eval_text, tokenize, string_to_int)

    
    print("training set size chars :")
    char_count = len(training_text)
    print(char_count)
    token_count = len(tokenized_training_data)
    print("training set size tokenized :")
    print(token_count)
    print("compression ratio:")
    print(char_count/token_count)
    time.sleep(3)
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
    if(should_load and use_tokenizer):
        model = load_checkpoint(model=model,checkpoint_path=model_to_load,device=device)
    # Entraînement
    if(should_train):
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
    starting_context = torch.tensor(tokenize(final_testing_sentence, string_to_int), dtype=torch.long, device=device).unsqueeze(0)
    generate_print_and_save_text(model, context_length, detokenize, int_to_string, max_new_token_number, 1, starting_context, 'generated_text.txt')
