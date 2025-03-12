import os
import torch
import time
from FileUtils import *

def save_checkpoint(model, loss, hyperparams, checkpoint_dir="checkpoints", base_name="gpt_wiki_bigram_two"):
    """
    Save the checkpoint with the hyperparameters in the name
    that intrinsically define the model.
    
    hyperparams: dictionary containing for example:
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