# --- Extract sub-batch ---
from datetime import datetime
import time

import torch
from datetime import timedelta

from models.ModelHandler import save_checkpoint


def get_batch(data_partition_name, training_data, evaluation_data, context_length, batch_size, device):
    data = training_data if data_partition_name == 'train' else evaluation_data
    max_offset = len(data) - context_length - 1
    random_start_offsets = torch.randint(max_offset, (batch_size,))
    input_tokens = torch.stack([data[offset:offset+context_length] for offset in random_start_offsets])
    solution_tokens = torch.stack([data[offset+1:offset+1+context_length] for offset in random_start_offsets])
    return input_tokens.to(device), solution_tokens.to(device)

# --- Loss evaluation functions ---
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

# --- Model classes ---








# --- Checkpoint saving and loading ---




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
            estimate_time(maximum_training_steps, starting_timer, step)
        
        random_input_tokens, solution_tokens = get_batch_func('train', training_data, evaluation_data, context_length, batch_size, device)
        logits, loss = model(random_input_tokens, solution_tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print('Training has finished :)')
    print(datetime.now())

def estimate_time(maximum_training_steps, starting_timer, step):
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