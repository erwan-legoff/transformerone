# --- Extract sub-batch ---
from datetime import datetime, timedelta
import time
import torch
import torch.nn.functional as F
import random

from models.ModelHandler import save_checkpoint

# --- Batching ---
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

# --- Long eval ---
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
import language_tool_python
tool = language_tool_python.LanguageTool("fr-FR")
# --- RL helpers ---
def safe_check(txt):
    try:
        return tool.check(txt)
    except Exception:
        return []
def reward_from_text(text: str) -> float:
    if "." in text:
        truncated = ".".join(text.split(".")[:-1])
    else:
        truncated = " ".join(text.split(" ")[:-1])

    KEEP_RULES = {"FR_SPELLING_RULE", "ACCORD_SUJET_VERBE"}
    matches = safe_check(truncated)
    err_count = sum(1 for m in matches if m.ruleId in KEEP_RULES)

    return -err_count

@torch.no_grad()
def sample_tokens(model, context, gen_len, temperature):
    model.eval()
    B = context.size(0)
    generated = []
    x = context.clone()

    max_ctx = model.position_embedding_table.num_embeddings  # == context_length

    for _ in range(gen_len):
        x_ctx = x[:, -max_ctx:]           # <-- rogne ici
        logits, _ = model(x_ctx, None)
        next_logits = logits[:, -1, :]
        if temperature != 1.0:
            next_logits = next_logits / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        generated.append(next_tok)
        x = torch.cat([x, next_tok], dim=1)

    return torch.cat(generated, dim=1) if generated else torch.empty(B, 0, dtype=context.dtype, device=context.device)

def recompute_logprob_sums(model, prompts, continuations):
    B, Tgen = continuations.size(0), continuations.size(1)
    if Tgen == 0:
        return torch.zeros(B, device=prompts.device)

    max_ctx = model.position_embedding_table.num_embeddings
    x = torch.cat([prompts, continuations[:, :-1]], dim=1)
    x = x[:, -max_ctx:]

    target = continuations.reshape(B, -1)
    model.eval()
    logits, _ = model(x, None)
    logits_gen = logits[:, -Tgen:, :]
    logprobs = F.log_softmax(logits_gen, dim=-1)
    lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return lp.sum(dim=1)


def apply_rl_step(model, prompts, detokenize_func, int_to_string, hyperparams, baseline):
    """Exécute un pas RL: génération, reward, logprobs, perte RL."""
    gen_len = int(hyperparams['rl_gen_len'])
    temperature = float(hyperparams['rl_temperature'])
    rl_weight = float(hyperparams['rl_weight'])
    beta = 0.9

    with torch.no_grad():
        continuations = sample_tokens(model, prompts, gen_len=gen_len, temperature=temperature)

    logprob_sums = recompute_logprob_sums(model, prompts, continuations)

    rewards = []
    for i in range(prompts.size(0)):
        full = torch.cat([prompts[i], continuations[i]], dim=0).tolist()
        text = detokenize_func(full, int_to_string)
        rewards.append(reward_from_text(text))
    rewards = torch.tensor(rewards, dtype=torch.float32, device=prompts.device)

    mean_r = rewards.mean().item()
    baseline = beta * baseline + (1.0 - beta) * mean_r
    advantages = rewards - baseline
    advantages = advantages.detach()

    rl_loss = -(advantages * logprob_sums).mean()
    return rl_weight * rl_loss, baseline, mean_r

# --- Training loop ---
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
    baseline = 0.0

    # hyperparams RL défaut
    hyperparams.setdefault('use_rl', True)
    hyperparams.setdefault('rl_weight', 0.01)
    hyperparams.setdefault('rl_interval', 400)
    hyperparams.setdefault('rl_min_steps', 200)
    hyperparams.setdefault('rl_gen_len', 50)
    hyperparams.setdefault('rl_temperature', 1)
    
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
            starting_context = torch.tensor(tokenize_func("Il est ", string_to_int), dtype=torch.long, device=device).unsqueeze(0)
            generate_and_print_text_func(model, context_length, detokenize_func, int_to_string, max_new_token_number_preview, 1, starting_context)
        
        if step % time_estimation_interval == 0 or step == maximum_training_steps - 1:
            estimate_time(maximum_training_steps, starting_timer, step)
        
        random_input_tokens, solution_tokens = get_batch_func('train', training_data, evaluation_data, context_length, batch_size, device)
        logits, ce_loss = model(random_input_tokens, solution_tokens)
        total_loss = ce_loss

        if hyperparams.get('use_rl', True) and step % hyperparams['rl_interval'] == 0 and step >= hyperparams['rl_min_steps']:
            rl_loss, baseline, mean_r = apply_rl_step(model, random_input_tokens, detokenize_func, int_to_string, hyperparams, baseline)
            total_loss = ce_loss + rl_loss
            print(f"[RL] step {step} | reward_mean={mean_r:.3f} baseline={baseline:.3f} ce={ce_loss.item():.4f} rl={rl_loss.item():.4f}")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
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
