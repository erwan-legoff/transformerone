from dataclasses import dataclass
import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import os
# Simple launch :
# python gpt2_originally_named.py
# DDp launch :
# torchrun --nproc_per_node=2 gpt2_originally_named.py
#Helper function for HellaSwag evaluation
# takes tokens, mask, and logits, returns index of the completion with lowest loss
def get_most_likely_completion(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    prediction_normalized = avg_loss.argmin().item()
    return prediction_normalized

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "gpt2_log.txt")
# open for writing to clear it
with open(log_file, "w") as f:
    f.write("")














class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #c_attention
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # c_projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANO_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd 

    def forward(self, input_tokens):
        B,T,C = input_tokens.size()

        qkv = self.c_attn(input_tokens)
        query, key, value = qkv.split(self.n_embd, dim=2)
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        # attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        # attention = attention.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # attention = F.softmax(attention, dim=-1)
        output_tokens = F.scaled_dot_product_attention(
        query, key, value, is_causal=True
      )
        output_tokens = output_tokens.transpose(1,2).contiguous().view(B,T,C)
        output_tokens = self.c_proj(output_tokens)
        return output_tokens

# FeedForwardNetwork
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # non_linearity
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANO_SCALE_INIT = 1

    def forward(self, input_tokens):
        input_tokens = self.c_fc(input_tokens)
        input_tokens = self.gelu(input_tokens)
        input_tokens = self.c_proj(input_tokens)
        return input_tokens
# AttentionBlock
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # layer_normalization_1
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # attention
        self.attn = CausalSelfAttention(config)
        # layer_normalization_2
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # feed_forward_network
        self.mlp = MLP(config)

    def forward(self, input_tokens):
        normalized_tokens = self.ln_1(input_tokens)
        attention_tokens = input_tokens + self.attn(normalized_tokens)

        normalized_tokens = self.ln_2(attention_tokens)
        mlp_tokens = attention_tokens + self.mlp(normalized_tokens)
        return mlp_tokens

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # token_embedding_weights
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position_embedding_weights
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # attention_blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # layer_normalization
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # type: ignore
        self.apply(self.init_weights)

    def init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANO_SCALE_INIT'):
            std *= (2*self.config.n_layer)** -0.5
        if(isinstance(module, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if(module.bias is not None):
               torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)   
    def forward(self, token_indices, targets = None):
        BATCH_SIZE, TIME_SIZE = token_indices.size()
        assert TIME_SIZE <= self.config.block_size, f"Cannot forward sequence of length {TIME_SIZE}, block size is {self.config.block_size}"
        pos = torch.arange(0, TIME_SIZE, dtype=torch.long, device=token_indices.device)
        position_embeddings = self.transformer.wpe(pos) # type: ignore
        token_embeddings = self.transformer.wte(token_indices) # type: ignore
        x = token_embeddings + position_embeddings
        # On se propage dans le transformer
        for block in self.transformer.h: # type: ignore
            x = block(x)

        # On se propage dans le dernier layer de normalization
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)

        loss = None 
        if(targets is not None):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss





    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # hugging face model
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        #create optimizer groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'bias' not in n]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2 or 'bias' in n]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        print(f"creating optimizer with {len(decay_params)} decay and {len(no_decay_params)} no_decay parameters")
        fused_avalaible = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avalaible and device == 'cuda'
        print(f"using fused AdamW: {use_fused} (fused available: {fused_avalaible}, device: {device})")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import os
import time

TOTAL_BATCH_SIZE = 524288  # 512K tokens per batch
B = 8
T = 1024
TRAIN_LOADER_BATCH_SIZE = 12
EVAL_INTERVAL = 100
HELLOSWAG_EVAL_INTERVAL = 100
GENERATE_INTERVAL = 50
CHECKPOINT_INTERVAL = 100
TEXT_PROMPT = "Java is a"
NUM_GENERATION_SEQUENCES = 4
MAX_GENERATION_LENGTH = 50
TOP_K = 50
GRAD_CLIP = 1.0

import tiktoken
import numpy as np


def initialize_distributed_mode():
    global ddp, rank, gpu_count, local_rank, master_process, device

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP only works with cuda"
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        gpu_count = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        print(f"DDP mode on. rank: {rank}, world_size: {gpu_count}, device: {device}")
        master_process = rank == 0
    else:
        rank = 0
        local_rank = 0
        gpu_count = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"single process mode, device: {device}")


def configure_gradient_accumulation():
    global grad_accumulation_steps

    assert TOTAL_BATCH_SIZE % (B * T * gpu_count) == 0, "total_batch_size must be a multiple of B*T*gpu_count"
    grad_accumulation_steps = TOTAL_BATCH_SIZE // (B * T * gpu_count)
    if master_process:
        print(f"grad accumulation steps: {grad_accumulation_steps} ")
        print(f"calculating {TOTAL_BATCH_SIZE} tokens per batch with B={B}, T={T}")
    print(f"grad_accumulation_steps: {grad_accumulation_steps}")

def load_tokens(filename):
    np_tokens = np.load(filename)
    pytorch_tokens = torch.tensor(np_tokens, dtype=torch.long)
    return pytorch_tokens

class DataLoaderLite:
    def __init__(self, B, T, process_rank, gpu_count, split: str = 'train') -> None:
        self.B = B 
        self.T = T 
        
        
        self.gpu_count = gpu_count
        self.process_rank = process_rank

        assert split in {'train', 'val'}

        # locate data directory relative to this file so the script can be run from repo root
        data_root = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")
        if not os.path.isdir(data_root):
            # fallback to a simple relative name if that path doesn't exist
            data_root = "edu_fineweb10B"

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        assert len(shards) > 0, "no data shards found"

        # store full paths to shards
        self.shards = [os.path.join(data_root, s) for s in shards]

        if master_process:
            print(f"found {len(self.shards)} data shards in {data_root}")
        self.reset()

        
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # start position depends on process rank so different GPUs read different offsets
        self.current_position = self.B * self.T * self.process_rank
        

    def next_batch(self):
        BATCH_SIZE, TIME_SIZE = self.B, self.T
        buffer = self.tokens[self.current_position : self.current_position+BATCH_SIZE*TIME_SIZE+1] # type: ignore
        inputs = buffer[:-1].view(BATCH_SIZE, TIME_SIZE)
        solutions = buffer[1:].view(BATCH_SIZE, TIME_SIZE)
        self.current_position += BATCH_SIZE * TIME_SIZE * self.gpu_count
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (BATCH_SIZE * TIME_SIZE * self.gpu_count + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            # reset position for this process/gpu
            self.current_position = self.B * self.T * self.process_rank

        return inputs, solutions


from torch.nn.parallel import DistributedDataParallel as DDP


def create_dataloaders():
    global train_loader, eval_loader

    train_loader = DataLoaderLite(B=TRAIN_LOADER_BATCH_SIZE, T=T, process_rank=rank, gpu_count=gpu_count, split='train')
    eval_loader = DataLoaderLite(B=TRAIN_LOADER_BATCH_SIZE, T=T, process_rank=rank, gpu_count=gpu_count, split='val')


def build_model():
    model = GPT(GPTConfig(vocab_size=50304))  # Random init
    model.to(device)
    model = torch.compile(model, fullgraph=True)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    print("Ca plante pas youhouu")
    return model


def get_raw_model(model):
    return model.module if ddp else model


times = []
toks = []
encoder = tiktoken.get_encoding("gpt2")
max_learning_rate = 6e-4 * 3
min_learning_rate = max_learning_rate / 30
warmup_steps = 200
max_steps = 19073


def get_learning_rate(step):
    if step < warmup_steps:
        return max_learning_rate * step / warmup_steps
    if step > max_steps:
        return min_learning_rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def generate_text(model):
    model.eval()
    tokens = encoder.encode(TEXT_PROMPT)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(NUM_GENERATION_SEQUENCES, 1)
    token_sentence = tokens.to(device)

    sample_random_generator = torch.Generator(device=device)
    sample_random_generator.manual_seed(42 + rank)
    while token_sentence.size(1) < MAX_GENERATION_LENGTH:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float16):
                logits, _ = model(token_sentence)
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            top_k_probabilities, top_k_indices = torch.topk(probabilities, TOP_K, dim=-1)
            random_number = torch.multinomial(top_k_probabilities, 1, generator=sample_random_generator)
            next_token_id = torch.gather(top_k_indices, -1, random_number)
            token_sentence = torch.cat((token_sentence, next_token_id), dim=1)

    for generation_step in range(NUM_GENERATION_SEQUENCES):
        tokens = token_sentence[generation_step, :MAX_GENERATION_LENGTH].tolist()
        decoded = encoder.decode(tokens)
        print("----- -----")
        print(decoded)
        print()


def run_validation(model):
    model.eval()
    eval_loader.reset()
    with torch.no_grad():
        eval_loss_accumulated = 0.0
        eval_loss_steps = 20
        for _ in range(eval_loss_steps):
            inputs, solutions = eval_loader.next_batch()
            inputs, solutions = inputs.to(device), solutions.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float16):
                _, loss = model(inputs, solutions)
            loss = loss / eval_loss_steps
            eval_loss_accumulated += loss.detach()
    if ddp:
        dist.all_reduce(eval_loss_accumulated, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"VALIDATION LOSS: {eval_loss_accumulated:.1f}")




def maybe_generate_text(model):
    if not master_process:
        return
    generate_text(model)


def perform_training_iteration(model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss_accumulated = 0.0

    for micro_step in range(grad_accumulation_steps):
        inputs, solutions = train_loader.next_batch()
        inputs, solutions = inputs.to(device), solutions.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float16):
            _, loss = model(inputs, solutions)
        loss = loss / grad_accumulation_steps
        loss_accumulated += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accumulated, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    return loss_accumulated, norm


def update_learning_rate(optimizer, step):
    lr = get_learning_rate(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def synchronize_device():
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.synchronize()


def log_step(step, loss_accumulated, norm, lr, dt, tokens_per_second):
    if not master_process:
        return
    print(f"step {step} | loss: {loss_accumulated:.1f} | norm: {norm:.2f} | learning {lr:.4e} | time:{dt:.0f}ms | {tokens_per_second:.0f}tokens/s")
    step_left = max_steps - (step + 1)
    time_left = step_left * (sum(times) / len(times)) / 1000
    h = math.floor(time_left / 3600)
    m = math.floor((time_left - h * 3600) / 60)
    s = math.floor(time_left - h * 3600 - m * 60)
    
    date = time.localtime(time.time() + time_left)
    print(f"finish in... steps: {step_left} | time: {h}h {m}m {s}s | date: {date.tm_mday}/{date.tm_mon} {date.tm_hour}:{date.tm_min}")

    with open(log_file, "a") as f:
        f.write(f"{step},{loss_accumulated:.4f}\n")


def run_training_loop(model, optimizer):
    times.clear()
    toks.clear()
    for step in range(max_steps):
        print()
        is_last_step = step == (max_steps - 1)
        t0 = time.time()
        if step % EVAL_INTERVAL == 0 or is_last_step:
            run_validation(model)
        if step % (GENERATE_INTERVAL) == 0 or is_last_step:
            maybe_generate_text(model)
        if step % HELLOSWAG_EVAL_INTERVAL == 0 or is_last_step:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % gpu_count != rank:
                    continue
                # render example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_completion(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long,device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long,device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            accuracy_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy norm: {num_correct_norm}/{num_total}={accuracy_norm:.2f}")
                with open(log_file, "a") as f:
                    f.write(f"{step},{accuracy_norm:.4f}\n")


        loss_accumulated, norm = perform_training_iteration(model, optimizer)
        lr = update_learning_rate(optimizer, step)
        if step > 0 and step % CHECKPOINT_INTERVAL == 0 and master_process:
            checkpoint_path = os.path.join(log_dir, f"gpt2_checkpoint_step{step}.pt")
            print(f"saving checkpoint to {checkpoint_path}")
            raw_model = get_raw_model(model)
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'val_loss': loss_accumulated,
            }
            torch.save(checkpoint, checkpoint_path)
        optimizer.step()
        synchronize_device()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_second = (train_loader.B * train_loader.T * grad_accumulation_steps * gpu_count) / (t1 - t0)
        times.append(dt)
        toks.append(tokens_per_second)
        log_step(step, loss_accumulated, norm, lr, dt, tokens_per_second)


def finalize_training():
    print(f"\nMoyenne temps/step: {sum(times)/len(times):.2f} ms")
    print(f"Moyenne tokens/s:   {sum(toks)/len(toks):.2f}")
    if ddp:
        destroy_process_group()


def main():
    initialize_distributed_mode()
    print(f"using device {device}")
    configure_gradient_accumulation()
    create_dataloaders()
    torch.set_float32_matmul_precision('medium')
    model = build_model()
    raw_model = get_raw_model(model)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)  # type: ignore
    run_training_loop(model, optimizer)
    finalize_training()


if __name__ == "__main__":
    main()















