from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_size % config.head_count == 0

        self.c_attention = nn.Linear(config.embedding_size, 3*config.embedding_size)

        self.c_projection = nn.Linear(config.embedding_size, config.embedding_size)

        self.head_count = config.head_count
        self.embedding_size = config.embedding_size 
        ones = torch.ones(config.context_length, config.context.length)
        mask = torch.tril(ones).view(1,1, config.context_length, config.context_length)
        self.register_buffer("mask", mask)

    def forward(self, input_tokens):
        B,T,C = input_tokens.size()

        qkv = self.c_attention(input_tokens)
        query, key, value = qkv.split(self.embedding_size, dim=2)
        key = key.view(B, T, self.head_count, C // self.head_count).transpose(1,2)
        query = query.view(B, T, self.head_count, C // self.head_count).transpose(1,2)
        value = value.view(B, T, self.head_count, C // self.head_count).transpose(1,2)

        attention = (query @ query.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attention = attention.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        output_tokens = attention @ value
        output_tokens = output_tokens.transpose(1,2).contiguous().view(B,T,C)
        output_tokens = self.c_projection(output_tokens)
        return output_tokens

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embedding_size, 4 * config.embedding_size)
        self.non_linearity = nn.GELU()
        self.c_projection = nn.Linear(4*config.embedding_size, config.embedding_size)

    def forward(self, input_tokens):
        input_tokens = self.c_fc(input_tokens)
        input_tokens = self.non_linearity(input_tokens)
        input_tokens = self.c_projection(input_tokens)
        return input_tokens

class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_normalization_1 = nn.LayerNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)
        self.layer_normalization_2 = nn.LayerNorm(config.embedding_size)
        self.feed_forward_network = FeedForwardNetwork(config)

    def forward(self, input_tokens):
        input_tokens_normalized = self.layer_normalization_1(input_tokens)
        input_tokens = input_tokens + self.attention(input_tokens_normalized)

        input_tokens_normalized = self.layer_normalization_2(input_tokens)
        input_tokens = input_tokens + self.feed_forward_network(input_tokens_normalized)
        return input_tokens

@dataclass
class GPT2Config:
    context_length: int = 1024
    vocabulary_size: int = 50257
    layer_count: int = 12
    head_count: int = 12
    embedding_size: int = 768
    
class GPT2(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            token_embedding_weights = nn.Embedding(config.vocabulary_size, config.embedding_size),
            position_embedding_weights = nn.Embedding(config.context_length, config.embedding_size),
            attention_blocks = nn.ModuleList([AttentionBlock(config) for _ in range(config.layer_count)]),
            layer_normalization = nn.LayerNorm(config.embedding_size)
        ))
        self.lm_head = nn.Linear(config.embedding_size, config.vocabulary_size, bias=False)