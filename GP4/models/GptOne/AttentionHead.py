import torch
import torch.nn as nn
from torch.nn import functional as F
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