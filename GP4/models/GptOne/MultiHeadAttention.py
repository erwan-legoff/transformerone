import torch.nn as nn
import torch
from models.GptOne import AttentionHead
class MultiHeadAttention(nn.Module):
    def __init__(self, head_count, head_size, embedding_dimension_count, context_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, embedding_dimension_count, context_length, dropout) for _ in range(head_count)])
        self.projection = nn.Linear(embedding_dimension_count, embedding_dimension_count)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tokens):
        self_attended_tokens = torch.cat([head(input_tokens) for head in self.heads], dim=-1)
        projection = self.projection(self_attended_tokens)
        projection = self.dropout(projection)
        return projection