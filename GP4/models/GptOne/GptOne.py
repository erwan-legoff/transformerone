import torch.nn as nn
import torch
from torch.nn import functional as F

from models.GptOne import AttentionThinkingBlock
class GptOne(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension_count, context_length, dropout, head_count, layer_count, device):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dimension_count)
        self.position_embedding_table = nn.Embedding(context_length, embedding_dimension_count)
        self.attention_thinking_blocks = nn.Sequential(
            *[AttentionThinkingBlock(embedding_dimension_count, head_count, context_length, dropout) for _ in range(layer_count)]
        )
        self.final_layer_normalization = nn.LayerNorm(embedding_dimension_count)
        self.language_modeling_head = nn.Linear(embedding_dimension_count, vocabulary_size)

    def forward(self, input_tokens, solution_tokens=None):
        batch_size, token_count = input_tokens.shape
        token_embeddings = self.token_embedding_table(input_tokens)
        position_embeddings = self.position_embedding_table(torch.arange(token_count, device=self.device))
        spatial_meaning_embedding = token_embeddings + position_embeddings
        spatial_meaning_embedding = self.attention_thinking_blocks(spatial_meaning_embedding)
        normalized_thought_embedding = self.final_layer_normalization(spatial_meaning_embedding)
        logits = self.language_modeling_head(normalized_thought_embedding)
        if solution_tokens is None:
            loss = None
        else:
            batch_size, token_count, channel_size = logits.shape
            logits = logits.view(batch_size * token_count, channel_size)
            solution_tokens = solution_tokens.view(batch_size * token_count)
            loss = F.cross_entropy(logits, solution_tokens)
        return logits, loss
