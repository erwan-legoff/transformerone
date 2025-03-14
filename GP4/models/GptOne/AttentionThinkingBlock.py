import torch.nn as nn
from models.GptOne import MultiHeadAttention
from models.GptOne import FeedForwardNetwork
import torch
from torch.nn import functional as F
class AttentionThinkingBlock(nn.Module):
    def __init__(self, embedding_dimension_count, head_count, context_length, dropout):
        super().__init__()
        head_size = embedding_dimension_count // head_count
        self.attention_network = MultiHeadAttention(head_count, head_size, embedding_dimension_count, context_length, dropout)
        self.feed_forward_network = FeedForwardNetwork(embedding_dimension_count, dropout)
        self.attention_layer_normalization = nn.LayerNorm(embedding_dimension_count)
        self.feed_forward_layer_normalization = nn.LayerNorm(embedding_dimension_count)

    def forward(self, input_tokens):
        normalized_input_tokens = self.attention_layer_normalization(input_tokens)
        attended_tokens = input_tokens + self.attention_network(normalized_input_tokens)
        normalized_attended_tokens = self.feed_forward_layer_normalization(attended_tokens)
        thought_attended_tokens = attended_tokens + self.feed_forward_network(normalized_attended_tokens)
        return thought_attended_tokens


    def generate(self, input_tokens, max_new_token_number, context_length):
        for _ in range(max_new_token_number):
            context_tokens = input_tokens[:, -context_length:]
            logits, _ = self(context_tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, next_token), dim=1)
        return input_tokens