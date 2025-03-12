import torch.nn as nn
class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dimension_count, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension_count, 4 * embedding_dimension_count),
            nn.ReLU(),
            nn.Linear(4 * embedding_dimension_count, embedding_dimension_count),
            nn.Dropout(dropout),
        )
    
    def forward(self, input_tokens):
        return self.network(input_tokens)