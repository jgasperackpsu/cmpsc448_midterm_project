"""
Phase 2 - Model definitions
===========================
Both classifiers share the same token-embedding configuration
(vocab_size x embed_dim, padding_idx = 0) so differences in results come from
the sequence encoder, not the input representation.

TextCNN  : embedding -> parallel 1D convs (several kernel sizes) -> ReLU
           -> global max-pool over time -> concat -> Dropout(0.5) -> Linear(K)
BiLSTM   : embedding -> bidirectional LSTM (packed, so padding is ignored)
           -> concat final forward & backward hidden states -> Dropout -> Linear(K)

K = number of LLM families in the data (derived from label_map.json).
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from config import PAD_IDX


class TokenEmbedding(nn.Module):
    """Shared embedding block used by both models."""

    def __init__(self, vocab_size, embed_dim=128, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x):                          # (B, L) -> (B, L, E)
        return self.dropout(self.embedding(x))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128, num_filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5, embed_dropout=0.0):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, embed_dim, embed_dropout)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.max_kernel = max(kernel_sizes)

    def forward(self, x, lengths=None):
        # pad very short batches so the widest kernel always fits
        if x.size(1) < self.max_kernel:
            x = nn.functional.pad(x, (0, self.max_kernel - x.size(1)), value=PAD_IDX)
        e = self.embed(x).transpose(1, 2)                       # (B, E, L)
        pooled = [torch.relu(conv(e)).amax(dim=2) for conv in self.convs]  # global max-pool
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128, hidden_dim=128,
                 num_layers=1, dropout=0.5, embed_dropout=0.0):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, embed_dim, embed_dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x, lengths):
        e = self.embed(x)
        packed = pack_padded_sequence(e, lengths.cpu().clamp(min=1), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * 2, B, H) -> last layer's forward (-2) and backward (-1)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(h))


def build_model(model_type, vocab_size, num_classes, **hp):
    """Factory used by the training / grid-search code."""
    embed_dim = hp.get("embed_dim", 128)
    if model_type == "cnn":
        return TextCNN(vocab_size, num_classes, embed_dim=embed_dim,
                       num_filters=hp.get("num_filters", 100),
                       kernel_sizes=tuple(hp.get("kernel_sizes", (3, 4, 5))),
                       dropout=hp.get("dropout", 0.5))
    if model_type == "rnn":
        return BiLSTMClassifier(vocab_size, num_classes, embed_dim=embed_dim,
                                hidden_dim=hp.get("hidden_dim", 128),
                                num_layers=hp.get("num_layers", 1),
                                dropout=hp.get("dropout", 0.5))
    raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
