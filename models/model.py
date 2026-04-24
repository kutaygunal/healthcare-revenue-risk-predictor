"""
PyTorch multi-task model:
- Structured tabular encoder (MLP)
- Clinical text encoder (Embedding → BiGRU → Attention)
- Two binary classification heads: denial + missed revenue
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_output, mask=None):
        # rnn_output: (batch, seq_len, hidden_dim)
        scores = self.attn(rnn_output).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), rnn_output).squeeze(1)  # (batch, hidden_dim)
        return context, weights

class RevenueRiskNet(nn.Module):
    def __init__(self, struct_dim, vocab_size, embed_dim=64, rnn_hidden=64,
                 rnn_layers=1, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.rnn_hidden = rnn_hidden

        # ---- Text branch ----
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim, rnn_hidden, num_layers=rnn_layers,
            batch_first=True, bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0
        )
        self.text_pool = AttentionPool(rnn_hidden * 2)
        self.text_proj = nn.Linear(rnn_hidden * 2, 64)

        # ---- Structured branch ----
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Combined ----
        self.combined = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Output heads ----
        self.denial_head = nn.Linear(64, 1)
        self.missed_head = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.zeros_(param)

    def forward(self, structured, text_indices):
        # structured: (batch, struct_dim)
        # text_indices: (batch, seq_len)
        batch_size = structured.size(0)

        # Text encoding
        mask = (text_indices != 0).float()  # padding mask
        emb = self.embedding(text_indices)  # (batch, seq_len, embed_dim)
        emb = self.text_dropout(emb)
        gru_out, _ = self.gru(emb)  # (batch, seq_len, hidden*2)
        text_context, text_attn = self.text_pool(gru_out, mask)  # (batch, hidden*2), (batch, seq_len)
        text_vec = F.relu(self.text_proj(text_context))  # (batch, 64)

        # Structured encoding
        struct_vec = self.struct_mlp(structured)  # (batch, 64)

        # Fuse
        fused = self.combined(torch.cat([struct_vec, text_vec], dim=1))  # (batch, 64)

        # Heads
        denial_logit = self.denial_head(fused)
        missed_logit = self.missed_head(fused)
        return denial_logit, missed_logit, text_attn

def bce_with_logits(y_hat, y):
    return F.binary_cross_entropy_with_logits(y_hat, y)

def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    """Optional focal loss for class imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()
