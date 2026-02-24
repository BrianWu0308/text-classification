import torch
import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class BiLSTMConfig:
    # general
    output_root: str = "outputs/bilstm"
    seed: int = 42
    val_size: float = 0.1
    labels: list[str] = field(
        default_factory=lambda: ["World", "Sports", "Business", "Sci/Tech"]
    )
    
    # training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3

    # data
    max_len: int = 200
    min_freq: int = 2
    max_vocab: int = 50000

    # model
    emb_dim: int = 128
    hidden_dim: int = 128
    num_classes: int = 4
    pad_idx: int = 0
    unk_idx: int = 1


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        emb_dim=128,
        hidden_dim=128,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        """
        input_ids: [B, T]
        """
        # [B, T, E]
        x = self.embedding(input_ids)

        # out: [B, T, 2H]
        out, (h_n, c_n) = self.lstm(x)

        h_forward = h_n[-2]
        h_backward = h_n[-1]

        h = torch.cat((h_forward, h_backward), dim=1)

        logits = self.fc(h)

        return logits