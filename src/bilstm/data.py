import re
import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

def tokenize(text: str):
    """
    Tokenizer:
    - lowercase
    - collapse whitespace
    - split by space
    """
    text = str(text).lower()

    # collapse all whitespace to single space, and strip leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text.split()


def build_vocab(texts, min_freq: int = 2, max_size: int = 50000):
    """
    Build a word-level vocab from texts.
    Returns (stoi, itos)
    """
    # count token frequencies
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    # build vocab with special tokens
    itos = ["<pad>", "<unk>"]

    # add tokens by frequency(descending order), respecting min_freq and max_size
    for w, c in counter.most_common():
        if c < min_freq:
            break
        itos.append(w)
        if len(itos) >= max_size:
            break

    stoi = {w: i for i, w in enumerate(itos)}

    return stoi, itos


def encode(text: str, stoi: dict, max_len: int):
    """
    Encode a text to a list of token ids, truncated to max_len.
    """
    unk = stoi["<unk>"]

    # convert tokens to ids, using unk for OOV tokens
    ids = [stoi.get(tok, unk) for tok in tokenize(text)]

    return ids[:max_len]


def collate_fn(batch, pad_id: int = 0):
    """
    Collate function for (x, y) batches using pad_sequence.
    batch: list of (Tensor[x_len], Tensor[])
    """
    # separate xs and ys, and pad xs to create x_pad of shape (batch_size, max_seq_len)
    xs, ys = zip(*batch)

    x_pad = pad_sequence(xs, batch_first=True, padding_value=pad_id)

    # stack labels into a single tensor of shape (batch_size,)
    y = torch.stack(ys)

    return x_pad, y


class TextDataset(Dataset):
    """
    texts: iterable of raw texts
    labels: iterable of labels (1~4 for AG News) or None
    """
    def __init__(self, texts, labels, stoi: dict, max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.stoi, self.max_len)
        x = torch.tensor(x, dtype=torch.long)

        # if labels is None, we assume it's inference and only return x
        if self.labels is None:
            return x
        
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return x, y
