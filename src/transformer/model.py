from dataclasses import dataclass, field
from transformers import AutoModelForSequenceClassification


@dataclass
class TransformerConfig:
    # general
    output_root: str = "outputs/transformer"
    seed: int = 42
    val_size: float = 0.1
    labels: list[str] = field(
        default_factory=lambda: ["World", "Sports", "Business", "Sci/Tech"]
    )

    # training
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # model
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 4

    # tokenizer
    max_length: int = 256


def build_model(cfg: TransformerConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
    )
    return model