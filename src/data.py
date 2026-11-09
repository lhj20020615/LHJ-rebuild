from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ERP_FEATURES: tuple[str, ...] = (
    "N400_peak",
    "N400_trough",
    "LPC_peak",
    "LPC_trough",
)
REQUIRED_COLUMNS = {"Sentence", "Score", *ERP_FEATURES}

# Each entry lists the ERP columns retained by that ablation condition.
ABLATIONS: dict[str, tuple[str, ...]] = {
    "full": ERP_FEATURES,
    "text_only": (),
    "wo_n400": ("LPC_peak", "LPC_trough"),
    "wo_lpc": ("N400_peak", "N400_trough"),
    "wo_peak": ("N400_trough", "LPC_trough"),
    "wo_trough": ("N400_peak", "LPC_peak"),
    "n400_peak_only": ("N400_peak",),
    "n400_trough_only": ("N400_trough",),
    "lpc_peak_only": ("LPC_peak",),
    "lpc_trough_only": ("LPC_trough",),
}


@dataclass(frozen=True)
class Sample:
    text: str
    erp: torch.Tensor
    label: int


def validate_frame(frame: pd.DataFrame, source: Path | str) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{source} contains no rows")


def load_samples(path: Path, ablation: str = "full") -> list[Sample]:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation {ablation!r}; choose from {sorted(ABLATIONS)}")
    frame = pd.read_excel(path)
    validate_frame(frame, path)
    retained = set(ABLATIONS[ablation])
    samples: list[Sample] = []
    for row_number, row in frame.iterrows():
        score = int(row["Score"])
        if score not in (1, 2, 3):
            raise ValueError(f"{path}: Score must be 1, 2, or 3; row {row_number} has {score}")
        erp = np.array(
            [float(row[name]) if name in retained else 0.0 for name in ERP_FEATURES],
            dtype=np.float32,
        )
        samples.append(Sample(str(row["Sentence"]), torch.from_numpy(erp), score - 1))
    return samples


def make_loader(
    samples: Sequence[Sample],
    tokenizer,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    max_length: int = 128,
) -> DataLoader:
    def collate(batch: Iterable[Sample]):
        batch = list(batch)
        texts = [item.text for item in batch]
        erp = torch.stack([item.erp for item in batch]).to(device)
        labels = torch.tensor([item.label for item in batch], dtype=torch.long, device=device)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        return encoded, erp, labels

    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

