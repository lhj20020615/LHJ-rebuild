from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer

from src.data import ABLATIONS, load_samples, make_loader
from src.model import LateFusionClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metrics(labels: list[int], logits: list[np.ndarray]) -> dict[str, float]:
    predicted = np.argmax(np.asarray(logits), axis=1)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision_macro": precision_score(labels, predicted, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predicted, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predicted, average="macro", zero_division=0),
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool, grad_accum: int):
    model.train(train)
    labels: list[int] = []
    logits: list[np.ndarray] = []
    total_loss = 0.0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, (encoded, erp, target) in enumerate(loader, start=1):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(**encoded, erp_feats=erp)
            loss = criterion(output, target)
        if train:
            (loss / grad_accum).backward()
            if step % grad_accum == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_loss += loss.detach().item() * target.numel()
        labels.extend(target.detach().cpu().tolist())
        logits.extend(output.detach().float().cpu().numpy())
    result = metrics(labels, logits)
    result["loss"] = total_loss / max(len(labels), 1)
    return result


def train_one_condition(args, ablation: str) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir / ablation
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_samples(args.data_dir / args.train_file, ablation)
    valid = load_samples(args.data_dir / args.val_file, ablation)
    test = load_samples(args.data_dir / args.test_file, ablation) if args.test_file else None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LateFusionClassifier(args.model_dir, use_lora=not args.no_lora).to(device)
    train_loader = make_loader(train, tokenizer, args.batch_size, device, True)
    valid_loader = make_loader(valid, tokenizer, args.batch_size, device, False)
    test_loader = make_loader(test, tokenizer, args.batch_size, device, False) if test else None
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_score = -1.0
    best_state = None
    rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True, args.grad_accum)
        with torch.no_grad():
            valid_metrics = run_epoch(model, valid_loader, criterion, optimizer, device, False, args.grad_accum)
            test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, False, args.grad_accum) if test_loader else {}
        row = {"ablation": ablation, "epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in valid_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        rows.append(row)
        if valid_metrics["accuracy"] > best_score:
            best_score = valid_metrics["accuracy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(f"[{ablation}] epoch={epoch} val_acc={valid_metrics['accuracy']:.4f} val_f1={valid_metrics['f1_macro']:.4f}", flush=True)

    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ERP-enhanced metaphor novelty classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-file", default="train_set.xlsx")
    parser.add_argument("--val-file", default="val_set.xlsx")
    parser.add_argument("--test-file", default="test_set.xlsx")
    parser.add_argument("--ablation", choices=["all", *ABLATIONS], default="full")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--no-lora", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    conditions = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    for condition in conditions:
        train_one_condition(args, condition)


if __name__ == "__main__":
    main()

