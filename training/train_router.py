"""Train a DistilBERT-based complexity classifier for prompt routing.

Fine-tunes distilbert-base-uncased for binary classification (simple vs complex).

Usage:
    python -m training.train_router --data training/data/router_dataset.csv --epochs 3
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def load_dataset(path: str) -> tuple[list[str], list[int]]:
    """Load prompts and labels from CSV."""
    prompts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["prompt"])
            labels.append(int(row["label"]))
    return prompts, labels


def train(
    data_path: str,
    output_dir: str = "models/router/distilbert-complexity",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    """Train the DistilBERT complexity classifier."""
    try:
        import torch
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, classification_report
    except ImportError as e:
        print(f"ML dependencies required: {e}")
        print("Install with: pip install torch transformers scikit-learn")
        return

    print("Loading dataset...")
    prompts, labels = load_dataset(data_path)
    print(f"Loaded {len(prompts)} samples (simple: {labels.count(0)}, complex: {labels.count(1)})")

    # Split: train / val / test
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        prompts, labels, test_size=(test_size + val_size), random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=test_size / (test_size + val_size),
        random_state=42, stratify=temp_labels
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Tokenize
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    # Create torch datasets
    class PromptDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = PromptDataset(train_encodings, train_labels)
    val_dataset = PromptDataset(val_encodings, val_labels)
    test_dataset = PromptDataset(test_encodings, test_labels)

    # Load model
    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="binary"),
        }

    # Train
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    print(classification_report(test_labels, test_preds, target_names=["simple", "complex"]))

    # Save model and tokenizer
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train the DistilBERT router classifier")
    parser.add_argument("--data", type=str, default="training/data/router_dataset.csv")
    parser.add_argument("--output", type=str, default="models/router/distilbert-complexity")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
