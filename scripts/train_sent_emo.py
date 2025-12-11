import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Setup paths and imports
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))  # so "bera" package can be imported

from bera.config import BERAConfig
from bera.model import BERA
from bera.tokenizer_utils import train_sentencepiece, load_sentencepiece
from bera.data_utils import BERADataset


def main():
    data_dir = BASE_DIR / "data"
    models_dir = BASE_DIR / "models"
    configs_dir = BASE_DIR / "configs"

    models_dir.mkdir(exist_ok=True)
    configs_dir.mkdir(exist_ok=True)

    dataset_path = data_dir / "dataset.csv"
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    text_col = "Review"
    sentiment_col = "Sentiment"
    emotion_col = "Emotion"

    # -----------------------------------------------------------------
    # Train or load SentencePiece tokenizer
    # -----------------------------------------------------------------
    sp_model_path = models_dir / "bera_spm.model"
    sp_vocab_path = models_dir / "bera_spm.vocab"

    if not sp_model_path.exists():
        print("Training SentencePiece tokenizer...")
        corpus_path = data_dir / "bera_text_corpus.txt"
        df[text_col].dropna().astype(str).to_csv(
            corpus_path, index=False, header=False
        )
        train_sentencepiece(
            input_file=str(corpus_path),
            model_prefix=str(models_dir / "bera_spm"),
            vocab_size=32000,
        )
    else:
        print("Using existing SentencePiece tokenizer.")

    sp = load_sentencepiece(str(sp_model_path))

    # -----------------------------------------------------------------
    # Label mappings
    # -----------------------------------------------------------------
    sentiment_labels = sorted(df[sentiment_col].dropna().unique().tolist())
    emotion_labels = sorted(df[emotion_col].dropna().unique().tolist())

    sentiment2id = {lbl: i for i, lbl in enumerate(sentiment_labels)}
    emotion2id = {lbl: i for i, lbl in enumerate(emotion_labels)}

    label_maps = {
        "sentiment2id": sentiment2id,
        "emotion2id": emotion2id,
    }

    label_map_path = configs_dir / "label_maps.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_maps, f, ensure_ascii=False, indent=2)

    print("Saved label maps to:", label_map_path)

    # -----------------------------------------------------------------
    # Train / val / test split
    # -----------------------------------------------------------------
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[sentiment_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[sentiment_col],
    )

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    # -----------------------------------------------------------------
    # Datasets & dataloaders
    # -----------------------------------------------------------------
    max_length = 128
    batch_size = 32

    train_dataset = BERADataset(
        train_df,
        sp_tokenizer=sp,
        sentiment2id=sentiment2id,
        emotion2id=emotion2id,
        max_length=max_length,
        text_col=text_col,
        sentiment_col=sentiment_col,
        emotion_col=emotion_col,
    )
    val_dataset = BERADataset(
        val_df,
        sp_tokenizer=sp,
        sentiment2id=sentiment2id,
        emotion2id=emotion2id,
        max_length=max_length,
        text_col=text_col,
        sentiment_col=sentiment_col,
        emotion_col=emotion_col,
    )
    test_dataset = BERADataset(
        test_df,
        sp_tokenizer=sp,
        sentiment2id=sentiment2id,
        emotion2id=emotion2id,
        max_length=max_length,
        text_col=text_col,
        sentiment_col=sentiment_col,
        emotion_col=emotion_col,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # -----------------------------------------------------------------
    # Model, loss, optimizer
    # -----------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    config = BERAConfig()
    model = BERA(
        config=config,
        num_sentiment_labels=len(sentiment2id),
        num_emotion_labels=len(emotion2id),
    ).to(device)

    sentiment_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # -----------------------------------------------------------------
    # Training + evaluation helpers
    # -----------------------------------------------------------------
    def train_one_epoch(epoch: int):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_labels = batch["sentiment_label"].to(device)
            emo_labels = batch["emotion_label"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            sent_loss = sentiment_criterion(out["sentiment_logits"], sent_labels)
            emo_loss = emotion_criterion(out["emotion_logits"], emo_labels)
            loss = sent_loss + emo_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(loader: DataLoader, split_name: str = "val"):
        model.eval()
        total_loss = 0.0

        all_sent_true, all_sent_pred = [], []
        all_emo_true, all_emo_pred = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"[{split_name}]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                sent_labels = batch["sentiment_label"].to(device)
                emo_labels = batch["emotion_label"].to(device)

                out = model(input_ids=input_ids, attention_mask=attention_mask)
                sent_loss = sentiment_criterion(out["sentiment_logits"], sent_labels)
                emo_loss = emotion_criterion(out["emotion_logits"], emo_labels)
                loss = sent_loss + emo_loss
                total_loss += loss.item()

                sent_pred = out["sentiment_logits"].argmax(dim=-1)
                emo_pred = out["emotion_logits"].argmax(dim=-1)

                all_sent_true.extend(sent_labels.cpu().tolist())
                all_sent_pred.extend(sent_pred.cpu().tolist())
                all_emo_true.extend(emo_labels.cpu().tolist())
                all_emo_pred.extend(emo_pred.cpu().tolist())

        avg_loss = total_loss / len(loader)
        sent_acc = accuracy_score(all_sent_true, all_sent_pred)
        sent_f1 = f1_score(all_sent_true, all_sent_pred, average="weighted")
        emo_acc = accuracy_score(all_emo_true, all_emo_pred)
        emo_f1 = f1_score(all_emo_true, all_emo_pred, average="weighted")

        print(
            f"{split_name} loss: {avg_loss:.4f} | "
            f"Sent acc: {sent_acc:.4f}, Sent F1: {sent_f1:.4f} | "
            f"Emo acc: {emo_acc:.4f}, Emo F1: {emo_f1:.4f}"
        )

        return avg_loss, sent_acc, sent_f1, emo_acc, emo_f1

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    num_epochs = 1
    best_val_loss = float("inf")
    ckpt_path = models_dir / "bera_sent_emo.pt"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(epoch)
        print(f"Train loss: {train_loss:.4f}")
        val_loss, *_ = evaluate(val_loader, split_name="val")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "label_maps": label_maps,
                },
                ckpt_path,
            )
            print(f"Saved best model to {ckpt_path}")

    # -----------------------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------------------
    print("Evaluating best model on test set...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate(test_loader, split_name="test")


if __name__ == "__main__":
    main()
