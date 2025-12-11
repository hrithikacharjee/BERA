import torch
from torch.utils.data import Dataset
import pandas as pd


class BERADataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        sp_tokenizer,
        sentiment2id=None,
        emotion2id=None,
        max_length=128,
        text_col="Review",
        sentiment_col="Sentiment",
        emotion_col="Emotion"
    ):
        self.df = dataframe.reset_index(drop=True)
        self.sp = sp_tokenizer
        self.sentiment2id = sentiment2id
        self.emotion2id = emotion2id
        self.max_length = max_length
        self.text_col = text_col
        self.sentiment_col = sentiment_col
        self.emotion_col = emotion_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])

        ids = self.sp.encode(text)
        ids = ids[:self.max_length]

        attention_mask = [1] * len(ids)

        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
            attention_mask += [0] * pad_len

        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if self.sentiment2id:
            item["sentiment_label"] = torch.tensor(
                self.sentiment2id.get(row[self.sentiment_col], -1),
                dtype=torch.long
            )

        if self.emotion2id:
            item["emotion_label"] = torch.tensor(
                self.emotion2id.get(row[self.emotion_col], -1),
                dtype=torch.long
            )

        return item
