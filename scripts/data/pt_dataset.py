import logging

from transformers import AutoTokenizer
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PtDataset(Dataset):
    def __init__(self, texts):
        self.data = texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PtDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    # handle knowledge distillation input containing query, docs and scores
    def __call__(self, batch):
        features = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        return features
