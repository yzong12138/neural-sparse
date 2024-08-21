import torch
import itertools
import logging

logger = logging.getLogger(__name__)


class KnowledgeDistillDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    # handle knowledge distillation input containing query, docs and scores
    def __call__(self, batch):
        q, docs, scores = zip(*batch)

        assert len(docs) == len(scores)
        docs = list(itertools.chain(*docs))

        q = self.tokenizer(
            list(q),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        docs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        result = {"query": q, "docs": docs, "scores": torch.tensor(scores)}

        # logger.info(f"""collator data fingerprint: {q["input_ids"].shape}, {result["scores"].mean()}""")

        return result
