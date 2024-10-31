import torch
import itertools
import logging

import transformers

logger = logging.getLogger(__name__)

PREFIXES = {"intfloat/multilingual-e5-large": ["query: ", "passage: "]}
EMPTY_PREFIX = ["", ""]


class KnowledgeDistillDataCollator:
    def __init__(self, tokenizer, max_length=512, teacher_tokenizer_ids=[]):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizers = [self.tokenizer] + [
            transformers.AutoTokenizer.from_pretrained(tokenizer_id)
            for tokenizer_id in teacher_tokenizer_ids
        ]
        self.prefixes = [EMPTY_PREFIX] + [
            PREFIXES.get(teacher_tokenizer_id, EMPTY_PREFIX)
            for teacher_tokenizer_id in teacher_tokenizer_ids
        ]
        logger.info(self.prefixes)
        logger.info(f"total tokenizers {len(self.tokenizers)}")

    # handle knowledge distillation input containing query, docs and scores
    def __call__(self, batch):
        q, docs, scores = zip(*batch)

        has_scores = list(scores)[0][0] is not None
        assert len(docs) == len(scores)
        docs = list(itertools.chain(*docs))

        result = {"query": [], "docs": []}

        for tokenizer, prefix in zip(self.tokenizers, self.prefixes):
            q_prefix, d_prefix = prefix
            result["query"].append(
                tokenizer(
                    [q_prefix + x for x in list(q)],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )
            result["docs"].append(
                tokenizer(
                    [d_prefix + x for x in docs],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )

        if has_scores:
            result["scores"] = torch.tensor(scores)

        return result


class PosNegsDataCollator:
    def __init__(self, tokenizer, max_length=512, teacher_tokenizer_ids=[]):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizers = [self.tokenizer] + [
            transformers.AutoTokenizer.from_pretrained(tokenizer_id)
            for tokenizer_id in teacher_tokenizer_ids
        ]

    def __call__(self, batch):
        q, pos, negs = zip(*batch)
        assert len(q) == len(pos)
        docs = []
        for p, neg in zip(pos, negs):
            docs.append(p)
            docs = docs + neg

        result = {"query": [], "docs": []}

        for tokenizer in self.tokenizers:
            result["query"].append(
                tokenizer(
                    list(q),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )
            result["docs"].append(
                tokenizer(
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )
        return result
