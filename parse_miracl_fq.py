import os
import json
import argparse

from datasets import load_dataset
from scripts.utils import get_logger_console

logger = get_logger_console(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("lang", type=str, help="Language code")
parser.add_argument("--source", type=str, default="miracl")

args = parser.parse_args()

lang = args.lang
source = args.source
logger.info(f"The provided language code is: {lang}, {source}")

dataset_source = "miracl/miracl-corpus"
if source == "tydi":
    dataset_source = "castorini/mr-tydi-corpus"

import torch
import os

from tqdm import tqdm
from transformers import AutoTokenizer


class IDFRepBuilder:
    def __init__(self, tokenizer, batch_size=100):
        self.tokenizer = tokenizer
        self.special_token_ids = [
            self.tokenizer.vocab[token]
            for token in self.tokenizer.special_tokens_map.values()
        ]
        self.batch_size = batch_size

    @staticmethod
    def batch(iterable, n=10):
        length = len(iterable)
        for ndx in range(0, length, n):
            yield iterable[ndx : min(ndx + n, length)]

    def build(self, text_list, return_fq=False):
        fq_tensor = torch.zeros(self.tokenizer.vocab_size)
        num_samples = len(text_list)
        print("start building idf representation")
        for text in tqdm(IDFRepBuilder.batch(text_list, 1000)):
            input_ids = self.tokenizer(
                text,
                return_attention_mask=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            for input_id in input_ids:
                fq_tensor[input_id] += 1

        fq_tensor[self.special_token_ids] = 0

        if return_fq:
            return fq_tensor

        zero_idxs = fq_tensor == 0
        nz_idxs = fq_tensor != 0

        idf = torch.log(1 + (num_samples - fq_tensor + 0.5) / (fq_tensor + 0.5))
        idf[zero_idxs] = 0

        return idf


class IDFRepWrapper:
    def __init__(self, tokenizer_type):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def get_idf(self, text_list=None, return_fq=False):
        builder = IDFRepBuilder(self.tokenizer)
        idf = builder.build(text_list, return_fq=return_fq)
        return idf


idf_wrapper = IDFRepWrapper("google-bert/bert-base-multilingual-uncased")
miracl_corpus = load_dataset(dataset_source, lang, trust_remote_code=True)["train"]
text_list = miracl_corpus["text"]
idf = idf_wrapper.get_idf(text_list, return_fq=True)

data = {"corpus": source, "lang": lang, "length": len(text_list), "fq_tensor": idf}

os.makedirs(f"idf-uncased/{source}/", exist_ok=True)
torch.save(data, f"idf-uncased/{source}/{lang}.bin")

logger.info(f"finish {lang}")
