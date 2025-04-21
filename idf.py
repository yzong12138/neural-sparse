from transformers import (
    AutoTokenizer,
)

import torch

from tqdm import tqdm
from transformers import AutoTokenizer
import json

from beir import util
from beir.datasets.data_loader import GenericDataLoader


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

    def build(self, text_list):
        fq_tensor = torch.zeros(len(self.tokenizer.vocab))
        num_samples = len(text_list)
        print("start building idf representation")
        for text in tqdm(IDFRepBuilder.batch(text_list, 10)):
            input_ids = self.tokenizer(
                text,
                return_attention_mask=False,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            for input_id in input_ids:
                fq_tensor[input_id] += 1

        fq_tensor[self.special_token_ids] = 0
        zero_idxs = fq_tensor == 0
        nz_idxs = fq_tensor != 0

        idf = torch.log(1 + (num_samples - fq_tensor + 0.5) / (fq_tensor + 0.5))
        idf[zero_idxs] = 0

        return idf


class IDFRepWrapper:
    def __init__(self, tokenizer_type):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def get_idf(self, text_list=None):
        builder = IDFRepBuilder(self.tokenizer)
        idf = builder.build(text_list)
        return idf


tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

beir_resource_url = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
)
local_data_path = util.download_and_unzip(beir_resource_url, "data/beir/")
corpus, queries, qrels = GenericDataLoader(data_folder=local_data_path).load(
    split="train"
)


def transform_str(s):
    try:
        s = s.encode("latin1").decode("utf-8")
    except:
        pass
    s = s.lower()
    s = " " + s.strip()
    return s


for k, v in tqdm(corpus.items()):
    corpus[k]["text"] = transform_str(v["text"])
text_list = []
for k, v in tqdm(corpus.items()):
    text_list.append(v["text"])
wrapper = IDFRepWrapper("answerdotai/ModernBERT-base")
idf = wrapper.get_idf(text_list)
idf_json = {}
for idx in range(idf.shape[0]):
    if idf[idx] != 0:
        idf_json[tokenizer._convert_id_to_token(idx)] = float(idf[idx])

with open("mdbert_idf.json", "w") as f:
    json.dump(idf_json, f)
