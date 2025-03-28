import torch
import re
import os
import transformers
import json


def has_uppercase(s):
    return any(c.isupper() for c in s)


# 1. 加载原始tokenizer和model
tokenizer = transformers.AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = transformers.AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-cased")

# 2. 找出要删除的token索引
vocab = tokenizer.get_vocab()
special_tokens = tokenizer.special_tokens_map.values()

new_vocab = {}
new_idx = 0
old_to_new_idx = {}

for old_idx in range(len(vocab)):
    token = tokenizer._convert_id_to_token(old_idx)
    if token not in special_tokens:
        if has_uppercase(token):
            if token.lower() in vocab:
                continue
            else:
                token = token.lower()
                if token in new_vocab:
                    continue

    new_vocab[token] = new_idx
    old_to_new_idx[old_idx] = new_idx
    new_idx += 1


tokenizer.save_pretrained("tmp")
os.makedirs("cased-del-model", exist_ok=True)

with open("tmp/tokenizer.json") as f:
    config = json.load(f)
config["model"]["vocab"] = new_vocab
with open("cased-del-model/tokenizer.json", "w") as f:
    json.dump(config, f, indent=4)


with open("tmp/tokenizer_config.json") as f:
    config = json.load(f)
config["do_lower_case"] = True
with open("cased-del-model/tokenizer_config.json", "w") as f:
    json.dump(config, f, indent=4)

# generate vocab.txt, special_tokens_map.json
tokenizer = transformers.AutoTokenizer.from_pretrained("cased-del-model")
tokenizer.save_pretrained("cased-del-model")

old_embeddings = model.bert.embeddings.word_embeddings.weight.data
new_embeddings = torch.zeros((len(new_vocab), old_embeddings.shape[1]))

for old_idx, new_idx in old_to_new_idx.items():
    new_embeddings[new_idx] = old_embeddings[old_idx]

model.bert.embeddings.word_embeddings = torch.nn.Embedding(
    len(new_vocab), old_embeddings.shape[1], _weight=new_embeddings
)


old_mlm_bias = model.cls.predictions.bias.data
old_mlm_decoder = model.cls.predictions.decoder.weight.data

new_mlm_bias = torch.zeros(len(new_vocab))
new_mlm_decoder = torch.zeros((len(new_vocab), old_mlm_decoder.shape[1]))

for old_idx, new_idx in old_to_new_idx.items():
    new_mlm_bias[new_idx] = old_mlm_bias[old_idx]
    new_mlm_decoder[new_idx] = old_mlm_decoder[old_idx]

model.cls.predictions.decoder = torch.nn.Linear(
    old_mlm_decoder.shape[1], len(new_vocab), bias=False
)
model.cls.predictions.decoder.weight.data = new_mlm_decoder
model.cls.predictions.bias = torch.nn.Parameter(new_mlm_bias)

# 6. 更新config
model.config.vocab_size = len(new_vocab)
model.save_pretrained("cased-del-model")
