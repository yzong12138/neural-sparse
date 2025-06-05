import json
import numpy as np
import matplotlib.pyplot as plt
from dynamo_utils import batch_get_vectors
from tqdm import tqdm

with open("search_res.json") as f:
    search_res = json.load(f)

print(search_res[0])

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader

beir_resource_url = (
    f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
)
local_data_path = util.download_and_unzip(beir_resource_url, "../data/beir")
corpus, queries, qrels = GenericDataLoader(data_folder=local_data_path).load(
    split="train"
)

q_ids = [res["q_id"] for res in search_res]
d_ids = [res["d_ids"] for res in search_res]

import pickle

with open("../data/query_to_id.bin", "rb") as f:
    query_to_id = pickle.load(f)

with open("../data/doc_to_id.bin", "rb") as f:
    doc_to_id = pickle.load(f)

from time import time


def transform_str(s):
    try:
        s = s.encode("latin1").decode("utf-8")
    except:
        pass
    return s


scores_bge_en_icl = []
for i in tqdm(range(len(search_res))):
    q = queries[q_ids[i]]
    docs = [corpus[d_id]["text"] for d_id in d_ids[i]]

    q = query_to_id[q]
    docs = [doc_to_id[transform_str(d)] for d in docs]

    q_vector = batch_get_vectors("vector_q", [q], region_name="us-west-2", model_id=2)
    d_vector = batch_get_vectors("vector", docs, region_name="us-west-2", model_id=2)
    score = np.dot(q_vector, d_vector.T).reshape(-1).astype(np.float32)
    scores_bge_en_icl.append(score)

scores_bge_en_icl = np.array(scores_bge_en_icl).astype(np.float64)
np.save("zeta_scores.bin.npy", scores_bge_en_icl)
