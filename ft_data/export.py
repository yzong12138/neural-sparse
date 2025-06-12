import json
import numpy as np
import matplotlib.pyplot as plt
import datasets
from tqdm import tqdm
import numpy as np

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

# score_sparse = np.load("sparse_scores.bin.npy")
# score_dense = np.load("gte_scores.bin.npy")
score_t5 = np.load("t5_scores.bin.npy")
score_minilm = np.load("minilm_scores.bin.npy")
score_bge_en_icl = np.load("bge_256_scores.bin.npy")
# score_gte_bi_15w = np.load("gte-bi-15w-scores.bin.npy")

score_inf = np.load("inf_scores.bin.npy")
score_bge_gemma2 = np.load("bge_gemma2_reranker_scores.npy")
score_qwen = np.load("qwen3_scores.bin.npy")


def transform_str(s):
    try:
        return s.encode("latin1").decode("utf-8")
    except:
        return s


def min_max_norm(x):
    min_val = x.min(axis=-1, keepdims=True)
    max_val = x.max(axis=-1, keepdims=True)
    return (x - min_val) / (max_val - min_val + 1e-8)  # 加上小量避免除零


def combine(*scores, weights=None):
    if weights is None:
        weights = [1] * len(scores)
    scores = [min_max_norm(score) * weights[i] for i, score in enumerate(scores)]
    scores = np.sum(scores, axis=0) / len(scores)
    return scores


def export_to_dataset(scores):
    run_res = dict(zip(q_ids, map(dict, map(zip, d_ids, scores))))

    all_data = []
    for q_id, doc_score_map in tqdm(run_res.items()):
        doc_score_items = sorted(
            doc_score_map.items(), key=lambda x: x[1], reverse=True
        )
        doc_ids = [item[0] for item in doc_score_items]
        score = [item[1] * 30 for item in doc_score_items]
        all_data.append(
            {
                "query": queries[q_id],
                "docs": [transform_str(corpus[doc_id]["text"]) for doc_id in doc_ids],
                "scores": score,
            }
        )
    return datasets.Dataset.from_list(all_data)


# score = combine(score_sparse,score_dense,score_t5,score_minilm,score_bge_en_icl, weights=[1,1,1,1,2])
# score = min_max_norm(score_bge_en_icl)
score = combine(
    score_qwen,
    score_bge_gemma2,
    score_inf,
    score_t5,
    score_minilm,
    score_bge_en_icl,
    weights=[1, 1, 1, 1, 1, 2],
)
ds = export_to_dataset(score)

print(ds[0])

ds.save_to_disk("../data/msmarco_ft_llm_reranker")
