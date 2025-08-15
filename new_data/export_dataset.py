import json
import numpy as np
import datasets
from beir.retrieval.evaluation import EvaluateRetrieval

def eval_score(scores, return_run_res = False, sample_num=1000000):
    run_res = dict(zip(q_ids[:sample_num], map(dict, map(zip, d_ids[:sample_num], scores[:sample_num]))))
    if return_run_res:
        return run_res
    else:
        return EvaluateRetrieval.evaluate(qrels, run_res, [10])
    
def min_max_norm(x):
    min_val = x.min(axis=-1, keepdims=True)
    max_val = x.max(axis=-1, keepdims=True)
    return (x - min_val) / (max_val - min_val + 1e-8)  # 加上小量避免除零

def combine(*scores, weights=None):
    if weights is None:
        weights = [1]*len(scores)
    scores = [min_max_norm(score)*weights[i] for i,score in enumerate(scores)]
    scores = np.sum(scores,axis=0)/sum(weights)
    return scores

score_dense = np.load("scores/msmarco_raw/gte-bi_scores.npy").astype(np.float64)
score_sparse = np.load("scores/msmarco_raw/sparse-v1_scores.npy").astype(np.float64)
score_minilm = np.load("scores/msmarco_raw/minilm_scores.npy").astype(np.float64)
score_t5 = np.load("scores/msmarco_raw/t5_scores.npy").astype(np.float64)

with open("marco_qrels.json") as f:
    qrels = json.load(f)
    
with open("q_ids.json") as f:
    q_ids = json.load(f)
    
with open("d_ids.json") as f:
    d_ids = json.load(f)



score = combine(score_sparse, score_minilm, score_t5, weights=[1,1,1])
print(eval_score(score, sample_num=2000))

score = score*30
ds = datasets.load_from_disk("msmarco_raw")
ds = ds.remove_columns(['q_id', 'd_ids', 'dynamo_q_id', 'dynamo_d_ids'])
ds = ds.add_column('scores', score.tolist())
ds.save_to_disk("msmarco_mts")


score = combine(score_minilm, score_t5, weights=[1,1])
print(eval_score(score, sample_num=2000))

score = score*30
ds = datasets.load_from_disk("msmarco_raw")
ds = ds.remove_columns(['q_id', 'd_ids', 'dynamo_q_id', 'dynamo_d_ids'])
ds = ds.add_column('scores', score.tolist())
ds.save_to_disk("msmarco_mt")