import os
import sys
import json
import numpy as np

sys.path.insert(0, "/home/zhichaog/neural-sparse")
from scripts.async_embedding_server import EmbeddingService
from scripts.model.sparse_encoders import SparseModel
from tqdm import tqdm

os.environ["AWS_REGION_NAME"] = "us-west-2"
emb_service = EmbeddingService()

max_id = 5620105
step = 1000
nan_idxs = []
for start in tqdm(range(0, max_id, step)):
    end = min(max_id, start + step)
    emb_service.register_task("vector_q", model_id=6, ids=list(range(end)))
    emb = emb_service.fetch_embedding("vector_q", model_id=6, ids=list(range(end)))
    idxs = np.isnan(emb).nonzero()[0].tolist()
    idxs = [idx + start for idx in idxs]
    if idxs != []:
        print(idxs)
    nan_idxs.extend(idxs)

with open("q_nan_idxs.json", "w") as f:
    json.dump(nan_idxs, f)
