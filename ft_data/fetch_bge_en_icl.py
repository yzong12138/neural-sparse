import json
import numpy as np
import matplotlib.pyplot as plt
from dynamo_utils import batch_get_vectors
from tqdm import tqdm
import argparse

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
import pickle
from time import time


def transform_str(s):
    try:
        s = s.encode("latin1").decode("utf-8")
    except:
        pass
    return s


def main(args):
    with open("search_res.json") as f:
        search_res = json.load(f)

    print(search_res[0])

    beir_resource_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
    local_data_path = util.download_and_unzip(beir_resource_url, "../data/beir")
    corpus, queries, qrels = GenericDataLoader(data_folder=local_data_path).load(
        split="train"
    )

    q_ids = [res["q_id"] for res in search_res]
    d_ids = [res["d_ids"] for res in search_res]

    with open("../data/query_to_id.bin", "rb") as f:
        query_to_id = pickle.load(f)

    with open("../data/doc_to_id.bin", "rb") as f:
        doc_to_id = pickle.load(f)

    all_scores = []
    for i in tqdm(range(len(search_res))):
        q = queries[q_ids[i]]
        docs = [corpus[d_id]["text"] for d_id in d_ids[i]]

        q = query_to_id[q]
        docs = [doc_to_id[transform_str(d)] for d in docs]

        q_vector = batch_get_vectors(
            "vector_q", [q], region_name=args.region, model_id=args.model_id
        )
        d_vector = batch_get_vectors(
            "vector", docs, region_name=args.region, model_id=args.model_id
        )
        print(q_vector.shape)
        print(d_vector.shape)
        score = np.dot(q_vector, d_vector.T).reshape(-1).astype(np.float32)
        all_scores.append(score)

    all_scores = np.array(all_scores).astype(np.float64)
    np.save(args.output, all_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int, default=3)
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--output", type=str, default="bge_256_scores.bin.npy")
    args = parser.parse_args()
    main(args)
