import torch.utils
import torch.utils.data
import yaml
import logging
import os
import json
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm

import pandas as pd
import asyncio

from scripts.data.dataset import MiraclCorpusDataset, KeyValueDataset
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_os_client, batch_search, set_logging, get_model
from scripts.args import ModelArguments, DataTrainingArguments, parse_args

import transformers
from transformers import (
    set_seed,
)
from datasets import load_dataset

from accelerate import Accelerator
from beir.retrieval.evaluation import EvaluateRetrieval

logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = parse_args()

    args_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }
    miracl_eval_dir = os.path.join(training_args.output_dir, "miracl_eval")
    os.makedirs(miracl_eval_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, "miracl_eval", "config.yaml"), "w"
    ) as file:
        yaml.dump(args_dict, file, sort_keys=False)

    set_logging(training_args, "eval_miracl.log")
    set_seed(training_args.seed)

    model = get_model(model_args)

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()

    datasets = data_args.miracl_datasets.split(",")
    result = {
        "dataset": datasets,
        "flops": [],
        "NDCG@10": [],
        "q_length": [],
        "d_length": [],
    }
    avg_res = dict()
    for dataset in datasets:
        miracl_dir = os.path.join(data_args.miracl_dir, dataset)
        corpus = load_dataset("miracl/miracl-corpus", dataset, trust_remote_code=True)[
            "train"
        ]
        with open(os.path.join(miracl_dir, "qrels.json")) as f:
            qrels = json.load(f)
        with open(os.path.join(miracl_dir, "queries.json")) as f:
            queries = json.load(f)

        transform_lambda = None
        if dataset == "zh":
            import zhconv

            logger.info("transform")
            transform_lambda = lambda text: zhconv.convert_for_mw(text, "zh-cn")
            for q_id, query_text in list(queries.items()):
                queries[q_id] = zhconv.convert_for_mw(query_text, "zh-cn")

        asyncio.run(
            ingest(
                dataset=MiraclCorpusDataset(
                    corpus=corpus, transform_lambda=transform_lambda
                ),
                model=model,
                out_dir=miracl_eval_dir,
                index_name=dataset,
                accelerator=accelerator,
                max_length=data_args.max_seq_length,
                batch_size=training_args.per_device_eval_batch_size,
            )
        )

        # search is only run on main process
        if accelerator.is_local_main_process:
            search_result = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir=miracl_eval_dir,
                    index_name=dataset,
                    max_length=data_args.max_seq_length,
                    batch_size=training_args.per_device_eval_batch_size,
                    delete=True,
                )
            )

            ndcg, map_, recall, p = EvaluateRetrieval.evaluate(
                qrels, search_result["run_res"], [1, 10]
            )
            logger.info(f"retrieve metrics for {dataset}: {ndcg, map_, recall, p}")
            result["NDCG@10"].append(ndcg["NDCG@10"])
            result["flops"].append(search_result["flops"])
            result["q_length"].append(search_result["q_length"])
            result["d_length"].append(search_result["d_length"])

        accelerator.wait_for_everyone()

    if accelerator.is_local_main_process and data_args.save_eval_result:
        df = pd.DataFrame(result)
        for key in ["flops", "q_length", "d_length", "NDCG@10"]:
            avg_res[key] = sum(result[key]) / len(result[key])
        df.to_csv(os.path.join(miracl_eval_dir, "miracl_statictics.csv"))
        with open(os.path.join(miracl_eval_dir, "avg_res.json"), "w") as f:
            json.dump(avg_res, f)


if __name__ == "__main__":
    main()
