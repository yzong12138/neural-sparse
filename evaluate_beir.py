import torch.utils
import torch.utils.data
import yaml
import logging
import os
import json
import random
import sys
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm

import pandas as pd
import asyncio

from scripts.model.sparse_encoders import (
    SparseModel,
    SparseEncoder,
    sparse_embedding_to_query,
)
from scripts.data.dataset import BEIRCorpusDataset, KeyValueDataset
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_os_client, batch_search
from scripts.args import ModelArguments, DataTrainingArguments

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from accelerate import Accelerator
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

logger = logging.getLogger(__name__)


def evaluate_dataset(dataset):
    pass


def main():
    # 1. parse args
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    args_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }
    beir_eval_dir = os.path.join(training_args.output_dir, "beir_eval")
    # Export to YAML
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(beir_eval_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, "beir_eval", "config.yaml"), "w"
    ) as file:
        yaml.dump(args_dict, file, sort_keys=False)

    # 2. set logs
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(training_args.output_dir, "eval_beir.log")
            ),
        ],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    accelerator = Accelerator(mixed_precision="fp16")
    idf = None
    if model_args.inf_free and model_args.idf_path:
        with open(model_args.idf_path) as f:
            idf = json.load(f)

    model = SparseModel(
        model_args.model_name_or_path, idf=idf, tokenizer_id=model_args.tokenizer_name
    )
    accelerator.wait_for_everyone()

    datasets = data_args.beir_datasets.split(",")
    result = {
        "dataset": datasets,
        "flops": [],
        "NDCG@10": [],
        "q_length": [],
        "d_length": [],
    }
    avg_res = dict()
    for dataset in datasets:
        # if the dataset wasn't download before, only download it on main process
        if accelerator.is_local_main_process:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            data_path = util.download_and_unzip(url, data_args.beir_dir)
        accelerator.wait_for_everyone()
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, data_args.beir_dir)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )

        asyncio.run(
            ingest(
                dataset=BEIRCorpusDataset(corpus=corpus),
                model=model,
                out_dir=beir_eval_dir,
                index_name=dataset,
                accelerator=accelerator,
                max_length=data_args.max_seq_length,
                batch_size=training_args.per_device_eval_batch_size,
                log_level=log_level,
            )
        )

        # search is only run on main process
        if accelerator.is_local_main_process:
            search_result = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir=beir_eval_dir,
                    index_name=dataset,
                    max_length=data_args.max_seq_length,
                    batch_size=training_args.per_device_eval_batch_size,
                    log_level=log_level,
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

    if accelerator.is_local_main_process:
        df = pd.DataFrame(result)
        for key in ["flops", "q_length", "d_length", "NDCG@10"]:
            avg_res[key] = sum(result[key]) / len(result[key])
        df.to_csv(os.path.join(beir_eval_dir, "beir_statictics.csv"))
        with open(os.path.join(beir_eval_dir, "avg_res.json"), "w") as f:
            json.dump(avg_res, f)


if __name__ == "__main__":
    main()
