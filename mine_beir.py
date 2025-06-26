import os
import json
import asyncio
import logging
import sys

from scripts.dataset.dataset import BEIRHfDataset
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_model
from scripts.args import ModelArguments, DataTrainingArguments

from datasets import Dataset, load_dataset
from transformers import HfArgumentParser
from accelerate import Accelerator
from beir import util
from beir.datasets.data_loader import GenericDataLoader


logging.basicConfig(
    level=20,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

    model_args, data_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()
    model = get_model(model_args)

    datasets = data_args.beir_datasets.split(",")

    for dataset in datasets:
        test_query_ids = set(load_dataset(f"BeIR/{dataset}-qrels")["test"]["query-id"])
        test_query_ids = set(str(x) for x in test_query_ids)
        queries_ds = load_dataset(f"BEIR/{dataset}", "queries", trust_remote_code=True)[
            "queries"
        ]
        assert isinstance(queries_ds[0]["_id"], str)
        queries_ds = queries_ds.filter(
            lambda sample: sample["_id"] not in test_query_ids
        )
        queries_ds = queries_ds.filter(lambda sample: sample["text"].strip() != "")
        if len(queries_ds) == 0:
            logger.info(f"skip {dataset}")
            continue
        logger.info(f"{dataset} remaining queries number: {len(queries_ds)}")

        queries = dict(zip(queries_ds["_id"], queries_ds["text"]))

        corpus_ds = BEIRHfDataset(
            load_dataset(f"BEIR/{dataset}", "corpus", trust_remote_code=True)["corpus"]
        )

        if not data_args.skip_ingest:
            asyncio.run(
                ingest(
                    dataset=corpus_ds,
                    model=model,
                    out_dir="tmp/out",
                    index_name=dataset,
                    accelerator=accelerator,
                    max_length=data_args.max_seq_length,
                    batch_size=50,
                )
            )

        if data_args.do_search and accelerator.is_local_main_process:
            search_result = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir="tmp/out",
                    index_name=dataset,
                    max_length=data_args.max_seq_length,
                    batch_size=50,
                    result_size=100,
                    return_text=True,
                )
            )

            data = []
            run_res = search_result["run_res"]
            for query, docs in run_res.items():
                data.append({"query": queries[query], "docs": docs})

            ds = Dataset.from_list(data)
            ds.save_to_disk(f"data/mine_beir/{dataset}")

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
