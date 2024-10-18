import os
import json
import asyncio

from scripts.data.dataset import BEIRCorpusDataset, MiraclCorpusDataset
from scripts.data.data_utils import get_corpus, get_queries
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_model
from scripts.args import ModelArguments, DataTrainingArguments, MiningArguments

from transformers import HfArgumentParser
from accelerate import Accelerator
from datasets import load_dataset, Dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MiningArguments))

    model_args, data_args, mine_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()
    model = get_model(model_args)

    datasets = mine_args.mine_datasets.split(",")
    source = mine_args.source
    for dataset in datasets:
        corpus = get_corpus(source, dataset)
        queries, query_pos_docs = get_queries(source, dataset)

        asyncio.run(
            ingest(
                dataset=MiraclCorpusDataset(corpus=corpus),
                model=model,
                out_dir="tmp/out",
                index_name=dataset,
                accelerator=accelerator,
                max_length=data_args.max_seq_length,
                batch_size=25,
            )
        )

        if accelerator.is_local_main_process:
            search_result = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir="tmp/out",
                    index_name=dataset,
                    max_length=data_args.max_seq_length,
                    batch_size=25,
                    result_size=100,
                    delete=True,
                )
            )

            run_res = search_result["run_res"]

            data = {"query_id": [], "query": [], "hits": [], "first_rank": []}

            for q_id in queries:
                hits = run_res[q_id]
                hits = list(hits.keys())
                pos_docs = set(query_pos_docs.get(q_id, []))

                if len(pos_docs) == 0:
                    first_rank = -1
                else:
                    first_rank = -2
                    for i, hit in enumerate(hits):
                        if hit in pos_docs:
                            first_rank = i
                            break

                data["query_id"].append(q_id)
                data["query"].append(queries[q_id])
                data["hits"].append(hits)
                data["first_rank"].append(first_rank)

            ds = Dataset.from_dict(data)
            ds.save_to_disk(f"mdata/raw/{source}/{dataset}")


if __name__ == "__main__":
    main()
