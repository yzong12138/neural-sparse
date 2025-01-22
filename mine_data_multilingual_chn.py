import os
import json
import sys
import asyncio
import logging
import datasets
import torch

from tqdm import tqdm
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_model
from scripts.args import ModelArguments, DataTrainingArguments

from transformers import HfArgumentParser
from accelerate import Accelerator

from torch.utils.data import Dataset

from scripts.model.sparse_encoders import (
    SparseModel,
    SparseEncoder,
    sparse_embedding_to_query,
)
from scripts.data.dataset import KeyValueDataset
from scripts.utils import batch_search, get_os_client


class MmarcoCorpusDataset(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_ids = list(corpus.keys())

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        key = self.corpus_ids[idx]
        value = self.corpus[key]
        return key, value


async def search(
    queries: dict,
    model: SparseModel,
    out_dir: str,
    index_name: str,
    max_length: int = 512,
    batch_size: int = 50,
    result_size: int = 15,
    inf_free: bool = True,
    delete: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    queries_dataset = KeyValueDataset(queries)
    dataloader = torch.utils.data.DataLoader(queries_dataset, batch_size=batch_size)

    query_encoder = SparseEncoder(
        sparse_model=model,
        max_length=max_length,
        do_count=True,
    )

    run_res = dict()
    for ids, texts in tqdm(dataloader):
        queries_encoded = query_encoder.encode(texts, inf_free=inf_free)

        search_results = await batch_search(
            queries=queries_encoded,
            index_name=index_name,
            endpoint_lambda=lambda index_name: f"""http://localhost:9200/{index_name}/_search""",
            get_query_lambda=lambda query: {
                "size": result_size,
                "query": sparse_embedding_to_query(query),
                "_source": ["id", "text"],
            },
            interval=0.001,
        )

        for i, (_id, res) in enumerate(zip(ids, search_results)):
            run_res[_id] = [
                [hit["_source"]["id"] for hit in res],
                [hit["_source"]["text"] for hit in res],
            ]

    if delete:
        client = get_os_client()
        client.indices.delete(index_name, params={"timeout": 1000})

    return run_res


def main():
    logging.basicConfig(
        level=20,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(mixed_precision="fp16")
    # accelerator.wait_for_everyone()
    model = get_model(model_args)

    for dataset in [
        "zh_NLI_data",
        "mmarco-french",
        "mmarco-chinese",
        "DuReader",
        "T2Ranking",
    ]:
        path = os.path.join("mdata/bge-m3-data/parsed", dataset)
        with open(os.path.join(path, "corpus.json"), "r") as f:
            corpus = json.load(f)
        with open(os.path.join(path, "queries.json"), "r") as f:
            queries = json.load(f)
        with open(os.path.join(path, "query_pos_docs.json"), "r") as f:
            query_pos_docs = json.load(f)

        # asyncio.run(
        #     ingest(
        #         dataset=MmarcoCorpusDataset(corpus=corpus),
        #         model=model,
        #         out_dir="tmp/out",
        #         index_name=dataset.lower(),
        #         accelerator=accelerator,
        #         max_length=data_args.max_seq_length,
        #         batch_size=25,
        #     )
        # )

        if 1 and accelerator.is_local_main_process:
            run_res = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir="tmp/out",
                    index_name=dataset.lower(),
                    max_length=data_args.max_seq_length,
                    batch_size=25,
                    result_size=60,
                    delete=False,
                )
            )

            data = {"query": [], "docs": [], "first_rank": []}

            for q_id in queries:
                hits, texts = run_res[q_id]
                pos_docs = set(query_pos_docs.get(q_id, []))

                if len(pos_docs) == 0:
                    first_rank = -1
                else:
                    first_rank = -2
                    for i, hit in enumerate(hits):
                        if hit in pos_docs:
                            first_rank = i
                            break

                data["query"].append(queries[q_id])
                data["docs"].append(texts)
                data["first_rank"].append(first_rank)

            ds = datasets.Dataset.from_dict(data)
            ds.save_to_disk(f"mdata/chn-pt/{dataset}")
            
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
