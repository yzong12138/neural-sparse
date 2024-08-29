import logging
import os
import torch
from tqdm import tqdm

from .model.sparse_encoders import SparseModel, SparseEncoder, sparse_embedding_to_query
from .data.dataset import KeyValueDataset
from .utils import batch_search

logger = logging.getLogger(__name__)


async def search(
    queries: dict,
    model: SparseModel,
    out_dir: str,
    index_name: str,
    max_length: int = 512,
    batch_size: int = 50,
    result_size: int = 15,
):
    device = model.backbone.device
    os.makedirs(out_dir, exist_ok=True)

    queries_dataset = KeyValueDataset(queries)
    dataloader = torch.utils.data.DataLoader(queries_dataset, batch_size=batch_size)

    query_encoder = SparseEncoder(
        sparse_model=model,
        max_length=max_length,
        device=device,
        do_count=True,
    )

    run_res = dict()
    for ids, texts in tqdm(dataloader):
        queries_encoded = query_encoder.encode(texts, inf_free=True)

        search_results = await batch_search(
            queries=queries_encoded,
            index_name=index_name,
            endpoint_lambda=lambda index_name: f"""http://localhost:9200/{index_name}/_search""",
            get_query_lambda=lambda query: {
                "size": result_size,
                "query": sparse_embedding_to_query(query),
                "_source": ["id"],
            },
            interval=0.001,
        )

        for i, (_id, res) in enumerate(zip(ids, search_results)):
            run_res[_id] = {hit["_source"]["id"]: hit["_score"] for hit in res}

    for query_id, doc_dict in run_res.items():
        if query_id in doc_dict:
            doc_dict.pop(query_id)

    count = query_encoder.count_tensor
    count = count / len(queries_dataset)
    count_doc = torch.load(
        os.path.join(out_dir, index_name + ".corpus.bin"), map_location=device
    )
    flops = float(torch.matmul(count, count_doc).cpu())
    q_length = float(count.sum().cpu())
    d_length = float(count_doc.sum().cpu())
    logger.info(
        f"Index_name: {index_name}, flops: {flops}, d_length:{d_length}, q_length:{q_length}"
    )

    return {
        "run_res": run_res,
        "flops": flops,
        "q_length": q_length,
        "d_length": d_length,
    }
