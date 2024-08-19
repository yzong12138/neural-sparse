import os
import asyncio
import aiohttp
from aiohttp import ClientTimeout
import json

import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from opensearchpy import OpenSearch


def is_ddp_enabled():
    if dist.is_initialized():
        return True
    return False


def get_os_client():
    return OpenSearch(
        hosts=["http://localhost:9200"], verify_certs=False, ssl_show_warn=False
    )


async def do_search(session, endpoint, query_body, post_process=None):
    url = endpoint
    body = query_body

    async with session.get(url, json=body, verify_ssl=False) as resp:
        response = await resp.json()
        if "error" in response:
            raise Exception(response["error"])

    hits = response["hits"]["hits"]
    if post_process is not None:
        hits = post_process(hits)
    return hits


async def do_bulk(bulk_body, session, endpoint="http://localhost:9200"):
    url = endpoint + "/_bulk"
    bulk_body = "\n".join(map(json.dumps, bulk_body)) + "\n"
    headers = {"Content-Type": "application/x-ndjson"}
    async with session.post(url, data=bulk_body, headers=headers) as resp:
        response = await resp.json()
        if "errors" not in response:
            print(response)
        assert response["errors"] == False

    return response


async def batch_search(
    queries,
    index_name,
    endpoint_lambda,
    get_query_lambda,
    post_process=None,
    interval=0.01,
):
    timeout = ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for query in queries:
            tasks.append(
                asyncio.create_task(
                    do_search(
                        session,
                        endpoint_lambda(index_name),
                        get_query_lambda(query),
                        post_process=post_process,
                    )
                )
            )
            await asyncio.sleep(interval)

        try:
            result = await asyncio.gather(*tasks)
        except Exception as e:
            print("error happens when querying index ", index_name)
            print(e)
            return {"error": str(e)}
    return result
