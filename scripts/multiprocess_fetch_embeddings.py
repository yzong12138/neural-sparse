import argparse
import logging
import math
import multiprocessing as mp
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
import torch

from .async_embedding_server import EmbeddingService

load_dotenv()


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(processName)s - %(message)s",
    )


def worker(
    process_index: int,
    start_id_inclusive: int,
    end_id_exclusive: int,
    batch_size: int,
    table_name: str,
    model_ids: List[int],
    max_workers: int,
) -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info(
        "Worker %s handling id range [%s, %s) with batch_size=%s, models=%s",
        process_index,
        start_id_inclusive,
        end_id_exclusive,
        batch_size,
        model_ids,
    )

    service = EmbeddingService(max_workers=max_workers)
    try:
        for batch_start in tqdm(range(start_id_inclusive, end_id_exclusive, batch_size)):
            batch_end = min(batch_start + batch_size, end_id_exclusive)
            ids = list(range(batch_start, batch_end))

            for model_id in model_ids:
                try:
                    service.register_task(table_name=table_name, model_id=model_id, ids=ids)
                    _ = service.fetch_embedding(
                        table_name=table_name,
                        model_id=model_id,
                        ids=ids,
                    )
                    _ = torch.tensor(_)
                except Exception:
                    logging.exception(
                        "Error fetching embeddings at start id %s for model_id=%s",
                        batch_start,
                        model_id,
                    )
    finally:
        service.shutdown()


def even_splits(
    start_id_inclusive: int, end_id_exclusive: int, processes: int
) -> List[tuple]:
    total = max(0, end_id_exclusive - start_id_inclusive)
    if processes <= 0:
        raise ValueError("processes must be > 0")
    if total == 0:
        return []

    base = total // processes
    remainder = total % processes
    splits = []
    current = start_id_inclusive
    for i in range(processes):
        size = base + (1 if i < remainder else 0)
        if size <= 0:
            continue
        start = current
        end = start + size
        splits.append((start, end))
        current = end
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "多进程获取embedding：按进程数均匀分配ID区间，每个进程内部按批次从DynamoDB获取。"
        )
    )
    parser.add_argument(
        "--processes",
        type=int,
        required=True,
        help="要启动的进程数",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        required=True,
        help="DynamoDB 表名",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        required=True,
        help="最大ID（不包含，区间为 [start-id, max-id)）",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="起始ID（包含，默认0）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="批大小（每次请求的ID数量，默认10000）",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        action="append",
        default=[],
        help="模型ID（可多次指定以支持多个模型，默认3）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="每个进程内的线程池大小（默认20）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="日志级别：DEBUG/INFO/WARN/ERROR（默认INFO）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    start_id = args.start_id
    max_id_exclusive = args.max_id
    if max_id_exclusive <= start_id:
        raise ValueError("--max-id 必须大于 --start-id")

    splits = even_splits(start_id, max_id_exclusive, args.processes)
    logger.info(
        "Dispatching %s processes over range [%s, %s): %s",
        args.processes,
        start_id,
        max_id_exclusive,
        splits,
    )

    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    for idx, (s, e) in enumerate(splits):
        p = ctx.Process(
            target=worker,
            name=f"worker-{idx}",
            args=(
                idx,
                s,
                e,
                args.batch_size,
                args.table_name,
                args.model_id,
                args.max_workers,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("All processes finished.")


if __name__ == "__main__":
    main()


