import os
import datasets
import logging
import numpy as np
import argparse

from tqdm import tqdm
from models import GET_SCORER

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_logger_console(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def select_dataset(dataset, total_processes, process_index):
    dataset_size = len(dataset)
    per_process_size = dataset_size // total_processes
    remainder = dataset_size % total_processes
    start_idx = process_index * per_process_size + min(process_index, remainder)
    end_idx = start_idx + per_process_size + (1 if process_index < remainder else 0)

    logger.info(f"Process {process_index}: processing indices {start_idx} to {end_idx}")
    return dataset.select(range(start_idx, end_idx))


def get_pairs(dataset):
    def make_pairs(batch):
        out_q, out_d = [], []
        for q, docs in zip(batch["query"], batch["docs"]):
            n = len(docs)
            # 重复 query n 次，然后 flatten docs
            out_q.extend([q] * n)
            out_d.extend(docs)
        return {"query": out_q, "doc": out_d}

    paired_ds = dataset.map(
        make_pairs,
        num_proc=3,
        batched=True,  # 批量处理
        remove_columns=dataset.column_names,  # 删除旧列
    )

    queries = paired_ds["query"]
    docs = paired_ds["doc"]
    pairs = list(zip(queries, docs))
    return pairs


logger = get_logger_console(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Worker process for scoring datasets")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the datasets directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output scores",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        required=True,
        help="Model identifier to use for scoring",
    )
    parser.add_argument(
        "--total_processes",
        type=int,
        required=True,
        help="Total number of processes",
    )
    parser.add_argument(
        "--process_rank",
        type=int,
        required=True,
        help="Rank of this process (0-indexed)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    identifier = args.identifier
    total_processes = args.total_processes
    process_rank = args.process_rank

    logger.info(
        f"Starting worker process {process_rank}/{total_processes} for dataset {dataset_name}"
    )

    # Initialize the scorer - use specific GPU based on process rank
    if HAS_TORCH and torch.cuda.is_available():
        gpu_id = process_rank % torch.cuda.device_count()
    else:
        gpu_id = process_rank

    scorer = GET_SCORER[identifier](f"cuda:{gpu_id}")

    # Load and select dataset portion for this process
    dataset = datasets.load_from_disk(os.path.join(dataset_path, dataset_name))
    dataset = select_dataset(dataset, total_processes, process_rank)

    logger.info(f"Process {process_rank} processing {len(dataset)} samples")

    if identifier.startswith("bge"):
        pairs = get_pairs(dataset)
        all_scores = scorer.score_pairs(pairs)
        all_scores = all_scores.reshape(-1, 100)
    else:
        all_scores = []
        for i in tqdm(range(len(dataset)), desc=f"Process {process_rank}"):
            # Use the scorer
            scores = scorer.score(**dataset[i])
            all_scores.append(scores)
        all_scores = np.array(all_scores)

    logger.info(
        f"Process {process_rank} computed scores with shape: {all_scores.shape}"
    )

    # Save scores for this process
    output_file = os.path.join(
        output_dir, dataset_name, identifier + f"_{process_rank}.npy"
    )
    np.save(output_file, all_scores)
    logger.info(f"Process {process_rank} saved scores to {output_file}")


if __name__ == "__main__":
    main()
