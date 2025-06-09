import os
import datasets
import logging
import numpy as np

from tqdm import tqdm
from accelerate import Accelerator
from models import GET_SCORER


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

    logger.info(f"{process_index} {start_idx} {end_idx}")
    return dataset.select(range(start_idx, end_idx))


logger = get_logger_console(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Score datasets using specified model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./",
        help="Path to the datasets directory (default: to_scores)",
    )
    parser.add_argument(
        "--all_datasets",
        type=str,
        default="scifact,hotpotqa,fever,nfcorpus,fiqa",
        help="Comma-separated list of datasets to process (default: scifact,hotpotqa,fever,nfcorpus,fiqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scores",
        help="Directory to save output scores (default: scores)",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default="minilm",
        help="Model identifier to use for scoring (default: minilm)",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    all_datasets = args.all_datasets
    output_dir = args.output_dir
    identifier = args.identifier

    accelerator = Accelerator(mixed_precision="fp16")
    total_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # Initialize the scorer
    scorer = GET_SCORER[identifier](f"cuda:{process_index}")

    for dataset_name in all_datasets.split(","):
        dataset = datasets.load_from_disk(os.path.join(dataset_path, dataset_name))
        dataset = select_dataset(dataset, total_processes, process_index)

        if accelerator.is_local_main_process:
            os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
        accelerator.wait_for_everyone()

        all_scores = []
        for i in tqdm(range(len(dataset))):
            # Use the scorer
            scores = scorer.score(**dataset[i])
            all_scores.append(scores)

        all_scores = np.array(all_scores)
        logger.info(all_scores.size)
        np.save(
            os.path.join(
                output_dir, dataset_name, identifier + f"_{process_index}.npy"
            ),
            all_scores,
        )

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            combined_scores = []

            # Read scores from each process in order
            for proc_idx in range(total_processes):
                proc_file = os.path.join(
                    output_dir, dataset_name, identifier + f"_{proc_idx}.npy"
                )
                proc_scores = np.load(proc_file)
                combined_scores.append(proc_scores)

                # Delete the individual file after reading
                os.remove(proc_file)

            # Concatenate all scores
            combined_scores = np.concatenate(combined_scores, axis=0)
            logger.info(f"Combined scores shape: {combined_scores.shape}")

            # Save the combined scores
            final_output_file = os.path.join(
                output_dir, dataset_name, identifier + f"_scores.npy"
            )
            np.save(final_output_file, combined_scores)
            logger.info(f"Saved combined scores to {final_output_file}")


if __name__ == "__main__":
    main()
