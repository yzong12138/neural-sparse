import pickle
import logging
import argparse
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from scripts.dynamo_utils import batch_store_vectors_binary


def get_logger_console(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Upload vector data to DynamoDB")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="/opt/dlami/nvme/bge/docs/7.bin",
        help="Path to the binary file containing vectors",
    )
    parser.add_argument(
        "--table", "-t", type=str, default="vector", help="DynamoDB table name"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    file_name = args.file
    table_name = args.table

    logger = get_logger_console(__name__)

    l = []
    with open(file_name, "rb") as f:
        while True:
            try:
                l.append(pickle.load(f))
            except EOFError:
                break

    logger.info(f"{file_name}: total lines {len(l)}")

    for data in tqdm(l):
        vector_ids, vectors = data
        batch_store_vectors_binary(table_name, vector_ids, vectors)


if __name__ == "__main__":
    main()
