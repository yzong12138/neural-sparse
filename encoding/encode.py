import os
import sys
import logging
import pickle
import random
import argparse

from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


def get_logger_console(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger_console(__name__)


class TextToIdDataset(Dataset):
    def __init__(self, text_to_id, from_id=-1):
        text_id_pairs = text_to_id.items()
        self.keys = sorted([_id for text, _id in text_id_pairs if _id >= from_id])
        self.data = {_id: text for text, _id in text_id_pairs if _id >= from_id}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        value = self.data[key]
        return key, value


class DDPDatasetWithRank(Dataset):
    def __init__(
        self, inner_dataset, local_rank, world_size, drop=False, shuffle=False
    ):
        self.inner_dataset = inner_dataset
        number_of_samples = len(self.inner_dataset)
        if drop:
            number_of_samples = number_of_samples - number_of_samples % world_size
        self.idxs = [
            i for i in range(number_of_samples) if i % world_size == local_rank
        ]
        if shuffle:
            state = random.getstate()
            random.seed(local_rank)
            random.shuffle(self.idxs)
            random.setstate(state)
        logger.info(
            f"local index {local_rank}, world size {world_size}, local sample number {len(self.idxs)}"
        )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.inner_dataset[self.idxs[idx]]


def get_model_config(model_type):
    """Return model-specific configuration based on model_type"""
    config = {
        "bge": {
            "model_id": "BAAI/bge-en-icl",
            "query_save_path": "/opt/dlami/nvme/bge-256/queries/",
            "doc_save_path": "/opt/dlami/nvme/bge-256/docs/",
            "module": "bge_en_icl"
        },
        "inf": {
            "model_id": "infly/inf-retriever-v1",
            "query_save_path": "/opt/dlami/nvme/inf-256/queries/",
            "doc_save_path": "/opt/dlami/nvme/inf-256/docs/",
            "module": "inf_retriever"
        },
        "zeta": {
            "model_id": "zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
            "query_save_path": "/opt/dlami/nvme/zeta/queries/",
            "doc_save_path": "/opt/dlami/nvme/zeta/docs/",
            "module": "zeta"
        }
    }
    return config[model_type]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate embeddings using specified model')
    parser.add_argument('--model_type', type=str, choices=['bge', 'inf', 'zeta'], 
                        required=True, help='Model type to use (bge, inf, or zeta)')
    parser.add_argument('--batch_size', type=int, default=30, 
                        help='Batch size for processing')
    parser.add_argument('--from_q_id', type=int, default=5604177, 
                        help='Starting query ID')
    parser.add_argument('--from_d_id', type=int, default=22527349, 
                        help='Starting document ID')
    parser.add_argument('--query_to_id_path', type=str, default="../data/query_to_id.bin", 
                        help='Path to query_to_id binary file')
    parser.add_argument('--doc_to_id_path', type=str, default="../data/doc_to_id.bin",
                        help='Path to doc_to_id binary file')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    model_config = get_model_config(args.model_type)
    model_id = model_config["model_id"]
    query_save_path = model_config["query_save_path"]
    doc_save_path = model_config["doc_save_path"]
    
    # Dynamically import the appropriate module
    module_name = model_config["module"]
    module = __import__(module_name, fromlist=['get_query_embeddings', 'get_document_embeddings'])
    get_query_embeddings = module.get_query_embeddings
    get_document_embeddings = module.get_document_embeddings
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.prepare(model)

    # Test the model with dummy input
    embedding = get_query_embeddings(
        ["hi " * 512] * args.batch_size, model=model, tokenizer=tokenizer
    )

    # Process queries
    with open(args.query_to_id_path, "rb") as f:
        query_to_id = pickle.load(f)

    ddp_dataset = DDPDatasetWithRank(
        TextToIdDataset(query_to_id, args.from_q_id),
        accelerator.local_process_index,
        accelerator.num_processes,
    )

    dataloader = DataLoader(ddp_dataset, batch_size=args.batch_size)
    os.makedirs(query_save_path, exist_ok=True)
    with open(
        os.path.join(query_save_path, f"{accelerator.local_process_index}.bin"), "wb"
    ) as f:
        for ids, texts in tqdm(dataloader):
            embedding = get_query_embeddings(texts, model=model, tokenizer=tokenizer)
            embedding = embedding.half().cpu().numpy()
            pickle.dump([ids.cpu().numpy().tolist(), embedding], f)

    # Process documents
    with open(args.doc_to_id_path, "rb") as f:
        doc_to_id = pickle.load(f)

    ddp_dataset = DDPDatasetWithRank(
        TextToIdDataset(doc_to_id, args.from_d_id),
        accelerator.local_process_index,
        accelerator.num_processes,
    )

    dataloader = DataLoader(ddp_dataset, batch_size=args.batch_size)
    os.makedirs(doc_save_path, exist_ok=True)
    with open(
        os.path.join(doc_save_path, f"{accelerator.local_process_index}.bin"), "wb"
    ) as f:
        for ids, texts in tqdm(dataloader):
            embedding = get_document_embeddings(texts, model=model, tokenizer=tokenizer)
            embedding = embedding.half().cpu().numpy()
            pickle.dump([ids.cpu().numpy().tolist(), embedding], f)


if __name__ == "__main__":
    main()
