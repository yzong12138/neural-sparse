import os
import sys
import logging
import pickle
import random

from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


method = "inf"
max_len = 256
if method == "bge_en_icl":
    from bge_en_icl import get_query_embeddings, get_document_embeddings

    query_save_path = "/opt/dlami/nvme/bge-256/queries/"
    doc_save_path = "/opt/dlami/nvme/bge-256/docs/"
    model_id = "BAAI/bge-en-icl"
elif method == "inf":
    from inf_retriever import get_query_embeddings, get_document_embeddings

    query_save_path = "/opt/dlami/nvme/inf-256/queries/"
    doc_save_path = "/opt/dlami/nvme/inf-256/docs/"
    model_id = "infly/inf-retriever-v1"
elif method == "zeta":
    from zeta import get_query_embeddings, get_document_embeddings

    query_save_path = "/opt/dlami/nvme/zeta-256/queries/"
    doc_save_path = "/opt/dlami/nvme/zeta-256/docs/"
    model_id = "zeta-alpha-ai/Zeta-Alpha-E5-Mistral"


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
    def __init__(self, text_to_id, last_id=-1):
        text_id_pairs = text_to_id.items()
        self.keys = sorted([_id for text, _id in text_id_pairs if _id >= last_id])
        self.data = {_id: text for text, _id in text_id_pairs if _id >= last_id}

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


query_to_id_path = "../data/query_to_id.bin"
doc_to_id_path = "../data/doc_to_id.bin"
batch_size = 30
last_q_id = -1
last_d_id = -1


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.prepare(model)

    embedding = get_query_embeddings(
        ["hi " * 512] * batch_size, model=model, tokenizer=tokenizer
    )

    with open(query_to_id_path, "rb") as f:
        query_to_id = pickle.load(f)

    ddp_dataset = DDPDatasetWithRank(
        TextToIdDataset(query_to_id, last_q_id),
        accelerator.local_process_index,
        accelerator.num_processes,
    )

    dataloader = DataLoader(ddp_dataset, batch_size=batch_size)
    os.makedirs(query_save_path, exist_ok=True)
    with open(
        os.path.join(query_save_path, f"{accelerator.local_process_index}.bin"), "wb"
    ) as f:
        for ids, texts in tqdm(dataloader):
            embedding = get_query_embeddings(
                texts, model=model, tokenizer=tokenizer, query_max_len=max_len
            )
            embedding = embedding.half().cpu().numpy()
            pickle.dump([ids.cpu().numpy().tolist(), embedding], f)

    with open(doc_to_id_path, "rb") as f:
        doc_to_id = pickle.load(f)

    ddp_dataset = DDPDatasetWithRank(
        TextToIdDataset(doc_to_id, last_d_id),
        accelerator.local_process_index,
        accelerator.num_processes,
    )

    dataloader = DataLoader(ddp_dataset, batch_size=batch_size)
    os.makedirs(doc_save_path, exist_ok=True)
    with open(
        os.path.join(doc_save_path, f"{accelerator.local_process_index}.bin"), "wb"
    ) as f:
        for ids, texts in tqdm(dataloader):
            embedding = get_document_embeddings(
                texts, model=model, tokenizer=tokenizer, doc_max_len=max_len
            )
            embedding = embedding.half().cpu().numpy()
            pickle.dump([ids.cpu().numpy().tolist(), embedding], f)


if __name__ == "__main__":
    main()
