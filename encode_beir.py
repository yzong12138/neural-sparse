import os
import sys
import logging
import pickle
import random
import json

from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from beir.datasets.data_loader import GenericDataLoader
from scripts.data.dataset import BEIRCorpusDataset
from scripts.model.sparse_encoders import SparseModel, SparseEncoder


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


beir_data_path = "data/beir"
dataset = "msmarco"
split = "train"
model_path = "models/SDD-001"
batch_size = 100
save_path = "encoding/sdd-001"
max_length = 256
use_l0 = False


def main():
    model = SparseModel(model_path, use_l0=use_l0)

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.prepare(model)
    sparse_encoder = SparseEncoder(
        sparse_model=model,
        max_length=max_length,
        do_count=False,
    )

    data_path = os.path.join(beir_data_path, dataset)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    ddp_dataset = DDPDatasetWithRank(
        BEIRCorpusDataset(corpus=corpus),
        accelerator.local_process_index,
        accelerator.num_processes,
    )

    dataloader = DataLoader(ddp_dataset, batch_size=batch_size)
    os.makedirs(save_path, exist_ok=True)
    with open(
        os.path.join(save_path, f"{accelerator.local_process_index}.json"), "w"
    ) as wf:
        for ids, texts in tqdm(dataloader):
            output = sparse_encoder.encode(texts)
            for _id, sv in zip(ids, output):
                wf.write(json.dumps({"id": _id, "sparse_embedding": sv}))
                wf.write("\n")

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        command = f"cat {save_path}/[0-9]*.json > {save_path}/{dataset}.json && rm {save_path}/[0-9]*.json"
        os.system(command)


if __name__ == "__main__":
    main()
