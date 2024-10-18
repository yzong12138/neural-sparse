import argparse
import os
import pickle
import torch
import transformers

from scripts.data.dataset import DDPDatasetWithRank, MiraclCorpusDataset
from scripts.data.data_utils import get_corpus, get_queries
from scripts.train.bi_encoder_wrapper import BiEncoderWrapper

from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset, load_dataset
from tqdm import tqdm


class ScorerDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, samples):
        assert isinstance(corpus, Dataset)
        assert isinstance(samples, Dataset)
        id_to_text = {}
        for data in tqdm(corpus):
            id_to_text[data["docid"]] = data["title"] + " " + data["text"]
        self.corpus = id_to_text
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample["query_id"],
            sample["query"],
            [self.corpus[hit] for hit in sample["hits"]],
        )


def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument("--datasets", default="zh", help="Dataset name")
    parser.add_argument("--source", default="miracl", help="Data source")
    parser.add_argument(
        "--max_length", type=int, default=256, help="Maximum length of input sequences"
    )
    parser.add_argument("--model_id", default="BAAI/bge-m3", help="Model ID")
    parser.add_argument("--model_type", default="dense", help="Model type")
    args = parser.parse_args()
    print(args)

    datasets = args.datasets
    source = args.source
    max_length = args.max_length
    model_id = args.model_id
    model_type = args.model_type

    model = BiEncoderWrapper.CLS_MAP[model_type].from_pretrained(model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    accelerator = Accelerator(mixed_precision="fp16")
    model = accelerator.prepare(model)
    accelerator.wait_for_everyone()
    for dataset in datasets:
        samples = Dataset.load_from_disk(f"mdata/raw/{source}/{dataset}")
        corpus = get_corpus(source, dataset)
        ddp_dataset = DDPDatasetWithRank(
            ScorerDataset(corpus, samples),
            accelerator.local_process_index,
            accelerator.num_processes,
        )

        output_dir = f"mdata/scores/{model_id.split('/')[-1]}/{args.dataset}/"
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, f"{accelerator.local_process_index}.pkl"), "wb"
        ) as f:
            for qid, query, docs in tqdm(ddp_dataset):
                features = tokenizer(
                    [query] + docs,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )

                features = {k: v.to(accelerator.device) for k, v in features.items()}
                with torch.no_grad():
                    reps = model(**features)
                    scores = torch.matmul(reps[0], reps[1:].t())
                data = (qid, scores.cpu().numpy().tolist())
                pickle.dump(data, f)


if __name__ == "__main__":
    main()
