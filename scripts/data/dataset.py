import json
import logging
import random

from tqdm import tqdm
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class KeyValueDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): The input data dictionary where the keys are IDs and values are the content.
        """
        self.keys = sorted(data_dict.keys())
        self.data = {key: data_dict[key] for key in self.keys}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        value = self.data[key]
        return key, value


class BEIRCorpusDataset(KeyValueDataset):
    def __init__(self, corpus):
        """
        Args:
            corpus (dict): The corpus dictionary where each key is a document ID, and the value is a dictionary with "title" and "text".
        """
        # Filter out corpus entries where both title and text are empty
        filtered_corpus = {
            key: value
            for key, value in corpus.items()
            if value["title"].strip() != "" or value["text"].strip() != ""
        }

        # Combine the title and text into a single string
        combined_corpus = {
            key: (value["title"] + " " + value["text"]).strip()
            for key, value in filtered_corpus.items()
        }

        # Initialize the base class with the combined corpus
        super().__init__(combined_corpus)


class DDPDatasetWithRank(Dataset):
    def __init__(self, inner_dataset, local_rank, world_size):
        self.inner_dataset = inner_dataset
        number_of_samples = len(self.inner_dataset)
        self.idxs = [
            i for i in range(number_of_samples) if i % world_size == local_rank
        ]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.inner_dataset[self.idxs[idx]]


class KnowledgeDistillDataset(Dataset):
    def __init__(self, all_data, sample_num=2):
        """
        Args:
            data: A list of dictionary which have "query", "docs" and "scores" list.
                An example: {"query":"hi","docs":["hello","hi"],"scores":[0.99,1]}
        """
        self.data = []
        logger.info(f"KnowledgeDistillDataset input data number: {len(all_data)}")
        assert sample_num >= 2

        for data in all_data:
            query = data["query"]
            texts = data["docs"]
            scores = data["scores"]
            assert len(texts) == len(scores)

            idxs = list(range(len(scores)))
            random.shuffle(idxs)
            for i in range(0, len(scores), sample_num):
                # we need to make sure all samples have same negative number
                # to split all documents and perform matmul
                if len(scores) - i < sample_num:
                    break
                sample_idxs = idxs[i : i + sample_num]
                self.data.append(
                    [
                        query,
                        [texts[idx] for idx in sample_idxs],
                        [scores[idx] for idx in sample_idxs],
                    ]
                )
        logger.info(
            f"KnowledgeDistillDataset after process data number: {len(self.data)}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MsMarcoKDDataset(KnowledgeDistillDataset):
    @staticmethod
    def transform_str(s):
        try:
            return s.encode("latin1").decode("utf-8")
        except:
            return s

    def __init__(self, score_dic_path, corpus=None, queries=None, sample_num=2):
        if corpus is None or queries is None:
            assert score_dic_path is not None
            beir_resource_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
            local_data_path = util.download_and_unzip(beir_resource_url, "data/beir/")
            corpus, queries, qrels = GenericDataLoader(
                data_folder=local_data_path
            ).load(split="train")

        with open(score_dic_path) as f:
            score_dic = json.load(f)

        for k, v in tqdm(corpus.items()):
            corpus[k]["text"] = MsMarcoKDDataset.transform_str(v["text"])

        all_data = []
        for q_id in tqdm(score_dic.keys()):
            doc_ids = score_dic[q_id]["doc_id"]
            scores = score_dic[q_id]["score"]
            texts = [corpus[doc_id]["text"] for doc_id in doc_ids]
            data = {"query": queries[q_id], "docs": texts, "scores": scores}
            all_data.append(data)

        super().__init__(all_data=all_data, sample_num=sample_num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PosNegsDataset(Dataset):
    def __init__(self, data, neg_sample_num=3):
        """
        Args:
            data: A list of dictionary which have "query", "pos" and "negs" list.
                An example: {"query":"hi","pos":"hello","negs":["hi","world"]}
                "negs" is optional
        """
        assert neg_sample_num >= 1
        self.data = []
        logger.info(f"input data number: {len(data)}")
        for dict_ in data:
            dict_["negs"] = dict_["negs"]
            for i in range(0, len(dict_.get("negs", [])), neg_sample_num):
                if len(dict_["negs"]) - i < neg_sample_num:
                    break
                self.data.append(
                    [
                        dict_["query"],
                        dict_["pos"],
                        dict_.get("negs", [])[i : i + neg_sample_num],
                    ]
                )
        logger.info(f"after process data number: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
