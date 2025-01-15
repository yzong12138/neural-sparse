import os
import json
import datasets

from tqdm import tqdm


class MmarcoDataParser:
    def __init__(self, lang, root_dir="mdata/bge-m3-data"):
        self.lang = lang
        self.root_dir = root_dir

    def load_data(self):
        self.triplets_ds = datasets.load_dataset(
            "unicamp-dl/mmarco", self.lang, trust_remote_code=True
        )["train"]
        self.queries_ds = datasets.load_dataset(
            "unicamp-dl/mmarco", f"queries-{self.lang}", trust_remote_code=True
        )["train"]
        self.collection_ds = datasets.load_dataset(
            "unicamp-dl/mmarco", f"collection-{self.lang}", trust_remote_code=True
        )["collection"]

    def build_dataset(self):
        corpus_id_to_text = {}
        corpus_text_to_id = {}
        queries_id_to_text = {}
        queries_text_to_id = {}
        query_pos_docs = {}
        for data in tqdm(self.collection_ds):
            _id = str(data["id"])
            text = str(data["text"])
            corpus_id_to_text[_id] = text
            corpus_text_to_id[text] = _id
        for data in tqdm(self.queries_ds):
            _id = str(data["id"])
            text = str(data["text"])
            queries_id_to_text[_id] = text
            queries_text_to_id[text] = _id
        for data in tqdm(self.triplets_ds):
            q_id = queries_text_to_id[data["query"]]
            pos_id = corpus_text_to_id[data["positive"]]
            if q_id not in query_pos_docs:
                query_pos_docs[q_id] = []
            if pos_id not in query_pos_docs[q_id]:
                query_pos_docs[q_id].append(pos_id)

        for key in list(queries_id_to_text.keys()):
            if key not in query_pos_docs:
                queries_id_to_text.pop(key)

        self.corpus = corpus_id_to_text
        self.queries = queries_id_to_text
        self.query_pos_docs = query_pos_docs

    def save(self):
        os.makedirs(
            os.path.join(self.root_dir, "parsed", "mmarco-" + self.lang), exist_ok=True
        )
        with open(
            os.path.join(self.root_dir, "parsed", "mmarco-" + self.lang, "corpus.json"),
            "w",
        ) as f:
            json.dump(self.corpus, f)
        with open(
            os.path.join(
                self.root_dir, "parsed", "mmarco-" + self.lang, "queries.json"
            ),
            "w",
        ) as f:
            json.dump(self.queries, f)
        with open(
            os.path.join(
                self.root_dir, "parsed", "mmarco-" + self.lang, "query_pos_docs.json"
            ),
            "w",
        ) as f:
            json.dump(self.query_pos_docs, f)


parser = MmarcoDataParser("chinese")
parser.load_data()
parser.build_dataset()
parser.save()

parser = MmarcoDataParser("french")
parser.load_data()
parser.build_dataset()
parser.save()
