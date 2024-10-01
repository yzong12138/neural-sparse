import os
import json
import datasets
import argparse
from scripts.utils import get_logger_console

logger = get_logger_console(__name__)

split = "dev"
tydi_lang_map = {}

parser = argparse.ArgumentParser()
parser.add_argument("lang", type=str, help="Language code")
parser.add_argument("--source", type=str, default="miracl")

args = parser.parse_args()

lang = args.lang
source = args.source
logger.info(f"The provided language code is: {lang}. Source: {source}")

dataset_source = "miracl/miracl"
if source == "tydi":
    dataset_source = "castorini/mr-tydi"


miracl = datasets.load_dataset(dataset_source, lang, trust_remote_code=True)

dataset = miracl[split]
qrels = {}
for data in dataset:
    qrels[data["query_id"]] = {
        pos_doc["docid"]: 1 for pos_doc in data["positive_passages"]
    }

queries = {}
for data in dataset:
    queries[data["query_id"]] = data["query"]

os.makedirs(f"mdata/miracl_eval/{lang}/", exist_ok=True)
with open(f"mdata/miracl_eval/{lang}/qrels.json", "w") as f:
    json.dump(qrels, f)
with open(f"mdata/miracl_eval/{lang}/queries.json", "w") as f:
    json.dump(queries, f)

miracl_corpus = datasets.load_dataset(
    "miracl/miracl-corpus", lang, trust_remote_code=True
)["train"]
logger.info(f"finish {lang}")
