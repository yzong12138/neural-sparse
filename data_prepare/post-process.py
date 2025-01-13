import os
import datasets
from scripts.data.data_utils import get_corpus, cached
from tqdm import tqdm


@cached
def get_corpus_id_to_text(source, dataset, concat_title=True):
    corpus = get_corpus(source, dataset)
    id_to_text = {}
    for data in tqdm(corpus):
        text = data["title"] + " " + data["text"] if concat_title else data["text"]
        id_to_text[data["docid"]] = text
    return id_to_text


def trans(sample):
    sample.pop("query_id")
    sample.pop("first_rank")
    sample["docs"] = [corpus_id_to_text[hit] for hit in sample.pop("hits")]
    return sample


raw_dir = "raw-v3"
target_dir = "kd-v3"
all_source = os.listdir(f"mdata/{raw_dir}")

for source in all_source:
    all_data = os.listdir(f"mdata/{raw_dir}/{source}")

    for dataset in all_data:
        ds = datasets.Dataset.load_from_disk(f"mdata/{raw_dir}/{source}/{dataset}")
        print(
            dataset,
            sum([1 if r == -2 or r > 20 else 0 for r in ds["first_rank"]]),
            sum([1 if r != -1 else 0 for r in ds["first_rank"]]),
        )
        corpus_id_to_text = get_corpus_id_to_text(source, dataset)
        ds = ds.map(trans, num_proc=30)
        ds.save_to_disk(f"mdata/{target_dir}/{source}-{dataset}")

# dedup
from datasets import Dataset, concatenate_datasets

language_map = {
    "english": "en",
    "arabic": "ar",
    "bengali": "bn",
    "finnish": "fi",
    "indonesian": "id",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
    "swahili": "sw",
    "telugu": "te",
}

for tydi_lang, miracl_lang in language_map.items():
    tydi_dataset = Dataset.load_from_disk(f"mdata/{target_dir}/tydi-{tydi_lang}")
    miracl_dataset = Dataset.load_from_disk(f"mdata/{target_dir}/miracl-{miracl_lang}")

    duplicate_query_set = set(tydi_dataset["query"]) & set(miracl_dataset["query"])
    filtered_tydi = tydi_dataset.filter(lambda x: x["query"] not in duplicate_query_set)
    combined_dataset = concatenate_datasets([miracl_dataset, filtered_tydi])

    print(
        miracl_lang,
        len(tydi_dataset),
        len(miracl_dataset),
        len(duplicate_query_set),
        len(combined_dataset),
    )
    combined_dataset.save_to_disk(f"mdata/{target_dir}-dedup/{miracl_lang}")

for lang in "es fa fr zh hi".split(" "):
    os.system(f"cp -r mdata/{target_dir}/miracl-{lang} mdata/{target_dir}-dedup/{lang}")
