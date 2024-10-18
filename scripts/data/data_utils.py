from datasets import load_dataset


def get_corpus(source, dataset):
    if source == "miracl" or source == "tydi":
        dataset = load_dataset(
            "miracl/miracl-corpus"
            if source == "miracl"
            else "castorini/mr-tydi-corpus",
            dataset,
            trust_remote_code=True,
            split="train",
        )
        return dataset


def get_queries(source, dataset):
    # return a tuple of (dict<query_id, query_text> and dict<query_id, list<pos_doc_id>>)
    queries = {}
    query_pos_docs = {}

    if source == "miracl" or source == "tydi":
        datasets = load_dataset(
            "miracl/miracl" if source == "miracl" else "castorini/mr-tydi",
            dataset,
            trust_remote_code=True,
        )
        for key, dataset in datasets.items():
            if key == "dev":
                continue
            for data in dataset:
                if data["query_id"] in queries:
                    print(key, data["query_id"])
                queries[data["query_id"]] = data["query"]
                query_pos_docs[data["query_id"]] = [
                    doc["docid"] for doc in data.get("positive_passages", [])
                ]

        return queries, query_pos_docs
