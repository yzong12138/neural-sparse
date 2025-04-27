import os
import datasets
import pickle

query_to_id_path = "../data/query_to_id.bin"
doc_to_id_path = "../data/doc_to_id.bin"

with open(query_to_id_path, "rb") as f:
    query_to_id = pickle.load(f)

with open(doc_to_id_path, "rb") as f:
    doc_to_id = pickle.load(f)

for data_dir in ["../data/pt-kd/", "../data/beir-kd/"]:
    data_names = os.listdir(data_dir)
    for data_name in data_names:
        dataset_path = os.path.join(data_dir, data_name)
        ds = datasets.load_from_disk(dataset_path)
        data = ds.to_dict()
        q_id = [query_to_id[q] for q in data["query"]]
        d_ids = [[doc_to_id[d] for d in docs] for docs in data["docs"]]
        data["q_id"] = q_id
        data["d_ids"] = d_ids
        ds2 = datasets.Dataset.from_dict(data)

        write_path = os.path.join("../data/pt-kd-ids", data_name)
        ds2.save_to_disk(write_path)

        print(write_path)
