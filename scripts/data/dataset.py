from torch.utils.data import Dataset


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
