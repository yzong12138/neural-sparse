from dataclasses import dataclass, field, asdict
from typing import Optional

beir_datasets = "trec-covid,nfcorpus,nq,hotpotqa,fiqa,arguana,webis-touche2020,dbpedia-entity,scidocs,fever,climate-fever,scifact,quora"


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=256)
    train_file: Optional[str] = field(default=None)
    beir_dir: str = field(default="data/beir")
    # dataset name split by comma
    beir_datasets: str = field(default="nfcorpus,trec-covid")

    def __post_init__(self):
        return


@dataclass
class ModelArguments:
    inf_free: bool = field(default=True)
    model_name_or_path: str = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    idf_path: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path
        return
