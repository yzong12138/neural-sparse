import sys
import os

from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments

beir_datasets = "trec-covid,nfcorpus,nq,hotpotqa,fiqa,arguana,webis-touche2020,dbpedia-entity,scidocs,fever,climate-fever,scifact,quora"


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=512)
    train_file: Optional[str] = field(default=None)
    beir_dir: str = field(default="data/beir")
    # dataset name split by comma
    beir_datasets: str = field(default=beir_datasets)
    training_mode: str = field(default="kd-marco")
    kd_sample_num: int = field(default=2)
    use_in_batch_negatives: bool = field(default=False)
    flops_d_lambda: float = field(default=1e-3)
    flops_d_T: float = field(default=10000)
    flops_q_lambda: float = field(default=None)
    flops_q_T: float = field(default=None)
    ranking_loss_weight: float = field(default=1)

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


def parse_args():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    return model_args, data_args, training_args
