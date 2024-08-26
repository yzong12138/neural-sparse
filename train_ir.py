import torch.utils
import torch.utils.data
import yaml
import logging
import os
import json
import random
import sys
import torch
from dataclasses import dataclass, field, asdict

from scripts.train.trainer import SparseModelTrainer
from scripts.train.loss import KLDivLoss
from scripts.data.dataset import KnowledgeDistillDataset, MsMarcoKDDataset
from scripts.data.collator import KnowledgeDistillDataCollator
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_os_client, batch_search, set_logging, get_model
from scripts.args import ModelArguments, DataTrainingArguments, parse_args

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = parse_args()
    os.makedirs(training_args.output_dir, exist_ok=True)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        os.system(
            f"""cp {sys.argv[1]} {os.path.join(training_args.output_dir, "train_config.yaml")}"""
        )
    else:
        args_dict = {
            "model_args": asdict(model_args),
            "data_args": asdict(data_args),
            "training_args": asdict(training_args),
        }
        with open(os.path.join(training_args.output_dir, "config.yaml"), "w") as file:
            yaml.dump(args_dict, file, sort_keys=False)

    set_logging(training_args, "train.log")
    set_seed(training_args.seed)

    model = get_model(model_args)
    loss_functions = []

    if data_args.training_mode.startswith("kd"):
        if data_args.training_mode.split("-")[-1] == "marco":
            dataset = MsMarcoKDDataset(
                score_dic_path=data_args.train_file,
                sample_num=data_args.kd_sample_num,
            )
            # with open("data/toy_kd.json") as f:
            #     data = json.load(f)
            # dataset = KnowledgeDistillDataset(data, sample_num=data_args.kd_sample_num)
            loss_functions.append(
                KLDivLoss(
                    in_batch_negatives=data_args.use_in_batch_negatives,
                    weight=data_args.ranking_loss_weight,
                )
            )

        data_collator = KnowledgeDistillDataCollator(
            model.tokenizer, data_args.max_seq_length
        )

    # split to eval
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=training_args.warmup_steps,
                                                num_training_steps=training_args.max_steps)

    trainer = SparseModelTrainer(
        model_args=model_args,
        data_args=data_args,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        loss_functions=loss_functions,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()


if __name__ == "__main__":
    main()
