import yaml
import logging
import os
import json
import sys
from dataclasses import asdict

from scripts.train.trainer import SparseModelTrainer
from scripts.train.loss import KLDivLoss, InfoNCELoss
from scripts.data.dataset import (
    KnowledgeDistillDataset,
    MsMarcoKDDataset,
    PosNegsDataset,
)
from scripts.data.collator import KnowledgeDistillDataCollator, PosNegsDataCollator
from scripts.utils import set_logging, get_model
from scripts.args import parse_args

from transformers import set_seed
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
            # with open(data_args.train_file) as f:
            #     data = json.load(f)
            # dataset = KnowledgeDistillDataset(
            #     data, sample_num=data_args.sample_num_one_query
            # )
            dataset = MsMarcoKDDataset(
                score_dic_path=data_args.train_file,
                sample_num=data_args.sample_num_one_query,
            )
            data_collator = KnowledgeDistillDataCollator(
                model.tokenizer, data_args.max_seq_length
            )
        elif data_args.training_mode.split("-")[-1] == "ensemble":
            with open(data_args.train_file) as f:
                data = f.readlines()
            data = [json.loads(line) for line in data]
            dataset = PosNegsDataset(
                data, neg_sample_num=data_args.sample_num_one_query
            )
            data_collator = PosNegsDataCollator(
                model.tokenizer, data_args.max_seq_length
            )
        else:
            with open(data_args.train_file) as f:
                data = f.readlines()
            data = [json.loads(line) for line in data]
            dataset = KnowledgeDistillDataset(
                data, sample_num=data_args.sample_num_one_query
            )
            data_collator = KnowledgeDistillDataCollator(
                model.tokenizer, data_args.max_seq_length
            )

        loss_functions.append(
            KLDivLoss(
                use_in_batch_negatives=data_args.use_in_batch_negatives,
                weight=data_args.ranking_loss_weight,
            )
        )

    elif data_args.training_mode.startswith("infonce"):
        with open(data_args.train_file) as f:
            data = f.readlines()
        data = [json.loads(line) for line in data]
        dataset = PosNegsDataset(data, neg_sample_num=data_args.sample_num_one_query)
        loss_functions.append(
            InfoNCELoss(
                weight=data_args.ranking_loss_weight,
                use_in_batch_negatives=data_args.use_in_batch_negatives,
            )
        )

        data_collator = PosNegsDataCollator(model.tokenizer, data_args.max_seq_length)

    # split to eval
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )

    trainer = SparseModelTrainer(
        model_args=model_args,
        data_args=data_args,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        loss_functions=loss_functions,
        optimizers=(optimizer, scheduler),
    )

    if data_args.training_mode.split("-")[-1] == "ensemble":
        trainer.set_bi_encoder_teacher()
    trainer.train()


if __name__ == "__main__":
    main()
