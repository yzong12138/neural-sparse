import yaml
import logging
import os
import json
import sys
from dataclasses import asdict

from scripts.train.trainer import SparseModelTrainer
from scripts.train.loss import KLDivLoss, InfoNCELoss, MarginMSELoss
from scripts.data.dataset import (
    KnowledgeDistillDataset,
    MsMarcoKDDataset,
    PosNegsDataset,
    MiraclTrainingDataset,
    CombinedDataset,
    CombinedRandomSampler,
    DDPDatasetWithRank,
    get_miracl_datasets,
)
from scripts.data.collator import KnowledgeDistillDataCollator, PosNegsDataCollator
from scripts.utils import set_logging, get_model
from scripts.args import parse_args, miracl_datasets, tydi_datasets

from datasets import Dataset
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

    datasets = []
    datasets = datasets + get_miracl_datasets(
        "miracl",
        miracl_datasets,
        data_args.sample_num_one_query,
        data_args.miracl_from_disk,
    )
    datasets = datasets + get_miracl_datasets(
        "tydi",
        tydi_datasets,
        data_args.sample_num_one_query,
        data_args.miracl_from_disk,
    )

    datasets = [
        DDPDatasetWithRank(
            dataset,
            training_args.local_process_index,
            training_args.world_size,
            drop=True if training_args.world_size != 1 else False,
            shuffle=True if training_args.world_size != 1 else False,
        )
        for dataset in datasets
    ]
    dataset = CombinedDataset(datasets)

    if data_args.training_mode.startswith("kd"):
        if data_args.training_mode.split("-")[-1] == "ensemble":
            if data_args.miracl_from_disk:
                logger.info("knowledge distill data collator")
                data_collator = KnowledgeDistillDataCollator(
                    model.tokenizer,
                    data_args.max_seq_length,
                    teacher_tokenizer_ids=data_args.kd_ensemble_teacher_kwargs.get(
                        "model_ids", []
                    ),
                )
            else:
                data_collator = PosNegsDataCollator(
                    model.tokenizer,
                    data_args.max_seq_length,
                    teacher_tokenizer_ids=data_args.kd_ensemble_teacher_kwargs.get(
                        "model_ids", []
                    ),
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

        for loss_type in data_args.kl_div_loss:
            loss_cls = KLDivLoss if loss_type == "kldiv" else MarginMSELoss
            logger.info(loss_cls)
            loss_functions.append(
                loss_cls(
                    use_in_batch_negatives=data_args.use_in_batch_negatives,
                    weight=data_args.ranking_loss_weight,
                )
            )

    elif data_args.training_mode.startswith("infonce"):
        data_collator = PosNegsDataCollator(model.tokenizer, data_args.max_seq_length)

        loss_functions.append(
            InfoNCELoss(
                weight=data_args.ranking_loss_weight,
                use_in_batch_negatives=data_args.use_in_batch_negatives,
            )
        )

    # split to eval
    if model_args.idf_requires_grad == False or data_args.idf_lr is None:
        optimizer = AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
    else:
        idf_vector_params = [model.idf_vector]
        other_params = [
            param for param in model.parameters() if param is not model.idf_vector
        ]
        param_groups = [
            {"params": idf_vector_params, "lr": data_args.idf_lr},
            {"params": other_params, "lr": training_args.learning_rate},
        ]
        optimizer = AdamW(param_groups, weight_decay=training_args.weight_decay)
        logger.info(f"idf_vector lr: {data_args.idf_lr}")

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
