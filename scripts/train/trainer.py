import os
import logging
import torch

from ..model.sparse_encoders import SparseModel

from transformers import Trainer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    def __init__(self, sparse_model, inf_free=True):
        super().__init__()
        self.sparse_model = sparse_model
        self.inf_free = inf_free

    def forward(self, inputs):
        d_rep = self.sparse_model(
            inf_free=False,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        q_rep = self.sparse_model(
            inf_free=self.inf_free,
            input_ids=inputs["q_input_ids"],
            attention_mask=inputs["q_attention_mask"],
        )
        return d_rep, q_rep

    def save(self, output_dir, **kwargs):
        self.sparse_model.backbone.save_pretrained(output_dir, **kwargs)
        self.sparse_model.tokenizer.save_pretrained(output_dir)


class SparseModelTrainer(Trainer):
    def __init__(self, model_args, data_args, loss_functions, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.loss_functions = loss_functions
        self.ranking_loss_moving_avg = 0
        kwargs["model"] = ModelWrapper(kwargs["model"])
        super().__init__(**kwargs)

    def flops_value(self, representation, group_num=1):
        # representation size: batch_size * vocab_dim
        # if ddp: all_representation size: n * batch_size * vocab_dim
        # we need to take the representations from other devices into calculation
        # but detach their grad. We only consider the grad for representation at local device
        # note representation is one element of all_representations
        # group num: how many semantic similar documents representations in one batch
        if self.accelerator.num_processes > 1:
            all_representation = self.accelerator.gather(representation)
            batch_size = representation.shape[0]
            local_index = self.accelerator.local_process_index
            base_representation = (
                torch.cat(
                    [
                        all_representation[: local_index * batch_size],
                        all_representation[local_index * batch_size + batch_size :],
                    ]
                )
                .detach()
                .clone()
            )
            representation = torch.cat([base_representation, representation])
            representation = representation.reshape(
                -1, group_num, representation.shape[-1]
            )
            return torch.sum(torch.mean(torch.abs(representation), dim=0) ** 2)*self.accelerator.num_processes
        else:
            representation = representation.reshape(
                -1, group_num, representation.shape[-1]
            )
            return torch.sum(torch.mean(torch.abs(representation), dim=0) ** 2)

    def get_lambda(self, lambda_value, lambda_T):
        if self.state.global_step >= lambda_T:
            return lambda_value
        step = self.state.global_step + 1
        return lambda_value * (step / lambda_T) ** 2

    def compute_loss(self, model: SparseModel, inputs, return_outputs=False):
        flops_loss = 0

        # we construct this input and model wrapper to make it work for both data parallel and DDP
        model_wrapper_input = {
            "q_input_ids": inputs["query"]["input_ids"],
            "q_attention_mask": inputs["query"]["attention_mask"],
            "input_ids": inputs["docs"]["input_ids"],
            "attention_mask": inputs["docs"]["attention_mask"],
        }

        d_rep, q_rep = model(model_wrapper_input)
        d_flops = self.flops_value(d_rep, d_rep.shape[0] // q_rep.shape[0])
        flops_loss += d_flops * self.get_lambda(
            self.data_args.flops_d_lambda, self.data_args.flops_d_T
        )

        if not self.model_args.inf_free:
            flops_loss += self.flops_value(q_rep) * self.get_lambda(
                self.data_args.flops_q_lambda, self.data_args.flops_q_T
            )

        ranking_loss = 0
        for loss_function in self.loss_functions:
            ranking_loss += loss_function.get_loss(
                q_rep=q_rep, d_rep=d_rep, inputs=inputs, accelerator=self.accelerator
            )
        self.ranking_loss_moving_avg = (
            0.01 * ranking_loss.item() + 0.99 * self.ranking_loss_moving_avg
        )

        loss = ranking_loss + flops_loss
        outputs = {
            "q_rep": q_rep,
            "d_rep": d_rep,
        }

        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(
                f"ranking loss moving avg:{self.ranking_loss_moving_avg}, d_flops: {d_flops}, flops_loss: {flops_loss} avg doc length: {(d_rep>0).sum()/d_rep.shape[0]}"
            )
            
        logger.info(
                f"step :{self.state.global_step}, d rep: {self.accelerator.gather(d_rep).sum():.4f}, {self.model.sparse_model.backbone.bert.embeddings.word_embeddings.weight.sum():.4f}"
            )

        assert 0

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        # self.accelerator.wait_for_everyone()

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        return self.accelerator.prepare(DataLoader(train_dataset, batch_size=self._train_batch_size, shuffle=False,
                                                   collate_fn=data_collator))