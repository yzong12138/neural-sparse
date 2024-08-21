import os
import logging
import torch

from .loss import FLOPS
from ..model.sparse_encoders import SparseModel

from transformers import Trainer

logger = logging.getLogger(__name__)

class ModelWrapper(torch.nn.Module):
    def __init__(self, sparse_model, inf_free=True):
        super().__init__()
        self.sparse_model = sparse_model
        self.inf_free = inf_free
        
    def forward(self, inputs):
        d_rep = self.sparse_model(inf_free=False, **inputs["docs"])
        q_rep = self.sparse_model(inf_free=self.inf_free, **inputs["query"])
        return d_rep, q_rep
    
    def save(self, output_dir, **kwargs):
        self.sparse_model.backbone.save_pretrained(
            output_dir,
            **kwargs
        )
        self.sparse_model.tokenizer.save_pretrained(output_dir)
        

class SparseModelTrainer(Trainer):
    def __init__(self, model_args, data_args, loss_functions, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.loss_functions = loss_functions
        kwargs["model"] = ModelWrapper(kwargs["model"])
        super().__init__(**kwargs)

    def compute_loss(self, model: SparseModel, inputs, return_outputs=False):
        flops_loss = 0

        d_rep, q_rep = model(inputs)
        flops_loss += FLOPS.flops_value(d_rep) * FLOPS.get_lambda(
            self.data_args.flops_d_lambda,
            self.data_args.flops_d_T,
            self.state.global_step,
        )
        
        if not self.model_args.inf_free:
            flops_loss += FLOPS.flops_value(q_rep) * FLOPS.get_lambda(
                self.data_args.flops_q_lambda,
                self.data_args.flops_q_T,
                self.state.global_step,
            )

        ranking_loss = 0
        for loss_function in self.loss_functions:
            ranking_loss += loss_function.get_loss(
                q_rep=q_rep, d_rep=d_rep, inputs=inputs
            )

        loss = ranking_loss + flops_loss
        outputs = {
            "q_rep": q_rep,
            "d_rep": d_rep,
        }

        logger.info(
            f"input_shape:{q_rep.shape, d_rep.shape} ranking loss:{ranking_loss}, flops_loss:{flops_loss}"
        )

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.accelerator.unwrap_model(self.model).save(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
        )
