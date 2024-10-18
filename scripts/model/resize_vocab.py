import torch
import torch.nn.functional as F

from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import DistilBertConfig, DistilBertForMaskedLM
from transformers.models.distilbert.modeling_distilbert import *


class ResizeBertConfig(DistilBertConfig):
    model_type = "ResizeBert"

    def __init__(self, vocab_resize=10000, **kwargs):
        super().__init__(**kwargs)
        self.vocab_resize = vocab_resize


class ResizeBertForMaskedLM(DistilBertForMaskedLM):
    config_class = ResizeBertConfig

    def __init__(self, config):
        super(DistilBertForMaskedLM, self).__init__(config)

        self.activation = get_activation(config.activation)

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.vocab_projector_resize = nn.Linear(config.dim, config.vocab_resize)
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()
        print("resize vocab for distill bert", self.vocab_projector_resize.weight.shape)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(
            prediction_logits
        )  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector_resize(
            prediction_logits
        )  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(
                prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )
