import torch
import torch.nn as nn
import transformers
import itertools
import logging

from .decomposition_bert import DecompBertConfig, DecompBertForMaskedLM
from ..args import QRepEnum
from transformers import AutoModelForMaskedLM, BertForMaskedLM, AutoConfig

logger = logging.getLogger(__name__)


class TokenWeightingModel(nn.Module):
    def __init__(
        self, backbone, hidden_dim=512, tie_weights=False, parameter_path=None
    ):
        super().__init__()

        logger.info(f"{hidden_dim}, {tie_weights}, {parameter_path}")

        origin_embedding = backbone.get_input_embeddings()
        if tie_weights:
            self.embedding = origin_embedding
        else:
            self.embedding = nn.Embedding(
                origin_embedding.weight.size(0), origin_embedding.weight.size(1)
            )
            self.embedding.weight.data.copy_(
                origin_embedding.weight.data.detach().clone()
            )

        self.fc1 = nn.Linear(backbone.config.dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.fc1.weight.data.normal_(mean=0.0, std=backbone.config.initializer_range)
        self.fc2.weight.data.normal_(mean=0.0, std=backbone.config.initializer_range)
        self.relu = nn.ReLU()

        if parameter_path is not None:
            state_dict = torch.load(parameter_path)
            self.load_state_dict(state_dict, strict=True)

        # self.weight_lower_bound = weight_lower_bound
        # self.weight_upper_bound = weight_upper_bound

    @property
    def token_weights(self):
        hidden = self.relu(self.fc1(self.embedding.weight))
        hidden = self.relu(self.fc2(hidden))
        weights = hidden.squeeze(-1)
        weights = weights / (weights.max().detach()/8)
        return weights
        # with torch.no_grad():
        #     min_weight = weights.min()
        #     max_weight = weights.max()

        # weights = ((weights - min_weight) / (max_weight - min_weight) * 9.9) + 0.1


class SparseModel(torch.nn.Module):
    def __init__(
        self,
        model_id,
        idf=None,
        tokenizer_id=None,
        split_batch=1,
        q_rep_strategy=QRepEnum.IDF.value,
        twm_args={},
    ):
        super().__init__()

        AutoConfig.register("DecompBert", DecompBertConfig)
        AutoModelForMaskedLM.register(DecompBertConfig, DecompBertForMaskedLM)

        if tokenizer_id is None:
            tokenizer_id = model_id
        self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(model_id)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
        self.special_token_ids = [
            self.tokenizer.vocab[token]
            for token in self.tokenizer.special_tokens_map.values()
        ]

        if q_rep_strategy in [QRepEnum.IDF.value, QRepEnum.LEARNED_IDF.value]:
            idf_vector = [1.0] * self.tokenizer.vocab_size
            if idf is not None:
                logger.info(f"set idf to the model. q_rep_strategy : {q_rep_strategy}")
                for token, weight in idf.items():
                    _id = self.tokenizer._convert_token_to_id_with_added_voc(token)
                    idf_vector[_id] = weight
            self.idf_vector = torch.nn.Parameter(
                torch.tensor(idf_vector),
                requires_grad=q_rep_strategy == QRepEnum.LEARNED_IDF.value,
            )
        elif q_rep_strategy == QRepEnum.LEARNED_WEIGHT.value:
            logger.info(f"q_rep_strategy : {q_rep_strategy}")
            self.token_weighting_model = TokenWeightingModel(self.backbone, **twm_args)

        self.split_batch = split_batch
        self.q_rep_strategy = q_rep_strategy

    def forward(self, inf_free=False, **kwargs):
        # input kwargs is the features from tokenizer
        if inf_free:
            return self._encode_inf_free(**kwargs)
        else:
            return self._encode(**kwargs)

    def _encode(self, **kwargs):
        if self.split_batch == 1:
            output = self.backbone(**kwargs)[0]
            values, _ = torch.max(
                output * kwargs.get("attention_mask").unsqueeze(-1), dim=1
            )
            values = torch.log(1 + torch.relu(values))
            return values

        batch_size = kwargs["input_ids"].size(0)
        split_sizes = [batch_size // self.split_batch] * self.split_batch
        remainder = batch_size % self.split_batch
        for i in range(remainder):
            split_sizes[i] += 1

        outputs = []
        start = 0
        assert isinstance(self.backbone, BertForMaskedLM)

        attention_mask = kwargs.get("attention_mask")
        output = self.backbone.bert(**kwargs)[0]
        output = self.backbone.cls.predictions.transform(output)
        for split_size in split_sizes:
            end = start + split_size
            values = self.backbone.cls.predictions.decoder(output[start:end])
            values, _ = torch.max(
                values * attention_mask[start:end].unsqueeze(-1), dim=1
            )
            outputs.append(values)
            start = end

        output = torch.cat(outputs, dim=0)
        output = torch.log(1 + torch.relu(output))
        return output

    def _encode_inf_free(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0]
        out = torch.zeros(
            batch_size, self.tokenizer.vocab_size, device=input_ids.device
        )
        out[torch.arange(batch_size).unsqueeze(-1), input_ids] = 1
        out[:, self.special_token_ids] = 0

        if self.q_rep_strategy == QRepEnum.LEARNED_WEIGHT.value:
            return out * torch.relu(self.token_weighting_model.token_weights)
        else:
            return out * torch.relu(self.idf_vector)


class SparsePostProcessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.id_to_token = ["" for i in range(tokenizer.vocab_size)]
        for token, _id in tokenizer.vocab.items():
            self.id_to_token[_id] = token

    def __call__(self, sparse_vector):
        sparse_vector[:, 0] = 1
        sample_indices, token_indices = torch.nonzero(sparse_vector, as_tuple=True)
        non_zero_values = sparse_vector[(sample_indices, token_indices)].tolist()
        number_of_tokens_for_each_sample = torch.bincount(sample_indices).cpu().tolist()
        tokens = [self.id_to_token[_id] for _id in token_indices.tolist()]

        output = []
        end_idxs = list(itertools.accumulate([0] + number_of_tokens_for_each_sample))
        for i in range(len(end_idxs) - 1):
            token_strings = tokens[end_idxs[i] : end_idxs[i + 1]]
            weights = non_zero_values[end_idxs[i] : end_idxs[i + 1]]
            output.append(dict(zip(token_strings[1:], weights[1:])))
        return output


class SparseEncoder:
    def __init__(self, sparse_model, max_length, do_count=True):
        self.model = sparse_model
        self.tokenizer = sparse_model.tokenizer
        self.post_processor = SparsePostProcessor(tokenizer=sparse_model.tokenizer)
        self.do_count = do_count
        self.max_length = max_length
        self.device = self.model.backbone.device
        self.count_tensor = torch.zeros(self.tokenizer.vocab_size).to(self.device)

    def reset_count(self):
        self.count_tensor = torch.zeros(self.tokenizer.vocab_size).to(self.device)

    def encode(self, texts, inf_free=False):
        features = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=self.max_length,
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        with torch.no_grad():
            output = self.model(inf_free=inf_free, **features)
        if self.do_count:
            self.count_tensor += (output > 0).long().sum(dim=0)
        output = self.post_processor(output)
        return output


def sparse_embedding_to_query(token_weight_map, field_name="text_sparse"):
    clause_list = []
    for token, weight in token_weight_map.items():
        clause_list.append(
            {
                "rank_feature": {
                    "field": f"{field_name}.{token}",
                    "boost": weight,
                    "linear": {},
                }
            }
        )
    return {"bool": {"should": clause_list}}
