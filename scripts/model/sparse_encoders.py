import torch
import torch.nn.functional as F
import transformers
import itertools


class SparseModel(torch.nn.Module):
    def __init__(self, model_id, idf=None, tokenizer_id=None):
        super().__init__()

        if tokenizer_id is None:
            tokenizer_id = model_id
        self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(model_id)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
        self.special_token_ids = [
            self.tokenizer.vocab[token]
            for token in self.tokenizer.special_tokens_map.values()
        ]

        idf_vector = [1] * self.tokenizer.vocab_size
        if idf is not None:
            for token, weight in idf.items():
                _id = self.tokenizer._convert_token_to_id_with_added_voc(token)
                idf_vector[_id] = weight
        self.idf_vector = torch.nn.Parameter(
            torch.tensor(idf_vector), requires_grad=False
        )

    def forward(self, **kwargs):
        # input kwargs is the features from tokenizer
        output = self.backbone(**kwargs)[0]
        values, _ = torch.max(
            output * kwargs.get("attention_mask").unsqueeze(-1), dim=1
        )
        values = torch.log(1 + torch.relu(values))
        values[:, self.special_token_ids] = 0
        return values

    def encode(self, **kwargs):
        return self.forward(**kwargs)

    def encode_inf_free(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0]
        out = torch.zeros(
            batch_size, self.tokenizer.vocab_size, device=input_ids.device
        )
        out[torch.arange(batch_size).unsqueeze(-1), input_ids] = 1
        out[:, self.special_token_ids] = 0
        return out * self.idf_vector


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
    def __init__(self, sparse_model, max_length, device, do_count=True):
        self.model = sparse_model
        self.tokenizer = sparse_model.tokenizer
        self.post_processor = SparsePostProcessor(tokenizer=sparse_model.tokenizer)
        self.do_count = do_count
        self.max_length = max_length
        self.device = device
        self.count_tensor = torch.zeros(self.tokenizer.vocab_size).to(self.device)

    def reset_count(self):
        self.count_tensor = torch.zeros(self.tokenizer.vocab_size).to(self.device)

    def accelerator_prepare(self, accelerator):
        accelerator.prepare(self.model)
        return

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
        if inf_free:
            output = self.model.encode_inf_free(**features)
        else:
            with torch.no_grad():
                output = self.model(**features)
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
