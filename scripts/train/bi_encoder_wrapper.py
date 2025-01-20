import torch
import torch.nn.functional as F
import transformers
from scripts.utils import gather_rep
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)


class SparseModel(torch.nn.Module):
    @staticmethod
    def from_pretrained(path):
        return SparseModel(path)

    def __init__(self, model_id):
        super().__init__()
        self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(model_id)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.special_token_ids = [
            self.tokenizer.vocab[token]
            for token in self.tokenizer.special_tokens_map.values()
        ]

    def forward(self, **kwargs):
        output = self.backbone(**kwargs)[0]
        values, _ = torch.max(
            output * kwargs.get("attention_mask").unsqueeze(-1), dim=1
        )
        values = torch.log(1 + torch.relu(values))
        values[:, self.special_token_ids] = 0
        return values


class DenseModel(torch.nn.Module):
    @staticmethod
    def from_pretrained(path):
        return DenseModel(path)

    @staticmethod
    def get_dense_embedding(output):
        sentence_embeddings = output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings

    def __init__(self, model_id):
        super().__init__()
        self.backbone = transformers.AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )

    def forward(self, **kwargs):
        output = self.backbone(**kwargs)
        return DenseModel.get_dense_embedding(output)


class BGEM3Model(torch.nn.Module):
    VOCAB_SIZE = 250002
    unused_tokens = [0, 1, 2, 3]
    split_batch = 6

    @staticmethod
    def from_pretrained(path):
        return BGEM3Model(path)

    @staticmethod
    def get_dense_embedding(hidden_state):
        return F.normalize(hidden_state[:, 0], dim=-1)

    @staticmethod
    def get_sparse_embedding(hidden_state, input_ids, sparse_linear):
        token_weights = torch.relu(sparse_linear(hidden_state))

        batch_size = input_ids.size(0)
        split_sizes = [batch_size // BGEM3Model.split_batch] * BGEM3Model.split_batch
        remainder = batch_size % BGEM3Model.split_batch
        for i in range(remainder):
            split_sizes[i] += 1
        outputs = []
        start = 0
        for split_size in split_sizes:
            end = start + split_size
            sparse_embedding = torch.zeros(
                split_size,
                input_ids.size(1),
                BGEM3Model.VOCAB_SIZE,
                dtype=token_weights.dtype,
                device=token_weights.device,
            )
            sparse_embedding = torch.scatter(
                sparse_embedding,
                dim=-1,
                index=input_ids[start:end].unsqueeze(-1),
                src=token_weights[start:end],
            )
            sparse_embedding = torch.max(sparse_embedding, dim=1).values
            sparse_embedding[:, BGEM3Model.unused_tokens] *= 0.0
            outputs.append(sparse_embedding)
            start = end

        output = torch.cat(outputs, dim=0)
        return output

    @staticmethod
    def get_colbert_embedding(hidden_state, mask, colbert_linear):
        colbert_vecs = colbert_linear(hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return F.normalize(colbert_vecs, dim=-1)

    def __init__(self, model_id):
        super().__init__()
        self.backbone = transformers.AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )

        sparse_cached_path = hf_hub_download(
            repo_id="BAAI/bge-m3", filename="sparse_linear.pt"
        )
        self.sparse_linear = torch.nn.Linear(in_features=1024, out_features=1)
        self.sparse_linear.load_state_dict(torch.load(sparse_cached_path), strict=True)

        colbert_cached_path = hf_hub_download(
            repo_id="BAAI/bge-m3", filename="colbert_linear.pt"
        )
        self.colbert_linear = torch.nn.Linear(in_features=1024, out_features=1024)
        self.colbert_linear.load_state_dict(
            torch.load(colbert_cached_path), strict=True
        )

    def forward(self, **kwargs):
        output = self.backbone(**kwargs)[0]
        return (
            BGEM3Model.get_dense_embedding(output),
            BGEM3Model.get_sparse_embedding(
                output, kwargs["input_ids"], self.sparse_linear
            ),
            BGEM3Model.get_colbert_embedding(
                output, kwargs["attention_mask"], self.colbert_linear
            ),
        )


class BiEncoderWrapper:
    CLS_MAP = {"sparse": SparseModel, "dense": DenseModel, "m3": BGEM3Model}

    def __init__(
        self,
        types,
        model_ids,
        score_scale=30,
        use_in_batch_negatives=False,
        weights=None,
    ):
        assert len(types) == len(model_ids)
        assert len(types) != 0
        self.score_scale = score_scale
        self.use_in_batch_negatives = use_in_batch_negatives
        self.models = []
        self.accelerator = None
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1] * len(model_ids)
        logger.info(self.weights)

        for i, (type, model_id) in enumerate(zip(types, model_ids)):
            cls = BiEncoderWrapper.CLS_MAP[type]
            model = cls.from_pretrained(model_id)
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def get_scores_batch(self, q_features_list, d_features_list):
        assert len(q_features_list) == len(self.models)
        scores = 0

        for i, model in enumerate(self.models):
            if not isinstance(self.accelerator.unwrap_model(model), BGEM3Model):
                q_rep = model(**q_features_list[i])
                d_rep = model(**d_features_list[i])
                if not self.use_in_batch_negatives:
                    d_rep = d_rep.reshape(q_rep.shape[0], -1, d_rep.shape[-1])
                    score = torch.bmm(
                        d_rep, q_rep.reshape(q_rep.shape[0], q_rep.shape[-1], 1)
                    ).squeeze()
                else:
                    d_rep = gather_rep(d_rep, self.accelerator)
                    score = torch.matmul(q_rep, d_rep.t())
            else:
                q_rep_dense, q_rep_sparse, q_rep_colbert = model(**q_features_list[i])
                d_rep_dense, d_rep_sparse, d_rep_colbert = model(**d_features_list[i])
                if self.use_in_batch_negatives:
                    L = d_rep_colbert.shape[1]
                    seq_length = torch.tensor([L], device=d_rep_colbert.device)
                    seq_length = self.accelerator.gather(seq_length)
                    max_seq_length = seq_length.max().item()
                    padding_size = (0, 0, 0, max_seq_length - L, 0, 0)
                    d_rep_colbert = F.pad(d_rep_colbert, padding_size)
                    d_rep_colbert = self.accelerator.gather(d_rep_colbert)
                    d_rep_dense = self.accelerator.gather(d_rep_dense)
                    d_rep_sparse = self.accelerator.gather(d_rep_sparse)

                    score_dense = torch.matmul(q_rep_dense, d_rep_dense.t())
                    score_sparse = torch.matmul(q_rep_sparse, d_rep_sparse.t())

                    token_scores = torch.einsum(
                        "qin,pjn->qipj", q_rep_colbert, d_rep_colbert
                    )
                    score_colbert, _ = token_scores.max(-1)
                    score_colbert = score_colbert.sum(1) / q_features_list[i][
                        "attention_mask"
                    ][:, 1:].sum(-1, keepdim=True)

                    score = score_dense + score_colbert + 0.3 * score_sparse
                else:
                    raise NotImplementedError

            max_t = score.max(dim=1).values
            min_t = score.min(dim=1).values
            score = (score - min_t.unsqueeze(-1)) / (
                (max_t - min_t + 1e-6).unsqueeze(-1)
            )
            scores += score * self.weights[i]

        scores = scores / sum(self.weights)
        return scores * self.score_scale
