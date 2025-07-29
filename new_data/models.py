import os
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.async_embedding_server import EmbeddingService
from scripts.model.sparse_encoders import SparseModel


class DenseEmbeddingScorer:
    def __init__(self, model_id, device, max_seq_length=256):
        self.model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": "float16"},
        )
        self.model.max_seq_length = max_seq_length

    def score(self, query, docs, **kwargs):
        embeddings = self.model.encode(
            [query] + docs, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = torch.matmul(embeddings[0], embeddings[1:].T)
        return scores.cpu().numpy()


class SparseEmbeddingScorer:
    def __init__(self, model_id, device, max_seq_length=256):
        self.model = SparseModel(model_id, use_l0=False)
        self.model = self.model.to(device)
        self.max_seq_length = max_seq_length
        self.device = device

    def score(self, query, docs, **kwargs):
        texts = [query] + docs
        features = self.model.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=self.max_seq_length,
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        with torch.cuda.amp.autocast(), torch.no_grad():
            embeddings = self.model(**features)
        scores = torch.matmul(embeddings[0], embeddings[1:].T)
        return scores.cpu().numpy()


class RemoteEmbeddingScorer:
    def __init__(self, model):
        self.model = model
        self.embedding_service = EmbeddingService()

    def score(self, q_id, d_ids, **kwargs):
        self.embedding_service.register_task(
            table_name="vector_q", model_id=self.model, ids=[q_id]
        )
        self.embedding_service.register_task(
            table_name="vector", model_id=self.model, ids=d_ids
        )
        q_emb = self.embedding_service.fetch_embedding(
            table_name="vector_q",
            model_id=self.model,
            ids=[q_id],
        )[0]
        d_embs = self.embedding_service.fetch_embedding(
            table_name="vector",
            model_id=self.model,
            ids=d_ids,
        )
        scores = np.matmul(q_emb, d_embs.T)
        return scores


class MinilmScorer:
    def __init__(self, model_id, device, max_seq_length=256):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = self.model.to(device)
        self.max_seq_length = max_seq_length
        self.device = device

    def score(self, query, docs, **kwargs):
        # 预先截断查询和文档
        tokenized_query = self.tokenizer(
            query,
            add_special_tokens=False,
            max_length=self.max_seq_length,
            truncation=True,
        )
        tokenized_docs = [
            self.tokenizer(
                doc,
                add_special_tokens=False,
                max_length=self.max_seq_length,
                truncation=True,
            )
            for doc in docs
        ]

        # 手动构建输入
        queries = [self.tokenizer.decode(tokenized_query["input_ids"])] * len(docs)
        truncated_docs = [
            self.tokenizer.decode(doc["input_ids"]) for doc in tokenized_docs
        ]

        # 重新tokenize，这次带特殊标记并完成完整处理
        features = self.tokenizer(
            queries,
            truncated_docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # 移动张量到设备
        features = {k: v.to(self.device) for k, v in features.items()}

        # 获取评分
        self.model.eval()
        with torch.cuda.amp.autocast(), torch.no_grad():
            scores = self.model(**features).logits
            scores = scores.squeeze(-1)

        return scores.cpu().numpy()


class T5Scorer:
    def __init__(self, model_id, device, max_seq_length=256):
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.tokenizer = T5Tokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id).half()
        self.model = self.model.to(device)
        self.max_seq_length = max_seq_length
        self.device = device

        # 获取T5模型中"true"和"false"的token ID
        self.token_false_id = self.tokenizer.get_vocab()["▁false"]
        self.token_true_id = self.tokenizer.get_vocab()["▁true"]

        # 查询文档模板
        self.pattern = "Query: {query} Document: {document} Relevant:"

    @torch.no_grad()
    def greedy_decode(
        self, input_ids, length, attention_mask=None, return_last_logits=True
    ):
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
        ).to(input_ids.device)
        encoder_outputs = self.model.get_encoder()(
            input_ids, attention_mask=attention_mask
        )
        next_token_logits = None
        for _ in range(length):
            model_inputs = self.model.prepare_inputs_for_generation(
                decode_ids,
                encoder_outputs=encoder_outputs,
                past=None,
                attention_mask=attention_mask,
                use_cache=True,
            )
            model_inputs.pop("past")
            outputs = self.model(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            decode_ids = torch.cat(
                [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
            )
        if return_last_logits:
            return decode_ids, next_token_logits
        return decode_ids

    def score(self, query, docs, **kwargs):
        # 预先截断查询和文档
        tokenized_query = self.tokenizer(
            query,
            add_special_tokens=False,
            max_length=self.max_seq_length,
            truncation=True,
        )
        tokenized_docs = [
            self.tokenizer(
                doc,
                add_special_tokens=False,
                max_length=self.max_seq_length,
                truncation=True,
            )
            for doc in docs
        ]

        # 手动构建输入
        query = self.tokenizer.decode(tokenized_query["input_ids"])
        truncated_docs = [
            self.tokenizer.decode(doc["input_ids"]) for doc in tokenized_docs
        ]

        # 创建文本对
        texts = [
            self.pattern.format(query=query, document=doc) for doc in truncated_docs
        ]

        # 批量编码
        features = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
            return_attention_mask=True,
        )

        # 移动到设备
        input_ids = features["input_ids"].to(self.device)
        attn_mask = features["attention_mask"].to(self.device)

        # 使用greedy_decode获取logits
        self.model.eval()
        with torch.cuda.amp.autocast():
            _, batch_scores = self.greedy_decode(
                input_ids, length=1, attention_mask=attn_mask, return_last_logits=True
            )

        # 获取true和false的分数
        batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]

        # 计算relevance score (true_score - false_score)
        scores = batch_scores[:, 1] - batch_scores[:, 0]

        return scores.cpu().numpy()


class BGEReranker:
    def __init__(self, model_id, device, max_seq_length=256):
        from FlagEmbedding import LightWeightFlagLLMReranker

        self.model = LightWeightFlagLLMReranker(model_id, use_fp16=True, device="cpu")
        self.device = device
        self.max_seq_length = max_seq_length

    def score(self, query, docs, **kwargs):
        # Format input pairs as required by the reranker: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc] for doc in docs]

        # Compute scores using the reranker
        scores = self.model.compute_score_single_gpu(
            pairs,
            cutoff_layers=[28],
            compress_ratio=2,
            compress_layers=[24, 40],
            query_max_length=self.max_seq_length,
            max_length=self.max_seq_length,
            device=self.device,
        )

        return np.array(scores).reshape(-1)


class Qwen3Reranker:
    def __init__(self, model_id, device, max_seq_length=256):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )
            .to(device)
            .eval()
        )

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_seq_length = max_seq_length
        self.device = device

        # 定义前缀和后缀
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )

        # 默认指令
        self.instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

    def format_instruction(self, query, doc):
        """格式化输入文本"""
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
        """处理输入对并添加前缀、后缀tokens"""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_seq_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_seq_length
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs):
        """计算logits并返回得分"""
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().cpu().numpy()
        return scores

    def score(self, query, docs, **kwargs):
        """对查询和文档进行打分"""
        # 创建查询-文档对
        pairs = [self.format_instruction(query, doc) for doc in docs]
        # 处理输入
        inputs = self.process_inputs(pairs)
        # 计算得分
        with torch.cuda.amp.autocast():
            scores = self.compute_logits(inputs)
        return scores


GET_SCORER = {
    "gte-bi": lambda device: DenseEmbeddingScorer(
        model_id="Alibaba-NLP/gte-large-en-v1.5", device=device
    ),
    "sparse-v1": lambda device: SparseEmbeddingScorer(
        model_id="opensearch-project/opensearch-neural-sparse-encoding-v1",
        device=device,
    ),
    "llm-0": lambda device: RemoteEmbeddingScorer("0"),
    "llm-1": lambda device: RemoteEmbeddingScorer("1"),
    "llm-2": lambda device: RemoteEmbeddingScorer("2"),
    "llm-3": lambda device: RemoteEmbeddingScorer("3"),
    "llm-4": lambda device: RemoteEmbeddingScorer("4"),
    "llm-5": lambda device: RemoteEmbeddingScorer("5"),
    "minilm": lambda device: MinilmScorer(
        model_id="cross-encoder/ms-marco-MiniLM-L12-v2",
        device=device,
    ),
    "t5": lambda device: T5Scorer(
        model_id="castorini/monot5-3b-msmarco-10k",
        device=device,
    ),
    "bge_gemma2_reranker": lambda device: BGEReranker(
        model_id="BAAI/bge-reranker-v2.5-gemma2-lightweight",
        device=device,
    ),
    "qwen3_reranker": lambda device: Qwen3Reranker(
        model_id="Qwen/Qwen3-Reranker-4B",
        device=device,
    ),
}
