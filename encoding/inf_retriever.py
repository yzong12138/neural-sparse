import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_query_embeddings(queries, query_max_len=512, model=None, tokenizer=None):
    """
    Generate embeddings for queries with instruction format

    Args:
        queries: List of query strings
        query_max_len: Maximum token length
        model: The model to use for embeddings
        tokenizer: The tokenizer to use

    Returns:
        Normalized query embeddings tensor
    """
    task = "Given a web search query, retrieve relevant passages that answer the query"
    formatted_queries = [get_detailed_instruct(task, query) for query in queries]

    # Tokenize the queries
    query_batch_dict = tokenizer(
        formatted_queries,
        max_length=query_max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Move to the same device as the model
    query_batch_dict = {k: v.to(model.device) for k, v in query_batch_dict.items()}

    # Generate embeddings
    with torch.no_grad():
        query_outputs = model(**query_batch_dict)
        query_embeddings = last_token_pool(
            query_outputs.last_hidden_state, query_batch_dict["attention_mask"]
        )

    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    return query_embeddings


def get_document_embeddings(documents, doc_max_len=512, model=None, tokenizer=None):
    """
    Generate embeddings for documents (no instruction format)

    Args:
        documents: List of document strings
        doc_max_len: Maximum token length
        model: The model to use for embeddings
        tokenizer: The tokenizer to use

    Returns:
        Normalized document embeddings tensor
    """
    # Tokenize the documents
    doc_batch_dict = tokenizer(
        documents,
        max_length=doc_max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Move to the same device as the model
    doc_batch_dict = {k: v.to(model.device) for k, v in doc_batch_dict.items()}

    # Generate embeddings
    with torch.no_grad():
        doc_outputs = model(**doc_batch_dict)
        doc_embeddings = last_token_pool(
            doc_outputs.last_hidden_state, doc_batch_dict["attention_mask"]
        )

    # Normalize embeddings
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

    return doc_embeddings
