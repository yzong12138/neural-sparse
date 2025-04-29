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


def get_document_embeddings(documents, doc_max_len=512, model=None, tokenizer=None):
    """
    Generate embeddings for documents using the Zeta-Alpha-E5-Mistral model.

    Args:
        documents (List[str]): List of document texts to embed
        doc_max_len (int): Maximum token length for documents
        model: Pre-loaded model (if None, will load the default model)
        tokenizer: Pre-loaded tokenizer (if None, will load the default tokenizer)

    Returns:
        torch.Tensor: Normalized document embeddings
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "zeta-alpha-ai/Zeta-Alpha-E5-Mistral"
            )
        if model is None:
            model = AutoModel.from_pretrained("zeta-alpha-ai/Zeta-Alpha-E5-Mistral")

    # Tokenize documents
    doc_batch_dict = tokenizer(
        documents,
        max_length=doc_max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Move tensors to model's device if available
    if hasattr(model, "device"):
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


def get_query_embeddings(
    queries, query_max_len=512, task_description=None, model=None, tokenizer=None
):
    """
    Generate embeddings for queries using the Zeta-Alpha-E5-Mistral model.

    Args:
        queries (List[str]): List of query texts to embed
        query_max_len (int): Maximum token length for queries
        task_description (str): Description of the task (if None, uses default)
        model: Pre-loaded model (if None, will load the default model)
        tokenizer: Pre-loaded tokenizer (if None, will load the default tokenizer)

    Returns:
        torch.Tensor: Normalized query embeddings
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "zeta-alpha-ai/Zeta-Alpha-E5-Mistral"
            )
        if model is None:
            model = AutoModel.from_pretrained("zeta-alpha-ai/Zeta-Alpha-E5-Mistral")

    # Use default task description if none provided
    if task_description is None:
        task_description = "Given a claim about climate change, retrieve documents that support or refute the claim"

    # Format queries using the instructional template
    formatted_queries = [
        get_detailed_instruct(task_description, query) for query in queries
    ]

    # Tokenize queries
    query_batch_dict = tokenizer(
        formatted_queries,
        max_length=query_max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Move tensors to model's device if available
    if hasattr(model, "device"):
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
