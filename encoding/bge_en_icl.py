import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# Constants from sample code
TASK_DESCRIPTION = (
    "Given a web search query, retrieve relevant passages that answer the query."
)
EXAMPLES = [
    {
        "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
        "query": "what is a virtual interface",
        "response": "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes.",
    },
    {
        "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
        "query": "causes of back pain in female for a week",
        "response": "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management.",
    },
]


# Utility functions
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
    return f"<instruct>{task_description}\n<query>{query}"


def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f"<instruct>{task_description}\n<query>{query}\n<response>{response}"


def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len
        - len(tokenizer("<s>", add_special_tokens=False)["input_ids"])
        - len(tokenizer("\n<response></s>", add_special_tokens=False)["input_ids"]),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer("\n<response>", add_special_tokens=False)["input_ids"]
    new_max_length = (
        len(prefix_ids) + len(suffix_ids) + query_max_len + 8
    ) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs["input_ids"])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + "\n<response>"
    return new_max_length, new_queries


def get_query_embeddings(queries, query_max_len=512, model=None, tokenizer=None):
    formatted_queries = [
        get_detailed_instruct(TASK_DESCRIPTION, query) for query in queries
    ]

    # Prepare examples
    formatted_examples = [
        get_detailed_example(e["instruct"], e["query"], e["response"]) for e in EXAMPLES
    ]
    examples_prefix = "\n\n".join(formatted_examples) + "\n\n"

    # Process queries with examples
    new_query_max_len, new_queries = get_new_queries(
        formatted_queries, query_max_len, examples_prefix, tokenizer
    )

    # Tokenize and get embeddings
    query_batch_dict = tokenizer(
        new_queries,
        max_length=new_query_max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    query_batch_dict = {k: v.to(model.device) for k, v in query_batch_dict.items()}

    with torch.no_grad():
        query_outputs = model(**query_batch_dict)
        query_embeddings = last_token_pool(
            query_outputs.last_hidden_state, query_batch_dict["attention_mask"]
        )

    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    return query_embeddings


def get_document_embeddings(documents, doc_max_len=512, model=None, tokenizer=None):
    doc_batch_dict = tokenizer(
        documents,
        max_length=doc_max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    doc_batch_dict = {k: v.to(model.device) for k, v in doc_batch_dict.items()}

    with torch.no_grad():
        doc_outputs = model(**doc_batch_dict)
        doc_embeddings = last_token_pool(
            doc_outputs.last_hidden_state, doc_batch_dict["attention_mask"]
        )

    # Normalize embeddings
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

    return doc_embeddings
