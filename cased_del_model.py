import os
import json
import torch
import transformers
from collections import defaultdict


def has_uppercase(text):
    """Check if a string contains uppercase characters."""
    return any(c.isupper() for c in text)


def create_case_insensitive_bert_model(
    source_model_name="google-bert/bert-base-cased",
    output_dir="case-insensitive-bert-model",
    similarity_threshold=None,
    debug_file="merged_tokens.jsonl",
):
    """
    Create a case-insensitive version of a BERT model by merging similar uppercase
    and lowercase token embeddings.

    Args:
        source_model_name: The source model to convert
        output_dir: Directory to save the converted model
        similarity_threshold: Cosine similarity threshold for merging embeddings
        debug_file: File to save information about merged tokens
    """
    print(f"Loading model and tokenizer from {source_model_name}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(source_model_name)
    model = transformers.AutoModelForMaskedLM.from_pretrained(source_model_name)

    # Get original embeddings and vocabulary
    original_embeddings = model.bert.embeddings.word_embeddings.weight.data
    original_vocab = tokenizer.get_vocab()
    special_tokens = tokenizer.special_tokens_map.values()

    # Map tokens to keep and those to merge
    print("Identifying tokens to merge...")
    new_vocab = {}
    old_to_new_idx = {}
    merged_tokens = defaultdict(list)
    new_idx = 0

    # First pass: identify tokens to merge
    for old_idx in range(len(original_vocab)):
        token = tokenizer._convert_id_to_token(old_idx)

        # Skip special tokens and handle uppercase tokens
        if token not in special_tokens:
            if has_uppercase(token):
                if token.lower() in original_vocab:
                    lowercase_id = tokenizer._convert_token_to_id_with_added_voc(
                        token.lower()
                    )
                    uppercase_id = tokenizer._convert_token_to_id_with_added_voc(token)

                    # Calculate similarity between embeddings
                    similarity = (
                        original_embeddings[lowercase_id]
                        @ original_embeddings[uppercase_id].T
                    )
                    if (
                        similarity_threshold is None
                        or similarity.max() >= similarity_threshold
                    ):
                        merged_tokens[lowercase_id].append(uppercase_id)
                continue

        # Token to keep
        new_vocab[token] = new_idx
        old_to_new_idx[old_idx] = new_idx
        new_idx += 1

    # Save information about merged tokens for debugging
    if debug_file:
        save_merged_tokens_info(
            tokenizer, original_embeddings, merged_tokens, debug_file
        )

    # Create temporary tokenizer first
    temp_dir = "tmp_tokenizer"
    tokenizer.save_pretrained(temp_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Update tokenizer files
    update_tokenizer_files(temp_dir, output_dir, new_vocab)

    # Load and verify the new tokenizer
    new_tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir)
    new_tokenizer.save_pretrained(output_dir)

    # Update model embeddings and projection layers
    update_model_embeddings(
        model, original_embeddings, old_to_new_idx, merged_tokens, len(new_vocab)
    )

    # Save the modified model
    model.config.vocab_size = len(new_vocab)
    model.save_pretrained(output_dir)

    print(f"Model successfully converted and saved to {output_dir}")
    return model, new_tokenizer, old_to_new_idx


def save_merged_tokens_info(tokenizer, embeddings, merged_tokens, output_file):
    """Save information about merged tokens to a file for analysis."""
    print(f"Saving merged token information to {output_file}...")
    with open(output_file, "w") as f:
        f.write(json.dumps(merged_tokens) + "\n")
        for base_id, merged_ids in merged_tokens.items():
            all_ids = [base_id] + merged_ids
            tokens = [tokenizer._convert_id_to_token(id) for id in all_ids]
            token_embeddings = embeddings[all_ids]
            similarities = token_embeddings[0] @ token_embeddings.T

            f.write(json.dumps([tokens, similarities.numpy().tolist()]))
            f.write("\n")


def update_tokenizer_files(temp_dir, output_dir, new_vocab):
    """Update tokenizer configuration files with the new vocabulary."""
    print("Updating tokenizer configuration...")

    # Update tokenizer.json
    with open(f"{temp_dir}/tokenizer.json") as f:
        config = json.load(f)
    config["model"]["vocab"] = new_vocab
    with open(f"{output_dir}/tokenizer.json", "w") as f:
        json.dump(config, f, indent=4)

    # Update tokenizer_config.json
    with open(f"{temp_dir}/tokenizer_config.json") as f:
        config = json.load(f)
    config["do_lower_case"] = True
    with open(f"{output_dir}/tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=4)


def update_model_embeddings(
    model, original_embeddings, old_to_new_idx, merged_tokens, new_vocab_size
):
    """
    Update model embeddings and MLM projection layers to account for merged tokens.
    """
    print("Updating model embeddings and projection layers...")

    # Update word embeddings
    new_embeddings = torch.zeros((new_vocab_size, original_embeddings.shape[1]))
    for old_idx, new_idx in old_to_new_idx.items():
        if old_idx in merged_tokens:
            # Average embeddings for merged tokens
            all_ids = [old_idx] + merged_tokens[old_idx]
            new_embeddings[new_idx] = torch.mean(original_embeddings[all_ids], dim=0)
        else:
            new_embeddings[new_idx] = original_embeddings[old_idx]

    # Replace embedding layer
    model.bert.embeddings.word_embeddings = torch.nn.Embedding(
        new_vocab_size, original_embeddings.shape[1], _weight=new_embeddings
    )

    # Update MLM head (bias and decoder)
    old_mlm_bias = model.cls.predictions.bias.data
    old_mlm_decoder = model.cls.predictions.decoder.weight.data

    new_mlm_bias = torch.zeros(new_vocab_size)
    new_mlm_decoder = torch.zeros((new_vocab_size, old_mlm_decoder.shape[1]))

    for old_idx, new_idx in old_to_new_idx.items():
        new_mlm_bias[new_idx] = old_mlm_bias[old_idx]
        if old_idx in merged_tokens:
            all_ids = [old_idx] + merged_tokens[old_idx]
            new_mlm_decoder[new_idx] = torch.mean(old_mlm_decoder[all_ids], dim=0)
        else:
            new_mlm_decoder[new_idx] = old_mlm_decoder[old_idx]

    # Replace MLM projection layers
    model.cls.predictions.decoder = torch.nn.Linear(
        old_mlm_decoder.shape[1], new_vocab_size, bias=False
    )
    model.cls.predictions.decoder.weight.data = new_mlm_decoder
    model.cls.predictions.bias = torch.nn.Parameter(new_mlm_bias)


def get_mapping_and_cluster_tokens():
    with open("old_to_new_idx.json") as f:
        idx_map = json.load(f)

    with open("merged_tokens.jsonl") as f:
        merged_tokens = json.loads(f.readline())

    indices_0 = list(idx_map.values())
    indices_1 = list(idx_map.keys())

    for key, values in merged_tokens.items():
        for value in values:
            if value == int(key):
                continue
            indices_0.append(idx_map[key])
            indices_1.append(value)

    indices_0 = [int(x) for x in indices_0]
    indices_1 = [int(x) for x in indices_1]

    new_idx_num = dict()
    for key, vs in merged_tokens.items():
        vs = [v for v in vs if v != int(key)]
        if vs == []:
            continue
        new_idx_num[idx_map[key]] = len(vs) + 1

    indices = torch.stack([torch.tensor(indices_0), torch.tensor(indices_1)])
    values = torch.ones_like(torch.tensor(indices_0)).float()

    for i, c_id in enumerate(indices[0]):
        c_id = int(c_id)
        values[i] /= new_idx_num.get(c_id, 1)

    mapping = torch.sparse_coo_tensor(
        indices, values, size=(len(idx_map), model.config.vocab_size)
    ).coalesce()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    cluster_to_tokens = {}

    for k, v in idx_map.items():
        cluster_to_tokens[v] = tokenizer._convert_id_to_token(int(k))

    torch.save(mapping, "cased-mapping.bin")
    torch.save(cluster_to_tokens, "cluster_to_tokens.bin")


if __name__ == "__main__":
    # Example usage
    model, tokenizer, old_to_new_idx = create_case_insensitive_bert_model(
        source_model_name="google-bert/bert-base-cased",
        output_dir="case-insensitive-bert",
        similarity_threshold=None,
        debug_file="merged_tokens.jsonl",
    )
    with open("old_to_new_idx.json", "w") as f:
        json.dump(old_to_new_idx, f)
