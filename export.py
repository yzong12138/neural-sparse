from transformers import AutoModelForMaskedLM, AutoTokenizer

source_dir = "output/baselines-pt/with-01-llm/ft3-0.08-l0-200/checkpoint-50000"
target_dir = "1-click"

model = AutoModelForMaskedLM.from_pretrained(source_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=True)
model.save_pretrained(target_dir)

if (
    str(model.__class__)
    == "<class 'transformers_modules.Alibaba-NLP.new-impl.40ced75c3017eb27626c9d4ea981bde21a2662f4.modeling.NewForMaskedLM'>"
):
    import torch

    original_weight = model.new.embeddings.word_embeddings.weight.data[
        : tokenizer.vocab_size
    ].clone()
    model.new.embeddings.word_embeddings = torch.nn.Embedding(
        tokenizer.vocab_size,
        model.config.hidden_size,
        padding_idx=tokenizer.pad_token_id,
    )
    with torch.no_grad():
        model.new.embeddings.word_embeddings.weight.copy_(original_weight)

    original_weight = model.lm_head.decoder.weight.data[: tokenizer.vocab_size].clone()
    original_bias = model.lm_head.decoder.bias.data[: tokenizer.vocab_size].clone()
    model.lm_head.decoder = torch.nn.Linear(
        in_features=model.config.hidden_size,
        out_features=tokenizer.vocab_size,
        bias=True,
    )
    with torch.no_grad():
        model.lm_head.decoder.weight.copy_(original_weight)
        model.lm_head.decoder.bias.copy_(original_bias)

    model.config.vocab_size = tokenizer.vocab_size
    model.config._name_or_path = target_dir
    model.config.auto_map = {
        "AutoConfig": "configuration.NewConfig",
        "AutoModel": "modeling.NewModel",
        "AutoModelForMaskedLM": "modeling.NewForMaskedLM",
        "AutoModelForMultipleChoice": "modeling.NewForMultipleChoice",
        "AutoModelForQuestionAnswering": "modeling.NewForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling.NewForSequenceClassification",
        "AutoModelForTokenClassification": "modeling.NewForTokenClassification",
    }
    model.save_pretrained(target_dir)

tokenizer.save_pretrained(target_dir)
