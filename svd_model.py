import argparse
import torch
import transformers

from scripts.model.decomposition_bert import DecompBertConfig, DecompBertForMaskedLM


def svd(model, K=256):
    original_decoder_weight = model.cls.predictions.decoder.weight.data
    U, S, Vh = torch.linalg.svd(original_decoder_weight, full_matrices=False)
    U_k = U[:, :K]
    S_k = torch.diag(S[:K])
    Vh_k = Vh[:K, :]
    reduced_decoder_weight_1 = torch.matmul(U_k, S_k)
    reduced_decoder_weight_2 = Vh_k
    bias = model.cls.predictions.decoder.bias

    model.cls.predictions.decoder = torch.nn.Sequential(
        torch.nn.Linear(
            reduced_decoder_weight_2.size(1),
            reduced_decoder_weight_2.size(0),
            bias=False,
        ),
        torch.nn.Linear(
            reduced_decoder_weight_1.size(1),
            reduced_decoder_weight_1.size(0),
            bias=True,
        ),
    )

    model.cls.predictions.decoder[0].weight = torch.nn.Parameter(
        reduced_decoder_weight_2
    )
    model.cls.predictions.decoder[1].weight = torch.nn.Parameter(
        reduced_decoder_weight_1
    )
    model.cls.predictions.decoder[1].bias = bias
    model.cls.predictions.decoder.bias = bias
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--k", type=int, default=256)
args = parser.parse_args()

model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_id)
svd(model, args.k)

config = DecompBertConfig.from_pretrained(
    args.model_id, vocab_intermediate_size=args.k, tie_word_embeddings=False
)
model_svd = DecompBertForMaskedLM(config)
model_svd.load_state_dict(model.state_dict())
model_svd.save_pretrained(args.output_dir, safe_serialization=False)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
tokenizer.save_pretrained(args.output_dir)
