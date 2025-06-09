set -e

torchrun --nproc_per_node=8 scorer.py --identifier minilm --all_datasets msmarco_raw
torchrun --nproc_per_node=8 scorer.py --identifier gte-bi --all_datasets msmarco_raw
torchrun --nproc_per_node=8 scorer.py --identifier sparse-v1 --all_datasets msmarco_raw

# torchrun --nproc_per_node=8 scorer.py --identifier llm-3 --all_datasets msmarco_raw
# torchrun --nproc_per_node=8 scorer.py --identifier llm-4 --all_datasets msmarco_raw
# torchrun --nproc_per_node=8 scorer.py --identifier llm-5 --all_datasets msmarco_raw