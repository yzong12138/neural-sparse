set -e

# torchrun --nproc_per_node=8 scorer.py --identifier bge_lw_reranker --all_datasets msmarco_ft3 --dataset_path ../data/


python scorer_runner.py \
    --dataset_path ../data/ \
    --all_datasets "msmarco_ft3" \
    --identifier bge_lw_reranker \
    --num_processes 8