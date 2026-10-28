set -e

# torchrun --nproc_per_node=8 scorer.py --identifier bge_lw_reranker --all_datasets msmarco_ft3 --dataset_path ../data/


python scorer_runner.py \
    --all_datasets "msmarco_st" \
    --identifier minilm-l6 \
    --num_processes 8

python scorer_runner.py \
    --all_datasets "msmarco_st" \
    --identifier t5 \
    --num_processes 8