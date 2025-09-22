#!/bin/bash
set -e

N_DEVICES=8;
MODEL_NAME="mdbert-wbcmasfp-5-200k";
PT_OUTPUT_DIR="output/sota/pt/${MODEL_NAME}";
FT_OUTPUT_DIR="output/sota/pt/${MODEL_NAME}/ft";

CONFIG_PT_CONTENT="""
logging_steps: 50
# log_level_replica: info

inf_free: true
model_name_or_path: ${MODEL_NAME}
tokenizer_name: ${MODEL_NAME}
idf_path: idf.json
use_l0: false

max_seq_length: 256
train_file_dir: [data/pt-kd-ids, data/pt-marco-half]
data_type: kd-ids
loss_types: [kldiv]
sample_num_one_query: 3
use_in_batch_negatives: true
flops_d_lambda: 0.01
flops_d_T: 10000
ranking_loss_weight: 1
kd_ensemble_teacher_kwargs:
  types: ["dense", "sparse", "remote", "remote"]
  model_ids: ["Alibaba-NLP/gte-large-en-v1.5","opensearch-project/opensearch-neural-sparse-encoding-v1","3", "4"]
  teacher_tokenizer_ids: ["Alibaba-NLP/gte-large-en-v1.5","opensearch-project/opensearch-neural-sparse-encoding-v1","3", "4"]
  score_scale: 30
first_rank_thresh: 10

output_dir: ${PT_OUTPUT_DIR}
per_device_eval_batch_size: 50
per_device_train_batch_size: 15

log_level: info
max_steps: 300000
fp16: true
learning_rate: 0.00002
weight_decay: 0.01
lr_scheduler_type: linear
warmup_steps: 6000
save_strategy: steps
save_steps: 200000
dataloader_drop_last: true
"""
rm config_pt.yaml
echo "$CONFIG_PT_CONTENT" > config_pt.yaml

CONFIG_BASELINE_CONTENT="""
logging_steps: 50
# log_level_replica: info

inf_free: true
model_name_or_path: ${PT_OUTPUT_DIR}/checkpoint-300000
tokenizer_name: ${PT_OUTPUT_DIR}/checkpoint-300000
idf_path: idf.json
use_l0: true

max_seq_length: 256
train_file: data/msmarco_llm_scores
data_type: kd
loss_types: [kldiv]
sample_num_one_query: 2
use_in_batch_negatives: false
flops_d_lambda: 0.08
flops_d_T: 10000
ranking_loss_weight: 1
flops_threshold: 150
swap_times: 5
score_scale: 1.2

output_dir: ${FT_OUTPUT_DIR}
per_device_eval_batch_size: 50
per_device_train_batch_size: 30

log_level: info
max_steps: 50000
fp16: true
learning_rate: 0.00002
weight_decay: 0.01
lr_scheduler_type: linear
warmup_steps: 6000
save_strategy: steps
save_steps: 50000
dataloader_drop_last: true
max_grad_norm: null
"""
rm config_baseline.yaml
echo "$CONFIG_BASELINE_CONTENT" > config_baseline.yaml

export AWS_REGION_NAME="us-west-2";

torchrun --nproc_per_node=${N_DEVICES} train_ir.py config_pt.yaml

torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
  --model_name_or_path ${PT_OUTPUT_DIR}/checkpoint-300000 \
  --inf_free true \
  --output_dir ${PT_OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 50 \
  --idf_path idf.json \
  --beir_datasets nfcorpus,trec-covid \
  --use_l0 false

torchrun --nproc_per_node=${N_DEVICES} train_ir.py config_baseline.yaml

torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
  --model_name_or_path ${FT_OUTPUT_DIR}/checkpoint-50000 \
  --inf_free true \
  --output_dir ${FT_OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 5 \
  --idf_path idf.json \
  --use_l0 true \
  --max_seq_length 8192