set -e

N_DEVICES=8
OUTPUT_DIR="output/baselines/bi"

torchrun --nproc_per_node=${N_DEVICES} train_ir.py config_marco.yaml

for step in {100000,}
do
torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free false \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 50
done