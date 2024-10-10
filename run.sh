set -e

N_DEVICES=8
OUTPUT_DIR="output/multi-idf-uncased-norm"

python train_ir.py config.yaml

for step in {50000,}
do
torchrun --nproc_per_node=${N_DEVICES} evaluate_miracl.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free true \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 20 \
  --idf_path multilingual_idf_uncased_norm.json
done