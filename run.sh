N_DEVICES=8
OUTPUT_DIR="output/distill-baseline"

torchrun --nproc_per_node=${N_DEVICES} train_ir.py config.yaml

for step in {100000,}
do
torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free true \
  --idf_path idf.json \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 50
done