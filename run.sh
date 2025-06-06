N_DEVICES=8;
OUTPUT_DIR="models/ft3-150-oldmarco";
step=50000;

torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free true \
  --idf_path idf.json \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 5 \
  --max_seq_length 8192