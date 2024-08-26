torchrun --nproc_per_node=8 train_ir.py config.yaml

OUTPUT_DIR="output/dp_test5"
torchrun --nproc_per_node=8 evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-50000 \
  --inf_free true \
  --idf_path idf.json \
  --output_dir ${OUTPUT_DIR} \
  --log_level info