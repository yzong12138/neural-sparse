set -e

CONFIG_PATH='config_kd.yaml';
OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

for step in {50000,}
do
torchrun --nproc_per_node=${N_DEVICES} evaluate_miracl.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free true \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 25
done
