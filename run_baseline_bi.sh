set -e


CONFIG_PATH='config_baseline_bi.yaml';
OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS


PREPROCESS_FUNC=$(grep "^preprocess_func:" "$CONFIG_PATH" | sed "s/^preprocess_func: //")
if [ -n "$PREPROCESS_FUNC" ]; then
    PREPROCESS_FUNC=$PREPROCESS_FUNC
else
    PREPROCESS_FUNC="null"
fi
echo "PREPROCESS_FUNC: $PREPROCESS_FUNC"


torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH


for step in {$MAX_STEPS,}
do
torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free false \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 50 \
  --preprocess_func $PREPROCESS_FUNC \
  --do_search false \
  --use_two_phase true

torchrun --nproc_per_node=1 evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free false \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --per_device_eval_batch_size 50 \
  --preprocess_func $PREPROCESS_FUNC \
  --skip_ingest true \
  --use_two_phase true
done