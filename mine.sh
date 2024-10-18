set -e

torchrun --nproc_per_node=8 mine_data_multilingual.py \
  --model_name_or_path output/kd-bge-m3/checkpoint-50000/ \
  --inf_free true \
  --source miracl \
  --mine_datasets bn,te,es,fr,id,hi,ru,ar,zh,fa,ja,fi,sw,ko,en

torchrun --nproc_per_node=8 mine_data_multilingual.py \
  --model_name_or_path output/kd-bge-m3/checkpoint-50000/ \
  --inf_free true \
  --source tydi \
  --mine_datasets arabic,bengali,english,finnish,indonesian,japanese,korean,russian,swahili,telugu