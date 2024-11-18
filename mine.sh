set -e

torchrun --nproc_per_node=8 mine_data_multilingual.py \
  --model_name_or_path output/kd/idf-minagg-2e-3-kt/checkpoint-50000 \
  --inf_free true \
  --source miracl \
  --mine_datasets zh,bn,te,es,fr,id,hi,ru,ar,fa,ja,fi,sw,ko,en

torchrun --nproc_per_node=8 mine_data_multilingual.py \
  --model_name_or_path output/kd/idf-minagg-2e-3-kt/checkpoint-50000 \
  --inf_free true \
  --source tydi \
  --mine_datasets arabic,bengali,english,finnish,indonesian,japanese,korean,russian,swahili,telugu

# Loop through each language

# for model_id in {'BAAI/bge-m3','Alibaba-NLP/gte-multilingual-base','intfloat/multilingual-e5-large'}
# do
# torchrun --nproc_per_node=8 scorer_multilingual.py \
#   --model_id $model_id \
#   --source miracl \
#   --datasets 'bn,te,es,fr,id,hi,ru,ar,zh,fa,ja,fi,sw,ko,en'

# torchrun --nproc_per_node=8 scorer_multilingual.py \
#   --model_id $model_id \
#   --source tydi \
#   --datasets 'arabic,bengali,english,finnish,indonesian,japanese,korean,russian,swahili,telugu'
# done