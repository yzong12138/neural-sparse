set -e

torchrun --nproc_per_node=8 mine_beir.py \
  --model_name_or_path output/baselines-pt/with-34-llm-003/ft3-150/checkpoint-50000 \
  --inf_free true \
  --idf_path idf.json \
  --beir_dir data/beir \
  --do_search false

torchrun --nproc_per_node=1 mine_beir.py \
  --model_name_or_path output/baselines-pt/with-34-llm-003/ft3-150/checkpoint-50000 \
  --inf_free true \
  --idf_path idf.json \
  --beir_dir data/beir \
  --skip_ingest true