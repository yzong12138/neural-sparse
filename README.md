## Prepare the environment

### Conda environment
```
conda create -n neural-sparse python=3.9
conda activate neural-sparse
conda install pytorch==1.11.0 cudatoolkit=10.2 -c pytorch
conda install numpy
conda install -c conda-forge accelerate transformers datasets
pip install opensearch-py beir
```

### OpenSearch service
To evaluate search relevance or mine hard negatives, run an OpenSearch node at local device. It can be accessed at `http://localhost:9200` without username/password(security disabled). For more details, please check [OpenSearch doc](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/tar/). Here are steps to start a node without security:
1. Follow the step1 and step2 in above documentation.
2. Modify `/path/to/opensearch-2.16.0/config/opensearch.yml`, add this line: `plugins.security.disabled: true`
3. Start a tmux session so the OpenSearch won't stop after the terminal is close `tmux new -s opensearch`. In the tmux session, run `cd /path/to/opensearch-2.16.0` and `./bin/opensearch`.
4. The sevice is running. Run `curl -X GET http://localhost:9200` to test.

### An example of fine-tuning on BEIR scifact
Here is an example of fine-tuning the `opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini` model at BEIR scifact.

1. Generate training data.
   1. `python demo_train_data.py`(data parallel) or `torchrun --nproc_per_node=${N_DEVICES} demo_train_data.py`(distributed data parallel) with configs.
   2. This will generate training data of hard negatives at `data/scifact_train.jsonl`
```
torchrun --nproc_per_node=${N_DEVICES} demo_train_data.py \
  --model_name_or_path opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini \
  --inf_free true \
  --idf_path idf.json \
  --beir_dir data/beir
```
2. Run training.
   1. `python train_ir.py {config_file}`(data parallel) or `torchrun --nproc_per_node=${N_DEVICES} train_ir.py config.yaml`(distributed data parallel)
   2. If training using infoNCE loss, use config_infonce.yaml
   3. If training using ensembled teacher models, using config_kd.yaml
3. Run evaluation on the test set.
```
for step in {500,1000,1500,2000}
do
OUTPUT_DIR="output/test"
torchrun --nproc_per_node=8 evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free true \
  --idf_path idf.json \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --beir_datasets scifact \
  --per_device_eval_batch_size 50
done
```

## Run with infoNCE loss
Run with data parallel:
```
python train_ir.py config_infonce.yaml
```
Run with distributed data parallel:
```
# the number of GPU
N_DEVICES=8
torchrun --nproc_per_node=${N_DEVICES} train_ir.py config_infonce.yaml
```


Data file is a jsonl file, each line is a data sample like this:
```json
{
    "query":"xxx xxx xxx",
    "pos":"xxxx xxxx xxxx",
    "negs": ["xxx", "xxx", "xxx", "xxx"],
}
```

An example to train on BEIR scifact dataset then evaluate the checkpoints:
```
N_DEVICES=8
torchrun --nproc_per_node=${N_DEVICES} train_ir.py config_infonce.yaml

for step in {500,1000,1500}
do
OUTPUT_DIR="output/test"
torchrun --nproc_per_node=8 evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free true \
  --idf_path idf.json \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --beir_datasets scifact \
  --per_device_eval_batch_size 50
done
```

## Run with knowledge distillation (scores pre-computed)
Run with data parallel:
```
python train_ir.py config.yaml
```
Run with distributed data parallel:
```
# the number of GPU
N_DEVICES=8
torchrun --nproc_per_node=${N_DEVICES} train_ir.py config.yaml
```

To change hyperparameters, please check the args in config file.

Data file is a jsonl file, each line is a data sample like this:
```json
{
    "query":"xxx xxx xxx",
    "docs": ["xxx", "xxx", "xxx", "xxx"],
    "scores": [1.0, 5.0, 9.0, 4.4]
}
```
