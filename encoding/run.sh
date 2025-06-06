torchrun --nproc_per_node=8 encode.py --model_type bge
torchrun --nproc_per_node=8 encode.py --model_type inf
python runner.py -m 3 -d /opt/dlami/nvme/bge-256
python runner.py -m 4 -d /opt/dlami/nvme/inf-256