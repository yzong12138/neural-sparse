conda create -n neural-sparse python=3.9
conda install pytorch==1.11.0 cudatoolkit=10.2 -c pytorch
conda install numpy
conda install -c conda-forge accelerate transformers datasets
pip install evaluate
pip install opensearch-py beir