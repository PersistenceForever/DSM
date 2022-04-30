# DSM
The pytorch implementation of Question Generation over Knowledge Base via Modeling Diverse Subgraphs with Meta-learner.

Requirements
====
Create a virtual environment first via:
```
$ conda activate -n your_env_name python 3.8.5 pip
```
Install all the required tools using the following command:
```
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

$ conda install -c dglteam dgl-cuda11.1

$ pip install -r requirements.txt
```

Running the code
====
(1) Create a folder to store results.
```
$ mkdir save
```

(2) To run the example, execute:
```
$ sh run_pubmed.sh python bart_train_f2.py --epoch 30 --input_dir dataset/PQ --output_dir './output_PQ_base_new' --update_lr 5e-5 --meta_lr 3e-5 --model_name_or_path './bart-base'
```

Notes: the experimental results and optimal hyper-parameters could be somewhat different under different environments (e.g., different devices and different versions of PyTorch), you can use the suggested method introduced in our paper to choose the combination of hyper-parameters.

