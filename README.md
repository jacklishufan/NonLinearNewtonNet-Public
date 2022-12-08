# Non-Linear Graph Attention for Molecular Energy Prediction #

## Overview ##

CS 182 Final Project Fall 2022 at University of California, Berkeley

Authors: Jiahsu Liang, Shufan Li, Yifan Zhang, and Divyam Goel

Summary: Real-world atomistic dynamics are dictated by complex quantum mechanical interactions. NewtonNet, a Newtonian message passing network for deep learning of interatomic potentials and forces, has proven to reduce such computational complexity.
In this project, we will be applying a deep neural network approach to the field of quantum chemistry. Specifically, we will be expanding the existing model, NewtonNet, to aim to achieve better prediction results on molecular energies for popular datasets such as Ani-ccx and MD-17 by combining latent force computations with three-body angular interactions using nonlinear attention from Equiformer.
 

## Installation and Dependencies ##

We recommend using conda environment to install dependencies of this library first.
Please install (or load) conda and then proceed with the following commands:

    conda create --name newtonnet python=3.7
    conda activate newtonnet
    conda install -c conda-forge numpy scipy scikit-learn pandas ase tqdm
    pip install pyyaml

You also need to install Pytorch based on your hardware (we support both cpu and gpu) and the command line 
provided on the official website: https://pytorch.org/get-started/locally/

Now, you can install NewtonNet in the conda environment by cloning this repository:

    git@github.com:jacklishufan/NonLinearNewtonNet-Public.git

and then runnig the following command inside the NewtonNet repository (where you have access to setup.py):

    pip install -e .

Once you finished installations succesfully, you will be able to run NewtonNet modules
anywhere on your computer as long as the `newtonnet` environment is activated.


## Getting Started

Download datasets from following websites:

Ani-1 data set: https://figshare.com/collections/_/3846712

Ani-ccx data set: https://springernature.figshare.com/collections/The_ANI-1ccx_and_ANI-1x_data_sets_coupled-cluster_and_density_functional_theory_properties_for_molecules/4712477

MD17 data set: https://paperswithcode.com/dataset/md17

Set `train_path` and `test_path` to paths to your local data folders in `config_NaN.yml`

To launch experiments with multiple GPUs:

Set `CUDA_VISIBLE_DEVICES=4,5,6,7` or your GPU device numbers in `config_NaN.yml`
```
python train.py -c config_NaN.yml -p ani
```

To launch experiments with one GPU:

Set `CUDA_VISIBLE_DEVICES=0` in `config_NaN.yml`
```  
python train.py -c config_NaN.yml -p ani
```

Note: you can run the model after editing `config_NaN.yml`  (`device: ` should be set to the actual number of gpus you have locally)


## Guidelines 

- You can find several run files inside the scripts directory that rely on the implemented modules in the NewtonNet library. 

- The run scripts need to be accompanied with a yaml configuration file where you can adjust hyperparameters such as batch size.

- The documentation of the modules are available at most cases. Please look up local classes or functions
and consult with the docstrings in the code.

- You can specify which dataset and dataloader you want to use by changing parse arguments. They can be found in train.py.

## Changes from NewtonNet Codebase.
1. We implement three class in `newtonnet/models/newtonnet.py`: `NonLinearAttention`,`LinearAttention` and `NonLinearAttentionThreeBody`.

2. We implement dataloaders in `newtonnet/data/parse_raw.py` for ANI-1ccx and MD17 dataset with ccsd energy. 

3. We update the config sturcture in `scripts/config_NaN.yml`. In particular, we added following keys:
```
  nonlinear_attention: on     # wheather to add nonlinear attention
  attention_heads: 32       # number of heads
  three_body: on             # wheather to enable three-body interaction
```

To reproduce ablation result of Linear attention, please set the following 
```
  nonlinear_attention: linear    
  attention_heads: 32      
  three_body: off  
```

To reproduce our three body method, use 
```
  nonlinear_attention: on    
  attention_heads: 32      
  three_body: on  
```

To reproduce our two body method, use 
```
  nonlinear_attention: on    
  attention_heads: 32      
  three_body: off
```

To reproduce baseline NewtonNet, use 
```
  nonlinear_attention: off    
  attention_heads: 1 
  three_body: off
```

## Full Codebase Strctures (Inherited from Original NewtonNet Project) ##
```
├──  cli
│    └── newtonnet_train.py                     - command line interface for the training
│
│
├──  newtonnet                                  - this folder contains the model architecture. 
│   └── data                                    - this folder contains the data loader.
│       └── __init__.py    
│       └── loader.py   
│       └── neighbors.py   
│       └── parse_raw.py                        - this file contains our implementation of custom data loader classes.
│       └── pyanitools.py    
│   └── layers                                  - this folder contains the layers and transformations used by the model.
│       └── __init__.py    
│       └── activations.py   
│       └── batchrenorm.py   
│       └── cutoff.py    
│       └── dense.py   
│       └── representations.py   
│       └── scalers.py    
│       └── shells.py   
│   └── mdoels                                  - this folder contains the model itself
│       └── __init__.py    
│       └── newtonnet.py                        - this files contains our implementation of non-linear attention and three-body update layer
│   └── trains                                  - this folder contains the train loops.
│       └── hooks
│           └── __init__.py
│           └── visualizers.py
│       └── __init__.py   
│       └── trainer.py  
│   └── utils                                   - this folder contains the utilies used by the model.
│       └── __init__.py    
│       └── utility.py   
|
|
├──  scripts  
│    └── config_NaN.yml                         - here's config file for our method
│    └── config_NewtonNet.yml                   - config file for vanilla NewtonNet
│    └── launch.sh       		                
│    └── run.py                                  
│    └── train.py                               - here's the script to start training
```

## License ##

MIT License

Copyright (c) 2021 THGLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Contributing

Any kind of enhancement or contribution is welcomed.


## Acknowledgments

We're grateful for the help provided by the original authors of NewtonNet paper from THG Lab at University of California, Berkeley. 


