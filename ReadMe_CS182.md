# Readme #

## Overview ##

CS 182 Final Project Fall 2022

Authors: Jiahsu Liang, Divyam Goel, Yifan Zhang, and Shufan Li

Summary: Real world atomistic dynamics are dictated by complex quantum mechanical interactions. NewtonNet, a Newtonian message passing network for deep learning of interatomic potentials and forces, has been proven to reduce such computational complexity. 
In this project, we will be applying deep neural network approach to the field of quantum chemistry. Specifically, we will be expanding the existing model, NewtonNet, to aim to achieve better prediction results on molecule energies for popular dataset such as Ani-ccx and MD-17 by combining latent force computations with three body angular interactions and 4 body di-headral interactions using nonlinear attention from Equiformer [this one may be changed depends on demand].  

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

    git@github.com:jacklishufan/NewtonNet.git

and then runnig the following command inside the NewtonNet repository (where you have access to setup.py):

    pip install -e .

Once you finished installations succesfully, you will be able to run NewtonNet modules
anywhere on your computer as long as the `newtonnet` environment is activated.


## Getting Started

To launch experiments with multiple GPUs:

set `CUDA_VISIBLE_DEVICES=4,5,6,7` or your GPU device numbers in `config_ani.yml`
```
python train.py -c config_ani.yml -p ani
```

To launch experiments with one GPU:

set `CUDA_VISIBLE_DEVICES=0` in `config_ani.yml`
```
python train.py -c config_ani.yml -p ani
```

Note: you can run the model after editing `config_ani.yml`  (`device: ` should be set to the actual number of gpus you have locally)


## Guidlines

To run Ani-CCX data,

python train.py -c config_aniccx.yml -p ani_ccx -k ccsd


## Codebase Strctures ##
```
├──  cli
│    └── newtonnet_train.py  - command line interface for the training
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  newtonnet              - this folder contains the model
│   └── data                - this folder contains the data loader.
│       └── __init__.py    
│       └── loader.py   
│       └── neighbors.py   
│       └── parse_raw.py    
│       └── pyanitools.py    
│   └── layers                - this file contains the layers used by the model.
│       └── __init__.py    
│       └── activations.py   
│       └── batchrenorm.py   
│       └── cutoff.py    
│       └── dense.py   
│       └── representations.py   
│       └── scalers.py    
│       └── shells.py   
│   └── mdoels                - this file contains the train loops.
│       └── __init__.py    
│       └── newtonnet.py   
│       └── batchrenorm.py   
│       └── cutoff.py    
│       └── dense.py   
│       └── representations.py   
│       └── scalers.py    
│       └── shells.py   
│   └── trains                - this folder contains the train loops.
│       └── hooks
│           └── __init__.py
│           └── visualizers.py
│       └── __init__.py   
│       └── trainer.py  
│   └── utils                - this folder contains the utilies used by the model.
│       └── __init__.py    
│       └── utility.py   
|
|
├──  scripts  
│    └── config.yml             - here's the datasets folder that is responsible for all data handling.
│    └── config_ani.yml         - here's the data preprocess folder that is responsible for all data augmentation.
│    └── config_aniccx.yml  		   - here's the file to make dataloader.
│    └── config_aniccx_nonlinear_atten.yml      - here's the file 
│    └── config_aniccx_test.yml             - here's the datasets folder that is responsible for all data handling.
│    └── config_md17.yml            - here's the data preprocess folder that is responsible for all data augmentation.
│    └── launch.sh       		   - here's the file to make dataloader.
│    └── run.py                 - here's the file 
│    └── train.py               - here's the file 
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


## Future Work ##

TBD


## Contributing
Any kind of enhancement or contribution is welcomed.


## Acknowledgments

We're grateful for the help provided by [XXX], who are the authors of NewtonNet paper and THG Lab at Berkeley.
We're also grateful for the computing resources provided by Berkeley Artifical Intelligence Reserach and NERSC. 


