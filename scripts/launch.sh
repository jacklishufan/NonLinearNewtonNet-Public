CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py -c config_md17.yml -p md17 -n 32;
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py -c config_md17_3.yml -p md17 -n 16;
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py -c config_md17_4.yml -p md17 -n 4;