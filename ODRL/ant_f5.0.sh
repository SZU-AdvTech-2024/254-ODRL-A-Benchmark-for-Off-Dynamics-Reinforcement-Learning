#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --policy DARC --env ant-friction --shift_level 5.0 --seed 0 --mode 0 --dir ant_f5.0 >> train_log.txt
CUDA_VISIBLE_DEVICES=0 python train.py --policy VGDF --env ant-friction --shift_level 5.0 --seed 0 --mode 0 --dir ant_f5.0 >> train_log.txt
CUDA_VISIBLE_DEVICES=0 python train.py --policy PAR --env ant-friction --shift_level 5.0 --seed 0 --mode 0 --dir ant_f5.0 >> train_log.txt
CUDA_VISIBLE_DEVICES=0 python train.py --policy SAC --env ant-friction --shift_level 5.0 --seed 0 --mode 0 --dir ant_f5.0 >> train_log.txt
CUDA_VISIBLE_DEVICES=0 python train.py --policy SAC_IW --env ant-friction --shift_level 5.0 --seed 0 --mode 0 --dir ant_f5.0 >> train_log.txt

CUDA_VISIBLE_DEVICES=0 python train.py --policy CQL_SAC --env ant-friction --shift_level 0.5 --seed 1 --mode 1 --dir runs