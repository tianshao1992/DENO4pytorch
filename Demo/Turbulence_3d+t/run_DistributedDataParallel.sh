#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5, OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12356 ./Demo/Turbulence_3d+t/run_train_DDP.py --model_name Transformer+40_1e-9 --batch_size=3 --total_epoch=81 --weight_decay=0 &

CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12358 ./Demo/Turbulence_3d+t/run_train_DDP.py --model_name FNO+40 --batch_size=12 --total_epoch=81 --weight_decay=1e-8 &

