#!/bin/bash

N_GPUS=$(nvidia-smi -L | wc -l)
ROOT='/data/nas/workspace/jupyter/Data/'
TRAIN_LOG_DIR='runs'
DATASET_ARGS="--data-root ${ROOT}/datapng --data-fold 0 --data-split-file ${ROOT}/xy/stratified_10_fold.json"

#TRAIN_ARGS='@configs/retina.cfg'
TRAIN_ARGS='@configs/efficientdet.cfg --warn-unmatched-bbox --loss-box-weight=1 --fg-iou-threshold=0.5 --bg-iou-threshold=0.4 --arch=efficientdet-x2 --basenet=Efficientnet-b2 --sync-bn --weights=../9870_best_metric.model.pth'

mkdir -p $TRAIN_LOG_DIR

# for single GPU
#python train.py --log-dir $TRAIN_LOG_DIR $DATASET_ARGS $TRAIN_ARGS 2>&1 | tee -a $TRAIN_LOG_DIR/train.log
# for mulit-GPUs
python -m torch.distributed.launch --nproc_per_node $N_GPUS  train.py --log-dir $TRAIN_LOG_DIR $DATASET_ARGS $TRAIN_ARGS 2>&1 | tee -a $TRAIN_LOG_DIR/train.log
#python -m torch.distributed.launch --nproc_per_node $N_GPUS  train.py --resume ../cervical_cancer-f35929bd/runs/args.yml --batch-size=8 --max-epochs=150 $DATASET_ARGS $TRAIN_ARGS 2>&1 | tee -a $TRAIN_LOG_DIR/train.log